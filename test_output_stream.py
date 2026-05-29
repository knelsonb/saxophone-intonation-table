"""Acceptance net for the v0.6.3 engine OUTPUT mirror (parity Sprint 1).

The output path in ``sax_audio_engine`` mirrors the long-hardened input path:
worker-thread open with a join timeout + cancelled-flag orphan disposal,
sample-rate negotiation, device-vanished -> DEVICE_DISCONNECTED, hot-plug
auto-recover, and a SEPARATE ``_out_transitioning`` guard so output lifecycle
can never corrupt the input state machine (D5). This file is the real
acceptance net Gandalf asked for: it drives that lifecycle with a fake
``sounddevice`` so no PortAudio / audio hardware is required.

Two fake-installation patterns, matching the existing suite:
  * module-level fake ``sounddevice`` installed BEFORE importing the engine
    (so AUDIO_OK is True at import) — same as test_hotplug_recovery.py.
  * the orphan-disposal test drives ``_try_open_output`` directly with a slow
    fake OutputStream — same shape as test_orphan_stream_cleanup.py.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Install a fake sounddevice BEFORE importing the engine. Extends the input
# fake with the OUTPUT surface: OutputStream, check_output_settings, and
# max_output_channels on the device dicts.
# ---------------------------------------------------------------------------
_fake_sd = types.ModuleType('sounddevice')

_state: dict = {
    'devices': [],
    'hostapis': [{'name': 'Windows WASAPI'}],
    'default_device': (None, None),   # (input_idx, output_idx)
    'out_opens': [],                  # (device_index, samplerate) per OutputStream
    'fail_open': False,
}


def _query_devices(idx=None):
    if idx is None:
        return list(_state['devices'])
    return _state['devices'][idx]


def _query_hostapis():
    return list(_state['hostapis'])


class _DefaultNs:
    @property
    def device(self):
        return _state['default_device']


_fake_sd.default = _DefaultNs()


def _check_output_settings(device=None, channels=None, dtype=None,
                           samplerate=None):
    return None   # accept every rate so negotiation always yields candidates


def _check_input_settings(device=None, channels=None, dtype=None,
                          samplerate=None):
    return None


class _FakeOutputStream:
    def __init__(self, samplerate, blocksize, channels, dtype, callback,
                 device):
        _state['out_opens'].append((device, int(samplerate)))
        if _state['fail_open']:
            raise RuntimeError('simulated output open failure')
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeInputStream:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


_fake_sd.query_devices = _query_devices
_fake_sd.query_hostapis = _query_hostapis
_fake_sd.check_output_settings = _check_output_settings
_fake_sd.check_input_settings = _check_input_settings
_fake_sd.OutputStream = _FakeOutputStream
_fake_sd.InputStream = _FakeInputStream

sys.modules['sounddevice'] = _fake_sd

if 'sax_audio_engine' in sys.modules:
    del sys.modules['sax_audio_engine']
_engine_mod = importlib.import_module('sax_audio_engine')

AudioEngine = _engine_mod.AudioEngine
AudioEngineState = _engine_mod.AudioEngineState
AudioEngineError = _engine_mod.AudioEngineError
DeviceSelection = _engine_mod.DeviceSelection


# ---------------------------------------------------------------------------
# Device descriptors.
# ---------------------------------------------------------------------------
_SPEAKERS = {
    'name': 'Speakers (Realtek)',
    'hostapi': 0,
    'max_input_channels': 0,
    'max_output_channels': 2,
    'default_samplerate': 48000.0,
}
_FIIO_OUT = {
    'name': 'FIIO DSP Audio',
    'hostapi': 0,
    'max_input_channels': 0,
    'max_output_channels': 2,
    'default_samplerate': 48000.0,
}


def _reset(devices, *, default_out=None, fail_open=False):
    _state['devices'] = list(devices)
    _state['out_opens'] = []
    _state['fail_open'] = fail_open
    _state['default_device'] = (None, default_out)


def _new_engine():
    """A fresh engine. AudioEngine() does not open anything in __init__."""
    return AudioEngine()


# ---------------------------------------------------------------------------
# 0. Sanity — the fake makes the engine think audio is available.
# ---------------------------------------------------------------------------
def test_engine_loaded_with_fake_sounddevice():
    assert _engine_mod.AUDIO_OK, 'fake sounddevice failed to load'


def test_query_output_devices_filters_to_output_capable():
    _reset([
        {'name': 'A Mic', 'hostapi': 0, 'max_input_channels': 1,
         'max_output_channels': 0, 'default_samplerate': 44100.0},
        _SPEAKERS,
    ])
    outs = _engine_mod.query_output_devices()
    names = [d.name for d in outs]
    assert 'Speakers (Realtek)' in names, 'output-capable device must appear'
    assert 'A Mic' not in names, 'input-only device must be filtered out'


# ---------------------------------------------------------------------------
# 1. open_output_device — success path.
# ---------------------------------------------------------------------------
def test_open_default_output_reaches_running():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    ok = eng.open_output_device(None)
    assert ok is True, 'opening the default output should succeed'
    assert eng.output_running is True
    assert eng.active_output_device is not None
    assert eng.active_output_device.name == 'Speakers (Realtek)'
    assert _state['out_opens'], 'an OutputStream should have been constructed'
    # opened mono float32 per the engine contract
    dev_idx, sr = _state['out_opens'][-1]
    assert sr > 0


def test_open_output_by_selection_resolves_named_device():
    _reset([_SPEAKERS, _FIIO_OUT], default_out=0)
    eng = _new_engine()
    ok = eng.open_output_device(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    assert ok is True
    assert eng.active_output_device.name == 'FIIO DSP Audio', (
        'selection must resolve to the named device, not the default')


def test_open_output_with_no_devices_fails_no_device():
    _reset([])
    eng = _new_engine()
    ok = eng.open_output_device(None)
    assert ok is False
    assert eng.output_running is False
    assert eng.last_output_error == AudioEngineError.NO_DEVICE
    assert _state['out_opens'] == [], 'must not attempt to open with no devices'


def test_open_output_failure_does_not_touch_input_state():
    """D5: the separate _out_transitioning guard + output status fields mean
    an output failure must NEVER mutate the input state machine."""
    _reset([])
    eng = _new_engine()
    input_state_before = eng.state
    input_err_before = eng.last_error
    eng.open_output_device(None)   # fails NO_DEVICE on the OUTPUT side
    assert eng.state == input_state_before, (
        'output open must not change the input engine state')
    assert eng.last_error == input_err_before, (
        'output failure must not write the input last_error field')


# ---------------------------------------------------------------------------
# 2. stop_output.
# ---------------------------------------------------------------------------
def test_stop_output_tears_down_and_clears_running():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    assert eng.output_running is True
    eng.stop_output()
    assert eng.output_running is False
    assert eng.last_output_error == AudioEngineError.NONE


def test_stop_output_preserves_active_device_as_last_selection():
    """Persistence invariant (war-council-verified — sax_audio_engine.py:532):
    stop_output() flips LIVENESS (output_running -> False) but PRESERVES
    active_output_device, which is the last user SELECTION and the hot-plug
    recovery target — not a liveness flag. Clearing it here would silently
    break auto-recovery; this locks it so a future 'cleanup' can't regress it.
    (The input side is symmetric: _teardown_stream nulls _stream but leaves
    active_device; that persistence is exercised by test_hotplug_recovery.py's
    retry/hotplug paths, which recover via the pinned device.)"""
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    selected = eng.active_output_device
    assert selected is not None and eng.output_running is True
    eng.stop_output()
    assert eng.output_running is False, "liveness (output_running) must flip off"
    assert eng.active_output_device is selected, (
        "active_output_device must PERSIST across stop (last selection / "
        "recovery target); only output_running is liveness")


def test_open_stop_reopen_output_cycle_is_clean():
    """A full open -> stop -> reopen cycle must leave the engine running again —
    a clean stop never wedges the output lifecycle."""
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    assert eng.open_output_device(None) is True
    eng.stop_output()
    assert eng.output_running is False
    assert eng.open_output_device(None) is True, "reopen after a clean stop must succeed"
    assert eng.output_running is True


# ---------------------------------------------------------------------------
# 3. Test tone — the Sprint-1 acceptance vehicle.
# ---------------------------------------------------------------------------
def test_start_test_tone_returns_none_when_not_running():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    # No output stream open yet → nothing would be heard.
    assert eng.start_test_tone(440.0) is None
    assert eng.mixer.active_sources() == 0


def test_start_test_tone_registers_on_mixer_when_running():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    handle = eng.start_test_tone(440.0)
    assert handle is not None
    assert eng.mixer.active_sources() == 1, 'tone should be registered once'


def _drain_release(mixer, blocks: int = 200) -> None:
    """Render enough blocks to complete a test-tone release fade. stop/replace
    now RELEASE the tone (a ~60 ms click-free fade) instead of hard-cutting it,
    so the Mixer reaps the source a block after its tail reaches zero — pump it
    until that happens."""
    mb = getattr(mixer, '_max_block', 1024)
    buf = np.zeros(mb, dtype=np.float32)
    for _ in range(blocks):
        mixer.render(buf, mb)


def test_start_test_tone_is_idempotent_not_stacking():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)
    eng.start_test_tone(660.0)   # replaces: releases the first (it fades out)
    # A replace releases the previous tone (now fading) and registers the new
    # one, so two may briefly coexist; once the released one fades it is reaped
    # and the count settles at exactly one. The invariant is "doesn't stack
    # unbounded", not "instantly one".
    _drain_release(eng.mixer)
    assert eng.mixer.active_sources() == 1, (
        'repeated start_test_tone must settle to a single source, not stack')


def test_stop_test_tone_releases_then_reaps():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)
    eng.stop_test_tone()
    # stop now RELEASES the tone (click-free fade) rather than hard-cutting it,
    # so it stays registered, fading, until the Mixer reaps it.
    assert eng.mixer.active_sources() == 1, (
        'stop_test_tone releases (fades) the tone; it lingers until faded out')
    assert eng._test_tone_handle is None, (
        'the engine clears its handle immediately even though the tone fades')
    _drain_release(eng.mixer)
    assert eng.mixer.active_sources() == 0, (
        'the faded test tone must be reaped by the Mixer')


def test_get_sounding_output_midis_empty_for_unpitched_test_tone():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)   # TestToneSource is unpitched (midi=None)
    assert eng.get_sounding_output_midis() == frozenset(), (
        'the test tone is unpitched, so it must not appear in sounding MIDIs')


# ---------------------------------------------------------------------------
# 4. Output callback — pulls the mixer, never raises.
# ---------------------------------------------------------------------------
def test_output_callback_pulls_mixer_block():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)
    sr = int(eng.output_samplerate)
    frames = int(eng.output_block_size)
    cb = eng._make_output_callback(sr)
    outdata = np.zeros((frames, 1), dtype=np.float32)
    cb(outdata, frames, None, None)   # PortAudio-shaped (frames, channels)
    assert np.any(np.abs(outdata) > 1e-4), (
        'callback should fill the device buffer from the mixer test tone')


def test_output_callback_emits_silence_on_mixer_error():
    """The output callback must never raise into PortAudio; on any internal
    error it fills the block with silence rather than garbage."""
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    cb = eng._make_output_callback(int(eng.output_samplerate))

    # Force the mixer to raise from render to exercise the guard.
    def boom(*a, **k):
        raise RuntimeError('mixer exploded')
    eng.mixer.render = boom

    frames = int(eng.output_block_size)
    outdata = np.full((frames, 1), 7.0, dtype=np.float32)
    cb(outdata, frames, None, None)   # must NOT raise
    assert np.all(outdata == 0.0), 'callback must zero the block on error'


# ---------------------------------------------------------------------------
# 5. Hot-plug — mirror of the input refresh_devices contract.
# ---------------------------------------------------------------------------
def test_output_hotplug_disconnect_while_running():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    assert eng.output_running is True
    # The active device vanishes from the enumeration.
    _state['devices'] = []
    eng.refresh_output_devices()
    assert eng.output_running is False
    assert eng.last_output_error == AudioEngineError.DEVICE_DISCONNECTED


def test_output_hotplug_new_device_auto_recovers():
    _reset([], default_out=None)
    eng = _new_engine()
    eng.open_output_device(None)            # fails NO_DEVICE
    assert eng.output_running is False
    assert eng.last_output_error == AudioEngineError.NO_DEVICE
    # A device is plugged in; the poller must auto-recover to RUNNING.
    _state['devices'] = [_SPEAKERS]
    _state['default_device'] = (None, 0)
    eng.refresh_output_devices()
    assert eng.output_running is True, (
        f'auto-recover should open the new output device, '
        f'state err={eng.last_output_error}')
    assert eng.active_output_device.name == 'Speakers (Realtek)'


# ---------------------------------------------------------------------------
# 6. Orphan disposal — a slow open that overruns the join timeout must be
#    stopped + closed by the worker, not leaked. Mirror of
#    test_orphan_stream_cleanup for the OUTPUT path.
# ---------------------------------------------------------------------------
def test_output_open_timeout_disposes_orphan_stream():
    timeout = _engine_mod.HOST_API_OPEN_TIMEOUT_S
    created: list = []

    class _SlowOutputStream:
        def __init__(self, **kwargs):
            time.sleep(timeout + 1.5)   # overrun the join timeout
            self.started = self.stopped = self.closed = False
            created.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    dev = _engine_mod.query_output_devices()[0]

    orig = _engine_mod.sd.OutputStream
    _engine_mod.sd.OutputStream = _SlowOutputStream
    try:
        t0 = time.monotonic()
        ok, kind, msg = eng._try_open_output(dev, samplerate=48000)
        elapsed = time.monotonic() - t0
    finally:
        _engine_mod.sd.OutputStream = orig

    assert not ok, 'a wedged open must report failure'
    assert kind == AudioEngineError.HOSTAPI_FAILURE, kind
    assert 'timed out' in msg, msg
    assert elapsed < timeout + 1.0, (
        f'_try_open_output should return near the {timeout:.1f}s timeout, '
        f'not wait for the slow open ({elapsed:.2f}s)')

    # Wait for the worker to finish its slow open and dispose the orphan.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if created and created[-1].closed:
            break
        time.sleep(0.05)
    assert created, 'the worker never constructed a stream'
    s = created[-1]
    assert s.started, 'worker should have started the stream before cancel check'
    assert s.stopped, 'orphaned output stream was not stopped'
    assert s.closed, 'orphaned output stream was not closed'


# ---------------------------------------------------------------------------
# 7. Open failure (constructor raises) is reported, not raised.
# ---------------------------------------------------------------------------
def test_open_output_construct_failure_reports_state():
    _reset([_SPEAKERS], default_out=0, fail_open=True)
    eng = _new_engine()
    ok = eng.open_output_device(None)
    assert ok is False, 'a failing OutputStream construct must yield False'
    assert eng.output_running is False
    assert eng.last_output_error != AudioEngineError.NONE


# ---------------------------------------------------------------------------
# 8. D3 coordination END-TO-END — the payoff: Sprint-1 coordination firing
#    against a real sounding source, via Gandalf's deterministic
#    coordination_step seam (no mic audio needed).
# ---------------------------------------------------------------------------
class _PitchedSource:
    """A registerable MixerSource that reports a sounding MIDI (silent audio is
    fine — active_midi is the wire D3 reads)."""

    def __init__(self, midi):
        self._midi = midi

    def render(self, out, frames, t0):
        return  # contributes no audio; only its active_midi matters here

    @property
    def active_midi(self):
        return self._midi


class _DuckConsumer:
    def __init__(self):
        self.level = 1.0
        self.history = []

    def set_duck_target(self, level):
        self.level = float(level)
        self.history.append(float(level))


def test_sounding_output_midis_reflects_pitched_source():
    eng = _new_engine()
    eng.mixer.register(_PitchedSource(60))
    eng.mixer.register(_PitchedSource(None))   # unpitched contributes nothing
    assert eng.get_sounding_output_midis() == frozenset({60})


def test_coordination_vote_excludes_and_ducks_on_sustained_match():
    """The end-to-end D3 contract: a sounding output MIDI is vote-excluded
    every hop, and sustained mic-detection of it (>=3 consecutive) ducks the
    output via the attached consumer — then clears on release."""
    eng = _new_engine()
    eng.mixer.register(_PitchedSource(60))
    consumer = _DuckConsumer()
    eng.attach_duck_consumer(consumer)

    # Vote-exclude is immediate (every hop the note is sounding).
    assert 60 in eng.coordination_step(60), "sounding MIDI must be vote-excluded"

    # Sustain the match: by >=3 consecutive hops the duck must engage.
    for _ in range(4):
        eng.coordination_step(60)
    assert consumer.level < 1.0, (
        f"sustained mic-bleed of the sounding note must duck the output, "
        f"level={consumer.level}")

    # Release: detection clears -> duck glides back toward open.
    for _ in range(40):
        eng.coordination_step(None)
    assert consumer.level == pytest.approx(1.0, abs=1e-3), (
        f"duck must release toward 1.0 after leakage clears, got {consumer.level}")


def test_coordination_inert_when_nothing_sounding():
    """Pre-drone (no pitched output source) the coordinator is INERT: no
    exclusion, no duck — so S1/S2 behaviour is unchanged. This locks that the
    D3 wiring is dormant until a drone/pitch-pipe actually sounds."""
    eng = _new_engine()
    consumer = _DuckConsumer()
    eng.attach_duck_consumer(consumer)
    excluded = eng.coordination_step(60)          # 60 detected, but nothing sounds
    assert excluded == frozenset(), "nothing sounding -> nothing to vote-exclude"
    for _ in range(5):
        eng.coordination_step(60)
    assert consumer.level == 1.0, "no sounding output -> duck never engages"


def test_detach_duck_consumer_stops_delivery():
    eng = _new_engine()
    eng.mixer.register(_PitchedSource(60))
    consumer = _DuckConsumer()
    eng.attach_duck_consumer(consumer)
    eng.detach_duck_consumer(consumer)
    for _ in range(5):
        eng.coordination_step(60)
    assert consumer.history == [], "a detached consumer must receive no duck targets"


# ---------------------------------------------------------------------------
# 9. Tape-deck input-recording tap (Sprint 4 — Gandalf's bounded-prealloc
#    seam). The deck is the FIRST input-side consumer (S1-3 were output): the
#    input hot path slice-assigns every mic frame into a bounded buffer while
#    armed. Driven via feed_input_frames (the real _on_input body) with a fake
#    OPEN input stream — no device, no PortAudio. Mirrors the output orphan
#    test, input-side. Pinned to Gandalf's landed API (engine 834-933 / 1486 /
#    1804-2011).
# ---------------------------------------------------------------------------
class _FakeOpenInputStream:
    """Stands in for an open mic stream — what input_running /
    start_input_recording gate on. The tap captures the frames we FEED, not
    anything from this object, so it only needs the stop/close teardown
    surface (so the orphan test can assert disposal)."""

    def __init__(self):
        self.stopped = False
        self.closed = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _recording_engine(samplerate=48000):
    """A fresh engine with a (fake) OPEN input stream so input_running is True
    and start_input_recording can arm. The capture buffer is sized from
    self.samplerate, so pin it to a known rate."""
    eng = _new_engine()
    eng.samplerate = samplerate
    eng._stream = _FakeOpenInputStream()
    return eng


def test_input_running_reflects_open_stream():
    eng = _new_engine()
    assert eng.input_running is False, "no stream -> not running (honest probe)"
    eng._stream = _FakeOpenInputStream()
    assert eng.input_running is True


def test_start_input_recording_false_when_input_not_open():
    eng = _new_engine()            # no input stream
    assert eng.start_input_recording(10.0) is False, "no mic -> no false 'recording'"
    assert eng.is_input_recording() is False


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_start_input_recording_rejects_nonpositive_or_nonfinite(bad):
    eng = _recording_engine()
    assert eng.start_input_recording(bad) is False
    assert eng.is_input_recording() is False


def test_record_feed_stop_is_byte_exact():
    """The core capture contract: every fed mic frame is captured UNMODIFIED
    (no gain/resample at the tap) and returned in order — feed K blocks, stop,
    assert the take equals their concatenation byte-for-byte."""
    eng = _recording_engine(samplerate=48000)
    assert eng.start_input_recording(10.0) is True
    assert eng.is_input_recording() is True
    rng = np.random.default_rng(0)
    fed = [rng.standard_normal(n).astype(np.float32) * 0.5
           for n in (480, 256, 1024, 17, 480)]
    for block in fed:
        eng.feed_input_frames(block)
    take, sr, truncated = eng.stop_input_recording()
    assert sr == 48000
    assert truncated is False
    assert np.array_equal(take, np.concatenate(fed)), (
        "the take must be the fed frames concatenated, byte-for-byte unmodified")
    assert eng.is_input_recording() is False, "stop disarms the tap"


def test_recorded_frame_count_grows_monotonically_then_resets():
    """recorded_frame_count is the continuous-capture observable: it grows by
    exactly the frames fed, then resets after stop."""
    eng = _recording_engine()
    eng.start_input_recording(10.0)
    assert eng.recorded_frame_count() == 0
    eng.feed_input_frames(np.zeros(100, dtype=np.float32))
    assert eng.recorded_frame_count() == 100
    eng.feed_input_frames(np.zeros(50, dtype=np.float32))
    assert eng.recorded_frame_count() == 150
    eng.stop_input_recording()
    assert eng.recorded_frame_count() == 0, "stop resets the counter"


def test_capture_continues_through_emissions_pause():
    """The tap sits BEFORE the emissions-pause gate (Sauron + Treebeard
    ratified): an A4-remap pitch-detection pause must NOT punch a hole in the
    take. Frames fed while paused are still captured."""
    eng = _recording_engine()
    eng.start_input_recording(10.0)
    eng.feed_input_frames(np.ones(100, dtype=np.float32) * 0.2)
    eng._emissions_paused = True                 # simulate the A4-remap pause
    eng.feed_input_frames(np.ones(64, dtype=np.float32) * 0.3)
    eng._emissions_paused = False
    eng.feed_input_frames(np.ones(36, dtype=np.float32) * 0.4)
    assert eng.recorded_frame_count() == 200, (
        "frames fed during an emissions pause must still be captured "
        "(the tap is before the pause gate)")
    take, _, truncated = eng.stop_input_recording()
    assert len(take) == 200 and truncated is False


def test_cap_hit_truncates_and_auto_disarms():
    """At deck_max_seconds the bounded buffer fills: the tap clamps at the cap,
    auto-disarms (is_input_recording flips False even before stop — the deck
    pump() reads that as 'cap hit'), and the take comes back truncated=True at
    exactly the cap length."""
    sr = 48000
    eng = _recording_engine(samplerate=sr)
    assert eng.start_input_recording(0.01) is True   # 0.01s cap -> 480 samples
    cap = int(0.01 * sr)
    eng.feed_input_frames(np.ones(300, dtype=np.float32) * 0.5)
    assert eng.is_input_recording() is True, "under cap -> still armed"
    eng.feed_input_frames(np.ones(400, dtype=np.float32) * 0.5)   # overruns cap
    assert eng.is_input_recording() is False, "hitting the cap must auto-disarm"
    take, _, truncated = eng.stop_input_recording()
    assert truncated is True, "a capped take must surface truncated=True"
    assert len(take) == cap, f"take clamps at the cap ({cap}), got {len(take)}"


def test_abrupt_teardown_mid_record_retains_partial_flagged_truncated():
    """The Android close-path race, locked input-side: a stream teardown while
    recording (device switch / stop) must disarm WITHOUT orphaning the partial
    take, close the stream, and surface truncated=True on the next stop."""
    eng = _recording_engine()
    eng.start_input_recording(10.0)
    eng.feed_input_frames(np.ones(123, dtype=np.float32) * 0.3)
    stream = eng._stream
    eng._teardown_stream()                       # abrupt close mid-record
    assert eng._stream is None, "teardown must drop the stream reference"
    assert eng.input_running is False
    assert eng.is_input_recording() is False, "teardown disarms the tap (no orphan armed flag)"
    assert stream.stopped and stream.closed, "the orphaned input stream must be stopped + closed"
    take, _, truncated = eng.stop_input_recording()
    assert len(take) == 123, "the partial take is RETAINED, not orphaned"
    assert truncated is True, "a take cut short by teardown must be flagged truncated"


def test_stop_engine_disarms_recording():
    """engine.stop() (the GUI close path) tears down the stream and disarms —
    never leaves an armed tap pointed at a dead stream."""
    eng = _recording_engine()
    eng.start_input_recording(10.0)
    eng.feed_input_frames(np.ones(64, dtype=np.float32) * 0.2)
    eng.stop()
    assert eng.is_input_recording() is False
    assert eng.input_running is False


def test_fresh_arm_discards_previous_take():
    """A new start_input_recording discards any prior unfetched take (single
    take, mic-only parity)."""
    eng = _recording_engine()
    eng.start_input_recording(10.0)
    eng.feed_input_frames(np.ones(500, dtype=np.float32) * 0.3)
    eng.start_input_recording(10.0)              # re-arm without stopping
    assert eng.recorded_frame_count() == 0, "re-arm starts a fresh take"
    eng.feed_input_frames(np.ones(80, dtype=np.float32) * 0.1)
    take, _, _ = eng.stop_input_recording()
    assert len(take) == 80, "only the new take's frames are returned"


def test_input_tap_adds_no_scaling_alloc_when_armed():
    """The deck tap is slice-assign only -> ZERO per-call allocation. Measured
    as the DELTA between armed and unarmed feeds of the SAME block: the input
    pipeline's pre-existing per-call work (ring write + YIN, the #D6 baseline)
    is identical in both, so any retained difference is the tap itself. It must
    not grow per call. (Isolation gate, not an absolute floor — numpy view-
    boxing is a constant an absolute floor would flake on; same spirit as the
    mixer/drone non-scaling gates.)"""
    import tracemalloc
    eng = _recording_engine()
    block = np.ascontiguousarray(np.zeros(512, dtype=np.float32))
    for _ in range(5):
        eng.feed_input_frames(block)             # warm lazy allocs / caches
    tracemalloc.start()
    a = tracemalloc.take_snapshot()
    for _ in range(40):
        eng.feed_input_frames(block)             # unarmed baseline
    b = tracemalloc.take_snapshot()
    eng.start_input_recording(60.0)              # big cap — never truncates here
    for _ in range(5):
        eng.feed_input_frames(block)             # warm armed path
    c = tracemalloc.take_snapshot()
    for _ in range(40):
        eng.feed_input_frames(block)             # armed: pipeline + tap
    d = tracemalloc.take_snapshot()
    tracemalloc.stop()

    def _delta(x, y):
        return sum(s.size_diff for s in y.compare_to(x, "filename"))

    unarmed = _delta(a, b)
    armed = _delta(c, d)
    assert armed <= unarmed + 16384, (
        f"the armed deck tap retains allocation beyond the unarmed pipeline "
        f"baseline (unarmed={unarmed} armed={armed}) — capture must be "
        f"slice-assign into the preallocated sink, never per-call growth")


def test_input_unwrap_reuses_scratch_and_is_chronological():
    """The ring -> contiguous unwrap reuses a preallocated scratch instead of
    np.concatenate (which allocated a fresh ~64 KB array every callback), and
    get_buf_snapshot still returns samples in chronological order across the
    ring's wrap boundary."""
    eng = _recording_engine(samplerate=48000)
    b1 = np.full(512, 0.10, dtype=np.float32)
    b2 = np.full(512, 0.20, dtype=np.float32)
    eng.feed_input_frames(b1)
    buf_obj_1 = eng._buf
    eng.feed_input_frames(b2)
    buf_obj_2 = eng._buf
    assert buf_obj_1 is buf_obj_2, (
        "the unwrap must reuse ONE preallocated scratch, not reallocate")
    snap = eng.get_buf_snapshot()
    # Chronological: the newest block sits at the tail, the prior block before it.
    assert np.allclose(snap[-512:], 0.20), "latest block must be at the tail"
    assert np.allclose(snap[-1024:-512], 0.10), "the prior block must precede it"

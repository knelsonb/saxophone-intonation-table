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


def test_start_test_tone_is_idempotent_not_stacking():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)
    eng.start_test_tone(660.0)   # replaces, does not stack
    assert eng.mixer.active_sources() == 1, (
        'a second start_test_tone must replace, not add a second source')


def test_stop_test_tone_unregisters():
    _reset([_SPEAKERS], default_out=0)
    eng = _new_engine()
    eng.open_output_device(None)
    eng.start_test_tone(440.0)
    eng.stop_test_tone()
    assert eng.mixer.active_sources() == 0


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

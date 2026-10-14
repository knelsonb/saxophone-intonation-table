"""Drone output-pitch + output-rate-change validation (uses sim_harness).

Locks the device-switch pitch fix: when the output format changes, the mixer
calls the drone source's on_format_changed, which builds a new TinySoundFont
synth at the new rate (TSF fixes its rate at construction) and strikes the
held note on it, so the drone holds its target pitch instead of shifting by
12*log2(new/old) semitones (up to +2547 cents on a 44.1k->192k switch) --
catastrophic for a tuning reference. Also locks that shutdown() nulls the
source (no reuse of a closed synth).

Needs tinysoundfont + the bundled SF2 (the venv); skips cleanly without them.
The drone uses real instrument voices (vibrato / sample detune), so pitch
tolerances are generous -- the point is to catch the RATE-RATIO bug (hundreds
to thousands of cents), not to assert a sub-cent reference (the pitch PIPES are
the exact reference; see test_pitch_pipes)."""
from __future__ import annotations

import pytest

pytest.importorskip("tinysoundfont")
import sim_harness as H  # noqa: E402
from intonation.audio.sources import drone as D  # noqa: E402


class _FakeMixer:
    def __init__(self, sr):
        self.max_block = 2048
        self.samplerate = sr

    def register(self, s):
        pass

    def unregister(self, s):
        pass


def _drone(sr, voice="strings"):
    c = D.DroneController(_FakeMixer(sr), sr, voice_id=voice, volume=0.9,
                          a4=440.0, reference_midi=69, semitones=0)
    c.set_enabled(True)
    return c


def _pitch(src, sr):
    return H.fft_peak_hz(H.render_stream(src, 16384, warmup=14), sr)


def test_drone_sounds_near_a4_at_base_rate():
    c = _drone(44100)
    try:
        f = _pitch(c._source, 44100)
        assert abs(H.cents(f, 440.0)) < 25.0, f"drone A4 -> {f:.2f} Hz"
    finally:
        c.shutdown()


@pytest.mark.parametrize("sr", [48000, 88200, 192000])
def test_drone_pitch_rate_invariant_after_a_format_change(sr):
    c = _drone(44100)
    try:
        src = c._source
        assert src.samplerate == 44100
        src.on_format_changed(sr, 2048)
        # The same source keeps sounding, now synthesised at the new rate ...
        assert c._source is src and src.samplerate == sr
        # ... and the pitch must stay ~A4, NOT shift by the rate ratio
        # (the bug: 440*sr/44100 -> +147 .. +2547 cents).
        f = _pitch(src, sr)
        err = H.cents(f, 440.0)
        assert abs(err) < 40.0, (
            f"drone at {sr} Hz -> {f:.2f} Hz ({err:+.1f} ct) -- rate-shift bug?")
    finally:
        c.shutdown()


def test_drone_format_change_is_noop_on_same_or_invalid_format():
    c = _drone(44100)
    try:
        src0 = c._source
        syn0 = src0._syn
        src0.on_format_changed(44100, 2048)   # same format -> no new synth
        assert src0._syn is syn0
        src0.on_format_changed(0, 2048)       # invalid -> no-op
        src0.on_format_changed(-48000, 2048)
        src0.on_format_changed(48000, 0)
        assert src0._syn is syn0 and src0.samplerate == 44100
    finally:
        c.shutdown()


def test_drone_shutdown_nulls_source_then_reenable_rebuilds():
    c = _drone(44100)
    c.shutdown()
    assert c._source is None, "shutdown must null _source (no closed-synth reuse)"
    c.set_enabled(True)
    assert c._source is not None, "re-enable must build a fresh source"
    assert c._source.samplerate == 44100
    c.shutdown()


class _FakeEngine:
    """Minimal engine surface: D3 duck wiring + the output_running probe."""

    def __init__(self, output_running):
        self.output_running = output_running

    def attach_duck_consumer(self, src):
        pass

    def detach_duck_consumer(self, src):
        pass


def test_drone_disable_clears_active_midi_when_output_stopped():
    """Disabling the drone while the output stream is stopped must clear
    active_midi at once: no render runs to drain the tail, so otherwise the
    sounding MIDI sticks and the D3 coordinator vote-excludes it forever."""
    c = D.DroneController(_FakeMixer(44100), 44100, voice_id="strings",
                          a4=440.0, reference_midi=69, semitones=0,
                          engine=_FakeEngine(output_running=False))
    c.set_enabled(True)
    assert c._source.active_midi == 69
    c.set_enabled(False)
    assert c._source.active_midi is None, (
        "output stopped -> disable must force-silence (no stuck vote-exclude)")
    c.shutdown()


def test_drone_disable_preserves_tail_when_output_running():
    """With output running, disable does NOT force-silence — render drains the
    tail naturally, so active_midi is still reported right after disable (the
    tail is audible and correctly vote-excluded for ~50ms)."""
    c = D.DroneController(_FakeMixer(44100), 44100, voice_id="strings",
                          a4=440.0, reference_midi=69, semitones=0,
                          engine=_FakeEngine(output_running=True))
    c.set_enabled(True)
    c.set_enabled(False)
    assert c._source.active_midi == 69, (
        "output running -> tail preserved (render drains it), not force-silenced")
    c.shutdown()

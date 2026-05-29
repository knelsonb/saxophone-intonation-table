"""Drone output-pitch + samplerate-rebuild validation (uses sim_harness).

Locks the device-switch pitch fix: DroneController.set_samplerate now rebuilds
the TinySoundFont source at the new rate, so the drone holds its target pitch
instead of shifting by 12*log2(new/old) semitones (up to +2547 cents on a
44.1k->192k switch) -- catastrophic for a tuning reference. Also locks that
shutdown() nulls the source (no reuse of a closed synth).

Needs tinysoundfont + the bundled SF2 (the venv); skips cleanly without them.
The drone uses real instrument voices (vibrato / sample detune), so pitch
tolerances are generous -- the point is to catch the RATE-RATIO bug (hundreds
to thousands of cents), not to assert a sub-cent reference (the pitch PIPES are
the exact reference; see test_pitch_pipes)."""
from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("tinysoundfont")
import sim_harness as H          # noqa: E402
import sax_drone as D            # noqa: E402


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
def test_drone_pitch_rate_invariant_after_set_samplerate(sr):
    c = _drone(44100)
    try:
        assert c._source.samplerate == 44100
        c.set_samplerate(sr)
        # The source must be REBUILT at the new rate ...
        assert c._source.samplerate == sr, "set_samplerate did not rebuild source"
        # ... and the pitch must stay ~A4, NOT shift by the rate ratio
        # (the bug: 440*sr/44100 -> +147 .. +2547 cents).
        f = _pitch(c._source, sr)
        err = H.cents(f, 440.0)
        assert abs(err) < 40.0, (
            f"drone at {sr} Hz -> {f:.2f} Hz ({err:+.1f} ct) -- rate-shift bug?")
    finally:
        c.shutdown()


def test_drone_set_samplerate_is_noop_on_same_or_invalid_rate():
    c = _drone(44100)
    try:
        src0 = c._source
        c.set_samplerate(44100)   # same rate -> no rebuild
        assert c._source is src0
        c.set_samplerate(0)       # invalid -> no-op
        assert c._source is src0
        c.set_samplerate(-48000)  # invalid -> no-op
        assert c._source is src0
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

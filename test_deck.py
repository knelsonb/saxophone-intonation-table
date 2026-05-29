"""Acceptance net for ``sax_deck`` (parity Sprint 4 — the tape deck).

Pre-written against Sauron's FIRM sax_deck.py contract (channel msgs 3549/3550)
and Gandalf's FIRM engine input-tap contract (3541/3544), skip-guarded with
``importorskip('sax_deck')`` so it stays inert until the module lands and then
auto-activates with zero churn — the test-seam-handshake discipline that ran
zero-churn for the mixer / D3 / metronome / drone lanes.

Two firm tiers (both run on system-python — sax_deck is numpy + stdlib ``wave``
+ sax_mixer, no PyQt6 / sounddevice / tinysoundfont):

  * WAV tier — the pure module-level ``write_wav`` / ``read_wav`` (stdlib wave,
    float32<->int16): header correctness + roundtrip within int16 quantization
    + out-of-range clipping. This is the "export writes a valid WAV that opens
    externally" half of the acceptance criteria.
  * STATE-MACHINE tier — DeckController transitions incl. the interrupts,
    driven by a fake engine that mirrors the input-tap surface (no real mic).
    idle/recording/have-take, empty-take, re-record-discard, truncation,
    can_record() honesty, cap-hit finalize via pump().

The play()/DeckPlaybackSource/playback-end transitions are in the PART 2
section below — held pending Sauron's answers to 3 zero-churn questions
(DeckPlaybackSource ctor sig; whether play() needs engine.output_running;
on_state_changed arity). They land here the moment those are confirmed.
"""
from __future__ import annotations

import os
import wave

import numpy as np
import pytest

# int16 PCM scale sax_deck encodes/decodes with (symmetric: +1.0<->+32767).
# One LSB is 1/32767; round-trip error is bounded by half an LSB.
_PCM_LSB = 1.0 / 32767.0

sax_deck = pytest.importorskip(
    "sax_deck", reason="sax_deck not landed yet (Sauron's S4 lane); auto-activates on land")

from sax_mixer import Mixer  # noqa: E402  (after the skip-guard by design)


# ===========================================================================
# WAV tier — pure write_wav / read_wav (stdlib wave, float32 <-> int16).
# ===========================================================================
def test_write_wav_header_is_mono_16bit_pcm(tmp_path):
    """Exported take must be a mono 16-bit PCM WAV at the take's samplerate,
    with one frame per sample — the shape an external player expects."""
    sr = 44100
    samples = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
    p = tmp_path / "take.wav"
    sax_deck.write_wav(str(p), samples, sr)
    assert p.exists(), "write_wav must create the file"
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1, "mic take is mono (parity)"
        assert w.getsampwidth() == 2, "16-bit PCM"
        assert w.getframerate() == sr, "framerate must be the take samplerate"
        assert w.getnframes() == len(samples), "one frame per captured sample"


def test_wav_roundtrip_within_int16_quantization(tmp_path):
    """write_wav -> read_wav must reproduce the samples within one int16
    quantization step (1/32768) — the "playback reproduces the take" core
    acceptance, modulo the unavoidable 16-bit encode."""
    sr = 48000
    samples = (np.sin(2 * np.pi * 220 * np.arange(2400) / sr) * 0.8).astype(np.float32)
    p = tmp_path / "rt.wav"
    sax_deck.write_wav(str(p), samples, sr)
    back, got_sr = sax_deck.read_wav(str(p))
    assert got_sr == sr, "read_wav must report the stored samplerate"
    assert back.dtype == np.float32, "read_wav returns float32"
    assert len(back) == len(samples), "no samples lost in the roundtrip"
    assert np.max(np.abs(back - samples)) <= 1.0 / 32768 + 1e-7, (
        "roundtrip error must stay within one int16 quantization step")


def test_wav_clips_out_of_range_samples(tmp_path):
    """float32 outside [-1, 1] must CLIP at the int16 boundary, never wrap
    (a wrap would turn a loud peak into a full-scale opposite-sign glitch)."""
    sr = 44100
    samples = np.array([2.0, -2.0, 1.5, 0.0, -1.0], dtype=np.float32)
    p = tmp_path / "clip.wav"
    sax_deck.write_wav(str(p), samples, sr)
    back, _ = sax_deck.read_wav(str(p))
    # Clipping must be SIGN-PRESERVING (never wrap a loud peak to the opposite
    # rail). int16 is asymmetric ([-32768, 32767]); read_wav scales by 32767,
    # so a clipped -1.0 full-scale decodes to -32768/32767 ≈ -1.00003 — allow
    # that hair of overshoot. A WRAP would instead FLIP the sign, which the
    # magnitude checks below catch.
    assert np.all(back <= 1.0 + 1e-4) and np.all(back >= -1.0 - 1e-4), (
        "out-of-range samples must clip (sign-preserving), not wrap")
    assert back[0] > 0.99, "+2.0 clips toward +full-scale (no wrap)"
    assert back[2] > 0.99, "+1.5 clips toward +full-scale (no wrap)"
    assert back[1] < -0.99, "-2.0 clips toward -full-scale (no wrap)"
    assert back[3] == pytest.approx(0.0, abs=1e-4), "silence stays silent"


def test_read_wav_recovers_written_samplerate(tmp_path):
    """Different capture rates must round-trip their samplerate independently
    of the sample values (the deck resamples capture->output at playback, so
    the stored rate must be faithful)."""
    for sr in (44100, 48000, 96000):
        samples = (np.cos(np.arange(500) / 7.0) * 0.3).astype(np.float32)
        p = tmp_path / f"sr_{sr}.wav"
        sax_deck.write_wav(str(p), samples, sr)
        _, got = sax_deck.read_wav(str(p))
        assert got == sr, f"samplerate {sr} must round-trip, got {got}"


# ===========================================================================
# STATE-MACHINE tier — DeckController transitions (fake engine, no real mic).
# ===========================================================================
class _FakeEngine:
    """Minimal stand-in for the engine input-tap surface DeckController drives
    (Gandalf's firmed API, 3541/3544). No real mic / sounddevice."""

    def __init__(self, *, input_open=True, take=None, sr=48000, truncated=False):
        self.input_running = bool(input_open)
        self.output_running = True
        self._armed = False
        self._take = np.zeros(0, dtype=np.float32) if take is None else np.asarray(take, dtype=np.float32)
        self._sr = sr
        self._truncated = truncated
        self.calls: list[tuple] = []

    def start_input_recording(self, max_seconds):
        self.calls.append(("start", max_seconds))
        if not self.input_running:
            return False
        self._armed = True
        return True

    def stop_input_recording(self):
        self.calls.append(("stop", None))
        self._armed = False
        return (np.array(self._take, dtype=np.float32), self._sr, self._truncated)

    def is_input_recording(self) -> bool:
        return self._armed

    def recorded_frame_count(self) -> int:
        return len(self._take) if self._armed else 0


def _deck(engine, **kw):
    return sax_deck.DeckController(Mixer(max_block=2048), 48000, engine=engine, **kw)


def _deck_and_mixer(engine, **kw):
    """Like _deck but also hand back the Mixer, so playback tests can render
    and inspect what the registered DeckPlaybackSource actually produces."""
    m = Mixer(max_block=2048)
    return sax_deck.DeckController(m, 48000, engine=engine, **kw), m


def test_starts_idle():
    d = _deck(_FakeEngine())
    assert d.state == "idle", "a fresh deck must be idle (no phantom take)"


def test_idle_to_recording_on_start_record():
    eng = _FakeEngine(input_open=True)
    d = _deck(eng)
    assert d.start_record() is True
    assert d.state == "recording"
    assert eng.is_input_recording() is True, "start_record must arm the engine tap"


def test_start_record_false_when_input_not_open_stays_idle():
    """The honesty rule: if the mic can't open, start_record() returns False
    and the deck stays idle — no false 'recording' state for the dot to show."""
    eng = _FakeEngine(input_open=False)
    d = _deck(eng)
    assert d.start_record() is False
    assert d.state == "idle"
    assert eng.is_input_recording() is False


def test_stop_recording_to_have_take():
    take = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    eng = _FakeEngine(input_open=True, take=take, sr=48000, truncated=False)
    d = _deck(eng)
    d.start_record()
    d.stop()
    assert d.state == "have-take"
    assert d.take is not None
    assert np.array_equal(d.take, take), "the take must be the captured samples, unmodified"
    assert d.take_samplerate == 48000
    assert d.take_truncated is False


def test_stop_recording_with_empty_take_returns_to_idle():
    """A record that captured nothing must drop back to idle, not leave a
    zero-length phantom have-take the user could 'play'."""
    eng = _FakeEngine(input_open=True, take=np.zeros(0, dtype=np.float32))
    d = _deck(eng)
    d.start_record()
    d.stop()
    assert d.state == "idle"


def test_have_take_start_record_discards_old_take_and_rerecords():
    eng = _FakeEngine(input_open=True, take=np.ones(500, dtype=np.float32) * 0.3)
    d = _deck(eng)
    d.start_record()
    d.stop()
    assert d.state == "have-take"
    assert d.start_record() is True
    assert d.state == "recording", "re-record from have-take discards the old take"


def test_truncated_flag_surfaces_as_take_truncated():
    """If the engine hit deck_max_seconds and truncated the capture, the deck
    must surface it so the GUI can warn — not silently present a clipped take."""
    eng = _FakeEngine(input_open=True, take=np.ones(10, dtype=np.float32) * 0.1, truncated=True)
    d = _deck(eng)
    d.start_record()
    d.stop()
    assert d.take_truncated is True


def test_can_record_mirrors_engine_input_running():
    """can_record() is the honest pre-click probe (backed by engine.input_running)
    so Frodo can pre-disable the button before a doomed start_record()."""
    eng = _FakeEngine(input_open=True)
    d = _deck(eng)
    assert d.can_record() is True
    eng.input_running = False
    assert d.can_record() is False


def test_cap_hit_auto_disarm_finalizes_to_have_take_via_pump():
    """When the engine auto-disarms at deck_max_seconds, the SM is still
    'recording' until the GUI's pump() tick notices is_input_recording()==False
    and finalizes the take -> have-take (dot off — honesty). No audio-thread
    state transition; pump() owns it."""
    eng = _FakeEngine(input_open=True, take=np.ones(100, dtype=np.float32) * 0.2)
    d = _deck(eng)
    d.start_record()
    assert d.state == "recording"
    eng._armed = False  # engine hit the cap and auto-disarmed
    d.pump()
    assert d.state == "have-take", "pump() must finalize a cap-hit recording"
    assert d.take is not None and len(d.take) == 100


def test_on_state_changed_fires_on_transitions():
    """The GUI callback must fire on transitions so Frodo relabels buttons +
    drives the dot. Arity-agnostic (callback pulls .state itself)."""
    fired: list[bool] = []
    eng = _FakeEngine(input_open=True, take=np.ones(50, dtype=np.float32) * 0.2)
    d = _deck(eng, on_state_changed=lambda *a: fired.append(True))
    base = len(fired)
    d.start_record()
    d.stop()
    assert len(fired) >= base + 2, "each transition must notify the GUI callback"


@pytest.mark.parametrize("drive", ["idle", "recording", "have-take"])
def test_close_is_idempotent_and_never_raises(drive):
    """Teardown from any state must be idempotent + never raise (the Android
    deck close-path race). Must NOT close the engine's input stream (engine
    owns that) and must never orphan a mixer source."""
    eng = _FakeEngine(input_open=True, take=np.ones(50, dtype=np.float32) * 0.2)
    d = _deck(eng)
    if drive in ("recording", "have-take"):
        d.start_record()
    if drive == "have-take":
        d.stop()
    closer = getattr(d, "close", None) or getattr(d, "shutdown", None)
    assert closer is not None, "deck needs a close()/shutdown() teardown"
    closer()
    closer()  # twice — idempotent


# ===========================================================================
# DeckPlaybackSource tier — the MixerSource that replays a take.
# (Pinned to the LANDED ctor: DeckPlaybackSource(take, capture_sr, output_sr,
#  max_block, gain=1.0); active_midi always None; .finished; render additive.)
# ===========================================================================
def test_playback_source_active_midi_is_none():
    """Recorded audio is not a tuned pitch — active_midi MUST be None so the
    deck stays invisible to Mixer.sounding_midis()/the D3 coordinator (never
    vote-excluded, never ducked)."""
    src = sax_deck.DeckPlaybackSource(
        np.ones(100, dtype=np.float32) * 0.5, 48000, 48000, 2048)
    assert src.active_midi is None


def test_playback_source_direct_path_body_is_faithful_with_edge_fades():
    """Matched capture/output rate -> the BODY is a faithful additive copy; only
    the first/last EDGE_FADE_MS are ramped from/to silence (BACKLOG-DECK-FADE ->
    click-free playback). The exported WAV is unaffected (it writes the raw
    take) — this fade is live-playback only."""
    sr = 48000
    n = 4096
    # cos so the edges are NON-zero (cos(0)=1): proves the fade actually
    # attenuates them, not that the signal was already ~0 there.
    take = (np.cos(2 * np.pi * np.arange(n) / 64.0) * 0.7).astype(np.float32)
    src = sax_deck.DeckPlaybackSource(take, sr, sr, 8192)
    out = np.zeros(n, dtype=np.float32)
    src.render(out, n, 0)
    fade = int(round(sax_deck.EDGE_FADE_MS / 1000.0 * sr))  # 240 samples
    # Body (between the two ramps) is a bit-faithful copy.
    assert np.allclose(out[fade:n - fade], take[fade:n - fade], atol=1e-6), \
        "playback body is a faithful copy"
    # Edges ramp from / to ~silence (click-free): far below the 0.7 source edge.
    assert abs(out[0]) < 0.05 < abs(take[0]), "fade-in starts near zero"
    assert abs(out[-1]) < 0.05, "fade-out ends near zero"
    # The fade only attenuates — never amplifies past the source peak.
    assert np.max(np.abs(out)) <= np.max(np.abs(take)) + 1e-6, "fade never amplifies"


def test_playback_source_renders_audio_then_finishes():
    take = (np.sin(np.arange(500) / 5.0) * 0.4).astype(np.float32)
    src = sax_deck.DeckPlaybackSource(take, 48000, 48000, 2048)
    assert src.finished is False
    out = np.zeros(2048, dtype=np.float32)
    src.render(out, 2048, 0)
    assert np.max(np.abs(out)) > 1e-3, "playback must produce audio"
    assert src.finished is True, "a 500-sample take is done after a 2048-frame render"


def test_playback_source_empty_take_is_immediately_finished():
    src = sax_deck.DeckPlaybackSource(np.zeros(0, dtype=np.float32), 48000, 48000, 2048)
    assert src.finished is True, "a zero-length take has nothing to play"


def test_playback_source_resamples_when_rates_differ():
    """A 48 kHz take through a 24 kHz output consumes ~2 capture samples per
    output sample, so it finishes in ~half the output samples — and stays
    audible + bounded. Asserts the resample PROPERTY (bounded, finishes), not
    an exact sample count, so Sauron keeps interpolation-impl freedom."""
    n = 1000
    take = (np.sin(2 * np.pi * np.arange(n) / 50.0) * 0.5).astype(np.float32)
    src = sax_deck.DeckPlaybackSource(take, 48000, 24000, 4096)
    out = np.zeros(4096, dtype=np.float32)
    src.render(out, 4096, 0)
    assert np.max(np.abs(out)) > 1e-3, "downsampled playback is still audible"
    assert np.max(np.abs(out)) <= 1.0 + 1e-4, "playback stays bounded"
    assert src.finished is True, "2 capture samples/output -> 1000-sample take done in one block"


@pytest.mark.parametrize("cap,out_sr", [
    (48000, 44100), (44100, 48000), (96000, 44100), (44100, 96000),
], ids=["48->44.1k", "44.1->48k", "96->44.1k", "44.1->96k"])
def test_playback_resample_preserves_pitch(cap, out_sr):
    """The PURPOSE of the resample: a take captured at one rate and played
    through a DIFFERENT output rate must keep its pitch, not pitch-shift by the
    rate ratio. The naive no-resample bug would be 440*cap/out_sr -> +-147 to
    +-1347 cents on these pairs; the linear interp is pitch-exact (sub-cent).
    Complements test_playback_source_resamples_when_rates_differ (which only
    pins bounded+finishes, leaving impl freedom) with the pitch contract."""
    import sim_harness as H
    F = 440.0
    take = H.sine(F, cap, cap, amp=0.6)                       # 1 s of 440 Hz @ cap
    src = sax_deck.DeckPlaybackSource(take, cap, out_sr, 2048, gain=1.0)
    buf = H.render_stream(src, out_sr // 2, block=2048, warmup=2)  # 0.5 s, within take
    err = H.cents(H.fft_peak_hz(buf, out_sr), F)
    assert abs(err) < 5.0, (
        f"resample {cap}->{out_sr} shifted pitch {err:+.2f} cents (rate-ratio bug?)")


def test_playback_source_render_is_non_scaling_alloc():
    """The resample path preallocates all scratch in __init__; a render must
    add NO per-frame allocation. Non-scaling gate (frames-delta), same shape
    as the mixer/drone gates — NOT an absolute byte floor (numpy view-boxing
    is a constant per call, so an absolute floor flakes)."""
    import tracemalloc
    n = 400_000
    take = (np.sin(np.arange(n) / 11.0) * 0.3).astype(np.float32)
    # Rate mismatch -> exercise the allocation-prone resample path.
    src = sax_deck.DeckPlaybackSource(take, 48000, 44100, 4096)
    out = np.zeros(4096, dtype=np.float32)
    src.render(out, 1024, 0)  # warm up (lazy view-boxing on first call)
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(20):
        out[:] = 0.0
        src.render(out, 1024, 0)
    snap_small = tracemalloc.take_snapshot()
    for _ in range(20):
        out[:] = 0.0
        src.render(out, 4096, 0)
    snap_big = tracemalloc.take_snapshot()
    tracemalloc.stop()

    def _delta(a, b):
        return sum(s.size_diff for s in b.compare_to(a, "filename"))

    small = _delta(snap0, snap_small)
    big = _delta(snap_small, snap_big)
    # 4x the frames must NOT cost ~4x the bytes: scratch is preallocated, so
    # the big-block delta tracks the small-block delta within constant slack.
    assert big <= small + 8192, (
        f"render allocation scales with frames (small={small} big={big}) — "
        f"resample scratch must be preallocated, not per-block")


# ===========================================================================
# Playback transitions (play / replay / interrupt / stop / playback-end),
# exercised through a real Mixer so the registered source actually renders.
# ===========================================================================
def test_have_take_play_to_playing():
    take = (np.sin(np.arange(1000) / 5.0) * 0.4).astype(np.float32)
    d = _deck(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    assert d.state == "have-take"
    assert d.play() is True
    assert d.state == "playing"


def test_play_with_no_take_returns_false():
    d = _deck(_FakeEngine(input_open=True, take=np.zeros(0, dtype=np.float32)))
    assert d.play() is False
    assert d.state == "idle", "nothing to play -> no phantom playing state"


def test_recording_play_is_noop_false():
    d = _deck(_FakeEngine(input_open=True, take=np.ones(100, dtype=np.float32) * 0.2))
    d.start_record()
    assert d.play() is False, "can't play mid-record — stop first"
    assert d.state == "recording"


def test_deck_playback_invisible_to_sounding_midis():
    take = (np.sin(np.arange(1000) / 5.0) * 0.4).astype(np.float32)
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    d.play()
    assert m.sounding_midis() == frozenset(), (
        "deck playback (active_midi=None) must not appear in sounding_midis -> "
        "never vote-excluded, never ducked by D3")
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert np.max(np.abs(out)) > 1e-3, "the take should actually be playing"
    assert m.sounding_midis() == frozenset(), "still invisible mid-playback"


def test_playing_play_restarts_from_zero():
    take = (np.sin(np.arange(8000) / 7.0) * 0.4).astype(np.float32)
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    assert d.play() is True
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)              # consume ~2048 of 8000
    assert d.play() is True          # 'replay' — restart at 0
    assert d.state == "playing"
    out2 = np.zeros(2048, dtype=np.float32)
    m.render(out2, 2048)
    assert np.max(np.abs(out2)) > 1e-3, "replay restarts the take — audio flows from the top again"


def test_playing_stop_to_have_take_retains_take():
    take = (np.sin(np.arange(8000) / 7.0) * 0.4).astype(np.float32)
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    d.play()
    assert d.state == "playing"
    d.stop()
    assert d.state == "have-take"
    assert d.take is not None and len(d.take) == 8000, "stopping playback retains the take"
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert np.max(np.abs(out)) < 1e-6, "stopped-playback source must be unregistered (silent)"


def test_playing_start_record_interrupts_and_unregisters():
    take = (np.sin(np.arange(8000) / 7.0) * 0.4).astype(np.float32)
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    assert d.play() is True
    assert d.start_record() is True, "start_record must interrupt playback"
    assert d.state == "recording"
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert np.max(np.abs(out)) < 1e-6, (
        "the interrupted playback source must be unregistered first (no orphan, silent)")


def test_playback_end_transitions_to_have_take_only_via_pump():
    """The source exhausts on the AUDIO thread (render), but the state
    transition must happen on the GUI thread (pump) — never inside render."""
    take = (np.sin(np.arange(400) / 5.0) * 0.4).astype(np.float32)  # short
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    assert d.play() is True
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)              # 400-sample take exhausts within this block
    assert d.state == "playing", "render (audio thread) must NOT perform the transition"
    d.pump()                        # GUI thread observes finished -> have-take
    assert d.state == "have-take", "pump() retires the finished source -> have-take"
    assert d.take is not None, "the take is retained after playback ends"


def test_close_from_playing_unregisters_no_orphan():
    take = (np.sin(np.arange(8000) / 7.0) * 0.4).astype(np.float32)
    d, m = _deck_and_mixer(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    d.play()
    d.close()
    d.close()  # idempotent
    assert d.state == "idle"
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert np.max(np.abs(out)) < 1e-6, "close() must drop the playback source (no orphan)"


# ===========================================================================
# Export / load round-trip + scratch persistence (the "valid WAV that opens
# externally" + "restore last take" acceptance paths, end-to-end through the
# controller).
# ===========================================================================
def test_export_writes_valid_loadable_wav(tmp_path):
    take = (np.sin(2 * np.pi * 220 * np.arange(2000) / 48000) * 0.6).astype(np.float32)
    d = _deck(_FakeEngine(input_open=True, take=take, sr=48000))
    d.start_record()
    d.stop()
    p = tmp_path / "export.wav"
    assert d.export(str(p)) is True
    assert p.exists()
    # Opens with stdlib wave (i.e. externally) + round-trips within int16 quant.
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 48000
    back, sr = sax_deck.read_wav(str(p))
    assert sr == 48000
    assert len(back) == len(take)
    assert np.max(np.abs(back - take)) <= _PCM_LSB + 1e-6, "exported take round-trips within int16 quant"


def test_export_with_no_take_returns_false(tmp_path):
    d = _deck(_FakeEngine(input_open=True, take=np.zeros(0, dtype=np.float32)))
    p = tmp_path / "none.wav"
    assert d.export(str(p)) is False
    assert not p.exists(), "no take -> nothing written"


def test_load_take_restores_have_take(tmp_path):
    """Restore-last-session path: load a WAV -> have-take, samplerate + samples
    recovered within int16 quant."""
    take = (np.cos(np.arange(800) / 9.0) * 0.5).astype(np.float32)
    p = tmp_path / "saved.wav"
    sax_deck.write_wav(str(p), take, 44100)
    d = _deck(_FakeEngine(input_open=True))
    assert d.load_take(str(p)) is True
    assert d.state == "have-take"
    assert d.take_samplerate == 44100
    assert np.max(np.abs(d.take - take)) <= _PCM_LSB + 1e-6


def test_finalize_persists_scratch_when_dir_set(tmp_path):
    """With a scratch_dir, finalising a recording best-effort writes the take
    so it can be reloaded next session; last_take_path points at it."""
    take = np.ones(200, dtype=np.float32) * 0.3
    d = _deck(_FakeEngine(input_open=True, take=take, sr=48000),
              scratch_dir=str(tmp_path / "scratch"))
    d.start_record()
    d.stop()
    assert d.state == "have-take"
    assert d.last_take_path is not None
    assert os.path.exists(d.last_take_path), "finalised take must be persisted to scratch_dir"


def test_resample_last_sample_boundary_uses_take_end_not_penultimate():
    """ADVERSARIAL-SWEEP wave 1: when a resampled position lands exactly on the
    last take index (_n-1), the interpolation must yield take[_n-1], not the
    penultimate take[_n-2] (the clipped-idx vs floor-based-frac off-by-one). Use
    a tiny samplerate so EDGE_FADE_MS rounds to 0 — otherwise the boundary sample
    sits in the fade ramp and the error is masked toward silence."""
    take = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    # 50 Hz capture -> 100 Hz output: ratio 0.5, positions hit 3.0 (=_n-1) exactly;
    # 5 ms * 50 = 0.25 samples -> round -> 0 -> no edge fade.
    src = sax_deck.DeckPlaybackSource(take, 50, 100, 64)
    assert src._fade == 0, "test needs the edge fade disabled to observe the boundary"
    out = np.zeros(64, dtype=np.float32)
    src.render(out, 64, 0)
    # positions 0, 0.5, 1, 1.5, 2, 2.5, 3.0 -> 7 samples; out[6] is pos == 3.0.
    assert out[6] == pytest.approx(0.4, abs=1e-6), (
        f"boundary sample must be take[-1]=0.4, got {out[6]:.4f} "
        f"(0.3 is the off-by-one bug)")

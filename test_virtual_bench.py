"""Virtual verification bench — pitch + timing of every input and output,
measured from the ACTUAL rendered/detected signal with no audio hardware.

Operator standard: "nothing should ever feel off — we should be able to verify
the timing and pitch of every input and output virtually before sending this
into the wild." Most modules test their MATH (beat_onset_sample, midi_to_freq);
this bench renders the real sources through the real mixer (and feeds the real
engine) and MEASURES the result — the end-to-end "does it actually sound at the
right pitch, at the right time" gate.

Pure numpy (no tsf / no PortAudio) so it runs on system-python and the venv. The
drone-pitch + deck-timing + input-latency benches build on this in follow-ups.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import sim_harness as H
from sax_mixer import Mixer, TestToneSource
from sax_metronome import MetronomeController, beat_onset_sample, beats_per_bar

pp = pytest.importorskip("sax_pitch_pipes", reason="pitch pipes module")
PitchPipeSource = pp.PitchPipeSource

SR = 48000


# ---------------------------------------------------------------------------
# 0. Lock the onset primitive itself (so the timing checks below are trustworthy).
# ---------------------------------------------------------------------------
def test_onset_samples_finds_bursts_at_known_positions():
    sig = np.zeros(60000, dtype=np.float32)
    positions = [1000, 13000, 25000, 49000]
    n = 1500
    t = np.arange(n)
    burst = (np.sin(2 * np.pi * 1000 * t / SR) * np.exp(-t / 400.0) * 0.5).astype(np.float32)
    for p in positions:
        sig[p:p + n] += burst
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=4000)
    assert len(onsets) == len(positions), f"expected {len(positions)} onsets, got {onsets}"
    for det, true in zip(onsets, positions):
        assert abs(det - true) <= 5, f"onset {det} vs true {true}"


# ---------------------------------------------------------------------------
# 1. OUTPUT timing — the metronome's RENDERED clicks land on the BPM grid with
#    no drift. (beat_onset_sample's integer-drift proof is the math; this is the
#    audio a speaker would actually emit.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bpm", [60, 120, 200])
def test_metronome_rendered_clicks_on_bpm_grid_no_drift(bpm):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, SR, bpm=bpm)
    metro.start()
    interval = SR * 60.0 / bpm
    nbeats = 8
    sig = H.render_mixer(mixer, int(interval * nbeats + interval), block=mb)
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))
    assert len(onsets) >= nbeats, f"{len(onsets)} click onsets < {nbeats}: {onsets}"
    onsets = onsets[:nbeats]
    intervals = np.diff(onsets)
    assert np.all(np.abs(intervals - interval) <= 2), (
        f"rendered beat spacing must be {interval:.1f}±2 samples (no drift): "
        f"{intervals.tolist()}")
    # cumulative drift over all beats — the thing that "feels off" if it creeps
    drift = abs((onsets[-1] - onsets[0]) - interval * (nbeats - 1))
    assert drift <= 2, f"cumulative drift over {nbeats} beats @ {bpm}bpm = {drift} samples"
    # and the absolute grid matches beat_onset_sample (the math) within the
    # click's ~1 ms attack offset
    anchor = onsets[0]
    for i, o in enumerate(onsets):
        expected = anchor + beat_onset_sample(i, bpm, SR)
        assert abs(o - expected) <= 4, f"beat {i} @ {bpm}bpm: onset {o} vs grid {expected}"


# ---------------------------------------------------------------------------
# 2. OUTPUT pitch — the RENDERED test tone is actually at its target frequency
#    (independent FFT readback, not the source's own dphi).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [110.0, 220.0, 440.0, 880.0, 1760.0])
def test_test_tone_rendered_pitch_matches_target(f):
    tone = TestToneSource(f, SR, 2048, gain=0.3)
    sig = H.render_stream(tone, 16384, block=2048)
    det = H.fft_peak_hz(sig, SR)
    err = H.cents(det, f)
    assert abs(err) < 1.5, f"test tone {f}Hz rendered as {det:.2f}Hz ({err:+.2f}c)"


# ---------------------------------------------------------------------------
# 3. OUTPUT pitch — the RENDERED pitch-pipe pad sounds at midi_to_freq(midi,a4).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("midi", [45, 57, 69, 81, 93])  # A2..A6
def test_pitch_pipe_rendered_pitch_matches_target(midi):
    a4 = 440.0
    src = PitchPipeSource(midi, SR, 2048, a4=a4)
    sig = H.render_stream(src, 16384, block=2048, warmup=2)  # skip the 8 ms attack
    det = H.fft_peak_hz(sig, SR)
    true = pp.midi_to_freq(midi, a4)
    err = H.cents(det, true)
    assert abs(err) < 1.5, f"pitch pad midi={midi} ({true:.2f}Hz) rendered {det:.2f}Hz ({err:+.2f}c)"


# ---------------------------------------------------------------------------
# 4. INPUT pitch — a synthetic tone driven through the REAL engine pipeline
#    (silence gate -> YIN -> confirm/lock -> emit) confirms the right note + cents.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,tol", [("sine", 0.6), ("sax", 0.6), ("noisy", 2.0)],
                         ids=["sine", "sax", "noisy"])
@pytest.mark.parametrize("f,midi", [
    (110.0, 45), (220.0, 57), (440.0, 69), (659.25, 76), (880.0, 81)],
    ids=["A2", "A3", "A4", "E5", "A5"])
def test_input_pipeline_confirms_correct_pitch(f, midi, kind, tol):
    """INPUT: 5 notes x 3 timbres through the FULL engine pipeline (gate -> YIN
    -> confirm -> emit) confirm the right note. Clean tones measure sub-cent (so
    the old 2c band WAS wide) -> 0.6c; the 20 dB-SNR 'noisy' timbre is the
    realistic worst case -> 2c."""
    res = H.detect_tone(f, kind=kind, hops=24)
    assert res.emitted, f"{kind} {f}Hz must confirm + emit"
    assert res.dominant_midi == midi, f"{kind} {f}Hz -> midi {res.dominant_midi} (expected {midi})"
    assert abs(res.cents_error()) < tol, (
        f"{kind} {f}Hz cents err {res.cents_error():+.2f}c (tol {tol})")


# ---------------------------------------------------------------------------
# 5. INPUT timing — detection latency (the confirm/lock machinery is not
#    instant, but must be PROMPT). The input twin of onset timing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [220.0, 440.0, 880.0])
def test_input_detection_latency_bounded(f):
    hops = H.detection_latency_hops(f, kind="sax")
    assert hops is not None, f"{f}Hz never confirmed within the hop cap"
    assert hops <= 12, f"{f}Hz took {hops} hops to confirm (latency regression?)"


# ---------------------------------------------------------------------------
# 6. OUTPUT pitch — 5 notes x 3 drone VOICES (patches). The intended FUNDAMENTAL
#    is present + in tune, measured by peak_near_hz (a global FFT peak finds
#    whichever partial is loudest — measured an octave / +2792c off for these
#    rich SF2 voices — and YIN octave-errs on the organ's sub-octave; the app
#    never self-measures the drone, it knows the note from active_midi). The ~12c
#    band is FFT-on-rich-timbre MEASUREMENT spread, not the synth's tuning (tsf
#    set_tuning is exact). Voices chosen are the non-ensemble patches; the
#    ensemble ones (strings/warmpad) carry intentional chorus detune (~15-25c
#    measured at low notes) — a fat-drone feature, excluded from this precision
#    gate by design. tsf/venv only.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("voice", ["organ", "cello", "tenorsax"])
def test_drone_rendered_pitch_matches_reference(voice):
    pytest.importorskip("tinysoundfont", reason="drone synth is venv/CI only")
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id=voice, volume=0.7)
    c.set_enabled(True)                                    # one SF2 load per voice
    for midi, name in [(45, "A2"), (57, "A3"), (69, "A4"), (76, "E5"), (81, "A5")]:
        c.set_reference_midi(midi)
        assert c.source.active_midi == midi, f"{voice} must declare midi {midi}"
        sig = H.render_mixer(mixer, 32768, block=2048)[6000:]   # skip the note attack
        true = D.midi_to_freq(midi, 440.0)
        det = H.peak_near_hz(sig, true, SR)
        assert det > 0, f"drone {voice} {name}: no fundamental near {true:.1f}Hz"
        err = H.cents(det, true)
        assert abs(err) < 12.0, f"drone {voice} {name} ({true:.1f}Hz) -> {det:.1f}Hz ({err:+.1f}c)"


# ---------------------------------------------------------------------------
# 7. OUTPUT timing + pitch — a recorded take plays back for its full duration
#    at the right pitch, then stops on time. (Cross-rate resample pitch is also
#    in test_deck.py::test_playback_resample_preserves_pitch.)
# ---------------------------------------------------------------------------
def test_deck_playback_duration_and_pitch():
    from sax_deck import DeckPlaybackSource
    n, f = 24000, 440.0                                    # 0.5 s take @ A4
    take = (np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.5).astype(np.float32)
    src = DeckPlaybackSource(take, SR, SR, 2048)
    out = H.render_stream(src, n + 4096, block=2048)
    det = H.fft_peak_hz(out[300:n - 300], SR)              # skip the 5 ms edge fades
    assert abs(H.cents(det, f)) < 2.0, f"deck playback pitch {det:.2f}Hz vs {f}"
    assert H.max_abs(out[n + 1000:]) < 1e-3, "playback must STOP at the take's end (timing)"


# ---------------------------------------------------------------------------
# 8. OUTPUT timing — the metronome ACCENT GRID across 5 tempos x each stock time
#    signature. Verifies, from the rendered audio: (a) every click on the BPM
#    grid with no drift, and (b) the downbeat accent falls on every
#    beats_per_bar-th beat (the accent click is louder than the plain click).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("time_sig", ["2/4", "3/4", "4/4", "6/8"])
@pytest.mark.parametrize("bpm", [40, 80, 120, 160, 200])
def test_metronome_accent_grid_across_tempos_and_time_sigs(time_sig, bpm):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, SR, bpm=bpm, time_sig=time_sig)
    metro.start()
    n = beats_per_bar(time_sig)
    interval = SR * 60.0 / bpm
    nbeats = n * 3 + 1                                     # ~3 full bars + 1
    sig = H.render_mixer(mixer, int(interval * (nbeats + 1)), block=mb)
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))
    assert len(onsets) >= nbeats, f"{time_sig}@{bpm}: {len(onsets)} onsets < {nbeats}"
    onsets = onsets[:nbeats]
    # (a) grid: even spacing, no drift over all beats
    intervals = np.diff(onsets)
    assert np.all(np.abs(intervals - interval) <= 2), (
        f"{time_sig}@{bpm}: beat spacing drifts from {interval:.0f}: {intervals.tolist()}")
    # (b) accent pattern: every n-th beat (the downbeat) is LOUDER than the rest
    win = min(int(interval * 0.3), 1500)
    peaks = [float(np.max(np.abs(sig[o:o + win]))) for o in onsets]
    accented = [peaks[i] for i in range(nbeats) if i % n == 0]
    plain = [peaks[i] for i in range(nbeats) if i % n != 0]
    assert plain, f"{time_sig}: expected some non-downbeat beats"
    assert min(accented) > max(plain), (
        f"{time_sig}@{bpm}: downbeat accent not louder than plain beats — "
        f"accent~{min(accented):.2f} vs plain~{max(plain):.2f}")


# ---------------------------------------------------------------------------
# 9. OUTPUT pitch STABILITY — the organ drone is the steady PRECISION reference
#    (vibrato-free), centred on pitch. The sampled-instrument voices carry
#    baked-in vibrato/chorus (measured at A4: strings ~21c, tenorsax ~14c,
#    warmpad ~12c, cello ~7c spread) and are COLOUR voices, not precision
#    references -- documented here, not asserted. The pure-sine test-tone /
#    pitch-pipe (sections 2-3) are the sub-cent precision references.
# ---------------------------------------------------------------------------
def test_drone_organ_is_a_steady_reference():
    """The organ is the steadiest, most-centred drone voice — verified at concert
    A4, where FFT-based stability measurement of a rich tone is reliable: pitch
    spread < 6c (vibrato-free) and centre within 4c. (Low notes like A2 are
    FFT-resolution-limited for this measure — ~46c/bin at 110 Hz — so the lock is
    the reliably-measurable A4 point, not a claim the organ is untestable there.
    For sub-cent precision across the full range, the pure-sine test-tone /
    pitch-pipe are the references; the drone is a musical reference.)"""
    pytest.importorskip("tinysoundfont", reason="drone synth is venv/CI only")
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id="organ", volume=0.7)  # ref 69 = A4
    c.set_enabled(True)
    sig = H.render_mixer(mixer, 180000, block=2048)[20000:]   # ~3.3 s steady
    true = D.midi_to_freq(69, 440.0)
    mean_c, spread_c = H.pitch_stability(sig, true, SR)
    assert spread_c < 6.0, f"organ A4 pitch spread {spread_c:.1f}c — not steady (vibrato?)"
    assert abs(mean_c) < 4.0, f"organ A4 centre {mean_c:+.1f}c off"

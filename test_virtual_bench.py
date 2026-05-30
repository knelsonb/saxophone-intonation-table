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
from sax_metronome import MetronomeController, beat_onset_sample

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
    assert abs(err) < 2.0, f"test tone {f}Hz rendered as {det:.2f}Hz ({err:+.2f}c)"


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
    assert abs(err) < 2.0, f"pitch pad midi={midi} ({true:.2f}Hz) rendered {det:.2f}Hz ({err:+.2f}c)"


# ---------------------------------------------------------------------------
# 4. INPUT pitch — a synthetic tone driven through the REAL engine pipeline
#    (silence gate -> YIN -> confirm/lock -> emit) confirms the right note + cents.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f,midi", [
    (110.0, 45),    # A2
    (220.0, 57),    # A3
    (440.0, 69),    # A4
    (880.0, 81),    # A5
    (1318.5, 88),   # E6
])
def test_input_pipeline_confirms_correct_pitch(f, midi):
    res = H.detect_tone(f, kind="sax", hops=24)
    assert res.emitted, f"{f}Hz must confirm + emit through the full input pipeline"
    assert res.dominant_midi == midi, (
        f"{f}Hz -> midi {res.dominant_midi} (expected {midi})")
    assert abs(res.cents_error()) < 2.0, (
        f"{f}Hz input cents error {res.cents_error():+.2f}c")

"""Tests for the I/O simulation harness (sim_harness.py).

Two jobs:
  1. Prove the SIM TOOLING itself is trustworthy — detect_tone / feed_engine
     (input pipeline) and render_mixer (output path) do what they claim, so
     other suites + ad-hoc verification can rely on them.
  2. Demonstrate end-to-end verification the tooling unlocks: the full engine
     detection pipeline (gate -> YIN -> confirm/lock -> emit) reads the right
     note + cents, honours A4 and the mic-gain floor, and the real Mixer
     renders the right pitch.

Pure numpy + app modules (no Qt, no PortAudio) — always-runnable suite.
"""
from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip('numpy')
import numpy as np  # noqa: E402

import sim_harness as H  # noqa: E402
from sax_audio_engine import FILTER_PRESETS  # noqa: E402


# ── INPUT pipeline: detect_tone / feed_engine ──────────────────────────────

@pytest.mark.parametrize("freq, midi", [
    (110.0, 45),   # A2
    (220.0, 57),   # A3
    (440.0, 69),   # A4
    (880.0, 81),   # A5
])
def test_detect_tone_sax_reads_correct_note(freq, midi):
    r = H.detect_tone(freq, kind='sax')
    assert r.emitted
    assert r.dominant_midi == midi
    assert abs(r.cents_error()) < 5.0


def test_detect_tone_sine_also_locks():
    r = H.detect_tone(440.0, kind='sine')
    assert r.emitted and r.dominant_midi == 69
    assert abs(r.cents_error()) < 5.0


def test_detection_honours_a4_reference():
    # The SAME 440 Hz tone read against A4=442 is flat by ~7.85 cents, but
    # still note A4 (MIDI 69). Proves A4 plumbs through the live pipeline.
    r = H.detect_tone(440.0, kind='sine', a4=442.0)
    assert r.dominant_midi == 69
    assert -12.0 < r.median_cents < -3.0


def test_silence_emits_nothing():
    eng_freq = 0.0  # unused; feed pure silence directly
    from sax_audio_engine import AudioEngine
    eng = AudioEngine()
    res = H.feed_engine(eng, H.silence(H.DEFAULT_N * 12))
    assert not res.emitted
    assert res.dominant_midi is None
    assert math.isnan(res.cents_error())  # no target, nothing detected


def test_response_modes_all_detect_clean_tone():
    for mode in ('fast', 'normal', 'slow'):
        r = H.detect_tone(440.0, kind='sax', mode=mode)
        assert r.emitted, f"{mode} mode failed to lock a clean A440"
        assert r.dominant_midi == 69


def test_mic_gain_floor_via_sim():
    # A tone whose RMS sits below the 'normal' silence floor is GATED at unity
    # gain but CLEARS the floor once boosted — verified through the sim's input
    # path (cross-checks SETUP-MICGAIN end to end).
    floor = FILTER_PRESETS['normal']['rms_floor']
    amp = (floor * 0.5) * math.sqrt(2.0)   # RMS = 0.5 * floor
    assert not H.detect_tone(440.0, kind='sine', amp=amp, mic_gain=1.0).emitted
    assert H.detect_tone(440.0, kind='sine', amp=amp, mic_gain=8.0).emitted


def test_feed_engine_result_surface():
    r = H.detect_tone(440.0, kind='sax')
    # DetectionResult exposes a coherent, inspectable surface.
    assert isinstance(r.notes, list) and r.notes
    assert all(len(n) == 3 for n in r.notes)
    assert r.last_rms_db > -120.0          # meter moved off the floor
    assert r.locked_midi == 69


# ── OUTPUT path: render_mixer ──────────────────────────────────────────────

def test_render_mixer_test_tone_pitch():
    from sax_mixer import Mixer, TestToneSource
    sr, block = H.DEFAULT_SR, 2048
    mix = Mixer(max_block=block)
    mix.register(TestToneSource(440.0, sr, block, gain=0.3))
    out = H.render_mixer(mix, total=sr // 2, block=block)   # 0.5 s
    assert out.size == sr // 2
    assert abs(H.cents(H.fft_peak_hz(out, sr), 440.0)) < 5.0


def test_render_mixer_empty_is_silence():
    from sax_mixer import Mixer
    mix = Mixer(max_block=1024)
    out = H.render_mixer(mix, total=4096, block=1024)
    assert out.size == 4096
    assert H.max_abs(out) == 0.0


def test_render_mixer_sums_two_sources():
    # Two tones mixed → both spectral peaks present (real summing path).
    from sax_mixer import Mixer, TestToneSource
    sr, block = H.DEFAULT_SR, 2048
    mix = Mixer(max_block=block)
    mix.register(TestToneSource(440.0, sr, block, gain=0.25))
    mix.register(TestToneSource(660.0, sr, block, gain=0.25))
    out = H.render_mixer(mix, total=sr // 2, block=block)
    sp = np.abs(np.fft.rfft(out.astype(np.float64) * np.hanning(out.size)))
    freqs = np.fft.rfftfreq(out.size, 1.0 / sr)
    def _has_peak(target):
        k = int(np.argmin(np.abs(freqs - target)))
        band = sp[max(0, k - 3):k + 4]
        return sp[k] > 0 and sp[k] >= 0.25 * float(sp.max()) and band.size > 0
    assert _has_peak(440.0) and _has_peak(660.0)

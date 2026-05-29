"""YIN pitch-detection ACCURACY net — cents-tolerance contract.

Complements test_yin_baseline (exact-value ULP change-detector) with the
contract that actually matters for a tuner: yin_pitch must detect a synthetic
tone within a tight cents tolerance at the engine's real detection window.

This is the regression guard for the parabolic-interpolation SIGN BUG (fixed
2026-05-28): the inverted denominator shifted every refined lag the wrong way
and read ~+8 cents SHARP across the whole mid/high range, so a perfectly-tuned
A440 reported "+8 ct". The fix collapsed the tuning range to sub-cent accuracy.

Pure numpy; runs under system python.
"""
from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip('numpy')
import numpy as np  # noqa: E402

from sax_audio_engine import yin_pitch, DEFAULT_BLOCK_SIZE  # noqa: E402


def _cents(detected: float, true: float) -> float:
    return 1200.0 * math.log2(detected / true)


def _sine(f: float, sr: int, N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    return (0.5 * np.sin(2.0 * math.pi * f * n / sr)).astype(np.float32)


def _sax_like(f: float, sr: int, N: int) -> np.ndarray:
    """Five-harmonic saxophone-like tone (same recipe as test_yin_baseline)."""
    n = np.arange(N, dtype=np.float64)
    out = (0.50 * np.sin(2.0 * math.pi * 1 * f * n / sr)
           + 0.30 * np.sin(2.0 * math.pi * 2 * f * n / sr + 0.4)
           + 0.20 * np.sin(2.0 * math.pi * 3 * f * n / sr + 1.1)
           + 0.10 * np.sin(2.0 * math.pi * 4 * f * n / sr + 2.0)
           + 0.05 * np.sin(2.0 * math.pi * 5 * f * n / sr + 0.7))
    return (0.7 * out / float(np.max(np.abs(out)))).astype(np.float32)


# (frequency, cents tolerance). The whole tuning range is sub-1.5 cents; the
# top is integer-lag-limited at 44.1k but still < 2.5 cents. Generous margins
# over the measured post-fix errors (440: 0.07, 880: 0.34, 1760: 1.4, 2093: 1.9)
# so the test is a true regression guard, not a flaky tight bound.
_RANGE = [
    (82.41, 1.0),    # E2
    (110.0, 1.0),    # A2
    (146.83, 1.0),   # D3
    (220.0, 1.0),    # A3
    (261.63, 1.0),   # C4
    (329.63, 1.0),   # E4
    (440.0, 1.0),    # A4
    (587.33, 1.5),   # D5
    (880.0, 1.5),    # A5
    (1318.5, 2.5),   # E6
    (1760.0, 2.5),   # A6
    (2093.0, 3.0),   # C7
]


@pytest.mark.parametrize('f,tol', _RANGE, ids=[f'{f:g}Hz' for f, _ in _RANGE])
def test_sine_accuracy_44k(f: float, tol: float) -> None:
    det, _ap = yin_pitch(_sine(f, 44100, DEFAULT_BLOCK_SIZE), 44100)
    assert det > 0, f'{f} Hz sine not detected'
    err = _cents(det, f)
    assert abs(err) < tol, (
        f'{f} Hz sine -> {det:.3f} Hz ({err:+.2f} ct), tol +-{tol}')


@pytest.mark.parametrize('f,tol', _RANGE, ids=[f'{f:g}Hz' for f, _ in _RANGE])
def test_sax_like_accuracy_44k(f: float, tol: float) -> None:
    det, _ap = yin_pitch(_sax_like(f, 44100, DEFAULT_BLOCK_SIZE), 44100)
    assert det > 0, f'{f} Hz sax-like not detected'
    err = _cents(det, f)
    assert abs(err) < tol, (
        f'{f} Hz sax-like -> {det:.3f} Hz ({err:+.2f} ct), tol +-{tol}')


def test_a440_is_not_sharp() -> None:
    """The specific regression: a perfect A440 must read within +-0.5 cents,
    NOT +8 (the parabolic-interpolation sign bug)."""
    det, _ = yin_pitch(_sine(440.0, 44100, DEFAULT_BLOCK_SIZE), 44100)
    err = _cents(det, 440.0)
    assert abs(err) < 0.5, (
        f'A440 -> {det:.4f} Hz ({err:+.3f} ct) — parabolic sharp-bias regressed')


@pytest.mark.parametrize('sr', [48000, 96000])
def test_accuracy_holds_at_higher_rates(sr: int) -> None:
    det, _ = yin_pitch(_sine(440.0, sr, DEFAULT_BLOCK_SIZE), sr)
    assert abs(_cents(det, 440.0)) < 1.0


def test_no_systematic_sharp_bias() -> None:
    """The mean SIGNED cents error across the range must be ~0 — no systematic
    sharp/flat bias. The sign bug produced a mean of ~+7.6 cents."""
    errs = [_cents(yin_pitch(_sine(f, 44100, DEFAULT_BLOCK_SIZE), 44100)[0], f)
            for f, _ in _RANGE]
    mean_err = sum(errs) / len(errs)
    assert abs(mean_err) < 0.7, (
        f'mean signed error {mean_err:+.2f} ct across the range — systematic bias')


# ---------------------------------------------------------------------------
# C8 (MIDI 108) reachability — the table's top note must not be silently
# dropped by the post-YIN freq gate where the MIDI-range gate accepts it.
# Before the v0.6.x fix MAX_FREQ=4200 cleared C8 only at A4=440 (4186 Hz);
# at A4 >= ~441 (or a sharp-played C8) the freq gate rejected it.
# ---------------------------------------------------------------------------
def test_max_freq_covers_top_note_at_max_a4() -> None:
    from sax_audio_engine import MAX_FREQ
    # C8 + 50 cents (the C8/C#8 rounding boundary) at the top of the allowed
    # A4 range (450 Hz). The freq gate must clear this so it never pre-rejects
    # a note the MIDI-range gate (midi_max=108) would accept.
    c8_plus_50c_at_450 = 450.0 * 2.0 ** ((108.5 - 69) / 12.0)   # ~4409 Hz
    assert MAX_FREQ >= c8_plus_50c_at_450, (
        f'MAX_FREQ {MAX_FREQ} < C8+50c@A4=450 ({c8_plus_50c_at_450:.0f} Hz)')


@pytest.mark.parametrize('a4', [430.0, 440.0, 445.0, 450.0])
def test_c8_detected_and_passes_freq_gate(a4: float) -> None:
    """C8 (MIDI 108) is detected across the whole allowed A4 range AND the
    detected pitch is below MAX_FREQ (so the engine's freq gate accepts it)."""
    from sax_audio_engine import MAX_FREQ
    c8 = a4 * 2.0 ** ((108 - 69) / 12.0)
    det, _ap = yin_pitch(_sine(c8, 44100, DEFAULT_BLOCK_SIZE), 44100)
    assert det > 0, f'C8 at A4={a4} not detected'
    assert det < MAX_FREQ, (
        f'C8 at A4={a4} -> {det:.1f} Hz >= MAX_FREQ {MAX_FREQ}; freq gate drops it')
    # Reads as ~C8 within the coarse top-of-range integer-lag limit (~10-sample
    # period at 44.1 k, so a few cents is expected and acceptable for an extreme).
    assert abs(_cents(det, c8)) < 15.0, f'C8 at A4={a4} -> {_cents(det, c8):+.1f} ct'

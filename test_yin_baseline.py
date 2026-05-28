"""YIN regression baseline — exact (freq, ap) change-detector on 10 signals.

v0.11 (2026-05-28): RECAPTURED after the parabolic-interpolation SIGN FIX in
yin_pitch, which removed a ~+8-cent sharp bias across the mid/high range
(A440 had read 441.99 Hz). The baselines below are the CORRECTED outputs; the
meaningful cents-accuracy contract is enforced separately by test_yin_accuracy.

(Original baselines were captured 2026-05-26 from the FFT-exact implementation
on Linux/WSL2, Python 3.12.3, NumPy 2.4.6; the planned FFT rewrite is done.)

Why (freq, ap) is sufficient to lock internal behaviour
-------------------------------------------------------
yin_pitch's internal difference array is never exposed through the public
API. However, the empirical comparison in yin_compare.py demonstrates that
the FFT-exact formulation matches the list-comp diff array to ~3e-15 relative
error, producing bit-identical chosen tau across all 10 test signals.  If a
future swap passes all 10 (freq, ap) assertions at rel=1e-10, the internal
calculation must be effectively equivalent; any divergence in the diff array
large enough to change the pitch estimate will be caught here.

Threshold note
--------------
yin_compare.py used YIN_THRESHOLD=0.15 for its printed output.  These tests
call yin_pitch() with no threshold override, so the engine default
(YIN_THRESHOLD=0.12) applies.  The 0 dB noise case therefore selects a
different (lower) tau than the comparison script printed (27.5 Hz instead of
~62.9 Hz).  That is correct: we are locking the actual engine threshold, not
the comparison-script threshold.

Noise-floor locking
-------------------
The SNR=10 dB case (case 7) returns ~434.6 Hz — not 440 Hz.  The SNR=0 dB
case (case 8) returns ~27.5 Hz, well below the true pitch.  Both are known
YIN noise-floor failures.  These tests intentionally lock the *current*
performance including those failures.  Phase 2 must replicate them exactly
(bit-identical tau), not fix them.  If a future fix is desired it should be
done as a separate, deliberate change with its own test update.
"""
from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip('numpy')
import numpy as np  # noqa: E402 — only reached if importorskip passes

from sax_audio_engine import yin_pitch, MAX_FREQ  # noqa: E402


# ---------------------------------------------------------------------------
# Signal constructors — deterministic, float64 throughout.
# ---------------------------------------------------------------------------

def _sine(f: float, sr: int, N: int) -> np.ndarray:
    """Pure sine at frequency *f* Hz, amplitude 0.5."""
    n = np.arange(N, dtype=np.float64)
    return 0.5 * np.sin(2.0 * math.pi * f * n / sr)


def _sax_like(f: float, sr: int, N: int) -> np.ndarray:
    """Five-harmonic saxophone-like tone normalised to 0.7 peak.

    Amplitudes : 0.50 / 0.30 / 0.20 / 0.10 / 0.05
    Phases (rad): 0 / 0.4 / 1.1 / 2.0 / 0.7
    """
    n = np.arange(N, dtype=np.float64)
    out = (0.50 * np.sin(2.0 * math.pi * 1 * f * n / sr)
         + 0.30 * np.sin(2.0 * math.pi * 2 * f * n / sr + 0.4)
         + 0.20 * np.sin(2.0 * math.pi * 3 * f * n / sr + 1.1)
         + 0.10 * np.sin(2.0 * math.pi * 4 * f * n / sr + 2.0)
         + 0.05 * np.sin(2.0 * math.pi * 5 * f * n / sr + 0.7))
    peak = float(np.max(np.abs(out)))
    return 0.7 * out / peak


def _noisy(f: float, sr: int, N: int, snr_db: float,
           seed: int = 42) -> np.ndarray:
    """Sine at *f* Hz with additive Gaussian noise at *snr_db* dB SNR.

    The RNG is seeded with *seed* via ``np.random.default_rng`` so the
    signal is fully reproducible regardless of global numpy state.
    """
    s = _sine(f, sr, N)
    rms_s = float(np.sqrt(np.mean(s * s)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(N).astype(np.float64)
    rms_n = float(np.sqrt(np.mean(noise * noise)))
    scale = rms_s / (rms_n * (10.0 ** (snr_db / 20.0)))
    return s + scale * noise


# ---------------------------------------------------------------------------
# Captured baselines: (freq_hz, aperiodicity)
#
# Values produced by the current list-comp yin_pitch on:
#   Python 3.12.3, NumPy 2.4.6, Linux/WSL2, 2026-05-26.
#
# repr(float) precision is preserved so future float comparison is exact
# at machine epsilon; pytest.approx(rel=1e-10) grants a tiny tolerance
# band for platforms where the final parabolic-interpolation arithmetic
# differs in the last ULP.
# ---------------------------------------------------------------------------
EXPECTED: dict[str, tuple[float, float]] = {
    # RECAPTURED 2026-05-28 after the parabolic-interpolation SIGN FIX in
    # yin_pitch. The previous baselines encoded a ~+8-cent SHARP bias across
    # the mid/high range (440 -> 441.99, Bb4 466.16 -> 462.26 = -14.6 ct,
    # 1320 -> 1352.8 = +42 ct). These values now reflect the CORRECTED
    # sub-sample lag. Cents tolerances are pinned by test_yin_accuracy; this
    # dict is the exact ULP-level change-detector.
    #
    # Case 1 — 440 Hz sine, 44.1 kHz, N=2048  (now +0.07 ct)
    'sine_440_44k_2048':   (440.01713505904496, 9.831369110901914e-05),

    # Case 2 — 110 Hz sine, 44.1 kHz, N=4096 (low note, long lag window)
    'sine_110_44k_4096':   (110.00029525232397, 9.682805696105062e-07),

    # Case 3 — 80 Hz sine, 44.1 kHz, N=8192 (near MIN_FREQ=27 Hz)
    'sine_80_44k_8192':    (80.00009645595944, 3.884649607285126e-06),

    # Case 4 — 1320 Hz sine, 44.1 kHz, N=2048  (now +0.41 ct; was +42 ct)
    'sine_1320_44k_2048':  (1320.3159047828774, 0.002897142413242738),

    # Case 5 — sax-like 220 Hz, 44.1 kHz, N=4096
    'sax_220_44k_4096':    (220.0010070301474, 0.000295576297365886),

    # Case 6 — sax-like 466.1638 Hz (Bb4), 44.1 kHz, N=2048  (now +0.04 ct)
    'sax_466_44k_2048':    (466.1740793572939, 0.001022568161530901),

    # Case 7 — 440 Hz + Gaussian noise, SNR=10 dB, seed=42: ~438.7 Hz.
    # Noise-floor approximation (NOT the parabolic bug — a separate YIN-under-
    # noise limitation); the fix nudged it closer to 440 but it stays off.
    'noisy_10db_44k_4096': (438.6762243719445, 0.09081826424593484),

    # Case 8 — 440 Hz + Gaussian noise, SNR=0 dB, seed=42: ~27.5 Hz.
    # Severe noise-floor failure (CMNDF minimum collapses to a sub-harmonic
    # lag). Locked as-is; noise robustness would be a separate change.
    'noisy_0db_44k_4096':  (27.51689162464746, 0.35327101569334746),

    # Case 9 — 440 Hz sine at 192 kHz, N=16384 (high-rate path; now +0.003 ct)
    'sine_440_192k_16384': (440.0006650826611, 1.3536498904678472e-05),

    # Case 10 — sax-like 442 Hz (slightly sharp A), 44.1 kHz, N=4096
    'sax_442_44k_4096':    (442.0157797100345, 0.00030517132237832106),
}


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

# Each entry: (case_key, signal_factory, sr, N)
# signal_factory is a zero-argument callable that returns the np.ndarray.
_CASES = [
    ('sine_440_44k_2048',
     lambda: _sine(440.0, 44100, 2048),
     44100),
    ('sine_110_44k_4096',
     lambda: _sine(110.0, 44100, 4096),
     44100),
    ('sine_80_44k_8192',
     lambda: _sine(80.0, 44100, 8192),
     44100),
    ('sine_1320_44k_2048',
     lambda: _sine(1320.0, 44100, 2048),
     44100),
    ('sax_220_44k_4096',
     lambda: _sax_like(220.0, 44100, 4096),
     44100),
    ('sax_466_44k_2048',
     lambda: _sax_like(466.1638, 44100, 2048),
     44100),
    ('noisy_10db_44k_4096',
     lambda: _noisy(440.0, 44100, 4096, snr_db=10.0, seed=42),
     44100),
    ('noisy_0db_44k_4096',
     lambda: _noisy(440.0, 44100, 4096, snr_db=0.0, seed=42),
     44100),
    ('sine_440_192k_16384',
     lambda: _sine(440.0, 192000, 16384),
     192000),
    ('sax_442_44k_4096',
     lambda: _sax_like(442.0, 44100, 4096),
     44100),
]


@pytest.mark.parametrize(
    'case,signal_fn,sr',
    [(c, fn, sr) for c, fn, sr in _CASES],
    ids=[c for c, _fn, _sr in _CASES],
)
def test_yin_baseline(case: str, signal_fn, sr: int) -> None:
    """yin_pitch must return the captured (freq, ap) to rel=1e-10."""
    sig = signal_fn()
    freq, ap = yin_pitch(sig, sr)
    expected_freq, expected_ap = EXPECTED[case]

    assert freq == pytest.approx(expected_freq, rel=1e-10), (
        f"[{case}] freq mismatch: got {freq!r}, expected {expected_freq!r}"
    )
    assert ap == pytest.approx(expected_ap, rel=1e-10, abs=1e-12), (
        f"[{case}] aperiodicity mismatch: got {ap!r}, expected {expected_ap!r}"
    )


# ---------------------------------------------------------------------------
# Additional contract tests: return type and shape invariants.
# These complement the numeric baseline by catching refactors that change
# the return type without changing the values (e.g. returning ndarray scalars
# instead of plain Python floats would break the engine callback's
# ``float(ap)`` pattern if the cast were ever removed).
# ---------------------------------------------------------------------------

def test_yin_pitch_returns_two_element_tuple() -> None:
    sig = _sine(440.0, 44100, 2048)
    result = yin_pitch(sig, 44100)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_yin_pitch_values_are_numeric() -> None:
    """freq and ap must be Python-float-compatible (float() must not raise)."""
    sig = _sine(440.0, 44100, 2048)
    freq, ap = yin_pitch(sig, 44100)
    float(freq)  # raises TypeError if not numeric
    float(ap)


def test_yin_pitch_silence_behaviour() -> None:
    """Lock zero-signal behaviour: CMNDF is all-ones, argmin picks tmin,
    parabolic step is a no-op (d=0), so freq = sr/tmin.

    yin_pitch has NO explicit all-zeros guard — it falls through to the
    argmin fallback with every CMNDF value equal to 1.0.  The returned
    aperiodicity is 1.0 (worst possible), which is what the engine's
    audio callback checks (``ap > params['yin_thr']``) to gate this frame
    as unvoiced.  The exact freq is sr/tmin where tmin tracks MAX_FREQ,
    but the only invariant the engine actually cares about is ``ap=1.0``.
    """
    sig = np.zeros(4096, dtype=np.float64)
    freq, ap = yin_pitch(sig, 44100)
    tmin = max(1, int(44100 / MAX_FREQ))
    assert freq == pytest.approx(44100 / tmin, rel=1e-10)
    assert ap == 1.0


def test_yin_pitch_freq_nonnegative() -> None:
    """yin_pitch must never return a negative frequency."""
    sig = _sine(220.0, 44100, 4096)
    freq, ap = yin_pitch(sig, 44100)
    assert freq >= 0.0


def test_yin_pitch_ap_in_unit_interval() -> None:
    """Aperiodicity is a CMNDF value; must be in [0, 1] for periodic signals."""
    sig = _sine(440.0, 44100, 2048)
    _freq, ap = yin_pitch(sig, 44100)
    assert 0.0 <= ap <= 1.0


def test_yin_pitch_tmax_lte_tmin_returns_sentinel() -> None:
    """When fmin >= fmax after integer truncation, yin_pitch returns (0.0, 1.0).

    This exercises the guard:  ``if tmax <= tmin: return 0.0, 1.0``
    We force it by passing a 2-sample buffer — tmax = N//2 = 1 = tmin.
    """
    sig = np.array([0.5, -0.5], dtype=np.float64)
    freq, ap = yin_pitch(sig, 44100)
    assert freq == 0.0
    assert ap == 1.0


# ---------------------------------------------------------------------------
# __main__ block for interactive re-capture.
# Run:  /tmp/yin-venv/bin/python test_yin_baseline.py
# to reprint all baselines (useful when updating to a new NumPy version).
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Recapturing yin_pitch baselines ...')
    print(f'NumPy {np.__version__}')
    print()
    for case, signal_fn, sr in _CASES:
        sig = signal_fn()
        freq, ap = yin_pitch(sig, sr)
        print(f"    {case!r:30s}: ({freq!r}, {ap!r}),")

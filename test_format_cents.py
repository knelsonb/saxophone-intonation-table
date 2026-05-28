"""Phase-0 safety net for the two cents-formatting helpers.

Covered functions
-----------------
* ``format_cents(value_cents, freq_hz, sample_rate)`` in sax_intonation_gui
* ``_format_cents_label(value_cents, freq_hz, sample_rate)`` in sax_intonation_chart

Both were patched for NaN/inf guard bugs:
  v0.5.7.2  — format_cents: non-finite inputs raised ValueError in int(round(NaN))
  v0.5.7.8  — _format_cents_label: same class of bug, independently patched

Design notes
------------
* Both functions are pure string formatters; neither requires a QApplication.
  Importing the modules pulls in PyQt6 (for type annotations / constants used
  elsewhere in those files), but creates no Qt objects at import time.
  The QApplication in sax_intonation_gui is guarded by ``if __name__ == '__main__'``.

* The two implementations are intentionally independent (chart is dependency-free
  of the GUI module), but they must agree on output for every input.
  Every parametrized case runs both functions and asserts identical results.

* Precision tier arithmetic (from the derivation in sax_intonation_gui):
    floor_ct = 173.0 * freq_hz / sample_rate
    floor_ct <= 0.3  → TENTHS tier  → f"{sign}{abs(v):.1f}"
    floor_ct <= 0.7  → HALVES tier  → snap to nearest 0.5, f"{sign}{abs(v):.1f}"
    floor_ct >  0.7  → WHOLES tier  → round to int,        f"{sign}{abs(int(v))}"

  At sr=44100:
    freq <= 76 Hz  → TENTHS   (173*76/44100 ≈ 0.298)
    77–178 Hz      → HALVES   (173*77/44100 ≈ 0.302; 173*178/44100 ≈ 0.698)
    freq >= 179 Hz → WHOLES   (173*179/44100 ≈ 0.702)
    440 Hz (A4)    → WHOLES   (173*440/44100 ≈ 1.726) — the common live-tuner case

Sentinel
--------
Non-finite or non-positive freq_hz → return '–' (U+2013 EN DASH).
The same sentinel is used for non-finite value_cents.

Negative-zero
-------------
Both functions apply the test ``value >= 0`` to determine the sign character.
Python guarantees ``-0.0 >= 0 is True``, so -0.0 always formats as '+0...' not '-0...'.
The docstring of format_cents explicitly documents this as the intended behaviour.
"""
from __future__ import annotations

import math

import pytest

# ``sax_intonation_gui`` imports PyQt6 at module scope (no QApplication at
# import time — that is guarded by ``if __name__ == '__main__'``). Importing
# it headless therefore needs PyQt6 present. Without this guard, collection on
# a PyQt6-less env (the always-runnable numpy/logic suite) ABORTS the whole
# run instead of skipping just this file. Same pattern as test_vendor_prefix.py.
pytest.importorskip('PyQt6', reason='PyQt6 not installed; skipping cents-format tests')

from sax_intonation_gui import format_cents  # noqa: E402
from sax_intonation_chart import _format_cents_label  # noqa: E402


# ---------------------------------------------------------------------------
# Sentinel locked in by reading the source; used throughout the test table.
# If the sentinel ever changes, this constant makes the diff surgical.
# ---------------------------------------------------------------------------
SENTINEL = '–'   # '–' U+2013 EN DASH


# ---------------------------------------------------------------------------
# Convenience: run the same assertion against both implementations.
# ---------------------------------------------------------------------------

_BOTH = pytest.mark.parametrize(
    "fn",
    [format_cents, _format_cents_label],
    ids=["format_cents", "_format_cents_label"],
)


# ===========================================================================
# 1. Non-finite value_cents — NaN and both signed infinities
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("bad_cents", [
    float('nan'),
    float('inf'),
    float('-inf'),
], ids=["nan", "+inf", "-inf"])
def test_nonfinite_value_cents_returns_sentinel(fn, bad_cents):
    """v0.5.7.2 / v0.5.7.8 regression: int(round(NaN)) used to raise ValueError.

    All non-finite value_cents must return the canonical sentinel string without
    raising, regardless of freq_hz or sample_rate.
    """
    result = fn(bad_cents, 440.0, 44100)
    assert result == SENTINEL, (
        f"{fn.__name__}({bad_cents!r}, 440.0, 44100) = {result!r}; expected {SENTINEL!r}"
    )


@_BOTH
@pytest.mark.parametrize("bad_cents", [
    float('nan'),
    float('inf'),
    float('-inf'),
], ids=["nan", "+inf", "-inf"])
def test_nonfinite_value_cents_does_not_raise(fn, bad_cents):
    """Complementary to the sentinel check: the call must not raise at all."""
    try:
        fn(bad_cents, 440.0, 44100)
    except Exception as exc:
        pytest.fail(
            f"{fn.__name__}({bad_cents!r}, 440.0, 44100) raised {type(exc).__name__}: {exc}"
        )


# ===========================================================================
# 2. Non-finite or non-positive freq_hz (a4 parameter edge cases)
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("bad_freq", [
    float('nan'),
    float('inf'),
    float('-inf'),
    0.0,           # a4=0 from v0.5.7.3 scenario
    -1.0,          # a4 negative
    -440.0,        # large negative freq
], ids=["nan", "+inf", "-inf", "zero", "neg_1", "neg_440"])
def test_nonfinite_or_nonpositive_freq_returns_sentinel(fn, bad_freq):
    """freq_hz <= 0 or non-finite freq_hz must return sentinel, not crash.

    The v0.5.7.3 scenario: an instrument transposition produces a0=0 when
    the reference pitch is unset.  The guard must fire before any arithmetic
    on freq_hz.
    """
    result = fn(-12.5, bad_freq, 44100)
    assert result == SENTINEL, (
        f"{fn.__name__}(-12.5, {bad_freq!r}, 44100) = {result!r}; expected {SENTINEL!r}"
    )


# ===========================================================================
# 3. Zero and negative-zero at the normal (A4/44100 Hz, WHOLES tier) setting
# ===========================================================================

@_BOTH
def test_positive_zero_wholes_tier(fn):
    """0.0 at 440 Hz / 44100 Hz (WHOLES tier) must format as '+0'.

    The function always emits a sign so the live tuner readout is unambiguous.
    """
    assert fn(0.0, 440.0, 44100) == '+0'


@_BOTH
def test_negative_zero_wholes_tier_coerces_to_positive(fn):
    """Python -0.0 must produce '+0', not '-0'.

    The docstring of format_cents explicitly documents that 'Negative-zero
    floats coerce to +0... via the explicit >= 0 test.'  This test locks that
    contract so a future refactor that changes the sign-selection logic will
    break here rather than silently producing '-0' in the live readout.
    """
    assert fn(-0.0, 440.0, 44100) == '+0'


@_BOTH
def test_positive_zero_tenths_tier(fn):
    """0.0 at 76 Hz / 44100 Hz (TENTHS tier) must format as '+0.0'."""
    assert fn(0.0, 76.0, 44100) == '+0.0'


@_BOTH
def test_negative_zero_tenths_tier_coerces_to_positive(fn):
    """-0.0 at TENTHS tier must produce '+0.0', not '-0.0'."""
    assert fn(-0.0, 76.0, 44100) == '+0.0'


# ===========================================================================
# 4. Typical mid-range values — one case per precision tier
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("value_cents,freq_hz,sample_rate,expected", [
    # ── WHOLES tier (440 Hz, 44100 Hz; floor ≈ 1.726) ──────────────────────
    # round(-12.5) = -12 (banker's rounding: -12.5 → nearest even -12)
    (-12.5,    440.0, 44100, '-12'),
    # round(7.85) = 8
    ( 7.85,    440.0, 44100, '+8'),
    # round(50.0) = 50 (integer input, no ambiguity)
    (50.0,     440.0, 44100, '+50'),
    # round(-23.4) = -23
    (-23.4,    440.0, 44100, '-23'),
    # ── TENTHS tier (76 Hz, 44100 Hz; floor ≈ 0.298) ───────────────────────
    # 7.85 → f"{'+' if v>=0 else '-'}{abs(7.85):.1f}" → '+7.8'
    ( 7.85,    76.0, 44100, '+7.8'),
    (-12.5,    76.0, 44100, '-12.5'),
    # ── HALVES tier (77 Hz, 44100 Hz; floor ≈ 0.302) ───────────────────────
    # round(7.85 * 2) / 2 = round(15.7) / 2 = 16/2 = 8.0
    ( 7.85,    77.0, 44100, '+8.0'),
    # round(-12.5 * 2) / 2 = round(-25.0) / 2 = -25/2 = -12.5
    (-12.5,    77.0, 44100, '-12.5'),
], ids=[
    "wholes/-12.5",
    "wholes/+7.85",
    "wholes/+50",
    "wholes/-23.4",
    "tenths/+7.85",
    "tenths/-12.5",
    "halves/+7.85",
    "halves/-12.5",
])
def test_typical_midrange_values(fn, value_cents, freq_hz, sample_rate, expected):
    """Mid-range cent values in each precision tier must match the expected format.

    These cases form the backbone of the contract: any change to the formatting
    formula, tier thresholds, or sign logic will surface here first.
    """
    result = fn(value_cents, freq_hz, sample_rate)
    assert result == expected, (
        f"{fn.__name__}({value_cents!r}, {freq_hz!r}, {sample_rate}) = {result!r}; "
        f"expected {expected!r}"
    )


# ===========================================================================
# 5. Out-of-practical-range cent values
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("value_cents,expected", [
    # round(999.9) = 1000 (Python rounds half-to-even, but 999.9 is not a half)
    ( 999.9, '+1000'),
    (-500.0, '-500'),
], ids=["+999.9", "-500.0"])
def test_out_of_practical_range_values(fn, value_cents, expected):
    """Large-magnitude cent values (beyond ±100 ct, outside the piano range) must
    not raise and must produce a sign-prefixed integer string in WHOLES tier.

    These can arise from corrupted CSV import data or race conditions where a
    reset leaves a stale extreme value in the stats ring.
    """
    result = fn(value_cents, 440.0, 44100)
    assert result == expected, (
        f"{fn.__name__}({value_cents!r}, 440.0, 44100) = {result!r}; expected {expected!r}"
    )


# ===========================================================================
# 6. sample_rate edge cases
# ===========================================================================

@_BOTH
def test_sample_rate_zero_does_not_raise(fn):
    """sr=0 must not raise.

    Some code paths pass sr=0 during engine teardown before the real sample
    rate is known.  Both functions fall back to 44100 Hz when sample_rate is
    falsy (``float(sr) if sr else 44100.0``).
    """
    try:
        result = fn(7.85, 440.0, 0)
    except Exception as exc:
        pytest.fail(
            f"{fn.__name__}(7.85, 440.0, 0) raised {type(exc).__name__}: {exc}"
        )
    # With sr=0 -> fallback 44100 -> WHOLES tier at 440 Hz -> '+8'
    assert result == '+8', (
        f"{fn.__name__}(7.85, 440.0, 0) = {result!r}; expected '+8' (sr=0 fallback to 44100)"
    )


@_BOTH
def test_sample_rate_zero_equals_default_sr_output(fn):
    """sr=0 produces the same output as sr=44100 (the fallback value).

    If this ever diverges, the sr=0 teardown path will show different text
    than the normal running path, causing a visible flicker in the UI.
    """
    assert fn(7.85, 440.0, 0) == fn(7.85, 440.0, 44100)


@_BOTH
def test_sample_rate_tiny_does_not_raise(fn):
    """sr=8 Hz (pathologically small) must not raise.

    173 * 440 / 8 = 9515, well into WHOLES tier.  The result should be '+8'.
    """
    try:
        result = fn(7.85, 440.0, 8)
    except Exception as exc:
        pytest.fail(
            f"{fn.__name__}(7.85, 440.0, 8) raised {type(exc).__name__}: {exc}"
        )
    assert result == '+8'


@_BOTH
def test_sample_rate_gigahertz_does_not_raise(fn):
    """sr=1 GHz (hypothetically large) must not raise.

    173 * 440 / 1_000_000_000 ≈ 7.6e-5, deep in TENTHS tier.
    Result should be '+7.8' (abs(7.85) formatted to one decimal place).
    """
    try:
        result = fn(7.85, 440.0, 1_000_000_000)
    except Exception as exc:
        pytest.fail(
            f"{fn.__name__}(7.85, 440.0, 1_000_000_000) raised {type(exc).__name__}: {exc}"
        )
    assert result == '+7.8'


# ===========================================================================
# 7. Tier-boundary frequencies at sr=44100
#
# The tier boundaries are at floor_ct = 0.3 (TENTHS/HALVES) and 0.7 (HALVES/WHOLES).
# Integer-Hz frequencies that land cleanly in each tier without floating-point
# ambiguity (verified by direct comparison in Python):
#   76 Hz → floor = 173*76/44100 ≈ 0.2981  → TENTHS  (0.2981 <= 0.3)
#   77 Hz → floor = 173*77/44100 ≈ 0.3021  → HALVES  (0.3021 > 0.3, <= 0.7)
#  178 Hz → floor = 173*178/44100 ≈ 0.6983 → HALVES  (0.6983 <= 0.7)
#  179 Hz → floor = 173*179/44100 ≈ 0.7022 → WHOLES  (0.7022 > 0.7)
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("freq_hz,expected_tier_label,expected", [
    # Last frequency that stays in TENTHS: 76 Hz
    (76.0,  "tenths",  '+7.8'),
    # First frequency that crosses into HALVES: 77 Hz
    (77.0,  "halves",  '+8.0'),
    # Last frequency in HALVES: 178 Hz
    (178.0, "halves",  '+8.0'),
    # First frequency in WHOLES: 179 Hz
    (179.0, "wholes",  '+8'),
], ids=[
    "76Hz_tenths_upper_boundary",
    "77Hz_halves_lower_boundary",
    "178Hz_halves_upper_boundary",
    "179Hz_wholes_lower_boundary",
])
def test_tier_boundary_frequencies(fn, freq_hz, expected_tier_label, expected):
    """Frequencies at the tier boundaries must land in the documented tier.

    These tests guard against inadvertent changes to CENT_PREC_TENTHS_MAX (0.3)
    or CENT_PREC_HALVES_MAX (0.7) that would silently shift the display
    precision of every note in a given frequency range.
    """
    result = fn(7.85, freq_hz, 44100)
    assert result == expected, (
        f"{fn.__name__}(7.85, {freq_hz} Hz, 44100) = {result!r}; "
        f"expected {expected!r} ({expected_tier_label} tier)"
    )


# ===========================================================================
# 8. Return type is always str
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("value_cents,freq_hz,sample_rate", [
    (float('nan'), 440.0, 44100),
    (0.0,          440.0, 44100),
    (-12.5,        440.0, 44100),
    ( 7.85,        76.0,  44100),
    (-0.0,         77.0,  44100),
    (999.9,        440.0, 0),
], ids=[
    "nan_is_str",
    "zero_is_str",
    "negative_is_str",
    "tenths_is_str",
    "neg_zero_halves_is_str",
    "large_sr0_is_str",
])
def test_return_type_is_always_str(fn, value_cents, freq_hz, sample_rate):
    """The return value must always be a ``str``, never None, never a float."""
    result = fn(value_cents, freq_hz, sample_rate)
    assert isinstance(result, str), (
        f"{fn.__name__}({value_cents!r}, {freq_hz!r}, {sample_rate}) "
        f"returned {type(result).__name__!r}, expected 'str'"
    )


# ===========================================================================
# 9. The two implementations agree on every input
#
# The chart module is intentionally a copy of the GUI helper rather than an
# import (to keep the chart module dependency-free).  Any silent divergence
# between the two would produce different labels in the live tuner vs the
# exported chart PNG for the same session data.
# ===========================================================================

@pytest.mark.parametrize("value_cents,freq_hz,sample_rate", [
    # Non-finite sentinels
    (float('nan'),  440.0, 44100),
    (float('inf'),  440.0, 44100),
    (float('-inf'), 440.0, 44100),
    # Non-positive freq
    (0.0,  0.0,   44100),
    (0.0,  -1.0,  44100),
    # Zero and neg-zero, all three tiers
    (0.0,  440.0, 44100),
    (-0.0, 440.0, 44100),
    (0.0,  76.0,  44100),
    (-0.0, 76.0,  44100),
    (0.0,  77.0,  44100),
    (-0.0, 77.0,  44100),
    # Typical values, all three tiers
    (-12.5, 440.0, 44100),
    ( 7.85, 440.0, 44100),
    ( 50.0, 440.0, 44100),
    (-23.4, 440.0, 44100),
    ( 7.85, 76.0,  44100),
    (-12.5, 76.0,  44100),
    ( 7.85, 77.0,  44100),
    (-12.5, 77.0,  44100),
    # Out-of-range
    ( 999.9, 440.0, 44100),
    (-500.0, 440.0, 44100),
    # sr edge cases
    (7.85, 440.0, 0),
    (7.85, 440.0, 8),
    (7.85, 440.0, 1_000_000_000),
], ids=[
    "nan", "+inf", "-inf",
    "freq_zero", "freq_neg",
    "zero_wholes", "negzero_wholes",
    "zero_tenths", "negzero_tenths",
    "zero_halves", "negzero_halves",
    "wholes/-12.5", "wholes/+7.85", "wholes/+50", "wholes/-23.4",
    "tenths/+7.85", "tenths/-12.5",
    "halves/+7.85", "halves/-12.5",
    "oor/+999.9", "oor/-500",
    "sr_zero", "sr_8hz", "sr_1ghz",
])
def test_implementations_agree(value_cents, freq_hz, sample_rate):
    """format_cents and _format_cents_label must produce identical output.

    They share the same algorithm but live in separate modules.  Any divergence
    means the live-tuner readout and the exported chart PNG would show different
    numbers for the same measurement, which is a correctness bug.
    """
    gui_result   = format_cents(value_cents, freq_hz, sample_rate)
    chart_result = _format_cents_label(value_cents, freq_hz, sample_rate)
    assert gui_result == chart_result, (
        f"format_cents({value_cents!r}, {freq_hz!r}, {sample_rate}) = {gui_result!r}\n"
        f"_format_cents_label({value_cents!r}, {freq_hz!r}, {sample_rate}) = {chart_result!r}\n"
        "The two implementations disagree — one was patched without updating the other."
    )


# ===========================================================================
# 10. Sign contract — output always starts with '+' or '-'
#
# The function promises an explicit sign on every non-sentinel output so the
# live readout never shows an ambiguous bare number (e.g. '8' instead of '+8').
# ===========================================================================

@_BOTH
@pytest.mark.parametrize("value_cents,freq_hz,sample_rate", [
    (0.0,   440.0, 44100),
    (-0.0,  440.0, 44100),
    (7.85,  440.0, 44100),
    (-12.5, 440.0, 44100),
    (50.0,  76.0,  44100),
    (-23.0, 76.0,  44100),
    (7.85,  77.0,  44100),
    (-7.85, 77.0,  44100),
], ids=[
    "zero", "negzero",
    "positive_wholes", "negative_wholes",
    "positive_tenths", "negative_tenths",
    "positive_halves", "negative_halves",
])
def test_output_always_has_explicit_sign(fn, value_cents, freq_hz, sample_rate):
    """Every non-sentinel result must start with '+' or '-'.

    A bare unsigned result (e.g. '8' instead of '+8') would cause the
    live-tuner label to jump between signed and unsigned formats depending
    on whether the deviation is positive or negative, creating a flickering
    display when the player crosses the zero line.
    """
    result = fn(value_cents, freq_hz, sample_rate)
    assert result in (SENTINEL,) or result[0] in ('+', '-'), (
        f"{fn.__name__}({value_cents!r}, {freq_hz!r}, {sample_rate}) = {result!r} "
        f"has neither a sign prefix nor the sentinel"
    )


# ===========================================================================
# 11. Sentinel identity — the exact glyph character matters
#
# Other parts of the codebase test for the sentinel by string equality
# (e.g. ``mean_str = '–' if has_data else ...``).  If the sentinel ever
# changes from U+2013 EN DASH to something else (ASCII hyphen, em dash,
# 'N/A', etc.), those callers silently break.
# ===========================================================================

def test_sentinel_is_en_dash():
    """The sentinel glyph must be U+2013 EN DASH, not a hyphen-minus or em dash.

    Callers in the paint path compare the formatted string to the literal '–'
    to decide whether to paint a placeholder cell.  If the sentinel becomes
    a different character, those cells will incorrectly show the sentinel
    string as data.
    """
    assert SENTINEL == '–', "Update this test if the sentinel is intentionally changed."
    # Confirm both functions return exactly that character, not a look-alike.
    gui_result   = format_cents(float('nan'), 440.0, 44100)
    chart_result = _format_cents_label(float('nan'), 440.0, 44100)
    assert gui_result   == '–'
    assert chart_result == '–'
    # Ensure it is NOT the ASCII hyphen (U+002D) or the em dash (U+2014).
    assert gui_result   != '-'      # ASCII hyphen-minus
    assert gui_result   != '—' # EM DASH
    assert chart_result != '-'
    assert chart_result != '—'

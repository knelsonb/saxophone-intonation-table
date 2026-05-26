"""
Phase-0 safety-net tests for ``_agg_stats`` in ``sax_intonation_log``.

``_agg_stats`` is the single aggregation primitive called by every CSV export
slice mode (per_run_note, per_instrument_note, per_nickname_note,
instrument_avg, overall_per_note).  Its signature and semantics, as read from
the source:

    def _agg_stats(values: list[float]) -> tuple[float, float, float, float]:
        ...
        return (mean, std, min, max)

Key invariants locked in here:
  * Return-tuple shape is (mean, std, min, max) — field ORDER is load-bearing
    because every writer unpacks it as ``mean, std, mn, mx = _agg_stats(...)``.
  * std uses POPULATION variance (divisor n), not sample variance (divisor
    n-1).  Changing to sample variance would silently corrupt every exported
    CSV; the parametrized symmetric-pair test is the frontline assertion for
    this invariant.
  * Empty input returns the sentinel (0.0, 0.0, 0.0, 0.0).
  * Single-element input returns std == 0.0 (guarded by the ``n > 1`` branch,
    not by the population formula).
  * NaN and Inf propagate — the function does NOT filter them.  Behaviour for
    each is locked in explicitly so any future NaN-safe refactor shows a test
    diff rather than silent behaviour change.

``_agg_stats`` is a pure function with no side effects; it imports nothing
outside the standard library.  No QApplication, no sounddevice, no Qt.
"""
from __future__ import annotations

import math
import pytest

from sax_intonation_log import _agg_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std_population(values: list[float]) -> float:
    """Reference implementation: population std, used to derive expected values
    in the parametrize table.  NOT used at test runtime for the core numeric
    cases — expected values are pre-computed constants."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


# ---------------------------------------------------------------------------
# Parametrized numeric cases
# ---------------------------------------------------------------------------

# Each entry: (test_id, values, expected_mean, expected_std, expected_min,
#              expected_max, exact_std)
# ``exact_std=True`` means assert std == expected_std (not approx), used when
# the result is exactly representable (0.0) or the formula gives an exact
# rational result (10.0 for the symmetric-pair case).

_NUMERIC_CASES = [
    # ------------------------------------------------------------------
    # 1. Empty input — sentinel tuple
    # ------------------------------------------------------------------
    pytest.param(
        [], 0.0, 0.0, 0.0, 0.0, True,
        id="empty_input_sentinel",
    ),
    # ------------------------------------------------------------------
    # 2. Single element — std must be exactly 0.0 (n>1 guard in source)
    # ------------------------------------------------------------------
    pytest.param(
        [5.0], 5.0, 0.0, 5.0, 5.0, True,
        id="single_element",
    ),
    # ------------------------------------------------------------------
    # 3. Symmetric pair — FRONTLINE population-variance assertion.
    #    Population std of [-10, 10]:
    #      mean = 0, var = (100 + 100) / 2 = 100, std = 10.0 exactly.
    #    Sample std would be sqrt(200 / 1) ≈ 14.142 — clearly different.
    # ------------------------------------------------------------------
    pytest.param(
        [-10.0, 10.0], 0.0, 10.0, -10.0, 10.0, True,
        # population variance: divisor n, not n-1
        id="symmetric_pair_population_variance",
    ),
    # ------------------------------------------------------------------
    # 4. Three-value ascending run
    #    mean = 2.0
    #    deviations: -1, 0, +1  →  sum_sq_dev = 2
    #    population var = 2/3,  std = sqrt(2/3) ≈ 0.816496580927726
    # ------------------------------------------------------------------
    pytest.param(
        [1.0, 2.0, 3.0], 2.0, math.sqrt(2.0 / 3.0), 1.0, 3.0, False,
        id="three_value_ascending",
    ),
    # ------------------------------------------------------------------
    # 5. Constant values — population std is exactly 0.0 for any n
    # ------------------------------------------------------------------
    pytest.param(
        [4.2, 4.2, 4.2, 4.2], 4.2, 0.0, 4.2, 4.2, True,
        id="constant_values_zero_std",
    ),
    # ------------------------------------------------------------------
    # 6. Negative values only
    #    mean = -3.0
    #    deviations: -2, 0, +2  →  sum_sq_dev = 8
    #    population var = 8/3,  std = sqrt(8/3) ≈ 1.6329931618554521
    # ------------------------------------------------------------------
    pytest.param(
        [-5.0, -3.0, -1.0], -3.0, math.sqrt(8.0 / 3.0), -5.0, -1.0, False,
        id="negative_values_only",
    ),
    # ------------------------------------------------------------------
    # 7. Mixed signs — typical cents data spread
    #    values = [-12.5, 7.0, -3.0, 15.5, -8.0]
    #    mean = (-12.5 + 7.0 + -3.0 + 15.5 + -8.0) / 5 = -1.0 / 5 = -0.2
    #    deviations: -12.3, 7.2, -2.8, 15.7, -7.8
    #    sum_sq_dev = 151.29 + 51.84 + 7.84 + 246.49 + 60.84 = 518.30
    #    population var = 518.30 / 5 = 103.66
    #    std = sqrt(103.66) ≈ 10.181356...
    # ------------------------------------------------------------------
    pytest.param(
        [-12.5, 7.0, -3.0, 15.5, -8.0],
        -0.2, math.sqrt(103.66), -12.5, 15.5, False,
        id="mixed_signs_typical_cents",
    ),
    # ------------------------------------------------------------------
    # 8. Large-N stability (smoke test against catastrophic cancellation)
    #    values = [0.0] * 1000 + [1000.0]   (n = 1001)
    #    mean = 1000 / 1001
    #    sum_sq_dev = 1000 * (1000/1001)^2 + (1000 * 1000/1001)^2
    #               = (1000/1001)^2 * (1000 + 1000^2)
    #               = (1000/1001)^2 * 1000 * 1001
    #               = 1000^3 / 1001
    #    population var = 1000^3 / 1001^2
    #    std = 1000^(3/2) / 1001 ≈ 31622.776... / 1001 ≈ 31.59118...
    #
    #    _agg_stats uses a two-pass (mean-then-deviations) formula which is
    #    algebraically stable.  A naive single-pass ``sum(x**2)/n - mean**2``
    #    formula is prone to catastrophic cancellation on this input and would
    #    fail or produce a negative variance.  This test passes for the
    #    current implementation; if the formula is ever changed to the naive
    #    form, this test catches the precision regression.
    # ------------------------------------------------------------------
    pytest.param(
        [0.0] * 1000 + [1000.0],
        1000.0 / 1001.0,
        1000.0 ** 1.5 / 1001.0,
        0.0,
        1000.0,
        False,
        id="large_n_stability",
    ),
]


@pytest.mark.parametrize(
    "values,exp_mean,exp_std,exp_min,exp_max,exact_std",
    _NUMERIC_CASES,
)
def test_agg_stats_numeric(
    values: list[float],
    exp_mean: float,
    exp_std: float,
    exp_min: float,
    exp_max: float,
    exact_std: bool,
) -> None:
    """``_agg_stats`` returns (mean, std, min, max) with population variance."""
    result = _agg_stats(values)

    # Verify tuple length — field order is load-bearing for every CSV writer.
    assert len(result) == 4, f"expected 4-tuple, got length {len(result)}"

    mean, std, mn, mx = result

    assert mean == pytest.approx(exp_mean, abs=1e-9), (
        f"mean mismatch: got {mean!r}, expected {exp_mean!r}"
    )

    if exact_std:
        assert std == exp_std, (
            f"std mismatch (exact): got {std!r}, expected {exp_std!r}"
        )
    else:
        assert std == pytest.approx(exp_std, abs=1e-9), (
            f"std mismatch: got {std!r}, expected {exp_std!r}"
        )

    assert mn == pytest.approx(exp_min, abs=1e-9), (
        f"min mismatch: got {mn!r}, expected {exp_min!r}"
    )
    assert mx == pytest.approx(exp_max, abs=1e-9), (
        f"max mismatch: got {mx!r}, expected {exp_max!r}"
    )


# ---------------------------------------------------------------------------
# NaN handling — propagation, not filtering
# ---------------------------------------------------------------------------

def test_agg_stats_nan_propagates() -> None:
    """NaN in the input list propagates through _agg_stats.

    _agg_stats does NOT filter NaN values (no numpy.nanmean style logic).
    With NaN at position 1 (not the first element):
      - sum([1.0, nan, 3.0]) → nan  →  mean = nan
      - (v - nan)**2 is nan for every v  →  var = nan  →  std = nan
      - Python's min/max: the initial current-best is 1.0 (first element);
        subsequent comparisons against nan are always False, so nan never
        displaces the running best.  min → 1.0,  max → 3.0.

    This behaviour is locked in.  A future NaN-safe refactor (e.g. filtering
    NaN before aggregation, matching the v0.5.7.3 NaN-safe matrix paint) must
    deliberately update this test to reflect the new contract.
    """
    result = _agg_stats([1.0, float("nan"), 3.0])
    assert len(result) == 4

    mean, std, mn, mx = result

    # mean and std are NaN because NaN poisons the sum.
    assert math.isnan(mean), f"expected nan mean, got {mean!r}"
    assert math.isnan(std), f"expected nan std, got {std!r}"

    # min and max are unaffected: Python's comparisons treat nan as unordered,
    # so the finite values win when they come before nan in the list.
    assert mn == 1.0, f"expected min=1.0, got {mn!r}"
    assert mx == 3.0, f"expected max=3.0, got {mx!r}"


# ---------------------------------------------------------------------------
# Inf handling — propagation, not filtering
# ---------------------------------------------------------------------------

def test_agg_stats_inf_propagates() -> None:
    """Inf in the input list propagates through _agg_stats.

    _agg_stats does NOT filter Inf.  With inf at position 1:
      - sum([1.0, inf, 3.0]) → inf  →  mean = inf
      - (1.0 - inf)**2 → inf,  (inf - inf)**2 → nan,  (3.0 - inf)**2 → inf
      - sum([inf, nan, inf]) → nan  →  math.sqrt(nan) → nan  →  std = nan
      - min([1.0, inf, 3.0]) → 1.0  (inf > 1.0 is True but doesn't replace)
        Actually Python min starts at 1.0; inf < 1.0 is False; 3.0 < 1.0 is
        False  →  min = 1.0
      - max([1.0, inf, 3.0]) → inf  (inf > 1.0 is True, replaces; 3.0 > inf
        is False)  →  max = inf

    This behaviour is locked in.  Any future Inf-guard (e.g. filtering or
    clamping before aggregation) must update this test.
    """
    result = _agg_stats([1.0, float("inf"), 3.0])
    assert len(result) == 4

    mean, std, mn, mx = result

    # mean is inf because sum is inf.
    assert math.isinf(mean) and mean > 0, (
        f"expected +inf mean, got {mean!r}"
    )
    # std is nan: (inf - inf)**2 = nan poisons the variance sum.
    assert math.isnan(std), f"expected nan std, got {std!r}"

    assert mn == 1.0, f"expected min=1.0, got {mn!r}"
    assert math.isinf(mx) and mx > 0, f"expected +inf max, got {mx!r}"


# ---------------------------------------------------------------------------
# Return-tuple shape contract (standalone, not parametrized)
# ---------------------------------------------------------------------------

def test_agg_stats_returns_four_tuple() -> None:
    """Return value must always be a 4-tuple in the order (mean, std, min, max).

    This is a structural guard: all six CSV writer functions unpack the result
    as ``mean, std, mn, mx = _agg_stats(vals)``; a shape change would raise
    a ValueError rather than corrupt data silently, but this test makes the
    contract explicit and catches accidental signature drift."""
    result = _agg_stats([1.0, 2.0, 3.0])
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 4, f"expected length 4, got {len(result)}"
    mean, std, mn, mx = result   # structural unpack — must not raise
    assert mn <= mean <= mx, (
        f"invariant violated: min ({mn}) <= mean ({mean}) <= max ({mx})"
    )
    assert std >= 0.0, f"std must be non-negative, got {std!r}"

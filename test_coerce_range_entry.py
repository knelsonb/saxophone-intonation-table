"""Phase-0 safety net for ``_coerce_range_entry`` in ``sax_instruments``.

``_coerce_range_entry`` is the sole validation gate for every entry that
``load_range_overrides`` reads from the user-edited
``instrument_ranges.json``.  A defect here silently poisons the per-
instrument range cache that ``fingered_range()`` consults for every
intonation-table build.

No PyQt6 dependency — this module is pure Python.

Design notes
------------
* All tests use ``pytest.mark.parametrize`` so each row is an independent
  test case with its own ID in the pytest report.
* Cases that lock a *defect* (current behaviour that is technically wrong
  but must not regress invisibly) are clearly labelled.
* The rejection sentinel is ``None`` — not ``()`` and not a raised
  exception — except for the ``OverflowError`` cases documented below.

Discovered behaviour (locked, not fixed)
-----------------------------------------
``int(float('inf'))`` and ``int(-float('inf'))`` raise ``OverflowError``,
which is **not** a subclass of ``ValueError`` or ``TypeError``.  The
``except (TypeError, ValueError)`` guard in ``_coerce_range_entry`` does
not catch it, so ``[48, float('inf')]`` (and symmetric cases) propagate
an unhandled ``OverflowError`` to the caller instead of returning
``None``.  The separate parametrize table ``test_coerce_range_entry_inf``
locks this behaviour so any future silent-accept *or* proper guard
produces a visible test failure requiring deliberate sign-off.
"""

from __future__ import annotations

import math

import pytest

from sax_instruments import _coerce_range_entry


# ---------------------------------------------------------------------------
# 1. Main parametrize table — ``None``-or-tuple cases
# ---------------------------------------------------------------------------
# fmt: off
_CASES: list[tuple[object, tuple[int, int] | None]] = [
    # --- valid integer tuple / list ----------------------------------------
    pytest.param((48, 79),        (48, 79),  id="valid_int_tuple"),
    pytest.param([48, 79],        (48, 79),  id="valid_int_list"),

    # --- string numerics (tolerant parsing via int()) -----------------------
    pytest.param(["48", "79"],    (48, 79),  id="string_numerics"),
    pytest.param(["48", 79],      (48, 79),  id="mixed_string_int"),

    # --- float numerics — int() truncates toward zero, not round() ----------
    # int(48.9) == 48, int(79.9) == 79 — truncation, not rounding.
    pytest.param([48.0, 79.0],    (48, 79),  id="float_exact"),
    pytest.param([48.9, 79.9],    (48, 79),  id="float_truncated_not_rounded"),
    pytest.param([0.9, 126.9],    (0, 126),  id="float_boundary_truncation"),

    # --- boundary inclusivity (full MIDI range 0..127 accepted) -------------
    pytest.param((0, 127),        (0, 127),  id="boundary_both_endpoints"),
    pytest.param((0, 0),          (0, 0),    id="boundary_lo_eq_hi_at_zero"),
    pytest.param((127, 127),      (127, 127), id="boundary_lo_eq_hi_at_127"),

    # --- single-note range (lo == hi) — degenerate but valid ----------------
    pytest.param((60, 60),        (60, 60),  id="single_note_range"),

    # --- inverted range (lo > hi) rejected ----------------------------------
    pytest.param((79, 48),        None,      id="inverted_range"),
    pytest.param((127, 0),        None,      id="inverted_range_extremes"),

    # --- below MIDI range ---------------------------------------------------
    pytest.param((-1, 50),        None,      id="lo_below_zero"),
    pytest.param((-128, 50),      None,      id="lo_far_below_zero"),

    # --- above MIDI range ---------------------------------------------------
    pytest.param((50, 128),       None,      id="hi_above_127"),
    pytest.param((50, 255),       None,      id="hi_far_above_127"),

    # --- both bounds out of range -------------------------------------------
    pytest.param((-5, 200),       None,      id="both_out_of_range"),

    # --- non-numeric values -------------------------------------------------
    pytest.param(("foo", "bar"),  None,      id="non_numeric_strings"),
    pytest.param(("48", "x"),     None,      id="second_element_non_numeric"),
    pytest.param(("x", "79"),     None,      id="first_element_non_numeric"),

    # --- wrong length -------------------------------------------------------
    pytest.param((48,),           None,      id="single_element_tuple"),
    pytest.param([48],            None,      id="single_element_list"),
    pytest.param((48, 79, 100),   None,      id="three_element_tuple"),
    pytest.param([48, 79, 100],   None,      id="three_element_list"),

    # --- empty containers ---------------------------------------------------
    pytest.param([],              None,      id="empty_list"),
    pytest.param((),              None,      id="empty_tuple"),

    # --- None input (must not raise) ----------------------------------------
    pytest.param(None,            None,      id="none_input"),

    # --- dict input (JSON dicts with named keys are rejected) ---------------
    # The function only accepts list or tuple, so dicts → None.
    pytest.param({"lo": 48, "hi": 79},   None,  id="dict_input_named_keys"),
    pytest.param({0: 48, 1: 79},         None,  id="dict_input_int_keys"),

    # --- other scalar and container types ------------------------------------
    pytest.param(48,              None,      id="bare_int"),
    pytest.param("48,79",         None,      id="bare_string"),
    pytest.param(48.0,            None,      id="bare_float"),

    # --- NaN — int(float('nan')) raises ValueError, caught → None -----------
    pytest.param([float("nan"), 79],   None,  id="nan_lo"),
    pytest.param([48, float("nan")],   None,  id="nan_hi"),
    pytest.param([float("nan"), float("nan")], None, id="nan_both"),
]
# fmt: on


@pytest.mark.parametrize("value,expected", _CASES)
def test_coerce_range_entry(value, expected):
    result = _coerce_range_entry(value)
    assert result == expected


# ---------------------------------------------------------------------------
# 2. Return-type contract — accepted entries are always a tuple, never a list
# ---------------------------------------------------------------------------

def test_coerce_range_entry_returns_tuple_not_list():
    """The function must return a tuple, not a list, so callers can use it
    directly as a dict value or compare to baked ``(lo, hi)`` tuples."""
    result = _coerce_range_entry([48, 79])
    assert isinstance(result, tuple), (
        f"Expected tuple, got {type(result).__name__}"
    )


def test_coerce_range_entry_returns_ints_not_floats():
    """Values in the returned tuple must be plain ints, not floats, because
    they feed MIDI arithmetic downstream."""
    result = _coerce_range_entry([48.0, 79.0])
    assert result is not None
    lo, hi = result
    assert isinstance(lo, int) and isinstance(hi, int), (
        f"Expected (int, int), got ({type(lo).__name__}, {type(hi).__name__})"
    )


# ---------------------------------------------------------------------------
# 3. Truncation vs rounding — pin the semantics of float coercion
#
# int() truncates toward zero, so int(0.9) == 0 (not 1) and
# int(-0.9) == 0 (not -1).  If a user edits the JSON with fractional
# values, the function silently truncates.  This table makes that explicit
# so a future switch to round() is visible.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lo_f,hi_f,expected_lo,expected_hi", [
    (0.1,   126.9,  0,   126),   # truncation, not ceil/round
    (1.9,   79.0,   1,   79),    # lo truncated down from 1.9 → 1
    (48.0,  79.999, 48,  79),    # hi truncated down from 79.999 → 79
])
def test_coerce_range_entry_float_truncation(lo_f, hi_f, expected_lo, expected_hi):
    result = _coerce_range_entry([lo_f, hi_f])
    assert result == (expected_lo, expected_hi)


# ---------------------------------------------------------------------------
# 4. Infinity inputs — DEFECT locked, not fixed.
#
# ``int(float('inf'))`` raises ``OverflowError``, which is NOT a subclass
# of ``ValueError`` or ``TypeError``.  The ``except (TypeError, ValueError)``
# block in ``_coerce_range_entry`` does not catch it, so these inputs
# propagate an unhandled ``OverflowError`` instead of returning ``None``.
#
# This is a latent bug: a user-edited JSON entry such as
#   "eb_alto": [48, 1e308]    (rounds to inf in IEEE-754)
# would crash ``load_range_overrides`` instead of silently skipping the key.
#
# Tests below lock CURRENT behaviour (raises).  When the bug is fixed
# (add OverflowError to the except clause), these tests must be updated to
# expect ``None`` instead.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    pytest.param([48, float("inf")],         id="hi_pos_inf"),
    pytest.param([float("inf"), 79],         id="lo_pos_inf"),
    pytest.param([48, float("-inf")],        id="hi_neg_inf"),
    pytest.param([float("-inf"), 79],        id="lo_neg_inf"),
    pytest.param([float("inf"), float("inf")], id="both_inf"),
])
def test_coerce_range_entry_inf_raises_overflow(value):
    """DEFECT: inf inputs raise OverflowError instead of returning None.

    The except clause catches (TypeError, ValueError) but not OverflowError.
    Lock current behaviour.  When fixed, change to assert result is None."""
    with pytest.raises(OverflowError):
        _coerce_range_entry(value)


# ---------------------------------------------------------------------------
# 5. Interaction with fingered_range() — overrides never exceed 0..127
#
# Spot-check that a value returned by _coerce_range_entry satisfies the
# downstream contract: lo and hi are in [0, 127] and lo <= hi.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    (0, 127),
    (57, 96),
    (60, 60),
    [48, 79],
    ["52", "91"],
])
def test_coerce_range_entry_accepted_values_satisfy_downstream_contract(value):
    result = _coerce_range_entry(value)
    assert result is not None
    lo, hi = result
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert 0 <= lo <= 127
    assert 0 <= hi <= 127
    assert lo <= hi

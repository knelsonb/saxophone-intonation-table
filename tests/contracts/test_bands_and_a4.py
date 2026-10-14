"""Contracts for the cents bands and the autotune A4 suggestion.

cents_band classifies the whole-cent value the views print, so its boundaries sit
on Python's round-half-to-even: 4.5 prints as "+4" and is in tune, 5.5 prints as
"+6" and is close, 12.5 prints as "+12" and is close.

suggest_a4 is the maths behind the Autotune button: a sqrt(n)-weighted median of
the per-note mean deviations, turned into an A4 in whole Hz and clamped to the
430..450 range the A4 combo offers.
"""
from __future__ import annotations

import math
import random

import pytest

from intonation.core.pitch import cents_dev
from intonation.ui.instrument_controller import (
    A4_MAX_HZ,
    A4_MIN_HZ,
    AUTOTUNE_MIN_NOTES,
    AUTOTUNE_MIN_SAMPLES,
    A4Suggestion,
    suggest_a4,
)
from intonation.ui.theme import CLOSE_CENTS, IN_TUNE_CENTS, cents_band

# ---------------------------------------------------------------------------
# cents_band
# ---------------------------------------------------------------------------
_BAND_TABLE = [
    # in tune: the printed value is at most 5
    (0.0, 'in'), (-0.0, 'in'), (0.49, 'in'), (0.5, 'in'),
    (4.49, 'in'), (4.5, 'in'), (5.0, 'in'), (5.49, 'in'),
    # close: printed 6..12
    (5.5, 'close'), (5.51, 'close'), (6.0, 'close'), (11.99, 'close'),
    (12.0, 'close'), (12.49, 'close'), (12.5, 'close'),
    # off: printed 13 and beyond
    (12.51, 'off'), (13.0, 'off'), (13.5, 'off'), (50.0, 'off'), (1e9, 'off'),
]
_BAND_TABLE += [(-v, band) for v, band in _BAND_TABLE if v > 0]


@pytest.mark.parametrize('cents,band', _BAND_TABLE,
                         ids=[f'{c:+g}' for c, _ in _BAND_TABLE])
def test_cents_band_boundaries(cents: float, band: str) -> None:
    assert cents_band(cents) == band


def test_band_thresholds_are_5_and_12() -> None:
    assert (IN_TUNE_CENTS, CLOSE_CENTS) == (5, 12)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'),
                                   None, 'abc', '', object()])
def test_cents_band_treats_unusable_input_as_off(value) -> None:
    assert cents_band(value) == 'off'


@pytest.mark.parametrize('value,band', [('5', 'in'), ('12.4', 'close'), (True, 'in')])
def test_cents_band_accepts_numeric_text_and_bools(value, band: str) -> None:
    assert cents_band(value) == band


def test_cents_band_is_symmetric_around_zero() -> None:
    for tenth in range(0, 400):
        v = tenth / 10.0
        assert cents_band(v) == cents_band(-v), v


def test_cents_band_never_gets_better_as_the_error_grows() -> None:
    rank = {'in': 0, 'close': 1, 'off': 2}
    last = 0
    for hundredth in range(0, 3000):
        r = rank[cents_band(hundredth / 100.0)]
        assert r >= last, hundredth
        last = r


# ---------------------------------------------------------------------------
# suggest_a4: qualification
# ---------------------------------------------------------------------------
def _notes(*means: float, n: int = AUTOTUNE_MIN_SAMPLES) -> list[tuple[float, int]]:
    return [(m, n) for m in means]


def test_thresholds() -> None:
    assert AUTOTUNE_MIN_SAMPLES == 5
    assert AUTOTUNE_MIN_NOTES == 3
    assert (A4_MIN_HZ, A4_MAX_HZ) == (430, 450)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_fewer_than_three_qualified_notes_gives_none(count: int) -> None:
    assert suggest_a4(_notes(*([10.0] * count)), 440.0) is None


def test_three_qualified_notes_is_enough() -> None:
    result = suggest_a4(_notes(1.0, 2.0, 3.0), 440.0)
    assert isinstance(result, A4Suggestion)
    assert result.notes == 3


def test_a_note_needs_five_samples_to_qualify() -> None:
    four = AUTOTUNE_MIN_SAMPLES - 1
    assert suggest_a4(_notes(1.0, 2.0, 3.0, n=four), 440.0) is None
    # Two full notes plus many that are one sample short stay below three.
    mixed = _notes(1.0, 2.0) + _notes(3.0, 4.0, 5.0, 6.0, n=four)
    assert suggest_a4(mixed, 440.0) is None
    # The third full note tips it over, and only the full notes are counted.
    mixed.append((7.0, AUTOTUNE_MIN_SAMPLES))
    result = suggest_a4(mixed, 440.0)
    assert result is not None and result.notes == 3


def test_notes_with_no_samples_are_ignored() -> None:
    assert suggest_a4(_notes(1.0, 2.0) + [(0.0, 0)], 440.0) is None


def test_accepts_any_iterable() -> None:
    assert suggest_a4(iter(_notes(4.0, 5.0, 6.0)), 440.0).notes == 3
    assert suggest_a4(tuple(_notes(4.0, 5.0, 6.0)), 440.0).notes == 3


# ---------------------------------------------------------------------------
# suggest_a4: weighted median
# ---------------------------------------------------------------------------
def test_equal_weights_take_the_middle_note() -> None:
    result = suggest_a4(_notes(-10.0, 0.0, 30.0), 440.0)
    assert result.offset_ct == 0.0


def test_input_order_does_not_matter() -> None:
    forward = suggest_a4(_notes(-20.0, -5.0, 0.0, 8.0, 40.0), 440.0)
    shuffled = suggest_a4(_notes(8.0, 40.0, -20.0, 0.0, -5.0), 440.0)
    assert forward == shuffled
    assert forward.offset_ct == 0.0


def test_a_stray_note_does_not_move_the_median_but_moves_the_mean() -> None:
    result = suggest_a4(_notes(0.0, 0.0, 0.0, 0.0, 100.0), 440.0)
    assert result.offset_ct == 0.0
    assert result.mean_weighted_ct == pytest.approx(20.0)


def test_even_count_takes_the_lower_middle_note() -> None:
    """With two equally heavy halves the cumulative weight reaches half at the
    lower middle note; this pins that choice."""
    result = suggest_a4(_notes(0.0, 10.0, 20.0, 30.0), 440.0)
    assert result.offset_ct == 10.0


def test_weights_are_the_square_root_of_the_sample_count() -> None:
    """Two light notes (n=5, weight 2.24 each, 4.47 together) against a heavy
    one decide who holds more than half the weight."""
    light = [(0.0, 5), (10.0, 5)]
    # n=100 -> weight 10: the heavy note is the median.
    assert suggest_a4(light + [(20.0, 100)], 440.0).offset_ct == 20.0
    # n=25 -> weight 5, still above 4.47: the heavy note wins.
    assert suggest_a4(light + [(20.0, 25)], 440.0).offset_ct == 20.0
    # n=16 -> weight 4, below 4.47: the middle note wins. A linear weight
    # (16 against 5 + 5) would have chosen the heavy note.
    assert suggest_a4(light + [(20.0, 16)], 440.0).offset_ct == 10.0
    # Equal counts at any size: the middle note.
    assert suggest_a4([(0.0, 50), (10.0, 50), (20.0, 50)], 440.0).offset_ct == 10.0


def test_weighted_mean_uses_the_same_weights() -> None:
    stats = [(0.0, 5), (10.0, 5), (20.0, 100)]
    w = [math.sqrt(n) for _m, n in stats]
    expected = sum(m * wi for (m, _n), wi in zip(stats, w)) / sum(w)
    assert suggest_a4(stats, 440.0).mean_weighted_ct == pytest.approx(expected)


# ---------------------------------------------------------------------------
# suggest_a4: conversion to a whole-Hz A4
# ---------------------------------------------------------------------------
def test_zero_offset_returns_the_current_reference() -> None:
    result = suggest_a4(_notes(0.0, 0.0, 0.0), 440.0)
    assert (result.optimal, result.rounded, result.clamped) == (440.0, 440, 440)


def test_optimal_is_the_reference_shifted_by_the_offset() -> None:
    result = suggest_a4(_notes(12.0, 12.0, 12.0), 440.0)
    assert result.optimal == pytest.approx(440.0 * 2.0 ** (12.0 / 1200.0))
    assert result.rounded == 443


@pytest.mark.parametrize('a4,expected', [
    (439.5, 440), (440.5, 440), (441.5, 442), (442.5, 442), (443.5, 444),
    (449.5, 450), (430.5, 430),
])
def test_rounding_is_half_to_even(a4: float, expected: int) -> None:
    """A zero offset leaves ``optimal`` exactly at the reference, so these are
    true ties."""
    result = suggest_a4(_notes(0.0, 0.0, 0.0), a4)
    assert result.optimal == a4
    assert result.rounded == expected


@pytest.mark.parametrize('offset,rounded,clamped', [
    (100.0, 466, 450),     # +100 ct: far above the range
    (-100.0, 415, 430),    # -100 ct: far below
    (40.0, 450, 450),      # 440 * 2^(40/1200) = 450.2: rounds to the edge
    (-40.0, 430, 430),     # 429.9: rounds to the edge
    (33.0, 448, 448),
])
def test_result_is_clamped_to_430_450(offset: float, rounded: int, clamped: int) -> None:
    result = suggest_a4(_notes(offset, offset, offset), 440.0)
    assert (result.rounded, result.clamped) == (rounded, clamped)


@pytest.mark.parametrize('a4,clamped', [(430.0, 430), (450.0, 450),
                                        (429.6, 430), (450.4, 450)])
def test_clamp_boundaries_hold_for_the_reference_itself(a4: float, clamped: int) -> None:
    assert suggest_a4(_notes(0.0, 0.0, 0.0), a4).clamped == clamped


def test_clamped_never_leaves_the_range_and_follows_the_offset() -> None:
    last = A4_MIN_HZ
    for offset in range(-200, 201):
        r = suggest_a4(_notes(float(offset), float(offset), float(offset)), 440.0)
        assert A4_MIN_HZ <= r.clamped <= A4_MAX_HZ
        assert r.clamped >= last, offset      # sharper player never lowers A4
        last = r.clamped


# ---------------------------------------------------------------------------
# suggest_a4: sign consistency with the engine's cents convention
# ---------------------------------------------------------------------------
def test_sharp_player_gets_a_higher_a4_and_flat_player_a_lower_one() -> None:
    sharp = suggest_a4(_notes(12.0, 12.0, 12.0), 440.0)
    flat = suggest_a4(_notes(-12.0, -12.0, -12.0), 440.0)
    assert sharp.rounded > 440 > flat.rounded
    # Mirror-image players move A4 by the same amount, give or take rounding.
    assert abs((sharp.rounded - 440) - (440 - flat.rounded)) <= 1


@pytest.mark.parametrize('played_cents', [12.0, -12.0, 25.0, -7.0])
def test_suggestion_brings_the_players_pitch_closer_to_zero(played_cents: float) -> None:
    """End to end with the engine's own cents maths: a player holding a pitch
    ``played_cents`` away from A440 reads that deviation at A4=440, and a
    smaller one at the suggested A4."""
    freq = 440.0 * 2.0 ** (played_cents / 1200.0)
    _midi, measured = cents_dev(freq, 440.0)
    assert measured == pytest.approx(played_cents, abs=1e-6)
    suggestion = suggest_a4(_notes(measured, measured, measured), 440.0)
    _midi, after = cents_dev(freq, float(suggestion.clamped))
    assert abs(after) < abs(measured)
    assert abs(after) <= 0.5 * 1200.0 * math.log2(
        (suggestion.clamped + 1) / suggestion.clamped) + 1e-9


def test_suggestion_is_relative_to_the_reference_the_means_were_measured_at() -> None:
    """Means of +12 ct at A4=442 mean the player sits 12 ct above 442."""
    result = suggest_a4(_notes(12.0, 12.0, 12.0), 442.0)
    assert result.optimal == pytest.approx(442.0 * 2.0 ** (12.0 / 1200.0))
    assert result.rounded == 445


# ---------------------------------------------------------------------------
# Parity with the maths as it was written inline in on_autotune
# ---------------------------------------------------------------------------
def _inline_reference(stats, a4_current):
    """The calculation exactly as on_autotune carried it before extraction."""
    import numpy as np
    items = [(m, n) for m, n in stats if n >= 5]
    if len(items) < 3:
        return None
    means = np.array([m for m, _ in items])
    weights = np.array([math.sqrt(n) for _, n in items])
    order = np.argsort(means)
    cumw = np.cumsum(weights[order])
    offset_ct = float(means[order][np.searchsorted(cumw, cumw[-1] / 2.0)])
    mean_weighted = float(np.sum(means * weights) / np.sum(weights))
    a4_optimal = a4_current * (2.0 ** (offset_ct / 1200.0))
    a4_rounded = round(a4_optimal)
    return (len(items), offset_ct, mean_weighted, a4_optimal, a4_rounded,
            max(430, min(450, a4_rounded)))


def test_matches_the_original_inline_calculation_on_random_input() -> None:
    rng = random.Random(20261013)
    for _ in range(300):
        stats = [(rng.uniform(-45.0, 45.0), rng.randint(0, 80))
                 for _ in range(rng.randint(0, 25))]
        a4 = float(rng.randint(430, 450))
        expected = _inline_reference(stats, a4)
        got = suggest_a4(stats, a4)
        if expected is None:
            assert got is None
        else:
            assert (got.notes, got.offset_ct, got.mean_weighted_ct, got.optimal,
                    got.rounded, got.clamped) == expected

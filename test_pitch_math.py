"""Phase-0 safety net for the two pure-math helpers in sax_audio_engine:
``freq_to_midi`` and ``cents_dev``.

No PyQt6, no sounddevice — the module-level try/except in the engine
means those optional imports fail silently, so these tests run in any
venv that has numpy and pytest.
"""
from __future__ import annotations

import math

import pytest

from sax_audio_engine import A4_DEFAULT, cents_dev, freq_to_midi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et_freq(midi: int, a4: float = A4_DEFAULT) -> float:
    """Frequency of an equal-tempered MIDI note."""
    return a4 * 2.0 ** ((midi - 69) / 12.0)


# ---------------------------------------------------------------------------
# 1. Identity at A4 and its octaves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,expected_midi,expected_cents", [
    (440.0, 69, 0.0),
    (880.0, 81, 0.0),
    (220.0, 57, 0.0),
])
def test_cents_dev_identity_octaves(freq, expected_midi, expected_cents):
    midi, cents = cents_dev(freq, a4=440.0)
    assert midi == expected_midi
    assert cents == expected_cents


# ---------------------------------------------------------------------------
# 2. Equal-tempered sweep [21, 108] — full piano range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", range(21, 109))
def test_cents_dev_equal_tempered_roundtrip(m):
    f = _et_freq(m)
    midi, cents = cents_dev(f, a4=A4_DEFAULT)
    assert midi == m
    assert abs(cents) < 1e-9, (
        f"MIDI {m}: expected 0-cent deviation, got {cents} cents"
    )


# ---------------------------------------------------------------------------
# 3. ±50-cent boundary — banker's rounding is load-bearing here.
#
# Python 3 uses round-half-to-even, so:
#   +50ct of A4: mf = 69.5 -> round(69.5) = 70  (70 is even)
#                result = (70, -50.0)
#   -50ct of A4: mf = 68.5 -> round(68.5) = 68  (68 is even)
#                result = (68, +50.0)
#
# Any future refactor that changes this (e.g. to round-half-away-from-zero)
# will break these assertions, which is intentional — the GUI interprets the
# sign of the cents deviation as "sharp" vs "flat" from the displayed MIDI.
# ---------------------------------------------------------------------------

def test_cents_dev_50_cents_sharp_of_A4():
    f = 440.0 * 2.0 ** (50.0 / 1200.0)
    mf_raw = freq_to_midi(f, a4=440.0)
    assert mf_raw == 69.5   # confirm exact IEEE-754 result
    midi, cents = cents_dev(f, a4=440.0)
    assert midi == 70       # round(69.5) == 70  (banker's rounding, 70 is even)
    assert abs(cents - (-50.0)) < 1e-6


def test_cents_dev_50_cents_flat_of_A4():
    f = 440.0 * 2.0 ** (-50.0 / 1200.0)
    mf_raw = freq_to_midi(f, a4=440.0)
    assert mf_raw == 68.5   # confirm exact IEEE-754 result
    midi, cents = cents_dev(f, a4=440.0)
    assert midi == 68       # round(68.5) == 68  (banker's rounding, 68 is even)
    assert abs(cents - 50.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. ±N-cent battery across a range of small offsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cents_offset", [1, 10, 25, 49])
def test_cents_dev_positive_offset_below_50(cents_offset):
    f = 440.0 * 2.0 ** (cents_offset / 1200.0)
    midi, cents = cents_dev(f, a4=440.0)
    assert midi == 69
    assert abs(cents - cents_offset) < 1e-6, (
        f"offset +{cents_offset}ct: expected ~+{cents_offset}, got {cents}"
    )


@pytest.mark.parametrize("cents_offset", [1, 10, 25, 49])
def test_cents_dev_negative_offset_below_50(cents_offset):
    f = 440.0 * 2.0 ** (-cents_offset / 1200.0)
    midi, cents = cents_dev(f, a4=440.0)
    assert midi == 69
    assert abs(cents - (-cents_offset)) < 1e-6, (
        f"offset -{cents_offset}ct: expected ~-{cents_offset}, got {cents}"
    )


# ---------------------------------------------------------------------------
# 5. Slight detuning (real-world A=442 scenario)
# ---------------------------------------------------------------------------

def test_cents_dev_442_from_440():
    midi, cents = cents_dev(442.0, a4=440.0)
    expected_cents = 1200.0 * math.log2(442.0 / 440.0)   # ≈ 7.851
    assert midi == 69
    assert abs(cents - expected_cents) < 1e-3


# ---------------------------------------------------------------------------
# 6. A4 sensitivity — same note, different reference pitch
# ---------------------------------------------------------------------------

def test_cents_dev_identity_with_non_standard_a4():
    midi, cents = cents_dev(441.0, a4=441.0)
    assert midi == 69
    assert cents == 0.0


def test_cents_dev_slight_sharp_against_441():
    midi, cents = cents_dev(442.0, a4=441.0)
    expected_cents = 1200.0 * math.log2(442.0 / 441.0)   # ≈ 3.921
    assert midi == 69
    assert abs(cents - expected_cents) < 1e-3


# ---------------------------------------------------------------------------
# 7. Octave wraparound — MIDI 68 (G#4 / Ab4)
# ---------------------------------------------------------------------------

def test_cents_dev_octave_wraparound_midi_68():
    f = 440.0 * 2.0 ** ((68 - 69) / 12.0)
    midi, cents = cents_dev(f, a4=440.0)
    assert midi == 68
    assert abs(cents) < 1e-9


# ---------------------------------------------------------------------------
# 8. freq_to_midi return type and value
# ---------------------------------------------------------------------------

def test_freq_to_midi_returns_float():
    result = freq_to_midi(440.0)
    assert isinstance(result, float)
    assert result == 69.0


# ---------------------------------------------------------------------------
# 9. freq_to_midi monotonic — ascending frequency -> ascending MIDI float
# ---------------------------------------------------------------------------

def test_freq_to_midi_monotonic():
    freqs = [_et_freq(m) for m in range(21, 109)]
    midis = [freq_to_midi(f) for f in freqs]
    for i in range(len(midis) - 1):
        assert midis[i] < midis[i + 1], (
            f"monotonicity broken between MIDI {21 + i} and {22 + i}: "
            f"{midis[i]} >= {midis[i + 1]}"
        )


# ---------------------------------------------------------------------------
# 10. Edge cases — lock current ValueError behaviour.
#
# math.log2(0) and math.log2(-1) raise ValueError in Python 3.
# Neither function catches it, so it propagates unchanged.
# These tests lock that behaviour so a future "guard invalid input" refactor
# is explicit and visible in the diff.
# ---------------------------------------------------------------------------

def test_freq_to_midi_raises_on_zero():
    with pytest.raises(ValueError):
        freq_to_midi(0.0)


def test_freq_to_midi_raises_on_negative():
    with pytest.raises(ValueError):
        freq_to_midi(-1.0)


def test_cents_dev_raises_on_zero():
    with pytest.raises(ValueError):
        cents_dev(0.0)


def test_cents_dev_raises_on_negative():
    with pytest.raises(ValueError):
        cents_dev(-440.0)


# ---------------------------------------------------------------------------
# 11. A4_DEFAULT constant — lock the value so a stray edit is caught
# ---------------------------------------------------------------------------

def test_a4_default_value():
    assert A4_DEFAULT == 440.0


# ---------------------------------------------------------------------------
# 12. cents_dev return-type contract
# ---------------------------------------------------------------------------

def test_cents_dev_return_types():
    midi, cents = cents_dev(440.0)
    assert isinstance(midi, int)
    assert isinstance(cents, float)


# ---------------------------------------------------------------------------
# 13. freq_to_midi default parameter uses A4_DEFAULT
# ---------------------------------------------------------------------------

def test_freq_to_midi_default_a4():
    assert freq_to_midi(A4_DEFAULT) == freq_to_midi(A4_DEFAULT, a4=A4_DEFAULT)
    assert freq_to_midi(440.0) == 69.0


def test_cents_dev_default_a4():
    assert cents_dev(A4_DEFAULT) == cents_dev(A4_DEFAULT, a4=A4_DEFAULT)

"""The stray-note rule behind the wrong-instrument hint, without the window.

intonation.core.stray_notes counts the different notes measured outside the
instrument's extended range and forgets each one after a window of playing.
The acceptance scenarios in tests/acceptance/test_wrong_instrument_hint.py
play this through the app; these pin the edges of the rule itself.
"""
from __future__ import annotations

from intonation.core.stray_notes import HINT_NOTES, WINDOW_READINGS, StrayNotes


def test_the_third_different_stray_raises_the_hint():
    strays = StrayNotes()
    assert not strays.add(30, in_range=False)
    assert not strays.add(30, in_range=False)       # the same note again
    assert not strays.add(31, in_range=False)
    assert strays.add(32, in_range=False)
    assert HINT_NOTES == 3


def test_in_range_readings_inside_the_window_forget_nothing():
    strays = StrayNotes(window_readings=10)
    strays.add(30, in_range=False)
    strays.add(31, in_range=False)
    for _ in range(7):
        strays.add(60, in_range=True)
    assert strays.add(32, in_range=False)


def test_a_stray_is_forgotten_once_the_window_has_passed():
    strays = StrayNotes(window_readings=10)
    strays.add(30, in_range=False)                  # clock 1
    strays.add(31, in_range=False)                  # clock 2
    for _ in range(9):                              # clocks 3..11
        strays.add(60, in_range=True)
    assert strays.counted() == 1                    # 30 has aged out, 31 not yet
    assert not strays.add(32, in_range=False)


def test_playing_a_stray_again_keeps_it_fresh():
    strays = StrayNotes(window_readings=10)
    strays.add(30, in_range=False)
    for _ in range(8):
        strays.add(60, in_range=True)
    strays.add(30, in_range=False)
    for _ in range(8):
        strays.add(60, in_range=True)
    assert strays.counted() == 1


def test_forget_where_and_clear():
    strays = StrayNotes()
    for midi in (30, 31, 40):
        strays.add(midi, in_range=False)
    strays.forget_where(lambda m: m < 35)
    assert strays.counted() == 1
    strays.clear()
    assert strays.counted() == 0


def test_the_window_is_about_twenty_seconds_of_readings():
    """One reading per ~46 ms analysis hop while a note sounds."""
    assert 18.0 <= WINDOW_READINGS * 0.046 <= 22.0

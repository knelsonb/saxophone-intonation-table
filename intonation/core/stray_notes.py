"""Stray-note counting behind the "is this the right instrument?" hint.

A stray is a measured note outside the selected instrument's extended range.
A few different strays close together suggest the wrong instrument is
selected. The same number spread over an hour of practice are squeaks and
detector slips on the right instrument, so a stray is forgotten once it is
older than a window of playing.

Time is counted in readings, one per analysis hop (about 46 ms) while a note
sounds. The window therefore measures playing time: rests do not age the
strays, and a test that feeds audio sees the same counts as a player does.

Pure Python with no Qt dependency, so the policy is testable on its own.
"""
from __future__ import annotations

from collections.abc import Callable

# This many DIFFERENT stray notes raise the hint. One long held note, repeated
# attacks on one note, or a single squeak can never raise it alone.
#
# A stray counts from its first reading. The detector confirms a pitch only
# after roughly 0.2 s of it in the default response mode (0.1 s in the fast
# mode, 0.5 s in the slow one), so blips shorter than that never get here. A
# per-note minimum on top would add little: a confirmed note arrives with a
# burst of several readings at once, so a reading count barely tells a 0.15 s
# blip from a 0.3 s note.
HINT_NOTES = 3

# Strays are forgotten after about 20 s of playing (20 s / 46 ms per reading).
# Someone on the wrong instrument plays three different strays well within
# that; a player who squeaks now and then on the right one does not.
WINDOW_READINGS = 435


class StrayNotes:
    """The recent stray notes of one instrument selection."""

    def __init__(self, hint_notes: int = HINT_NOTES,
                 window_readings: int = WINDOW_READINGS) -> None:
        self._hint_notes = hint_notes
        self._window = window_readings
        self._clock = 0                         # readings seen, in or out of range
        self._last_seen: dict[int, int] = {}    # sounding MIDI -> clock at its last reading

    def add(self, midi_sounding: int, in_range: bool) -> bool:
        """Record one reading. Returns whether enough different strays have
        now been heard within the window to raise the hint."""
        self._clock += 1
        self._expire()
        if in_range:
            return False
        self._last_seen[midi_sounding] = self._clock
        return self.counted() >= self._hint_notes

    def counted(self) -> int:
        """The different strays that currently count towards the hint."""
        return len(self._last_seen)

    def forget_where(self, covered: Callable[[int], bool]) -> None:
        """Forget the strays ``covered`` accepts, for example the notes an
        edited range now includes."""
        for midi in [m for m in self._last_seen if covered(m)]:
            del self._last_seen[midi]

    def clear(self) -> None:
        self._last_seen.clear()

    def _expire(self) -> None:
        horizon = self._clock - self._window
        for midi in [m for m, seen in self._last_seen.items() if seen <= horizon]:
            del self._last_seen[midi]

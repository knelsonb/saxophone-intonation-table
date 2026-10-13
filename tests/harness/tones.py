"""Synthetic microphone signals.

``Tone`` describes what a player does (a pitch, how far off it is, what it
sounds like); ``render(sr)`` turns it into float32 frames at the rate of the
input stream the app actually opened.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

import sim_harness

TIMBRES = ("sine", "sax", "weak_fundamental")

_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note_to_midi(name: str) -> int:
    """'A4' -> 69, 'Bb3' -> 58, 'F#5' -> 78. Concert (sounding) numbering."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"not a note name: {name!r}")
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pc = _PITCH_CLASS[letter] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    return (octave + 1) * 12 + pc


def midi_to_freq(midi: float, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((midi - 69.0) / 12.0)


# Harmonic amplitudes (fundamental first) and phases for the additive timbres
# that sim_harness does not already provide.
_WEAK_FUNDAMENTAL = ((0.10, 0.0), (0.55, 0.3), (0.40, 0.9), (0.28, 1.7), (0.16, 0.5))
_SAX = ((0.50, 0.0), (0.30, 0.4), (0.20, 1.1), (0.10, 2.0), (0.05, 0.7))


@dataclass(frozen=True)
class Tone:
    """One sustained note.

    ``note`` is a sounding (concert) pitch name; ``freq`` is an explicit
    frequency in Hz. Exactly one of them is given. ``cents`` detunes it.
    ``a4`` is the reference the *player* tunes to, independent of the app's A4
    setting, so a test can play a true 440 Hz A at an app set to 442.
    ``vibrato`` is ``(rate_hz, depth_cents)``.
    """

    note: str | None = None
    freq: float | None = None
    cents: float = 0.0
    timbre: str = "sine"
    seconds: float = 1.0
    amp: float = 0.5
    vibrato: tuple[float, float] | None = None
    a4: float = 440.0

    def __post_init__(self) -> None:
        if (self.note is None) == (self.freq is None):
            raise ValueError("give exactly one of note= or freq=")
        if self.timbre not in TIMBRES:
            raise ValueError(f"timbre must be one of {TIMBRES}")

    @property
    def hz(self) -> float:
        base = self.freq if self.freq is not None else midi_to_freq(
            note_to_midi(self.note), self.a4)
        return base * 2.0 ** (self.cents / 1200.0)

    def render(self, sr: int) -> np.ndarray:
        n = max(1, int(round(self.seconds * sr)))
        f0 = self.hz
        if self.vibrato is None and self.timbre == "sine":
            return sim_harness.sine(f0, sr, n, self.amp)
        if self.vibrato is None and self.timbre == "sax":
            return sim_harness.sax_like(f0, sr, n, self.amp)
        partials = _WEAK_FUNDAMENTAL if self.timbre == "weak_fundamental" else (
            _SAX if self.timbre == "sax" else ((1.0, 0.0),))
        t = np.arange(n, dtype=np.float64) / sr
        if self.vibrato is None:
            phase = 2.0 * math.pi * f0 * t
        else:
            rate, depth = self.vibrato
            # Integrated phase of f(t) = f0 * 2**(depth * sin(2*pi*rate*t) / 1200).
            inst = f0 * 2.0 ** (depth * np.sin(2.0 * math.pi * rate * t) / 1200.0)
            phase = 2.0 * math.pi * np.cumsum(inst) / sr
        out = np.zeros(n, dtype=np.float64)
        for k, (a, ph) in enumerate(partials, start=1):
            out += a * np.sin(k * phase + ph)
        peak = float(np.max(np.abs(out))) or 1.0
        return (self.amp * out / peak).astype(np.float32)

"""Plain data the harness hands back to tests.

These types are the whole vocabulary of observable app state. They hold only
strings, numbers and paths, never a widget or an app object, so a test that uses
them cannot depend on how the app is built.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BANDS = ("green", "amber", "red")


def note_spellings(cell: str) -> set[str]:
    """Every way a table cell's note name can be written.

    The app shows enharmonic pairs as 'F#/Gb5'. A test may say 'F#/Gb5', 'F#5'
    or 'Gb5'; all three name the same cell.
    """
    m = re.match(r"^(.*?)(-?\d+)$", cell.strip())
    if not m:
        return {cell}
    names, octave = m.group(1), m.group(2)
    out = {cell.strip()}
    for part in names.split("/"):
        out.add(f"{part}{octave}")
    return out


@dataclass(frozen=True)
class Row:
    """One note in the intonation table, as displayed."""
    fingered: str
    sounding: str
    mean: Optional[int]      # displayed integer cents; None when nothing was played
    std: Optional[int]
    n: int
    band: Optional[str]      # 'green' | 'amber' | 'red' | None (no data)

    @property
    def played(self) -> bool:
        return self.n > 0

    def is_named(self, name: str, by: Optional[str] = None) -> bool:
        cols = {"fingered": [self.fingered], "sounding": [self.sounding],
                None: [self.fingered, self.sounding]}[by]
        return any(name in note_spellings(c) for c in cols)


@dataclass(frozen=True)
class Reading:
    """What the live tuner shows."""
    note: str
    cents: float
    freq: float
    active: bool
    status_text: str

    @property
    def cents_shown(self) -> int:
        return int(round(self.cents))


@dataclass(frozen=True)
class AudioStatus:
    """What the app shows about its audio devices."""
    input_state: str                 # 'running' | 'failed' | 'stopped' | ...
    input_error: str                 # 'none' | 'device_disconnected' | 'device_busy' | ...
    input_device: str
    input_rate: int
    output_running: bool
    output_device: str
    output_rate: int
    banner_visible: bool
    banner_text: str
    chip_text: str
    status_text: str


@dataclass(frozen=True)
class MetronomeState:
    running: bool
    bpm: int
    time_sig: str
    volume: float
    unavailable_text: Optional[str]


@dataclass(frozen=True)
class DroneState:
    enabled: bool
    voice: str
    semitones: int
    sine_fallback: bool
    status_text: Optional[str]


@dataclass(frozen=True)
class DeckState:
    state: str                       # 'idle' | 'recording' | 'have_take' | 'playing'
    status_text: Optional[str]
    label: str


@dataclass(frozen=True)
class ImportResult:
    # 'imported' | 'nothing_new' | 'empty' | 'rejected' | 'io_error' | 'cancelled'
    status: str
    runs: int = 0
    measurements: int = 0
    skipped: int = 0
    message: str = ""


@dataclass(frozen=True)
class AutotuneResult:
    status: str                      # 'insufficient_data' | 'applied' | 'declined'
    message: str
    a4_before: int
    a4_after: int


class ExportFailed(AssertionError):
    """An export produced no file. ``message`` is what the app told the user."""

    def __init__(self, what: str, message: Optional[str]) -> None:
        super().__init__(f"{what} export produced no file; app said: {message!r}")
        self.message = message


@dataclass
class ExportModels:
    """Render models captured from the last PDF and chart export."""
    pdf: Optional[dict] = None
    chart: Optional[dict] = None
    csv_rows: Optional[int] = None
    paths: list[Path] = field(default_factory=list)

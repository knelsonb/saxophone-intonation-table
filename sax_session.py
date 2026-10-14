# =============================================================================
# sax_session.py — what the player is measuring, in one place
#
# SessionModel holds the values every part of the window has to agree on: the
# concert pitch (A4), the instrument, how notes are spelled (fingered or
# sounding), the horn's nickname, whether readings are recorded, and the
# per-note statistics behind the table. Each has one setter that emits a
# signal when the value really changes; the toolbar and SETUP combos, the
# engine, the drone, the pitch pipes and the history log follow those
# signals, and the table and export read the model. There is no second copy
# to fall out of step (an A4 combo saying 442 while the engine measured
# against 440 was the bug class this replaces).
#
# RunTracker opens a new history run whenever the meaning of a reading
# changes: another A4, another instrument, recording resumed, or a new horn
# nickname committed.
#
# Both live on the Qt main thread: readings reach the model through the
# engine's queued note signal, so nothing here needs a lock.
# =============================================================================
from __future__ import annotations

import datetime
import math
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

import sax_instruments
from sax_pitch import A4_DEFAULT, SAX_MIDI, cents_dev
from sax_stats import NoteStats

DISPLAY_MODES = ('griff', 'klingend')

# How long the table keeps highlighting the note last played.
ACTIVE_NOTE_HOLD_S = 1.5


class SessionModel(QObject):
    """The session's shared state; see the module comment."""

    a4_changed = pyqtSignal(float)
    instrument_changed = pyqtSignal(str)
    display_changed = pyqtSignal(str)
    nickname_changed = pyqtSignal(str)
    recording_changed = pyqtSignal(bool)

    def __init__(self, *, instrument: str, display: str = 'griff',
                 a4: float = A4_DEFAULT, nickname: str = '',
                 recording: bool = True, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._a4 = float(a4)
        self._instrument = instrument
        self._display = display if display in DISPLAY_MODES else 'griff'
        self._nickname = nickname
        self._recording = bool(recording)
        # Per-note statistics behind the table, keyed by sounding MIDI.
        self.stats: dict[int, NoteStats] = {}
        self._active_midi: Optional[int] = None
        self._active_at: Optional[datetime.datetime] = None

    # ── Values ───────────────────────────────────────────────────────────────
    @property
    def a4(self) -> float:
        return self._a4

    def set_a4(self, hz: float) -> None:
        hz = float(hz)
        if hz != self._a4:
            self._a4 = hz
            self.a4_changed.emit(hz)

    @property
    def instrument(self) -> str:
        return self._instrument

    def set_instrument(self, key: str) -> None:
        if key != self._instrument:
            self._instrument = key
            self.instrument_changed.emit(key)

    @property
    def display(self) -> str:
        """'griff' (fingered) or 'klingend' (sounding)."""
        return self._display

    def set_display(self, mode: str) -> None:
        if mode in DISPLAY_MODES and mode != self._display:
            self._display = mode
            self.display_changed.emit(mode)

    @property
    def nickname(self) -> str:
        """The nickname as typed (both editors show it)."""
        return self._nickname

    @property
    def label(self) -> str:
        """The nickname as a run label: without surrounding blanks."""
        return self._nickname.strip()

    def set_nickname(self, text: str) -> None:
        if text != self._nickname:
            self._nickname = text
            self.nickname_changed.emit(text)

    @property
    def recording(self) -> bool:
        return self._recording

    def set_recording(self, on: bool) -> None:
        on = bool(on)
        if on != self._recording:
            self._recording = on
            self.recording_changed.emit(on)

    # ── Instrument helpers ───────────────────────────────────────────────────
    @property
    def transposition(self) -> int:
        """Semitones from fingered to sounding for the current instrument."""
        return sax_instruments.transp_map().get(self._instrument, 0)

    def fingered_range(self) -> tuple[int, int]:
        return sax_instruments.fingered_range(self._instrument)

    def in_range(self, midi_sounding: int) -> bool:
        lo, hi = self.fingered_range()
        return lo <= midi_sounding - self.transposition <= hi

    # ── Statistics ───────────────────────────────────────────────────────────
    def add_reading(self, midi_sounding: int, cents: float, freq: float) -> None:
        self.stats.setdefault(midi_sounding, NoteStats()).add(cents, freq)
        self._active_midi = midi_sounding
        self._active_at = datetime.datetime.now()

    def active_midi(self) -> Optional[int]:
        """The note played last, while it is still fresh enough to highlight."""
        if (self._active_at is None
                or (datetime.datetime.now() - self._active_at).total_seconds()
                > ACTIVE_NOTE_HOLD_S):
            return None
        return self._active_midi

    def seed_expected_notes(self) -> None:
        """Add an empty row for every note in the instrument's fingered range,
        so the table shows the range before anything is played. Existing
        measurements are kept."""
        transp = self.transposition
        lo, hi = self.fingered_range()
        for fingered in range(lo, hi + 1):
            sounding = fingered + transp
            if sounding in SAX_MIDI and sounding not in self.stats:
                self.stats[sounding] = NoteStats()

    def rederive_stats(self, a4: float) -> None:
        """Re-grade every reading in the table against ``a4``. Cents are a
        function of frequency and A4; the frequencies behind the current
        table are kept, so nothing the player recorded is thrown away. Only
        the current table is used: the history log also holds earlier
        sessions, imports and deleted notes, which do not belong here."""
        remapped: dict[int, NoteStats] = {}
        for st in self.stats.values():
            for f in st.freqs:
                # Non-positive / non-finite freqs carry no tuning information
                # and would make freq_to_midi raise.
                if not math.isfinite(f) or f <= 0.0:
                    continue
                midi, cents = cents_dev(f, a4)
                if midi in SAX_MIDI:
                    remapped.setdefault(midi, NoteStats()).add(cents, f)
        self.stats = remapped

    def clear_stats(self) -> None:
        self.stats.clear()

    def discard(self, midi_sounding: int) -> None:
        self.stats.pop(midi_sounding, None)

    def has_played_notes(self) -> bool:
        # Rows seeded for the instrument range have n == 0 until played.
        return any(st.n > 0 for st in self.stats.values())


class RunTracker:
    """Keeps the history log's current run in step with the session: a new
    run (instrument, A4 and nickname latched) whenever one of them changes
    while recording, and none while recording is paused. ``enabled`` is
    False when there is no audio input at all, so nothing is ever logged."""

    def __init__(self, session: SessionModel, log, *, enabled: bool) -> None:
        self._session = session
        self._log = log
        self._enabled = bool(enabled)
        session.a4_changed.connect(self._on_reference_changed)
        session.instrument_changed.connect(self._on_reference_changed)
        session.recording_changed.connect(self._on_recording_changed)

    def set_log(self, log) -> None:
        """Follow another log (history was switched on mid-session)."""
        self._log = log

    def start_run(self) -> None:
        """Open a run for the current instrument, A4 and nickname. An empty
        previous run is coalesced by the log, so scrubbing a combo leaves no
        trace."""
        s = self._session
        if self._enabled and s.recording:
            self._log.start_run(instrument=s.instrument, a4_hz=s.a4, label=s.label)

    def nickname_committed(self) -> None:
        """The player finished editing the nickname. A nickname names a horn,
        so a new one starts a new run, as an instrument or A4 change does:
        relabelling the current run would move readings already taken on the
        previous horn under the new name. editingFinished also fires on a
        plain focus change; only an actual change of name opens a run."""
        if not (self._enabled and self._session.recording):
            return
        current = self._log.current_run()
        if current is not None and current.label == self._session.label:
            return
        self.start_run()

    def _on_reference_changed(self, *_args) -> None:
        self.start_run()

    def _on_recording_changed(self, on: bool) -> None:
        if on:
            self.start_run()
        else:
            self._log.end_run()

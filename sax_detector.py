"""Per-hop note detection: level gate, YIN, and the note filter.

``Detector.process`` takes one detection window (the newest ~370 ms of mic
audio, once per ~46 ms hop) and returns the readings that window confirms as
``NoteEvent`` objects. Between hops it carries the filter state:

* the *recent* window of per-hop (midi, freq) detections that votes on which
  note is sounding (``confirm`` matching hops out of the last ``window``);
* the *incumbent* (locked) note, which changes only on a confirmed vote;
* the *edge* guard: the first ``edge_hops`` hops after a lock are attack
  transient and produce nothing, and the last ``edge_hops`` readings before a
  note ends are release transient, so they wait in a *pending* FIFO and are
  dropped when the note ends.

Hop kinds:

* reject  silence, an aperiodic frame, a pitch outside the band or the MIDI
          range. Ends the held note: the pending FIFO is trimmed by the release
          guard and the survivors are emitted, then the vote restarts.
* skip    a pitch the coordination policy excludes (the drone or a pitch pipe
          is sounding that note, so the mic may be hearing the speaker). The
          hop casts no vote and changes nothing: one bleed hop must not end
          the player's note or discard its readings.
* vote    everything else.

Single owner, no locks: the engine's analysis consumer is the only caller, and
anything another thread wants changed (mode, A4, a reset) reaches it as a
value the consumer passes in or a reset it performs between hops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sax_yin import MAX_FREQ, MIN_FREQ, yin_pitch

__all__ = ["FILTER_MODE_DEFAULT", "FILTER_PRESETS", "Detector", "NoteEvent"]

# Filter-mode presets. Each hop is ~46 ms at every sample rate (the hop is
# sized in samples per rate so its duration stays the same), so `confirm` and
# `edge_hops` are quantized to that grid.
#
# Calibrated for three working-musician use cases:
#   fast    — live play / scale drills / immediate per-note feedback
#   normal  — practice & tuning long tones (the default)
#   slow    — instrument setup, repair, cataloging intonation maps
#
# First reading ~ (confirm + 2 * edge_hops) * hop
# Smoothing     ~ window * hop over the same-MIDI window
#
#   window     — recent valid detections kept for confirmation/median
#   confirm    — required matching-MIDI detections in the window
#   yin_thr    — YIN aperiodicity ceiling (lower = stricter)
#   rms_floor  — RMS gate below which the frame is silence
#   edge_hops  — attack/release transient guard (in hops)
FILTER_PRESETS = {
    'fast':   dict(window=3,  confirm=2, yin_thr=0.15, rms_floor=8e-5,  edge_hops=1),
    'normal': dict(window=5,  confirm=3, yin_thr=0.10, rms_floor=1.5e-4, edge_hops=2),
    'slow':   dict(window=10, confirm=6, yin_thr=0.07, rms_floor=3e-4,  edge_hops=4),
}
FILTER_MODE_DEFAULT = 'normal'

# Coordination hook: detected MIDI (None on a rejected hop) -> MIDIs to skip.
Coordinate = Callable[[Optional[int]], frozenset]


@dataclass(frozen=True)
class NoteEvent:
    """One confirmed reading.

    ``cents`` is graded against ``a4``, the reference in force when the hop was
    analysed, and ``a4_generation`` identifies that reference: the engine bumps
    it on every A4 change, so a consumer can tell a reading measured before a
    change from one measured after it even when both A4 values are equal.
    """
    midi: int
    freq: float
    cents: float
    a4: float = 440.0
    a4_generation: int = 0


def _cents_dev(f: float, a4: float) -> tuple[int, float]:
    # Same maths as sax_audio_engine.cents_dev; kept local so this module does
    # not import the engine that owns it.
    mf = 69.0 + 12.0 * math.log2(f / a4)
    mr = round(mf)
    return mr, (mf - mr) * 100.0


class Detector:
    """The detection state machine. Frames in, ``NoteEvent`` list out."""

    def __init__(self, mode: str = FILTER_MODE_DEFAULT,
                 midi_min: int = 21, midi_max: int = 108) -> None:
        self.mode = mode if mode in FILTER_PRESETS else FILTER_MODE_DEFAULT
        self.params = FILTER_PRESETS[self.mode]
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        # Diagnostics of the last processed hop (read by the engine).
        self.rms_db: float = -120.0
        self.ap: float = 1.0
        self.freq: float = 0.0
        self.shown_midi: Optional[int] = None
        self.reset()

    # ---- control ----------------------------------------------------------
    def reset(self) -> None:
        """Forget the held note and everything waiting to be emitted."""
        self._recent: list[tuple[int, float]] = []
        self._locked: Optional[int] = None
        self._hop_in_note = 0
        self._pending: list[NoteEvent] = []

    def set_mode(self, mode: str) -> None:
        """Switch filter preset; unknown names are ignored. Always resets."""
        if mode in FILTER_PRESETS:
            self.mode = mode
            self.params = FILTER_PRESETS[mode]
        self.reset()

    @property
    def locked_midi(self) -> Optional[int]:
        return self._locked

    # ---- one hop ----------------------------------------------------------
    def process(self, window: np.ndarray, sr: int, a4: float,
                a4_generation: int = 0, mic_gain: float = 1.0,
                coordinate: Optional[Coordinate] = None) -> list[NoteEvent]:
        """Analyse one detection window and return the readings it releases."""
        p = self.params
        rms = math.sqrt(float(np.dot(window, window)) / window.size) if window.size else 0.0
        # A corrupt or overflowed driver buffer reads NaN: skip the hop and
        # resume cleanly on the next one.
        if not math.isfinite(rms):
            return []
        # Mic gain scales the gate decision and the level meter only. The
        # window is never scaled, so YIN (which normalises by rms) and the cents
        # readout are unaffected.
        rms_eff = rms * mic_gain
        self.rms_db = 20.0 * math.log10(max(rms_eff, 1e-9))
        if rms_eff < p['rms_floor']:
            self.ap, self.freq = 1.0, 0.0
            return self._reject(coordinate)
        freq, ap = yin_pitch(window / (rms + 1e-9), sr)
        self.ap, self.freq = float(ap), float(freq)
        if ap > p['yin_thr'] or not (MIN_FREQ < freq < MAX_FREQ):
            return self._reject(coordinate)
        midi, _cents = _cents_dev(freq, a4)
        if not (self.midi_min <= midi <= self.midi_max):
            return self._reject(coordinate)
        if coordinate is not None and midi in coordinate(midi):
            # The output is sounding this note, so the mic may be hearing the
            # speaker: the hop casts no vote. It is not silence either, so the
            # player's held note and its waiting readings stay as they are.
            self.shown_midi = self._locked
            return []
        return self._vote(midi, float(freq), a4, a4_generation)

    # ---- hop kinds --------------------------------------------------------
    def _reject(self, coordinate: Optional[Coordinate]) -> list[NoteEvent]:
        self.shown_midi = self._locked
        out = self._end_note()
        if coordinate is not None:
            coordinate(None)             # keeps the duck's release ramp moving
        return out

    def _end_note(self) -> list[NoteEvent]:
        out: list[NoteEvent] = []
        if self._locked is not None or self._pending or self._recent:
            self._drop_release_edge()
            out.extend(self._pending)
            self._pending.clear()
        self._recent.clear()
        self._locked = None
        self._hop_in_note = 0
        return out

    def _drop_release_edge(self) -> None:
        keep = max(0, len(self._pending) - self.params['edge_hops'])
        del self._pending[keep:]

    def _vote(self, midi: int, freq: float, a4: float, gen: int) -> list[NoteEvent]:
        p = self.params
        out: list[NoteEvent] = []
        self.shown_midi = self._locked if self._locked is not None else midi
        self._recent.append((midi, freq))
        if len(self._recent) > p['window']:
            self._recent.pop(0)
        latest = self._recent[-1][0]
        if sum(1 for m, _f in self._recent if m == latest) < p['confirm']:
            return out
        if self._locked != latest:
            # A new note won the vote: the old note's last readings are its
            # release edge; emit whatever came before them, then relock.
            self._drop_release_edge()
            out.extend(self._pending)
            self._pending.clear()
            self._locked = latest
            self._hop_in_note = 0
        self._hop_in_note += 1
        if self._hop_in_note <= p['edge_hops']:
            return out                    # attack edge
        median_freq = float(np.median([f for m, f in self._recent if m == latest]))
        _m, median_cents = _cents_dev(median_freq, a4)
        self._pending.append(NoteEvent(latest, median_freq, median_cents, a4, gen))
        if len(self._pending) > p['edge_hops']:
            out.append(self._pending.pop(0))
        return out

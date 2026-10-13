"""Per-note running statistics behind the intonation table."""
from __future__ import annotations

import math
from array import array


class NoteStats:
    """Welford's online algorithm for mean and population variance.

    Replaces the v0.5.7 list-based accumulator.  O(1) time and O(1) memory
    per note regardless of how many cents readings accumulate.  Population
    std (ddof=0) is preserved to match np.std default and the _agg_stats
    contract locked by test_agg_stats.py.

    Thread-safety note: add() and the property readers are both called on
    the Qt main thread (add via the engine.note_event slot; readers via
    the table-refresh timer), so no lock is needed here.  The v0.5.7.3
    concurrent-clear NaN hazard is structurally gone: _mean and _m2 are
    plain Python floats updated by individual assignments; under CPython's
    GIL each assignment is atomic, and there is no window between a
    truthiness check and a length read where a concurrent clear could
    produce NaN from an empty sequence.
    """

    def __init__(self) -> None:
        self._n:    int   = 0
        self._mean: float = 0.0
        self._m2:   float = 0.0
        # Raw detected frequencies behind this row. A concert-pitch change
        # re-derives cents from these, so the table keeps exactly the notes it
        # held (after any Reset or per-note delete). 8 bytes per reading.
        self.freqs = array('d')

    def add(self, c: float, freq: float | None = None) -> None:
        if freq is not None:
            self.freqs.append(freq)
        self._n += 1
        delta        = c - self._mean
        self._mean  += delta / self._n
        delta2       = c - self._mean   # uses updated mean
        self._m2    += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / self._n) if self._n > 1 else 0.0

    @property
    def n(self) -> int:
        return self._n

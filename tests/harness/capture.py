"""What came out of the (fake) speaker."""
from __future__ import annotations

import numpy as np

import sim_harness


class OutputCapture:
    """A mono float32 recording of the output stream at sample rate ``sr``."""

    def __init__(self, samples: np.ndarray, sr: int) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)
        self.sr = int(sr)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return len(self) / float(self.sr) if self.sr else 0.0

    def slice(self, t0: float = 0.0, t1: float | None = None) -> "OutputCapture":
        a = int(round(t0 * self.sr))
        b = len(self) if t1 is None else int(round(t1 * self.sr))
        return OutputCapture(self.samples[a:b], self.sr)

    def rms(self) -> float:
        return sim_harness.rms(self.samples)

    def peak(self) -> float:
        return sim_harness.max_abs(self.samples)

    def is_silent(self, threshold: float = 1e-4) -> bool:
        return self.peak() < threshold

    def pitch(self, near: float | None = None, search_cents: float = 70.0) -> float:
        """Dominant frequency in Hz. With ``near``, the strongest peak within
        +/- ``search_cents`` of that frequency (use it for rich timbres, where
        the loudest partial need not be the fundamental)."""
        if near is None:
            return sim_harness.fft_peak_hz(self.samples, self.sr)
        return sim_harness.peak_near_hz(self.samples, near, self.sr, search_cents)

    def onsets(self, threshold: float = 1e-2, min_gap_s: float = 0.1) -> list[float]:
        """Start time (seconds) of every burst. ``min_gap_s`` merges the
        oscillations inside one click into a single onset."""
        idx = sim_harness.onset_samples(
            self.samples, threshold=threshold,
            min_gap=max(1, int(round(min_gap_s * self.sr))))
        return [i / float(self.sr) for i in idx]

    def onset_intervals(self, **kw) -> list[float]:
        o = self.onsets(**kw)
        return [o[i + 1] - o[i] for i in range(len(o) - 1)]

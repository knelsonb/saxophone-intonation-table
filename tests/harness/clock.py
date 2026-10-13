"""Deterministic time for the headless app.

``ManualClock`` replaces the wall clock for everything the harness controls:
the fake audio host pulls output blocks from it and the driver ticks the app's
QTimers from it. Nothing in the acceptance suite sleeps; a test that wants two
seconds to pass calls ``app.advance(2.0)``.
"""
from __future__ import annotations

from typing import Callable


class ManualClock:
    # Listeners see time in slices no longer than this, so a long advance
    # interleaves audio pulls and timer ticks the way real time would instead of
    # firing every timer once at the end.
    MAX_STEP_S = 0.05

    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)
        self._listeners: list[Callable[[float], None]] = []

    def subscribe(self, fn: Callable[[float], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def unsubscribe(self, fn: Callable[[float], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance a clock backwards")
        remaining = float(seconds)
        while remaining > 1e-12:
            step = min(self.MAX_STEP_S, remaining)
            remaining -= step
            self.now += step
            for fn in list(self._listeners):
                fn(step)

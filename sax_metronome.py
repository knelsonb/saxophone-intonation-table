"""Metronome: sample-accurate click scheduling + a two-voice numpy click synth.

Sprint 2 of the parity sprint. The metronome is the first real *output* consumer
of the Sprint-1 mixer foundation (see ``sax_mixer.py``). It owns no stream of its
own: it registers a :class:`ClickSource` on the shared :class:`~sax_mixer.Mixer`
and schedules each beat's click by ABSOLUTE sample index via
``mixer.schedule(abs_sample, event)``. The mixer's monotonic sample clock — not
any wall-clock timer — is the timebase, so timing is exact and blocksize-invariant.

Drift budget is ZERO (plan D2). The keystone rule, proven by Gandalf's
scheduling stress-test (non-dividing + alternating blocksizes, 64-beat runs, all
max|drift|=0):

    onset(beat) = round(beat_index * samplerate * 60 / bpm)

The beat onset is computed from the beat's INTEGER index every time and rounded
ONCE — never by accumulating a float interval (``next += interval``), which is
where microsecond error compounds into audible drift over minutes. See
:func:`beat_onset_sample`.

Control surface (``MetronomeController``): bpm / time-sig / click-volume / running
state, with ``set_bpm`` / ``nudge_bpm`` / ``set_time_sig`` / ``set_volume`` /
``register_tap`` / ``start`` / ``stop`` / ``toggle`` / ``is_running`` and an
``on_state_changed`` callback the GUI uses to flip the tab's running-dot and the
Start/Stop button. Live tempo / time-sig changes take effect on the NEXT beat
(the scheduler keeps only a one-beat rolling horizon and re-anchors on change).

This module imports only numpy (+ stdlib) — no Qt, no sounddevice — so the synth,
the onset math, the tap-tempo logic and the accent logic are all unit-testable
directly against a real ``Mixer`` with no audio hardware (see test_metronome.py).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, List, Optional

import numpy as np

try:  # canonical allowlist (single source of truth with the config/GUI selector)
    from sax_config import TIME_SIG_VALUES as VALID_TIME_SIGS
except Exception:  # pragma: no cover - standalone fallback if config unavailable
    VALID_TIME_SIGS = frozenset({"2/4", "3/4", "4/4", "6/8"})

__all__ = [
    "BPM_MIN", "BPM_MAX", "DEFAULT_BPM", "DEFAULT_TIME_SIG", "VALID_TIME_SIGS",
    "beat_onset_sample", "bpm_from_tap_times", "clamp_bpm", "clamp_volume",
    "ClickSource", "MetronomeController",
]

# Tempo / time-sig rails (mirror the Android parity scope + sax_config clamps).
BPM_MIN = 30
BPM_MAX = 300
DEFAULT_BPM = 100
DEFAULT_TIME_SIG = "4/4"

# Tap-tempo windowing (Android useMetronome parity).
_TAP_AVG_INTERVALS = 4     # average the last N inter-tap intervals
_TAP_RESET_GAP_S = 2.0     # a gap longer than this starts a fresh tap run


# ---------------------------------------------------------------------------
# Pure helpers — the drift-zero core + clamps + tap math. No state, no audio.
# ---------------------------------------------------------------------------
def beat_onset_sample(beat_index: int, bpm: float, samplerate: int) -> int:
    """Absolute sample offset (from beat 0) of beat ``beat_index``.

    ``round(beat_index * samplerate * 60 / bpm)`` — computed from the integer
    beat index and rounded ONCE. This is the drift-killer: because every beat is
    derived from its own index rather than by accumulating a float interval, the
    rounding error is bounded to <=0.5 sample for ALL beats and never compounds.
    A ``next += interval`` float accumulator would drift without bound.
    """
    return int(round(beat_index * samplerate * 60.0 / float(bpm)))


def clamp_bpm(bpm: float) -> int:
    """Coerce to an int BPM within [BPM_MIN, BPM_MAX] (matches sax_config)."""
    return max(BPM_MIN, min(BPM_MAX, int(round(bpm))))


def clamp_volume(volume: float) -> float:
    """Coerce to a float volume within [0.0, 1.0] (matches sax_config)."""
    v = float(volume)
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


def bpm_from_tap_times(times: List[float]) -> Optional[int]:
    """Derive a clamped BPM from a list of monotonically-increasing tap times.

    Averages the last :data:`_TAP_AVG_INTERVALS` inter-tap intervals of the
    CURRENT run, where a gap longer than :data:`_TAP_RESET_GAP_S` seconds starts
    a fresh run (taps before the last such gap are ignored). Returns ``None``
    until there are at least two taps in the current run (i.e. on the first tap
    of a run there is no interval yet). Pure and clock-free so tap-tempo is
    testable with synthetic timestamps.
    """
    if not times or len(times) < 2:
        return None
    # Start of the current run = just after the most recent >2s gap.
    start = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > _TAP_RESET_GAP_S:
            start = i
    run = times[start:]
    if len(run) < 2:
        return None
    intervals = [run[i] - run[i - 1] for i in range(1, len(run))]
    intervals = intervals[-_TAP_AVG_INTERVALS:]
    avg = sum(intervals) / len(intervals)
    if avg <= 0.0:
        return None
    return clamp_bpm(round(60.0 / avg))


def beats_per_bar(time_sig: str) -> int:
    """Number of beats per measure = the time-sig numerator (2/4->2 .. 6/8->6)."""
    try:
        return max(1, int(str(time_sig).split("/")[0]))
    except (ValueError, IndexError):
        return 4


# ---------------------------------------------------------------------------
# ClickSource — a two-voice (accent / beat) numpy click, a MixerSource.
# ---------------------------------------------------------------------------
class ClickSource:
    """A short enveloped click burst, placed sample-accurately by the mixer.

    Two pre-synthesised voices: an ACCENT click (the downbeat — higher + louder)
    and a plain BEAT click. The scheduled beat event calls :meth:`trigger`
    (which only records which voice and the intra-block start offset); the mixer
    then calls :meth:`render` in the SAME block, which writes the burst additively
    starting at that offset — the event-sets-state -> render-places-by-t0 pattern.

    Zero allocation in the steady state: between clicks ``render`` returns
    immediately. The voice envelopes are built once; a ``_pos`` cursor lets a
    click that overruns a block resume mid-envelope on the next render.
    ``active_midi`` is ``None`` — a click is unpitched, so it never participates
    in output-pitch vote-exclude / ducking.
    """

    __test__ = False  # name starts with neither Test* nor test_*; belt-and-braces

    def __init__(self, samplerate: int, volume: float = 1.0, *,
                 accent_freq: float = 1320.0, beat_freq: float = 880.0,
                 dur_ms: float = 38.0,
                 accent_peak: float = 0.92, beat_peak: float = 0.55):
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        self._samplerate = int(samplerate)
        self._volume = clamp_volume(volume)
        self._accent_freq = float(accent_freq)
        self._beat_freq = float(beat_freq)
        self._dur_ms = float(dur_ms)
        self._accent_peak = float(accent_peak)
        self._beat_peak = float(beat_peak)

        # Volume-independent base envelopes (built once).
        self._accent_base = self._synth(self._accent_freq, self._accent_peak)
        self._beat_base = self._synth(self._beat_freq, self._beat_peak)
        # Volume-scaled play buffers (rebuilt off the hot path on volume change).
        self._accent = self._accent_base * np.float32(self._volume)
        self._beat = self._beat_base * np.float32(self._volume)

        # Playback state. _buf is the voice currently sounding (or None);
        # _pos is samples already emitted; _start_off is the intra-block start
        # for the NEXT render (set by trigger, 0 thereafter).
        self._buf: Optional[np.ndarray] = None
        self._pos = 0
        self._start_off = 0
        # Set by request_stop(): once the in-flight burst drains, `finished`
        # goes True and the Mixer reaps this source (a click-free stop with no
        # mid-burst truncation pop). Cleared by arm() on (re)start.
        self._stopping = False

    # -- synthesis ----------------------------------------------------------
    def _synth(self, freq: float, peak: float) -> np.ndarray:
        """One enveloped sine burst: ~1ms linear attack, exponential decay."""
        n = max(1, int(round(self._samplerate * self._dur_ms / 1000.0)))
        t = np.arange(n, dtype=np.float64) / self._samplerate
        tau = max(self._dur_ms / 1000.0 / 4.0, 1e-4)   # decay time constant
        env = np.exp(-t / tau)
        atk = max(1, int(self._samplerate * 0.001))
        if atk < n:
            env[:atk] *= np.linspace(0.0, 1.0, atk)
        sig = np.sin(2.0 * math.pi * freq * t) * env * peak
        return sig.astype(np.float32)

    # -- volume / samplerate ------------------------------------------------
    def set_volume(self, volume: float) -> None:
        """Set click volume in [0, 1]. Rebuilds the scaled play buffers off the
        hot path; an in-flight click keeps its old buffer (no torn read)."""
        self._volume = clamp_volume(volume)
        self._accent = self._accent_base * np.float32(self._volume)
        self._beat = self._beat_base * np.float32(self._volume)

    @property
    def volume(self) -> float:
        return self._volume

    def set_samplerate(self, samplerate: int) -> None:
        """Re-synthesise the voices for a new sample rate (e.g. on stream open).
        Preserves click DURATION; an in-flight click keeps its old buffer."""
        if samplerate <= 0 or int(samplerate) == self._samplerate:
            return
        self._samplerate = int(samplerate)
        self._accent_base = self._synth(self._accent_freq, self._accent_peak)
        self._beat_base = self._synth(self._beat_freq, self._beat_peak)
        self.set_volume(self._volume)

    # -- trigger / MixerSource protocol ------------------------------------
    def trigger(self, accent: bool, offset: int = 0) -> None:
        """Start a click. ``accent`` picks the voice; ``offset`` is the intra-
        block sample at which it begins (the mixer passes the event's offset)."""
        self._buf = self._accent if accent else self._beat
        self._pos = 0
        self._start_off = max(0, int(offset))

    def request_stop(self) -> None:
        """Stop after the in-flight click finishes. The current burst plays out
        (no truncation pop); once it drains, ``finished`` is True and the Mixer
        reaps this source."""
        self._stopping = True

    def arm(self) -> None:
        """Clear a pending stop so a re-registered click isn't instantly reaped
        (called on metronome (re)start)."""
        self._stopping = False

    @property
    def finished(self) -> bool:
        """True once a requested stop's in-flight burst has fully drained — the
        Mixer reaps the source then, so the final click ends click-free."""
        return self._stopping and self._buf is None

    @property
    def active_midi(self) -> Optional[int]:
        return None  # unpitched

    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        """Additively write the in-flight click into ``out[:frames]``.

        No-op (zero allocation) when no click is sounding — the steady state.
        Writes ``[start_off, start_off + n)`` of the block, advancing ``_pos``;
        a click longer than one block resumes here next render.
        """
        buf = self._buf
        if buf is None or frames <= 0:
            return
        start = self._start_off
        if start >= frames:
            # Trigger offset beyond this block (defensive — shouldn't happen).
            self._start_off = start - frames
            return
        remaining = buf.shape[0] - self._pos
        n = remaining if remaining < (frames - start) else (frames - start)
        if n > 0:
            out[start:start + n] += buf[self._pos:self._pos + n]
            self._pos += n
        self._start_off = 0
        if self._pos >= buf.shape[0]:
            self._buf = None


# ---------------------------------------------------------------------------
# MetronomeController — state + the drift-zero scheduler.
# ---------------------------------------------------------------------------
class MetronomeController:
    """Owns metronome state and drives the click source through the mixer.

    Scheduling model (one-beat rolling horizon, drift budget zero):
      * ``start`` anchors beat 0 at the mixer's current clock and schedules it.
      * Each beat's event fires the click and schedules the NEXT beat, so a live
        BPM / time-sig change takes effect on the following beat (it is never
        locked in further ahead than one beat).
      * Within a constant-tempo segment, every onset is
        ``anchor + beat_onset_sample(beat - anchor_beat, bpm, sr)`` — derived
        from the integer index, never accumulated, so drift stays <=0.5 sample.
        A tempo change re-anchors the segment at the just-fired beat (one bounded
        re-round, no accumulation).

    Thread model: ``set_*`` / ``start`` / ``stop`` / ``register_tap`` are called
    from the GUI thread; the per-beat event (and thus ``_schedule_next``) runs on
    the mixer's render thread. State is plain ints/floats (atomic under the GIL);
    register/unregister and schedule are mixer-side thread-safe.
    """

    def __init__(self, mixer, samplerate: int, *,
                 bpm: int = DEFAULT_BPM, time_sig: str = DEFAULT_TIME_SIG,
                 volume: float = 1.0,
                 on_state_changed: Optional[Callable[[], None]] = None):
        self._mixer = mixer
        self._samplerate = int(samplerate)
        self.bpm = clamp_bpm(bpm)
        self.time_sig = time_sig if time_sig in VALID_TIME_SIGS else DEFAULT_TIME_SIG
        self.volume = clamp_volume(volume)
        self._on_state_changed = on_state_changed

        self._click = ClickSource(self._samplerate, self.volume)
        self._running = False
        self._taps: List[float] = []

        # Scheduler segment anchor (see class docstring).
        self._anchor = 0          # absolute sample of self._anchor_beat
        self._anchor_beat = 0     # global beat index at the anchor
        self._anchor_bpm = float(self.bpm)

    # -- introspection ------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def is_running(self) -> bool:
        return self._running

    @property
    def beats_per_bar(self) -> int:
        return beats_per_bar(self.time_sig)

    def is_accent(self, beat_index: int) -> bool:
        """The downbeat (beat 1 of the bar) is accented."""
        return beat_index % self.beats_per_bar == 0

    @property
    def click_source(self) -> ClickSource:
        """The underlying MixerSource (registerable/renderable for tests)."""
        return self._click

    # -- state setters (GUI thread) ----------------------------------------
    def set_bpm(self, bpm: float) -> None:
        nb = clamp_bpm(bpm)
        if nb != self.bpm:
            self.bpm = nb
            self._emit()

    def nudge_bpm(self, delta: int) -> None:
        self.set_bpm(self.bpm + int(delta))

    def set_time_sig(self, time_sig: str) -> None:
        if time_sig in VALID_TIME_SIGS and time_sig != self.time_sig:
            self.time_sig = time_sig
            self._emit()

    def set_volume(self, volume: float) -> None:
        nv = clamp_volume(volume)
        if nv != self.volume:
            self.volume = nv
            self._click.set_volume(nv)
            self._emit()

    def set_samplerate(self, samplerate: int) -> None:
        """Update for a new output rate (call before start, or it re-anchors on
        next start). Re-synthesises the click; timing uses the new rate."""
        if samplerate > 0:
            self._samplerate = int(samplerate)
            self._click.set_samplerate(self._samplerate)

    def register_tap(self, now: Optional[float] = None) -> Optional[int]:
        """Record a tap (default timestamp ``time.monotonic()``) and, once a run
        has >=2 taps, set + return the derived BPM; ``None`` on the first tap of
        a run. ``now`` is injectable so tap-tempo is testable without sleeps."""
        if now is None:
            now = time.monotonic()
        self._taps.append(float(now))
        # Keep just enough taps for the averaging window (+1 for the interval).
        if len(self._taps) > _TAP_AVG_INTERVALS + 1:
            self._taps = self._taps[-(_TAP_AVG_INTERVALS + 1):]
        bpm = bpm_from_tap_times(self._taps)
        if bpm is not None:
            self.set_bpm(bpm)
        return bpm

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._click.arm()              # clear any pending stop before re-register
        self._mixer.register(self._click)
        self._anchor = int(self._mixer.clock)
        self._anchor_beat = 0
        self._anchor_bpm = float(self.bpm)
        self._schedule_beat(0)
        self._emit()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Don't hard-unregister: that truncates an in-flight click burst (a
        # pop, up to ~0.5 amplitude mid-decay). Request stop instead — the burst
        # plays out, then the Mixer reaps the source once it reports finished.
        self._click.request_stop()
        self._emit()

    def toggle(self) -> None:
        self.stop() if self._running else self.start()

    # -- scheduler (render thread, except the initial start) ----------------
    def _onset(self, beat_index: int) -> int:
        return self._anchor + beat_onset_sample(
            beat_index - self._anchor_beat, self._anchor_bpm, self._samplerate)

    def _schedule_beat(self, beat_index: int) -> None:
        onset = self._onset(beat_index)
        accent = self.is_accent(beat_index)

        def event(offset: int, bi: int = beat_index, ac: bool = accent) -> None:
            # Fires on the render thread at the click's intra-block offset.
            if not self._running:
                return
            self._click.trigger(ac, offset)
            self._schedule_next(bi)

        if not self._mixer.schedule(onset, event):
            # Onset already in the past (e.g. a start/clock race or render
            # starvation). Re-anchor to the live clock and retry once so the
            # beat chain self-heals rather than stalling silently.
            self._anchor = int(self._mixer.clock) + 1
            self._anchor_beat = beat_index
            self._anchor_bpm = float(self.bpm)
            self._mixer.schedule(self._onset(beat_index), event)

    def _schedule_next(self, prev_beat_index: int) -> None:
        if not self._running:
            return
        # Re-anchor the segment if the tempo changed since it began, so the new
        # interval applies from the just-fired beat without drift accumulation.
        if float(self.bpm) != self._anchor_bpm:
            prev_onset = self._onset(prev_beat_index)
            self._anchor = prev_onset
            self._anchor_beat = prev_beat_index
            self._anchor_bpm = float(self.bpm)
        self._schedule_beat(prev_beat_index + 1)

    # -- notify -------------------------------------------------------------
    def _emit(self) -> None:
        cb = self._on_state_changed
        if cb is not None:
            try:
                cb()
            except Exception:
                # A misbehaving GUI callback must not break metronome control.
                pass

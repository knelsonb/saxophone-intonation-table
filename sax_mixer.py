"""
Pull-based software mixer for the audio-output foundation (Sprint 1).

Three of the four parity features (metronome, drone, pitch pipes) and the
tape deck's playback are *output* consumers. Rather than give each its own
stream, the engine owns ONE output path and a single numpy mixer that sums
the per-block contributions of however many sources are currently sounding.
Building the output path once, correctly, is the whole point of Sprint 1 —
see docs/parity-sprint-plan.md §2.

Design contract (locked in the parity sprint, "Gandalf's contract"):

* **Sample-accurate, drift budget zero.** The mixer owns a monotonic sample
  clock (``clock``) that advances by exactly ``frames`` on every ``render``.
  This counter — not any wall-clock timer — is the master timebase. Sources
  that must place events at an exact instant (the metronome click) schedule
  them by ABSOLUTE sample index via ``schedule(abs_sample, event)``; the
  mixer fires each one at its correct intra-block offset.

* **Half-open block intervals.** A ``render`` starting at ``t0`` for
  ``frames`` samples owns ``[t0, t0 + frames)``. An event at ``abs_sample ==
  t0 + frames`` belongs to the NEXT block (it fires at offset 0 there), which
  kills the classic double-fire-on-boundary bug.

* **Late events are dropped, not smeared.** An event scheduled in the past
  (``abs_sample < clock``) is dropped and ``dropped_events`` is incremented.
  Clamping it to "now" would shift timing and violate the zero-drift budget;
  silently skipping it would make scheduler starvation invisible. A rising
  ``dropped_events`` is the observable canary that the render thread is
  falling behind.

* **Blocksize-invariant.** Because scheduling is in absolute samples, an
  event at sample N lands at N regardless of how the stream happens to chunk
  its callbacks. The same schedule replayed at a different blocksize fires at
  the same absolute instants.

* **No allocation in the steady-state render path.** The accumulator is
  preallocated once; ``render`` zeroes a slice, sums sources in place, clamps
  in place, and copies into the caller's output buffer. Sources MUST do the
  same (pre-allocate their working buffers; write additively into the slice
  they're handed). ``TestToneSource`` below is the reference proving this is
  achievable, and is what the GUI's test-tone control drives. The plan's
  no-alloc bar is enforced by a tracemalloc gate in the test suite.

* **Source registry is a lock-snapshot.** ``register`` / ``unregister`` swap
  an immutable tuple of sources under a short lock; ``render`` copies the
  tuple reference under the same lock and iterates OUTSIDE it. The audio
  thread therefore never sees a half-updated registry and never holds the
  lock while doing numpy work — mirroring the buffer-snapshot discipline the
  input engine already uses (see sax_audio_engine.AudioEngine).

The mixer is mono. The whole app reasons about a single mono signal; the
engine's output callback fans the mono block out to however many channels
the device wants. Keeping the mixer mono keeps every source trivial.

This module imports only numpy — no sounddevice, no Qt — so the mix math and
the scheduling contract are unit-testable directly, with no audio hardware
and no mock (see test_mixer.py).
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

# A4 reference for the source helpers' midi<->freq math. Mirrors
# sax_audio_engine.A4_DEFAULT; the engine passes its live a4 to sources that
# care, so this is only the standalone default.
A4_DEFAULT = 440.0

# An event is any zero-cost-to-store callable invoked with the intra-block
# sample offset at which it fired. The metronome will pass a closure that
# starts a click envelope at that offset; tests pass a recorder. Keeping the
# event opaque keeps the mixer agnostic to what is being scheduled.
Event = Callable[[int], None]


@runtime_checkable
class MixerSource(Protocol):
    """Protocol every mixer source implements. Synth-agnostic: a numpy click
    generator, a TinySoundFont-backed drone, and the test tone all satisfy
    this identically, so the mixer never knows what kind of sound it sums.

    ``render`` MUST be additive (``out += contribution``), MUST NOT allocate
    in the steady state, and MUST NOT call back into PortAudio, acquire the
    engine lock, or block. It receives ``t0`` (the absolute sample index of
    the block's first frame) so a source triggered mid-block can compute its
    own start offset from a scheduled absolute sample.
    """

    def render(self, out: np.ndarray, frames: int, t0: int) -> None: ...

    @property
    def active_midi(self) -> Optional[int]:
        """The MIDI note this source is currently sounding, or None.

        Pitched sources (drone, pitch pipe) return their note so the input
        pitch detector can vote-exclude it and duck on suspicion (the
        mic-coordination policy: keep the tuner readout live, suppress
        leakage). Unpitched sources (metronome click) return None.
        """
        ...


class Mixer:
    """Sums pull-based sources into a mono output block, sample-accurately.

    Thread model: ``render`` runs on the PortAudio output worker thread.
    ``register`` / ``unregister`` / ``schedule`` are called from the GUI
    (Qt main) thread. The short ``_lock`` guards the source tuple and the
    event schedule; numpy work happens outside it.
    """

    def __init__(self, max_block: int, channels: int = 1):
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        # channels is stored for the engine's fan-out bookkeeping; the mixer
        # itself produces a mono block. See module docstring.
        self.channels = int(channels)
        self._max_block = int(max_block)

        self._lock = threading.Lock()
        # Immutable snapshot the render thread reads. Swapped wholesale on
        # register/unregister so a concurrent render sees either the old or
        # the new set, never a torn one.
        self._sources: tuple[MixerSource, ...] = ()

        # Preallocated mono accumulator. render() reuses _acc[:frames]; no
        # allocation happens on the hot path.
        self._acc = np.zeros(self._max_block, dtype=np.float32)

        # Monotonic master sample clock: absolute index of the NEXT frame to
        # be rendered. Advanced by exactly `frames` per render().
        self._clock = 0

        # Pending events, kept sorted ascending by (abs_sample, seq). seq is a
        # monotonic tiebreaker so two events at the same sample fire in
        # schedule order. Guarded by _lock.
        self._events: list[tuple[int, int, Event]] = []
        self._event_seq = 0

        # Observable canary: count of events dropped for being scheduled in
        # the past. A rising value means the render thread fell behind.
        self.dropped_events = 0

        # Clamp rails as numpy scalars. Passing Python floats to np.clip boxes
        # them into temporary 0-d arrays on (some) calls; pre-boxing once
        # keeps the render path at literally zero allocations.
        self._lo = np.float32(-1.0)
        self._hi = np.float32(1.0)

    # ---- clock ------------------------------------------------------------
    @property
    def clock(self) -> int:
        """Absolute sample index of the next frame render() will emit.

        Readable from any thread (a bare int read is atomic under the GIL).
        Tests drive render() with fixed blocksizes and assert events land at
        ``abs_sample - clock_at_block_start``.
        """
        return self._clock

    def reset_clock(self, value: int = 0) -> None:
        """Reset the sample clock and clear any pending schedule.

        Called by the engine when a fresh output stream opens, so absolute
        sample indices restart from a known origin alongside the new stream's
        own frame count. Not for use while a stream is running.
        """
        with self._lock:
            self._clock = int(value)
            self._events.clear()
            # dropped_events is a lifetime diagnostic — intentionally NOT
            # reset here, mirroring how the engine keeps overflow counts.

    @property
    def max_block(self) -> int:
        return self._max_block

    def resize(self, max_block: int) -> None:
        """Reallocate the accumulator for a new maximum block size.

        Called by the engine when an output stream opens at a blocksize that
        differs from the mixer's current size, so the zero-allocation fast
        path (which only triggers when ``frames == max_block``) stays armed at
        the actual stream blocksize. MUST be called with no stream running —
        it swaps the accumulator the render thread reads. A no-op when the
        size is unchanged.
        """
        max_block = int(max_block)
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        with self._lock:
            if max_block == self._max_block:
                return
            self._max_block = max_block
            self._acc = np.zeros(max_block, dtype=np.float32)

    # ---- source registry (lock-snapshot) ----------------------------------
    def register(self, source: MixerSource) -> MixerSource:
        """Add a source to the mix. Returns the source as an opaque handle
        for unregister(). Idempotent: registering an already-present source
        is a no-op (it is not summed twice)."""
        with self._lock:
            if source in self._sources:
                return source
            self._sources = self._sources + (source,)
        return source

    def unregister(self, handle: MixerSource) -> None:
        """Remove a previously-registered source. Silent if it isn't present
        (e.g. already removed) so double-stops can't raise."""
        with self._lock:
            if handle in self._sources:
                self._sources = tuple(
                    s for s in self._sources if s is not handle)

    def active_sources(self) -> int:
        """Count of currently-registered sources. Diagnostics / tests."""
        return len(self._sources)

    # ---- event scheduling --------------------------------------------------
    def schedule(self, abs_sample: int, event: Event) -> bool:
        """Schedule ``event`` to fire when the render covering ``abs_sample``
        runs. Returns True if accepted, False if dropped for being in the
        past (``abs_sample < clock``), in which case ``dropped_events`` is
        incremented.

        Scheduling is by ABSOLUTE sample index, so the fire instant is
        independent of blocksize. ``event`` is invoked with the intra-block
        offset ``abs_sample - t0`` at fire time, on the render thread.
        """
        abs_sample = int(abs_sample)
        with self._lock:
            if abs_sample < self._clock:
                self.dropped_events += 1
                return False
            seq = self._event_seq
            self._event_seq += 1
            # Insert keeping the list sorted ascending. The pending set is
            # tiny (a handful of upcoming clicks), so a linear insert is
            # cheaper than a heap and keeps fire order trivially correct.
            entry = (abs_sample, seq, event)
            lo, hi = 0, len(self._events)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._events[mid][:2] < entry[:2]:
                    lo = mid + 1
                else:
                    hi = mid
            self._events.insert(lo, entry)
        return True

    def pending_events(self) -> int:
        """Count of not-yet-fired scheduled events. Diagnostics / tests."""
        return len(self._events)

    # ---- the hot path ------------------------------------------------------
    def render(self, out: np.ndarray, frames: int) -> None:
        """Fill ``out[:frames]`` with the summed, clamped mix for the block
        ``[clock, clock + frames)``, then advance the clock by ``frames``.

        Order within a block:
          1. Fire every event due in ``[t0, t0 + frames)`` in ascending
             sample order (half-open: an event at ``t0 + frames`` waits for
             the next block). Firing updates source state, so it precedes
             rendering and a mid-block-triggered source places itself
             correctly via the ``t0`` it receives.
          2. Zero the accumulator slice.
          3. Sum every registered source additively (outside the lock).
          4. Clamp to [-1, 1] in place.
          5. Copy into the caller's buffer.

        Never raises for an over-long ``frames``; it clamps to ``max_block``
        and the caller's buffer governs the rest. No allocation occurs when
        no events are due (the steady-state test-tone / drone path).
        """
        if frames <= 0:
            return
        if frames > self._max_block:
            frames = self._max_block

        t0 = self._clock

        # 1. Fire due events. Snapshot+drain under the lock, invoke outside it
        #    so an event handler can't deadlock against schedule(). When the
        #    schedule is empty (the common case) nothing is allocated.
        if self._events:
            due: Optional[list[tuple[int, Event]]] = None
            end = t0 + frames
            with self._lock:
                while self._events and self._events[0][0] < end:
                    abs_sample, _seq, event = self._events.pop(0)
                    # An event already past t0 (shouldn't happen — schedule()
                    # rejects past samples and the clock only moves forward —
                    # but guard anyway) fires at offset 0 rather than negative.
                    offset = abs_sample - t0
                    if offset < 0:
                        offset = 0
                    if due is None:
                        due = []
                    due.append((offset, event))
            if due is not None:
                for offset, event in due:
                    try:
                        event(offset)
                    except Exception:
                        # A misbehaving event must not take down the audio
                        # callback; drop it and keep the stream alive.
                        pass

        # 2-3. Sum sources into the preallocated accumulator. When the block
        #    fills the whole accumulator (the steady-state case — the mixer is
        #    sized to the stream's blocksize), use the buffer WHOLE: a numpy
        #    slice such as self._acc[:frames] allocates a fresh view object
        #    every call, which a strict peak-allocation gate (plan §2) counts.
        #    Slicing is taken only for a rare short final block.
        full = (frames == self._max_block)
        acc = self._acc if full else self._acc[:frames]
        acc.fill(0.0)
        # Lock-free read of the source tuple: a reference read is atomic under
        # the GIL, and register/unregister swap the tuple WHOLE (never mutate
        # in place), so render sees either the old or new set, never a torn
        # one — without paying a lock acquire on the audio thread.
        for src in self._sources:
            try:
                src.render(acc, frames, t0)
            except Exception:
                # One bad source shouldn't silence the whole mix or crash the
                # callback; skip its contribution this block.
                continue

        # 3b. Reap fully-finished sources — a released test tone or pitch pad
        #     whose tail has faded to zero (MixerSource.finished documents this
        #     contract: "the mixer should drop it"). This is what lets a source
        #     self-retire with a click-free release instead of a hard-cut
        #     unregister. Cheap lock-free scan of the tiny source tuple; the
        #     lock is taken ONLY to swap the tuple WHOLE when something actually
        #     needs dropping (rare), so the steady state stays lock-free and
        #     allocation-free (a plain loop + getattr boxes nothing). A source
        #     with no `finished` attribute (the default) is never reaped.
        reap = False
        for s in self._sources:
            if getattr(s, "finished", False):
                reap = True
                break
        if reap:
            with self._lock:
                self._sources = tuple(
                    s for s in self._sources if not getattr(s, "finished", False))

        # 4. Clamp in place to the output rails. np.clip routes through a
        #    fromnumeric wrapper that allocates small dispatch objects on a
        #    fraction of calls; the minimum/maximum ufuncs with preboxed
        #    numpy-scalar bounds go straight through umath and allocate
        #    nothing. Two passes = clamp to [lo, hi].
        np.minimum(acc, self._hi, out=acc)
        np.maximum(acc, self._lo, out=acc)

        # 5. Hand off to the caller's buffer. np.copyto avoids allocating an
        #    out[:frames] slice view when the target is exactly the block size
        #    (the steady-state mono path); other shapes fall back to indexed
        #    assignment. A 2-D target gets the mono block broadcast to all
        #    channels.
        if out.ndim == 1:
            if out.shape[0] == frames:
                np.copyto(out, acc)
            else:
                out[:frames] = acc
        else:
            out[:frames, :] = acc.reshape(-1, 1)

        # Advance the master clock by exactly the block size.
        self._clock = t0 + frames

    # ---- coordination surface ---------------------------------------------
    def sounding_midis(self) -> frozenset[int]:
        """The set of MIDI notes currently being sounded by pitched sources.

        Consumed by the engine's ``get_sounding_output_midis`` so the input
        pitch detector can vote-exclude output leakage and duck on suspicion
        (keep the tuner readout live while the drone sounds — the
        mic-coordination policy). Unpitched sources contribute nothing.
        """
        with self._lock:
            sources = self._sources
        out: set[int] = set()
        for src in sources:
            try:
                m = src.active_midi
            except Exception:
                m = None
            if m is not None:
                out.add(int(m))
        return frozenset(out)


# ---------------------------------------------------------------------------
# Reference source: a pure-numpy sine test tone.
# ---------------------------------------------------------------------------
class TestToneSource:
    """A steady sine tone — the reference MixerSource and the Sprint-1
    acceptance vehicle ("a test tone plays through the mixer while the tuner
    still reads the mic").

    It demonstrates the zero-allocation discipline every real source must
    follow: all working buffers (the sample ramp, the phase scratch) are
    allocated once in ``__init__``; ``render`` runs entirely in place via
    numpy ``out=`` arguments and produces no garbage. Phase is tracked in
    radians and wrapped each block so it never grows without bound across a
    long-running tone.
    """

    # The leading "Test" in the (semantically correct) name "test tone"
    # collides with pytest's Test*-class collection heuristic. This marks the
    # class as not-a-test so pytest ignores it; the public name stays frozen
    # as published in Handshake #2.
    __test__ = False

    def __init__(self, freq: float, samplerate: int, max_block: int,
                 gain: float = 0.2, midi: Optional[int] = None,
                 attack_ms: float = 0.0, release_ms: float = 0.0):
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        self.freq = float(freq)
        self.samplerate = int(samplerate)
        self._max_block = int(max_block)
        self._gain = float(gain)
        self._midi = midi
        # Per-sample phase increment in radians. Kept as both a Python float
        # (for precise phase accumulation across many blocks) and a preboxed
        # numpy scalar (so the per-block multiply doesn't box it each call).
        self._dphi = 2.0 * math.pi * self.freq / self.samplerate
        self._dphi32 = np.float32(self._dphi)
        self._gain32 = np.float32(self._gain)
        self._phase = 0.0
        # 0-d holder for the current phase, updated in place via fill() each
        # block so the per-block phase add boxes no new scalar object.
        self._phase_box = np.zeros((), dtype=np.float32)
        # Preallocated scratch: an index ramp [0,1,2,...] and a work buffer
        # the sine is computed into. Reused every block; never reallocated.
        # float32 throughout so the additive mix into the float32 accumulator
        # needs no per-block dtype cast (an .astype() would allocate). Phase
        # is carried in a float64 scalar for long-run precision; only the
        # within-block ramp is float32, which is ample for one block.
        self._ramp = np.arange(self._max_block, dtype=np.float32)
        self._work = np.zeros(self._max_block, dtype=np.float32)

        # Optional click-free envelope (OPT-IN; default OFF so the reference
        # steady tone — and every test/use that builds it plainly — is
        # byte-identical). When enabled, the tone attacks from silence on the
        # first render and release() ramps it back down; once the tail reaches
        # zero the source reports finished=True so the Mixer auto-reaps it,
        # turning stop_test_tone into a fade instead of a hard-cut click.
        # Mirrors PitchPipeSource's envelope. _env_work is allocated only when
        # enveloped (the steady reference tone keeps its lean footprint).
        self._enveloped = (attack_ms > 0.0 or release_ms > 0.0)
        self._env = 0.0 if attack_ms > 0.0 else 1.0
        self._env_target = 1.0
        self._atk_step = 1.0 / max(1.0, attack_ms * self.samplerate / 1000.0)
        self._rel_step = 1.0 / max(1.0, release_ms * self.samplerate / 1000.0)
        self._done = False
        self._env_work = (np.zeros(self._max_block, dtype=np.float32)
                          if self._enveloped else None)

    # -- gain / duck --------------------------------------------------------
    def set_gain(self, g: float) -> None:
        """Set output gain in [0, 1]. The mic-coordination duck drives this
        (e.g. to ~0.3 briefly on suspicion of mic bleed)."""
        self._gain = max(0.0, min(1.0, float(g)))
        self._gain32 = np.float32(self._gain)

    @property
    def gain(self) -> float:
        return self._gain

    # -- release / lifecycle ------------------------------------------------
    def release(self) -> None:
        """Begin a click-free release fade (enveloped tones only). The tone
        keeps sounding until its tail reaches zero, then reports finished=True
        so the Mixer drops it. For a tone built WITHOUT an envelope this marks
        it finished immediately (same effect as the old hard unregister, but
        via the uniform reap path)."""
        if self._enveloped:
            self._env_target = 0.0
        else:
            self._done = True

    @property
    def finished(self) -> bool:
        """True once a released tone has fully faded — the Mixer reaps it."""
        return self._done

    # -- MixerSource protocol ----------------------------------------------
    @property
    def active_midi(self) -> Optional[int]:
        # A released-and-faded tone no longer sounds (so it stops vote-excluding
        # / ducking the moment its tail is gone); otherwise reports its midi.
        if self._done:
            return None
        return self._midi

    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        """Additively write ``frames`` samples of the sine into ``out``.

        Zero allocation: phase[k] = _phase + dphi*k is built into _work via
        in-place numpy ops on the preallocated ramp, the sine is taken in
        place, scaled by gain, and added to ``out``. ``t0`` is unused — a
        steady tone has no scheduled start — but is part of the protocol.
        """
        if self._done:
            return
        if frames <= 0 or self._gain == 0.0:
            # Still advance phase so a later un-duck stays phase-continuous.
            self._phase = (self._phase + self._dphi * frames) % (2.0 * math.pi)
            return
        if self._enveloped and self._env == 0.0 and self._env_target == 0.0:
            # Released and fully faded: retire (the Mixer reaps on finished).
            self._done = True
            return
        # Use the working buffers WHOLE when the block fills them (the steady
        # state); slicing self._work[:n] would allocate a view object every
        # call, which the strict peak-allocation gate counts. ramp/work/out
        # are all sliced together only for a short final block.
        if frames >= self._max_block:
            n = self._max_block
            ramp = self._ramp
            work = self._work
            tgt = out if (out.ndim == 1 and out.shape[0] == n) else out[:n]
        else:
            n = frames
            ramp = self._ramp[:n]
            work = self._work[:n]
            tgt = out[:n]
        # work = _phase + dphi * ramp, all in place on preboxed scalars.
        np.multiply(ramp, self._dphi32, out=work)
        self._phase_box.fill(self._phase)   # in-place; boxes no new scalar
        work += self._phase_box
        np.sin(work, out=work)
        work *= self._gain32
        if self._enveloped:
            # Apply the attack/release envelope. Steady (env == target) is a
            # single scalar multiply (or nothing at full gain); a transition
            # ramps the envelope per-sample across the block. The full-block
            # path uses the preallocated _env_work WHOLE, so the steady state
            # stays allocation-free (only a short final block slices a view).
            if self._env == self._env_target:
                if self._env != 1.0:
                    work *= np.float32(self._env)
            else:
                step = (self._atk_step if self._env_target > self._env
                        else -self._rel_step)
                env = (self._env_work if n == self._max_block
                       else self._env_work[:n])
                np.multiply(ramp, np.float32(step), out=env)
                env += np.float32(self._env)
                np.clip(env, 0.0, 1.0, out=env)
                work *= env
                self._env = float(env[n - 1])
                if abs(self._env - self._env_target) <= max(self._atk_step,
                                                            self._rel_step):
                    self._env = self._env_target
        # Additive mix — both float32, so this is a true in-place add with no
        # dtype cast and no temporary.
        tgt += work
        # Advance and wrap the phase for block-to-block continuity.
        self._phase = (self._phase + self._dphi * n) % (2.0 * math.pi)


# ---------------------------------------------------------------------------
# GainGlide — per-sample slew-limited de-zipper for a source's applied gain.
# ---------------------------------------------------------------------------
class GainGlide:
    """Smooths a gain that is *targeted* coarsely but must be *applied* without
    zipper noise.

    The D3 mic-coordination policy (sax_coordination) emits a duck gain once per
    detection hop (~46 ms) — a coarse step (e.g. 1.0 -> 0.30 in ~2 frames).
    Applying that straight to a sustained drone would step the gain by a large
    amount in a single sample = an audible click. GainGlide is the *mechanism*
    half of that split: it glides the ACTUAL applied gain toward the target
    SAMPLE-BY-SAMPLE, slew-limited, so the gain is continuous regardless of how
    coarsely ``set_target`` is called. The policy owns the macro envelope; this
    owns audio-rate smoothing — no second duck envelope.

    A source EMBEDS one of these and calls :meth:`apply` inside its ``render``
    (the only per-sample hook the mixer exposes). ``set_target`` is called from
    the control side (the engine's per-hop D3 wiring). Zero allocation in the
    steady state: the per-sample gain envelope is written into a preallocated
    buffer.
    """

    def __init__(self, max_block: int, samplerate: int, glide_ms: float = 5.0,
                 initial: float = 1.0):
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        # A full 0->1 swing completes in glide_ms; per-sample slew cap.
        glide_samples = max(1.0, samplerate * float(glide_ms) / 1000.0)
        self._max_step = 1.0 / glide_samples
        self._gain = float(initial)
        self._target = float(initial)
        self._max_block = int(max_block)
        # Preallocated per-sample gain envelope + an index ramp, reused each block.
        self._g = np.zeros(self._max_block, dtype=np.float32)
        self._ramp = np.arange(self._max_block, dtype=np.float32)

    def set_target(self, target: float) -> None:
        """Set the gain to glide toward, clamped to [0, 1]. Called per hop from
        the control side; cheap and lock-free (a single float store)."""
        self._target = max(0.0, min(1.0, float(target)))

    @property
    def gain(self) -> float:
        """The current applied gain (end of the last rendered block)."""
        return self._gain

    @property
    def target(self) -> float:
        return self._target

    def apply(self, block: np.ndarray, frames: int) -> float:
        """Scale ``block[:frames]`` in place by the gliding gain; return the
        gain at the block's end. Per-sample step never exceeds the slew cap, so
        a coarse target jump becomes a smooth ramp (no zipper). Zero allocation:
        the gain envelope is built in the preallocated ``_g`` buffer.
        """
        if frames <= 0:
            return self._gain
        if frames > self._max_block:
            frames = self._max_block
        g0 = self._gain
        tgt = self._target
        step = self._max_step
        delta = tgt - g0
        gview = self._g[:frames]
        if -step <= delta <= step:
            # Within one slew step of the target: snap + hold (no audible jump).
            gview.fill(np.float32(tgt))
            self._gain = tgt
        else:
            s = step if delta > 0 else -step
            n_reach = int(abs(delta) / step)   # samples to reach target at slew cap
            if n_reach >= frames:
                # Ramp the whole block; target not reached this block.
                np.multiply(self._ramp[:frames], np.float32(s), out=gview)
                gview += np.float32(g0 + s)    # +1 sample: i=0 has already moved
                self._gain = g0 + s * frames
            else:
                # Ramp to target over n_reach samples, then hold target.
                head = self._g[:n_reach]
                np.multiply(self._ramp[:n_reach], np.float32(s), out=head)
                head += np.float32(g0 + s)
                self._g[n_reach:frames] = np.float32(tgt)
                self._gain = tgt
        block[:frames] *= gview
        return self._gain

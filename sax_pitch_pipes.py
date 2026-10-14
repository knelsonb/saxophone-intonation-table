"""Pitch pipes: pure-numpy sine reference pads over the chromatic octave.

The simplest of the pitched output sources. A pitch pipe
is a steady sine at a chromatic reference note (C4-B4 by default), sounded at
the app's current A4 so it doubles as a tuning reference. Tap a pad to sustain
it, tap again to release; several can sound at once.

Unlike the metronome click (unpitched), a sustained pipe IS a pitched output —
so each :class:`PitchPipeSource` reports ``active_midi`` (audible-tail aware,
like the drone) and therefore participates in the D3 mic-coordination
vote-exclude the moment it sounds. Given the engine, the controller also
attaches each pad as a duck consumer, so mic bleed of a pad is ducked like
the drone's, and a player holding the pad's note in unison is measured. Pure
numpy, no Qt / no sounddevice, so the frequency math and the sources are
unit-testable directly on a Mixer.
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

from sax_mixer import GainGlide
from sax_pitch import A4_DEFAULT, midi_to_freq

# Default chromatic octave: C4 (MIDI 60) .. B4 (MIDI 71).
PITCH_PIPE_NOTES: List[int] = list(range(60, 72))

__all__ = [
    "A4_DEFAULT", "PITCH_PIPE_NOTES",
    "midi_to_freq", "pitch_pipe_freq",
    "PitchPipeSource", "PitchPipesController",
]


# ---------------------------------------------------------------------------
# Pure frequency math.
# ---------------------------------------------------------------------------
def pitch_pipe_freq(midi: float, a4: float = A4_DEFAULT) -> float:
    """Frequency of a pitch-pipe pad (alias of :func:`midi_to_freq`)."""
    return midi_to_freq(midi, a4)


# ---------------------------------------------------------------------------
# PitchPipeSource — one sustained sine pad with a click-free envelope.
# ---------------------------------------------------------------------------
class PitchPipeSource:
    """A single sustained sine pad, a MixerSource.

    Built on the zero-allocation sine discipline of ``sax_mixer.TestToneSource``
    (preallocated ramp + work buffers, phase carried in a float64 scalar and
    wrapped each block). A short linear attack/release envelope avoids clicks on
    tap/release; in the steady (fully-sounding) state the envelope is constant,
    so the render path is allocation-free, honouring the strict numpy-source bar.

    ``active_midi`` reports the pad's MIDI while it is sounding OR while its
    release tail is still audible, and ``None`` once it has faded out — the
    audible-tail semantics the D3 coordinator needs.
    """

    __test__ = False

    def __init__(self, midi: int, samplerate: int, max_block: int,
                 a4: float = A4_DEFAULT, gain: float = 0.18,
                 attack_ms: float = 8.0, release_ms: float = 80.0):
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        self._midi = int(midi)
        self.samplerate = int(samplerate)
        self._gain = float(gain)
        self._attack_ms = float(attack_ms)
        self._release_ms = float(release_ms)
        self.freq = midi_to_freq(self._midi, a4)

        # Phase is carried as a float for precise long-run accumulation; the
        # per-block multiply uses preboxed numpy scalars (_dphi32, _gain32)
        # so it allocates nothing.
        self._gain32 = np.float32(self._gain)
        self._phase = 0.0
        self._phase_box = np.zeros((), dtype=np.float32)

        # Envelope: current gain in [0,1], a target (1 = on, 0 = releasing), and
        # per-sample slopes. Starts at 0 and attacks to 1 (click-free tap-on).
        self._env = 0.0
        self._env_target = 1.0
        self._done = False  # True once a release has fully faded out
        # Serialises an A4 change (GUI) against a rate change (audio-control
        # thread): both recompute the steps from freq and rate. render()
        # never takes it.
        self._ctl = threading.Lock()
        self._set_rate_steps()
        # The D3 duck (engine coordination); open unless mic bleed is heard.
        self._duck = GainGlide(int(max_block), self.samplerate, initial=1.0)
        self._max_block = 0
        self._alloc(max_block)

    def _set_rate_steps(self) -> None:
        """Per-sample phase and envelope steps for the current rate and pitch.
        The render thread may read these mid-update; the worst case is one
        block at a mix of old and new step, which is inaudible."""
        dphi = 2.0 * math.pi * self.freq / self.samplerate
        self._dphi32 = np.float32(dphi)
        self._dphi = dphi
        self._atk_step = 1.0 / max(1.0, self._attack_ms * self.samplerate / 1000.0)
        self._rel_step = 1.0 / max(1.0, self._release_ms * self.samplerate / 1000.0)

    def _alloc(self, max_block: int) -> None:
        self._max_block = int(max_block)
        self._ramp = np.arange(self._max_block, dtype=np.float32)
        self._work = np.zeros(self._max_block, dtype=np.float32)
        self._env_work = np.zeros(self._max_block, dtype=np.float32)
        self._duck.reformat(self._max_block, self.samplerate)

    def on_format_changed(self, samplerate: int, max_block: int) -> None:
        """Keep sounding, at the same pitch, on a stream with another rate or
        block size. Only the per-sample steps change; the phase carries over."""
        samplerate, max_block = int(samplerate), int(max_block)
        if samplerate <= 0 or max_block < 1:
            return
        with self._ctl:
            rate_changed = samplerate != self.samplerate
            if rate_changed:
                self.samplerate = samplerate
                self._set_rate_steps()
            if rate_changed or max_block != self._max_block:
                self._alloc(max_block)

    def set_duck_target(self, level: float) -> None:
        """Engine D3 hook: per-hop duck gain target (in [duck_depth, 1.0])."""
        self._duck.set_target(level)

    def set_a4(self, a4: float) -> None:
        """Retune to a new concert pitch while sounding. The phase is
        continuous, so a held pad glides to the new pitch without a click."""
        with self._ctl:
            self.freq = midi_to_freq(self._midi, a4)
            self._set_rate_steps()

    # -- control ------------------------------------------------------------
    @property
    def midi(self) -> int:
        return self._midi

    def release(self) -> None:
        """Begin the release fade. The pad keeps sounding (and reporting
        active_midi) until the tail reaches zero, then goes inactive."""
        self._env_target = 0.0

    @property
    def finished(self) -> bool:
        """True once a released pad has fully faded (the mixer should drop it)."""
        return self._done

    @property
    def active_midi(self) -> Optional[int]:
        # Audible while sounding OR while the release tail still rings.
        if self._done:
            return None
        if self._env_target > 0.0 or self._env > 1e-4:
            return self._midi
        return None

    # -- MixerSource render -------------------------------------------------
    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        if frames <= 0 or self._done:
            return
        n = self._max_block if frames >= self._max_block else frames
        steady = (self._env == self._env_target)

        if steady and self._env == 0.0:
            # Fully released: nothing to add. Mark done so the owner unregisters.
            self._done = True
            return

        # Build the sine into _work (phase ramp -> sin), same as TestToneSource.
        if n == self._max_block:
            ramp, work, env = self._ramp, self._work, self._env_work
            tgt = out if (out.ndim == 1 and out.shape[0] == n) else out[:n]
        else:
            ramp, work, env = self._ramp[:n], self._work[:n], self._env_work[:n]
            tgt = out[:n]
        np.multiply(ramp, self._dphi32, out=work)
        self._phase_box.fill(self._phase)
        work += self._phase_box
        np.sin(work, out=work)
        work *= self._gain32

        if steady:
            # Constant envelope (== 1.0 here): single scalar multiply, no alloc.
            if self._env != 1.0:
                work *= np.float32(self._env)
        else:
            # Transition: per-sample linear ramp of the envelope across the block.
            step = self._atk_step if self._env_target > self._env else -self._rel_step
            np.multiply(ramp, np.float32(step), out=env)   # env = step*k
            env += np.float32(self._env)                    # env = cur + step*k
            np.clip(env, 0.0, 1.0, out=env)
            work *= env
            self._env = float(env[n - 1])
            # Snap to target once within one step (avoid float crawl).
            if abs(self._env - self._env_target) <= max(self._atk_step, self._rel_step):
                self._env = self._env_target

        # The duck only costs a pass over the block while it is engaged.
        if self._duck.gain != 1.0 or self._duck.target != 1.0:
            self._duck.apply(work, n)
        tgt += work
        self._phase = (self._phase + self._dphi * n) % (2.0 * math.pi)


# ---------------------------------------------------------------------------
# PitchPipesController — manage a set of pads on the mixer.
# ---------------------------------------------------------------------------
class PitchPipesController:
    """Owns the active pitch-pipe pads. ``toggle(midi)`` sounds or releases a
    pad; multiple may sound at once. Released pads are reaped (unregistered)
    once their tail fades. ``on_state_changed`` fires on any pad change so the
    GUI can re-light the modal's pads.
    """

    def __init__(self, mixer, samplerate: int, *, a4: float = A4_DEFAULT,
                 max_block: Optional[int] = None, engine=None,
                 on_state_changed: Optional[Callable[[], None]] = None):
        self._mixer = mixer
        # Duck wiring, as for the drone: a sounding pad is attached as a duck
        # consumer (engine.attach_duck_consumer) and detached once it is gone.
        self._engine = engine
        self._samplerate = int(samplerate)
        self._a4 = float(a4)
        self._max_block = int(max_block) if max_block else int(
            getattr(mixer, "max_block", 1024))
        self._on_state_changed = on_state_changed
        # midi -> live source (sounding or releasing).
        self._pads: Dict[int, PitchPipeSource] = {}

    def _format(self) -> tuple[int, int]:
        """(rate, block) to build a pad at: the mixer's output format once
        one is known, else what this controller was constructed with. Pads
        already sounding follow later format changes through the mixer."""
        sr = int(getattr(self._mixer, "samplerate", 0) or 0)
        if sr <= 0:
            return self._samplerate, self._max_block
        return sr, int(getattr(self._mixer, "max_block", self._max_block))

    def _emit(self) -> None:
        cb = self._on_state_changed
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def active_midis(self) -> frozenset[int]:
        """MIDIs currently held on (excludes pads that are only releasing)."""
        return frozenset(m for m, s in self._pads.items() if s._env_target > 0.0)

    def toggle(self, midi: int) -> bool:
        """Sound the pad if off, release it if on. Returns True if now sounding."""
        midi = int(midi)
        self._reap()
        src = self._pads.get(midi)
        if src is not None and src._env_target > 0.0:
            src.release()           # keeps releasing source registered until faded
            self._emit()
            return False
        # (Re)start a fresh pad — replaces a still-releasing one at this note.
        sr, mb = self._format()
        new = PitchPipeSource(midi, sr, mb, a4=self._a4)
        if src is not None:
            self._drop(src)
        self._pads[midi] = new
        self._mixer.register(new)
        if self._engine is not None:
            self._engine.attach_duck_consumer(new)
        self._emit()
        return True

    def _drop(self, src: PitchPipeSource) -> None:
        """Take a pad out of the mix and out of the duck."""
        self._mixer.unregister(src)
        if self._engine is not None:
            self._engine.detach_duck_consumer(src)

    def release(self, midi: int) -> None:
        src = self._pads.get(int(midi))
        if src is not None and src._env_target > 0.0:
            src.release()
            self._emit()

    def release_all(self) -> None:
        changed = False
        for s in self._pads.values():
            if s._env_target > 0.0:
                s.release()
                changed = True
        if changed:
            self._emit()

    def silence(self) -> None:
        """Drop every pad at once, with no release tail. For when the output
        is gone: nothing renders, so a fade would only play out (over a pad
        started afresh) once the output is back."""
        if not self._pads:
            return
        for src in self._pads.values():
            self._drop(src)
        self._pads.clear()
        self._emit()

    def set_a4(self, a4: float) -> None:
        """Update concert pitch. New pads sound at it, and pads already
        sounding (or fading out) are retuned to it phase-continuously, so the
        pad a player is tuning against always matches the app's A4. Ignores
        non-finite / non-positive input (would render NaN into the sine)."""
        a4 = float(a4)
        if not math.isfinite(a4) or a4 <= 0.0:
            return
        self._a4 = a4
        for src in self._pads.values():
            src.set_a4(a4)

    def _reap(self) -> None:
        """Unregister pads whose release tail has fully faded."""
        done = [m for m, s in self._pads.items() if s.finished]
        for m in done:
            self._drop(self._pads.pop(m))

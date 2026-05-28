"""
Tape deck: record one mic take, play it back through the mixer, export it.

Sprint-4 parity feature (Android ``useDeck``). The deck is the first
*input-side* consumer of the audio engine — Sprints 1-3 were all output
(metronome / drone / pitch pipes). The split, mirroring D3's
mechanism/policy divide (GainGlide vs OutputCoordinator):

* **Engine = pure capture mechanism** (Gandalf, sax_audio_engine). While
  armed, the input callback slice-assigns each mic frame into a bounded
  preallocated buffer — zero hot-path allocation, zero drop until the cap.
  ``start_input_recording(max_seconds) -> bool`` arms;
  ``stop_input_recording() -> (take, sr, truncated)`` hands back the whole
  take in one shot (no incremental drain); ``is_input_recording() -> bool``
  reports the armed flag; teardown disarms so an abrupt close / device
  switch can never orphan an armed flag at a dead stream.

* **DeckController = policy / state machine** (this module). It owns the
  idle → recording → have-take → playing state graph, arms/finalises the
  engine capture, drives a :class:`DeckPlaybackSource` on the mixer for
  playback, and encodes the take to a 16-bit-PCM WAV. It never touches the
  audio callback; the engine never touches it.

State transitions (interrupts called out):

    idle      --start_record()--> recording      arm; False+stay if mic shut
    have-take --start_record()--> recording      DISCARDS the old take
    playing   --start_record()--> recording      INTERRUPT: drop the source first
    recording --stop()--------->  have-take       finalise; empty take -> idle
    playing   --stop()--------->  have-take       drop the source; take kept
    have-take --play()--------->  playing         register a playback source
    playing   --play()--------->  playing         RESTART at 0 ("replay")
    recording --play()--------->  recording       no-op (False) — stop first
    <cap hit>                     -> have-take     engine auto-disarms; pump() sees it
    <playback end>                -> have-take     source.finished; pump() sees it
    close()/shutdown()            -> idle          disarm + drop source; never orphans

The single delicate cross-thread point: a playback source runs out on the
**audio** thread, and a recording hits its cap and auto-disarms on the
**audio** thread — but the *state transition* (and therefore the
``on_state_changed`` callback the GUI relabels buttons from) must happen on
the **GUI** thread. So neither transition is fired from the callback: the
source only flips a lock-free ``finished`` bool, and the engine only flips
its armed flag; :meth:`DeckController.pump` — called from the GUI's existing
repaint/poll tick — observes both and performs the transition. Net:
``on_state_changed`` always fires on the GUI thread, exactly like the drone
and metronome controllers, so the GUI never needs a ``QTimer.singleShot``
marshal.

Pure numpy + stdlib ``wave``; no Qt, no sounddevice. The frequency-free WAV
helpers and the sources are unit-testable directly on a Mixer with a fake
engine — no audio hardware, no mock stream (see test_deck.py).
"""

from __future__ import annotations

import math
import os
import wave
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

__all__ = [
    "DeckState",
    "write_wav", "read_wav",
    "DeckPlaybackSource",
    "DeckController",
]

# Symmetric 16-bit-PCM scale. Using the SAME constant for encode and decode
# makes the float32->int16->float32 round-trip tight (<= half an LSB,
# ~1.5e-5): +1.0 <-> +32767 and -1.0 <-> -32767 map exactly, with no rail
# clipping. (The asymmetric int16 floor -32768 is still accepted on encode as
# a clip guard for out-of-range inputs.)
_PCM16_SCALE = 32767.0


# ---------------------------------------------------------------------------
# Deck state — a str-Enum so callers can compare to either the enum member or
# the plain string. ``state is DeckState.RECORDING`` (tests) and
# ``state == 'recording'`` (the GUI's button-enable logic) are BOTH true, so
# there is one source of truth with no second stringly-typed mirror.
# ---------------------------------------------------------------------------
class DeckState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    HAVE_TAKE = "have-take"
    PLAYING = "playing"

    def __str__(self) -> str:  # so f-strings / logs print 'recording', not 'DeckState.RECORDING'
        return self.value


# ---------------------------------------------------------------------------
# Pure WAV I/O — stdlib ``wave``, mono, 16-bit PCM. No Qt, no engine.
# ---------------------------------------------------------------------------
def write_wav(path: str, samples, samplerate: int) -> None:
    """Write mono ``float32`` samples in [-1, 1] to a 16-bit-PCM WAV.

    nchannels=1, sampwidth=2, framerate=``samplerate``. Samples are rounded
    to the nearest int16 and clipped to the int16 range; values outside
    [-1, 1] are clamped rather than wrapping. An empty array writes a valid
    zero-frame header.
    """
    arr = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
    # f32 -> int16: round to nearest, clip to the int16 rails.
    scaled = np.rint(arr * _PCM16_SCALE)
    np.clip(scaled, -32768.0, 32767.0, out=scaled)
    pcm = scaled.astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(samplerate))
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a 16-bit-PCM WAV back to ``(float32 mono in [-1, 1], samplerate)``.

    Multi-channel files are down-mixed to mono by averaging channels (the
    deck only ever writes mono, so this is purely defensive). Raises
    ``ValueError`` for sample widths other than 16-bit — the deck's own files
    are always 16-bit; foreign widths are out of scope.
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(
            f"read_wav supports 16-bit PCM only, got sampwidth={sampwidth} bytes")
    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        # Trim a ragged tail (shouldn't happen for a well-formed file) before
        # reshaping, then average the channels down to mono.
        usable = (data.shape[0] // n_channels) * n_channels
        data = data[:usable].reshape(-1, n_channels).mean(axis=1)
    samples = data.astype(np.float32) / np.float32(_PCM16_SCALE)
    return samples, int(sr)


# ---------------------------------------------------------------------------
# DeckPlaybackSource — a MixerSource that replays a captured take.
# ---------------------------------------------------------------------------
class DeckPlaybackSource:
    """Plays a captured mono take back through the mixer, once.

    ``active_midi`` is ALWAYS ``None``: a recording is arbitrary audio, not a
    tuned pitch, so it must stay invisible to ``Mixer.sounding_midis`` / the
    D3 coordinator — deck playback is never vote-excluded and never ducked.

    The take was captured at the *input* device's sample rate; the mixer is
    pulled at the *output* device's rate. When those differ (the app allows a
    mic on one device and playback on another), naive sample-for-sample
    playback would pitch-shift the take — e.g. a 48 kHz take through a 44.1
    kHz output plays ~8.8% slow. So when the rates differ this source
    linear-interpolates from capture-rate to output-rate, keeping playback
    pitch-accurate. When the rates match it takes a direct copy fast path.

    Zero *scaling* allocation: every working buffer is preallocated once in
    ``__init__`` sized to ``max_block`` and reused; the absolute read position
    is carried in a float64 scalar so it stays integer-precise across a long
    take (a float32 position would lose 1-sample precision past ~16.7 M
    samples — minutes of audio). The handful of fixed-size slice views per
    block is constant, so a non-scaling tracemalloc gate passes (same
    calibration as the mixer / drone sources).
    """

    def __init__(self, take, capture_samplerate: int, output_samplerate: int,
                 max_block: int, gain: float = 1.0):
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        self._take = np.ascontiguousarray(
            np.asarray(take, dtype=np.float32).reshape(-1))
        self._n = int(self._take.shape[0])
        self.capture_samplerate = int(capture_samplerate)
        self.output_samplerate = int(output_samplerate)
        self._max_block = int(max_block)
        self._gain32 = np.float32(gain)

        # Capture samples consumed per output sample. r == 1.0 => no resample.
        self._ratio = (float(self.capture_samplerate) / float(self.output_samplerate)
                       if self.output_samplerate > 0 else 1.0)
        # Resample only when the rates genuinely differ AND we have >=2 samples
        # to interpolate between; otherwise the direct fast path.
        self._resample = (self.capture_samplerate != self.output_samplerate
                          and self._n >= 2)

        self._pos = 0.0          # absolute fractional read position, in take samples (f64)
        self._finished = (self._n == 0)

        # Fast-path scratch: one work buffer for (segment * gain).
        self._work = np.zeros(self._max_block, dtype=np.float32)
        # Resample scratch — allocated only when needed.
        if self._resample:
            self._ramp = np.arange(self._max_block, dtype=np.float64)
            self._pos_buf = np.zeros(self._max_block, dtype=np.float64)
            self._floor_buf = np.zeros(self._max_block, dtype=np.float64)
            self._frac64 = np.zeros(self._max_block, dtype=np.float64)
            self._frac32 = np.zeros(self._max_block, dtype=np.float32)
            self._idx = np.zeros(self._max_block, dtype=np.intp)
            self._g0 = np.zeros(self._max_block, dtype=np.float32)
            self._g1 = np.zeros(self._max_block, dtype=np.float32)

    # -- MixerSource protocol ----------------------------------------------
    @property
    def active_midi(self) -> Optional[int]:
        return None  # recorded audio is not a pitch — invisible to D3.

    @property
    def finished(self) -> bool:
        """True once the take has been fully played out. Read on the GUI
        thread by :meth:`DeckController.pump` to retire the source."""
        return self._finished

    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        if self._finished or frames <= 0:
            return
        if self._resample:
            self._render_resampled(out, frames)
        else:
            self._render_direct(out, frames)

    # -- direct (rates match) ----------------------------------------------
    def _render_direct(self, out: np.ndarray, frames: int) -> None:
        pos_i = int(self._pos)
        remaining = self._n - pos_i
        if remaining <= 0:
            self._finished = True
            return
        n = frames if frames < remaining else remaining
        if n > self._max_block:
            n = self._max_block
        work = self._work[:n]
        np.multiply(self._take[pos_i:pos_i + n], self._gain32, out=work)
        tgt = out if (out.ndim == 1 and out.shape[0] == n) else out[:n]
        tgt += work
        self._pos = float(pos_i + n)
        if self._pos >= self._n:
            self._finished = True

    # -- resampled (rates differ, linear interpolation) --------------------
    def _render_resampled(self, out: np.ndarray, frames: int) -> None:
        # Output samples producible before we run off the interpolatable end
        # (the last sample we can read is take[_n - 1], so the last index we
        # can use as the lower interp point is _n - 2).
        avail = (self._n - 1) - self._pos
        if avail <= 0.0:
            self._finished = True
            return
        n = frames if frames < self._max_block else self._max_block
        # k_max = how many output samples until pos exceeds _n - 1.
        k_max = int(avail / self._ratio) + 1
        if n > k_max:
            n = k_max
        if n <= 0:
            self._finished = True
            return

        ramp = self._ramp[:n]
        pos_buf = self._pos_buf[:n]
        np.multiply(ramp, self._ratio, out=pos_buf)   # ratio * k
        pos_buf += self._pos                            # absolute positions (f64)
        floor_buf = self._floor_buf[:n]
        np.floor(pos_buf, out=floor_buf)
        idx = self._idx[:n]
        np.copyto(idx, floor_buf, casting="unsafe")     # integer lower index
        np.clip(idx, 0, self._n - 2, out=idx)           # tail guard: idx+1 stays valid
        frac64 = self._frac64[:n]
        np.subtract(pos_buf, floor_buf, out=frac64)     # fractional part in [0, 1)
        frac = self._frac32[:n]
        np.copyto(frac, frac64, casting="unsafe")

        g0 = self._g0[:n]
        g1 = self._g1[:n]
        np.take(self._take, idx, out=g0)
        idx += 1
        np.take(self._take, idx, out=g1)

        # work = g0 + frac * (g1 - g0), then * gain.
        work = self._work[:n]
        np.subtract(g1, g0, out=work)
        work *= frac
        work += g0
        work *= self._gain32
        tgt = out if (out.ndim == 1 and out.shape[0] == n) else out[:n]
        tgt += work

        self._pos += self._ratio * n
        if self._pos >= self._n - 1:
            self._finished = True


# ---------------------------------------------------------------------------
# DeckController — the idle/recording/have-take/playing state machine.
# ---------------------------------------------------------------------------
class DeckController:
    """Owns the deck state machine, the captured take, and playback.

    Construction mirrors DroneController so the GUI wires it the same way::

        DeckController(mixer, samplerate, *, engine=self._engine,
                       max_seconds=cfg.deck_max_seconds,
                       scratch_dir=cfg.deck_scratch_dir,
                       on_state_changed=self._on_deck_state_changed)

    ``samplerate`` is the OUTPUT (mixer) rate at construction; the live output
    rate is re-read from the engine at ``play()`` time so a device switch that
    changed it is honoured. ``on_state_changed`` fires on every state
    transition — always on the GUI thread (see the module docstring and
    :meth:`pump`).
    """

    def __init__(self, mixer, samplerate: int, *, engine=None,
                 max_seconds: float = 300.0,
                 scratch_dir: Optional[str] = None,
                 on_state_changed: Optional[Callable[[], None]] = None):
        self._mixer = mixer
        self._samplerate = int(samplerate)
        self._engine = engine
        self._max_seconds = float(max_seconds)
        self._scratch_dir = scratch_dir
        self._on_state_changed = on_state_changed

        self._state = DeckState.IDLE
        self._take: Optional[np.ndarray] = None
        self._take_sr: int = 0
        self._take_truncated: bool = False
        self._playback: Optional[DeckPlaybackSource] = None
        self._last_take_path: Optional[str] = None

    # -- observable state ---------------------------------------------------
    @property
    def state(self) -> DeckState:
        return self._state

    @property
    def take(self) -> Optional[np.ndarray]:
        return self._take

    @property
    def take_samplerate(self) -> int:
        return self._take_sr

    @property
    def take_truncated(self) -> bool:
        return self._take_truncated

    @property
    def take_duration_s(self) -> float:
        if self._take is None or self._take_sr <= 0:
            return 0.0
        return self._take.shape[0] / float(self._take_sr)

    @property
    def last_take_path(self) -> Optional[str]:
        """Path of the most recently persisted scratch take, or None."""
        return self._last_take_path

    # -- internal helpers ---------------------------------------------------
    def _set_state(self, state: DeckState) -> None:
        if state != self._state:
            self._state = state
            cb = self._on_state_changed
            if cb is not None:
                try:
                    cb()
                except Exception:
                    # A misbehaving GUI callback must not wedge the deck.
                    pass

    def _max_block(self) -> int:
        return int(getattr(self._mixer, "max_block", 2048))

    def _current_output_sr(self) -> int:
        eng = self._engine
        if eng is not None:
            sr = int(getattr(eng, "output_samplerate", 0) or 0)
            if sr > 0:
                return sr
        return self._samplerate

    def _unregister_playback(self) -> None:
        """Drop the live playback source from the mixer (no state change).
        Idempotent; never raises."""
        src = self._playback
        self._playback = None
        if src is not None:
            try:
                self._mixer.unregister(src)
            except Exception:
                pass

    # -- recording ----------------------------------------------------------
    def can_record(self) -> bool:
        """Honest pre-click probe: is the input stream open so a recording
        could actually start? Backs the GUI's button-enable; ``start_record``
        is the authoritative post-click result."""
        eng = self._engine
        if eng is None:
            return False
        return bool(getattr(eng, "input_running", False))

    def start_record(self) -> bool:
        """idle/have-take/playing -> recording. Returns False (no state
        change beyond an interrupted playback) if the mic can't be armed, so
        the GUI never shows a false 'recording'."""
        eng = self._engine
        if eng is None or not self.can_record():
            return False
        # Recording preempts playback.
        if self._state == DeckState.PLAYING:
            self._unregister_playback()
        try:
            ok = bool(eng.start_input_recording(self._max_seconds))
        except Exception:
            ok = False
        if not ok:
            # Arm failed despite the probe (input vanished in the race) —
            # settle into a truthful non-recording state.
            self._set_state(DeckState.HAVE_TAKE if self._take is not None
                            else DeckState.IDLE)
            return False
        # Single take: a fresh recording discards the previous one.
        self._take = None
        self._take_sr = 0
        self._take_truncated = False
        self._set_state(DeckState.RECORDING)
        return True

    def _finalize_recording(self) -> None:
        """Pull the take out of the engine and settle into have-take (or idle
        if nothing was captured)."""
        eng = self._engine
        take, sr, truncated = None, 0, False
        if eng is not None:
            try:
                take, sr, truncated = eng.stop_input_recording()
            except Exception:
                take, sr, truncated = None, 0, False
        if take is not None and len(take) > 0:
            self._take = np.ascontiguousarray(
                np.asarray(take, dtype=np.float32).reshape(-1))
            self._take_sr = int(sr) if sr else int(
                getattr(eng, "samplerate", self._samplerate) or self._samplerate)
            self._take_truncated = bool(truncated)
            self._persist_scratch()
            self._set_state(DeckState.HAVE_TAKE)
        else:
            self._take = None
            self._take_sr = 0
            self._take_truncated = False
            self._set_state(DeckState.IDLE)

    # -- playback -----------------------------------------------------------
    def play(self) -> bool:
        """have-take/playing -> playing. Returns False with no state change if
        there is no take or we're mid-recording. Re-playing while already
        playing restarts the take from the beginning (the GUI 'replay')."""
        if self._take is None or self._state == DeckState.RECORDING:
            return False
        if self._state == DeckState.PLAYING:
            self._unregister_playback()      # restart from 0
        src = DeckPlaybackSource(self._take, self._take_sr,
                                 self._current_output_sr(), self._max_block())
        if src.finished:                     # degenerate (empty) take guard
            self._set_state(DeckState.HAVE_TAKE)
            return False
        self._playback = src
        self._mixer.register(src)
        self._set_state(DeckState.PLAYING)
        return True

    # -- context-sensitive stop --------------------------------------------
    def stop(self) -> None:
        """recording -> have-take (finalise) | playing -> have-take (drop the
        source). No-op in idle/have-take."""
        if self._state == DeckState.RECORDING:
            self._finalize_recording()
        elif self._state == DeckState.PLAYING:
            self._unregister_playback()
            self._set_state(DeckState.HAVE_TAKE)

    # -- GUI-thread heartbeat ----------------------------------------------
    def pump(self) -> None:
        """Call once per GUI repaint/poll tick. The ONLY place the
        audio-thread-driven transitions land, so ``on_state_changed`` always
        fires on the GUI thread:

        * playing: when the playback source has run out (``finished``), retire
          it and fall back to have-take.
        * recording: when the engine has auto-disarmed (the take hit
          ``deck_max_seconds``, or the input stream went away), finalise the
          take.

        Cheap no-op in idle/have-take.
        """
        st = self._state
        if st == DeckState.PLAYING:
            src = self._playback
            if src is None or src.finished:
                self._unregister_playback()
                self._set_state(DeckState.HAVE_TAKE if self._take is not None
                                else DeckState.IDLE)
        elif st == DeckState.RECORDING:
            eng = self._engine
            if eng is None:
                return
            try:
                armed = bool(eng.is_input_recording())
            except Exception:
                armed = True   # can't tell -> assume still recording, don't drop audio
            if not armed:
                self._finalize_recording()

    # -- export / persistence ----------------------------------------------
    def export(self, path: str) -> bool:
        """Write the current take to ``path`` as a 16-bit-PCM WAV. The GUI
        owns the file dialog and hands us the path. Returns success; never
        raises."""
        if self._take is None or self._take_sr <= 0:
            return False
        try:
            write_wav(path, self._take, self._take_sr)
            return True
        except Exception:
            return False

    def load_take(self, path: str) -> bool:
        """Load a take from a WAV (e.g. restore last session's take on
        startup) and settle into have-take. Returns success; never raises."""
        try:
            samples, sr = read_wav(path)
        except Exception:
            return False
        if samples is None or len(samples) == 0:
            return False
        self._take = np.ascontiguousarray(samples)
        self._take_sr = int(sr)
        self._take_truncated = False
        self._set_state(DeckState.HAVE_TAKE)
        return True

    def _persist_scratch(self) -> None:
        """Best-effort: write the just-finalised take to
        ``scratch_dir/last_take.wav`` so it can be reloaded next session.
        Silent on any failure — scratch persistence never blocks the deck."""
        if not self._scratch_dir or self._take is None or self._take_sr <= 0:
            return
        try:
            os.makedirs(self._scratch_dir, exist_ok=True)
            path = os.path.join(self._scratch_dir, "last_take.wav")
            write_wav(path, self._take, self._take_sr)
            self._last_take_path = path
        except Exception:
            pass

    # -- teardown -----------------------------------------------------------
    def shutdown(self) -> None:
        """Stop all deck audio activity cleanly. Disarms an in-flight
        recording and drops any playback source from the mixer so nothing is
        orphaned on an abrupt close. Idempotent; never raises. Does NOT close
        the engine's input stream — the engine owns that lifecycle."""
        if self._state == DeckState.RECORDING and self._engine is not None:
            try:
                self._engine.stop_input_recording()
            except Exception:
                pass
        self._unregister_playback()
        self._set_state(DeckState.IDLE)

    # Alias mirroring the GUI's close hooks on other controllers.
    close = shutdown

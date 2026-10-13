"""A fake audio system that speaks the subset of ``sounddevice`` the app uses.

The app touches ``sounddevice`` in one place (the audio engine) through:
``query_devices``, ``query_hostapis``, ``default.device``,
``check_input_settings``, ``check_output_settings``, ``InputStream``,
``OutputStream`` and ``PortAudioError``. ``FakeAudioHost.sounddevice_module()``
returns a module object with exactly that surface.

Nothing here runs a real thread or touches real audio. Streams record the
callback the app registers; the *test* drives them:

* ``pump_input(samples)`` feeds microphone frames to the input callback in
  blocks of the stream's block size.
* ``pump_output(n_blocks)`` / ``pull_output(seconds)`` call the output callback
  and append what the app rendered to a capture buffer.

Failure modes a device can be given (``fail=``):

* ``'busy'``      opening raises a PortAudio "device unavailable" error.
* ``'hang'``      opening blocks until ``release_hangs()`` (or a safety limit),
                  which the app's open timeout turns into a host-API failure.
* ``'open_rate'`` rate probes pass but the real open rejects the rate (a
                  "lying" driver).
"""
from __future__ import annotations

import threading
import types
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .capture import OutputCapture
from .clock import ManualClock

DEFAULT_HOSTAPI = "Fake Host API"


class PortAudioError(Exception):
    """Stand-in for ``sounddevice.PortAudioError``."""


class HarnessError(AssertionError):
    """Raised for harness misuse, so a bad test fails loudly and clearly."""


@dataclass
class FakeDevice:
    name: str
    max_input: int = 0
    max_output: int = 0
    rates: set = field(default_factory=lambda: {44100, 48000})
    default_rate: float = 44100.0
    fail: Optional[str] = None
    hostapi: int = 0
    present: bool = True

    def accepts(self, rate: float) -> bool:
        return int(rate) in self.rates


class FakeStream:
    """One open ``InputStream`` / ``OutputStream``."""

    def __init__(self, host: "FakeAudioHost", kind: str, device: FakeDevice,
                 samplerate: int, blocksize: int, channels: int, callback) -> None:
        self.host = host
        self.kind = kind
        self.device = device
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.channels = int(channels)
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        self.dead = False          # the device was unplugged under us

    @property
    def active(self) -> bool:
        return self.started and not self.stopped and not self.closed and not self.dead

    def start(self) -> None:
        if self.closed:
            raise PortAudioError("Stream is closed")
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        self._raise_if_dead()

    def close(self) -> None:
        self.closed = True
        self.stopped = True
        self._raise_if_dead()

    def abort(self) -> None:
        self.stop()

    def _raise_if_dead(self) -> None:
        # Real PortAudio reports an unplugged device when you stop/close it.
        if self.dead and self.host.raise_on_dead_close:
            raise PortAudioError("Error closing stream: Unanticipated host error "
                                 "[PaErrorCode -9999]: device removed")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<FakeStream {self.kind} {self.device.name!r} {self.samplerate} Hz "
                f"block={self.blocksize} active={self.active}>")


class _Default:
    """``sd.default`` — only ``.device`` is read by the app."""

    def __init__(self, host: "FakeAudioHost") -> None:
        self._host = host

    @property
    def device(self) -> tuple:
        return (self._host._default_index("in"), self._host._default_index("out"))

    @device.setter
    def device(self, value) -> None:  # pragma: no cover - the app never sets it
        pass


class FakeAudioHost:
    """Owns the device list, the open streams and the output capture."""

    OPEN_TIMEOUT_S = 0.25          # engine open timeout while this host is installed
    HANG_LIMIT_S = 3.0             # a 'hang' open gives up on its own after this
    MAX_CAPTURE_SECONDS = 300.0    # oldest output is discarded beyond this

    def __init__(self, clock: Optional[ManualClock] = None,
                 hostapi: str = DEFAULT_HOSTAPI) -> None:
        self.clock = clock
        self.hostapi = hostapi
        self.devices: list[FakeDevice] = []        # every device ever added
        self.streams: list[FakeStream] = []        # every stream ever opened
        self.open_attempts: list[tuple] = []       # (kind, device, rate, blocksize, outcome)
        self.raise_on_dead_close = True
        self.auto_pull = True                      # pull output as the clock advances
        self.dropped_input_frames = 0
        self._default_in: Optional[str] = None
        self._default_out: Optional[str] = None
        self._hang_gate = threading.Event()
        self._hung: list[threading.Thread] = []
        self._in_pending = np.zeros(0, dtype=np.float32)
        self._in_pending_stream: Optional[FakeStream] = None
        self._out_credit = 0.0
        self._out_segments: list[dict] = []
        self._module: Optional[types.ModuleType] = None
        if clock is not None:
            clock.subscribe(self._on_clock)

    # ------------------------------------------------------------------ topology
    def add_input(self, name: str, rates: Iterable[int] = (44100, 48000), *,
                  fail: Optional[str] = None,
                  default_rate: Optional[float] = None,
                  channels: int = 1) -> FakeDevice:
        return self._add(name, rates, fail, default_rate, max_input=channels)

    def add_output(self, name: str, rates: Iterable[int] = (44100, 48000), *,
                   fail: Optional[str] = None,
                   default_rate: Optional[float] = None,
                   channels: int = 2) -> FakeDevice:
        return self._add(name, rates, fail, default_rate, max_output=channels)

    def add_duplex(self, name: str, rates: Iterable[int] = (44100, 48000), *,
                   fail: Optional[str] = None,
                   default_rate: Optional[float] = None) -> FakeDevice:
        return self._add(name, rates, fail, default_rate, max_input=1, max_output=2)

    def _add(self, name, rates, fail, default_rate, max_input=0, max_output=0):
        if self._find(name, present_only=False) is not None:
            raise HarnessError(f"device {name!r} already exists")
        # The first device added for a direction becomes its default.
        rates = sorted({int(r) for r in rates})
        dev = FakeDevice(name=name, max_input=max_input, max_output=max_output,
                         rates=set(rates), fail=fail,
                         default_rate=float(default_rate or (rates[0] if rates else 44100)))
        self.devices.append(dev)
        if max_input and self._default_in is None:
            self._default_in = name
        if max_output and self._default_out is None:
            self._default_out = name
        return dev

    def device(self, name: str) -> FakeDevice:
        dev = self._find(name, present_only=False)
        if dev is None:
            raise HarnessError(f"no such device {name!r}")
        return dev

    def set_default(self, input: Optional[str] = None, output: Optional[str] = None) -> None:
        if input is not None:
            self._default_in = input
        if output is not None:
            self._default_out = output

    def set_rates(self, name: str, rates: Iterable[int]) -> None:
        """Change what a device accepts from now on (a settings change in the OS)."""
        self.device(name).rates = {int(r) for r in rates}

    def set_fail(self, name: str, mode: Optional[str]) -> None:
        self.device(name).fail = mode

    def remove_device(self, name: str) -> None:
        """Forget a device entirely (it can no longer be plugged back in)."""
        self.unplug(name)
        self.devices.remove(self.device(name))

    def unplug(self, name: str) -> None:
        """Remove a device. Open streams on it die; the app notices on its next
        device poll (advance the clock by a second)."""
        dev = self.device(name)
        dev.present = False
        for s in self.streams:
            if s.device is dev and not s.closed:
                s.dead = True

    def plug(self, name: str) -> None:
        self.device(name).present = True

    def _find(self, name: str, present_only: bool = True) -> Optional[FakeDevice]:
        for d in self.devices:
            if d.name == name and (d.present or not present_only):
                return d
        return None

    def _present(self) -> list[FakeDevice]:
        return [d for d in self.devices if d.present]

    def _index_of(self, dev: FakeDevice) -> int:
        return self._present().index(dev)

    def _default_index(self, direction: str) -> int:
        name = self._default_in if direction == "in" else self._default_out
        if name is None:
            return -1
        dev = self._find(name)
        if dev is None:
            return -1
        has = dev.max_input if direction == "in" else dev.max_output
        return self._index_of(dev) if has else -1

    def _resolve(self, device, direction: str) -> FakeDevice:
        present = self._present()
        if device is None or device == -1:
            idx = self._default_index(direction)
            if idx < 0:
                raise PortAudioError("Error querying device -1")
            return present[idx]
        try:
            return present[int(device)]
        except (IndexError, ValueError, TypeError):
            raise PortAudioError(f"Error querying device {device}") from None

    # ------------------------------------------------------------- sd surface
    def sounddevice_module(self) -> types.ModuleType:
        if self._module is not None:
            return self._module
        m = types.ModuleType("sounddevice")
        host = self
        m.PortAudioError = PortAudioError
        m.default = _Default(host)
        m.query_devices = host._query_devices
        m.query_hostapis = host._query_hostapis
        m.check_input_settings = host._check_input_settings
        m.check_output_settings = host._check_output_settings

        def InputStream(samplerate=None, blocksize=0, channels=1, dtype="float32",
                        callback=None, device=None, **_kw):
            return host._open("input", device, samplerate, blocksize, channels, callback)

        def OutputStream(samplerate=None, blocksize=0, channels=1, dtype="float32",
                         callback=None, device=None, **_kw):
            return host._open("output", device, samplerate, blocksize, channels, callback)

        m.InputStream = InputStream
        m.OutputStream = OutputStream
        self._module = m
        return m

    def _info(self, dev: FakeDevice) -> dict:
        return {"name": dev.name, "index": self._index_of(dev), "hostapi": dev.hostapi,
                "max_input_channels": dev.max_input, "max_output_channels": dev.max_output,
                "default_samplerate": dev.default_rate,
                "default_low_input_latency": 0.01, "default_low_output_latency": 0.01,
                "default_high_input_latency": 0.1, "default_high_output_latency": 0.1}

    def _query_devices(self, device=None, kind=None):
        present = self._present()
        if device is None:
            return [self._info(d) for d in present]
        try:
            return self._info(present[int(device)])
        except (IndexError, ValueError, TypeError):
            raise PortAudioError(f"Error querying device {device}") from None

    def _query_hostapis(self, index=None):
        apis = [{"name": self.hostapi, "devices": list(range(len(self._present()))),
                 "default_input_device": self._default_index("in"),
                 "default_output_device": self._default_index("out")}]
        return apis if index is None else apis[index]

    def _check(self, direction: str, device, samplerate) -> None:
        dev = self._resolve(device, direction)
        if (dev.max_input if direction == "in" else dev.max_output) < 1:
            raise PortAudioError("Invalid number of channels [PaErrorCode -9998]")
        if samplerate is not None and not dev.accepts(samplerate):
            raise PortAudioError("Invalid sample rate [PaErrorCode -9997]")

    def _check_input_settings(self, device=None, channels=None, dtype=None,
                              samplerate=None, **_kw):
        self._check("in", device, samplerate)

    def _check_output_settings(self, device=None, channels=None, dtype=None,
                               samplerate=None, **_kw):
        self._check("out", device, samplerate)

    # ----------------------------------------------------------------- streams
    def _open(self, kind: str, device, samplerate, blocksize, channels, callback) -> FakeStream:
        direction = "in" if kind == "input" else "out"
        dev = self._resolve(device, direction)
        what = f"Error opening {kind.capitalize()}Stream"
        outcome = "ok"
        try:
            if dev.fail == "hang":
                self._hung.append(threading.current_thread())
                self._hang_gate.wait(timeout=self.HANG_LIMIT_S)
            if dev.fail == "busy":
                outcome = "busy"
                raise PortAudioError(f"{what}: Device unavailable [PaErrorCode -9985]")
            if not dev.present:
                outcome = "gone"
                raise PortAudioError(f"{what}: Invalid device [PaErrorCode -9996]")
            if not dev.accepts(samplerate) or dev.fail == "open_rate":
                outcome = "bad-rate"
                raise PortAudioError(f"{what}: Invalid sample rate [PaErrorCode -9997]")
            stream = FakeStream(self, kind, dev, samplerate, blocksize, channels, callback)
            self.streams.append(stream)
            return stream
        finally:
            self.open_attempts.append(
                (kind, dev.name, int(samplerate), int(blocksize), outcome))

    @property
    def input_stream(self) -> Optional[FakeStream]:
        for s in reversed(self.streams):
            if s.kind == "input" and s.active:
                return s
        return None

    @property
    def output_stream(self) -> Optional[FakeStream]:
        for s in reversed(self.streams):
            if s.kind == "output" and s.active:
                return s
        return None

    def open_streams(self) -> list[FakeStream]:
        """Streams that were opened and never closed (a leak if the app is gone)."""
        return [s for s in self.streams if not s.closed]

    def release_hangs(self) -> None:
        """Let every open that is currently blocked on a 'hang' device finish.
        Opens started afterwards block again."""
        gate, self._hang_gate = self._hang_gate, threading.Event()
        gate.set()
        # Let the released opens run to completion so a test sees their final
        # state (the app disposes of a stream that finishes after it gave up).
        threads, self._hung = self._hung, []
        for t in threads:
            if t is not threading.current_thread():
                t.join(timeout=1.0)

    # --------------------------------------------------------------- input pump
    def pump_input(self, samples, status=None) -> int:
        """Deliver microphone ``samples`` to the active input stream in blocks of
        its block size. A trailing partial block is held for the next call.
        Returns the number of frames handed to the app (0 with no live stream:
        the audio is lost, as it would be)."""
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        stream = self.input_stream
        if stream is None:
            self.dropped_input_frames += arr.shape[0]
            self._in_pending = np.zeros(0, dtype=np.float32)
            return 0
        if self._in_pending_stream is not stream:
            self._in_pending = np.zeros(0, dtype=np.float32)
            self._in_pending_stream = stream
        buf = np.concatenate([self._in_pending, arr]) if self._in_pending.size else arr
        bs = stream.blocksize
        n_blocks = buf.shape[0] // bs
        for i in range(n_blocks):
            block = np.ascontiguousarray(buf[i * bs:(i + 1) * bs]).reshape(-1, 1)
            stream.callback(block, bs, None, status)
        self._in_pending = buf[n_blocks * bs:].copy()
        return n_blocks * bs

    # -------------------------------------------------------------- output pump
    def pump_output(self, n_blocks: int = 1) -> int:
        """Call the output callback ``n_blocks`` times and capture the result."""
        stream = self.output_stream
        if stream is None:
            return 0
        seg = self._segment_for(stream)
        for _ in range(int(n_blocks)):
            out = np.zeros((stream.blocksize, stream.channels), dtype=np.float32)
            stream.callback(out, stream.blocksize, None, None)
            seg["chunks"].append(out[:, 0].copy())
            seg["frames"] += stream.blocksize
        self._trim(seg)
        return int(n_blocks)

    def pull_output(self, seconds: float) -> int:
        """Pump as many whole output blocks as ``seconds`` of real time would
        have consumed; the fraction is carried to the next call."""
        stream = self.output_stream
        if stream is None:
            self._out_credit = 0.0
            return 0
        self._out_credit += seconds * stream.samplerate
        # 1e-6 frame of slack: 0.05 s * 48000 is not exactly 2400 in floating point.
        n = int((self._out_credit + 1e-6) // stream.blocksize)
        self._out_credit -= n * stream.blocksize
        return self.pump_output(n) if n else 0

    def _on_clock(self, dt: float) -> None:
        if self.auto_pull:
            self.pull_output(dt)

    def _segment_for(self, stream: FakeStream) -> dict:
        if not self._out_segments or self._out_segments[-1]["stream"] is not stream:
            self._out_segments.append({"stream": stream, "sr": stream.samplerate,
                                       "chunks": [], "frames": 0})
        return self._out_segments[-1]

    def _trim(self, seg: dict) -> None:
        limit = int(self.MAX_CAPTURE_SECONDS * seg["sr"])
        while seg["frames"] > limit and len(seg["chunks"]) > 1:
            seg["frames"] -= seg["chunks"].pop(0).shape[0]

    def output_segments(self) -> list[OutputCapture]:
        """One capture per output stream that has produced audio, oldest first.
        A device switch or a rate change starts a new segment."""
        segs = []
        for s in self._out_segments:
            data = np.concatenate(s["chunks"]) if s["chunks"] else np.zeros(0, np.float32)
            segs.append(OutputCapture(data, s["sr"]))
        return segs

    def output_capture(self, segment: int = -1) -> OutputCapture:
        """The capture for the newest (default) output stream. Empty when nothing
        has been pulled yet."""
        segs = self.output_segments()
        if not segs:
            stream = self.output_stream
            return OutputCapture(np.zeros(0, np.float32), stream.samplerate if stream else 48000)
        return segs[segment]

    def clear_output(self) -> None:
        self._out_segments.clear()
        self._out_credit = 0.0

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        """Release hung opens and drop clock hooks. Safe to call more than once."""
        self.release_hangs()
        if self.clock is not None:
            self.clock.unsubscribe(self._on_clock)

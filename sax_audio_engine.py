"""
Audio engine for the Intonation Analyzer.

Lifted out of sax_intonation_gui.py in v0.5.4 because the engine had
grown a state machine, a host-API fallback chain, a hot-plug poller,
and a thread-safety contract — none of which belong inside a window
constructor. Splitting the file makes the engine testable in isolation
and removes the temptation to reach into Qt widgets from the audio
callback.

Public surface:

* ``AudioEngine`` — pitch detection + filtering + lifecycle.
* ``AudioEngineState`` — INIT, ENUMERATING, OPENING, RUNNING, FAILED, STOPPED.
* ``AudioEngineError`` — NO_DEVICE, DEVICE_BUSY, DEVICE_DISCONNECTED,
  UNSUPPORTED_RATE, HOSTAPI_FAILURE, UNKNOWN.
* ``DeviceInfo`` / ``DeviceSelection`` — dataclasses for picker UI and
  persistence.
* ``AudioSignals`` — Qt signals: ``state_changed``,
  ``devices_changed``, ``note_detected``.

The engine *never* raises across its public API. Failures become state:
``self.state = FAILED`` plus ``self.last_error`` + ``self.last_error_message``,
emitted via ``signals.state_changed``. The GUI listens and renders a
banner.

Threading contract:

* Audio callback runs on a PortAudio worker thread. It briefly acquires
  ``self._lock`` to mutate filter state and the shared ``_buf``, then
  releases. YIN and FFT compute happen *outside* the lock.
* GUI thread reads via ``get_buf_snapshot()`` and ``get_diagnostics()``
  which both acquire the lock briefly and return immutable snapshots.
* The hot-plug poll runs on the Qt main thread; ``refresh_devices()``
  does not call into PortAudio from inside the audio callback.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    sd = None  # type: ignore
    AUDIO_OK = False

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    QT_OK = True
except Exception:
    QT_OK = False


# ---------------------------------------------------------------------------
# Constants — single source of truth shared with the GUI module.
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 44100
HOP_MS = 1000.0 * 2048.0 / 44100.0       # ~46 ms; preserved across rates
DEFAULT_HOP_SIZE = 2048
DEFAULT_BLOCK_SIZE = 16384
MIN_FREQ = 27.0
MAX_FREQ = 1400.0
YIN_THRESHOLD = 0.12
A4_DEFAULT = 440.0

# Filter-mode presets. Each callback fires every HOP_MS (~46 ms).
#   window     — recent valid detections kept for confirmation/median
#   confirm    — required matching-MIDI detections in the window
#   yin_thr    — YIN aperiodicity ceiling (lower = stricter)
#   rms_floor  — RMS gate below which the frame is silence
#   edge_hops  — attack/release transient guard (in hops)
FILTER_PRESETS = {
    'fast':   dict(window=2, confirm=2, yin_thr=0.16, rms_floor=8e-5,  edge_hops=1),
    'normal': dict(window=4, confirm=3, yin_thr=0.11, rms_floor=1.5e-4, edge_hops=1),
    'slow':   dict(window=7, confirm=5, yin_thr=0.08, rms_floor=3e-4,  edge_hops=2),
}
FILTER_MODE_DEFAULT = 'normal'

# Per-host-API attempt timeout. Legolas measured stream open at 30–400 ms
# typical; 800 ms is the upper end we'll wait before declaring the host
# API hung and moving on.
HOST_API_OPEN_TIMEOUT_S = 0.8

# Windows host-API preference order. WASAPI first because it's the
# modern shared-mode path that coexists with other apps. WDM-KS last,
# and only if the user explicitly opted in — see Aragorn memo on GLE 0xAA.
WIN_HOST_API_ORDER = ('Windows WASAPI', 'MME', 'Windows DirectSound')
WIN_HOST_API_ORDER_WITH_KS = WIN_HOST_API_ORDER + ('Windows WDM-KS',)

# Vendor regex for ranking external interfaces in the picker.
VENDOR_REGEX = (
    r'focusrite|scarlett|motu|apollo|universal audio|behringer|umc|'
    r'audient|evo|presonus|rme|babyface|steinberg|ur\d|tascam|zoom|'
    r'm-audio|m audio'
)


# ---------------------------------------------------------------------------
# State + error enums
# ---------------------------------------------------------------------------
class AudioEngineState(Enum):
    INIT = 'init'
    ENUMERATING = 'enumerating'
    OPENING = 'opening'
    RUNNING = 'running'
    FAILED = 'failed'
    STOPPED = 'stopped'


class AudioEngineError(Enum):
    NONE = 'none'
    NO_DEVICE = 'no_device'
    DEVICE_BUSY = 'device_busy'
    DEVICE_DISCONNECTED = 'device_disconnected'
    UNSUPPORTED_RATE = 'unsupported_rate'
    HOSTAPI_FAILURE = 'hostapi_failure'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class DeviceInfo:
    """One row in the device picker. Persistence uses (name, host_api)."""
    index: int                # PortAudio index — can shift, do not persist
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float


@dataclass(frozen=True)
class DeviceSelection:
    """Persistence-friendly identifier for an audio input device.

    ``samplerate=0`` means auto-negotiate (try 44100 → device default →
    a fallback list). ``host_api`` may be empty if the saved selection
    pre-dates the picker; resolution falls back to a system default in
    that case.
    """
    name: str = ''
    host_api: str = ''
    samplerate: int = 0


@dataclass
class AudioEngineDiagnostics:
    """Snapshot of the engine's last_* scalars, taken atomically under
    the engine lock and returned to the GUI thread for display."""
    rms_db: float = -120.0
    aperiodicity: float = 1.0
    freq: float = 0.0
    locked_midi: Optional[int] = None
    samplerate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    hop_size: int = DEFAULT_HOP_SIZE
    overflow_count: int = 0
    underflow_count: int = 0
    device_name: str = ''
    host_api: str = ''


# ---------------------------------------------------------------------------
# Qt signals — wrapped so the engine remains importable without PyQt6.
# ---------------------------------------------------------------------------
if QT_OK:
    class AudioSignals(QObject):
        # state, error_kind, message
        state_changed = pyqtSignal(object, object, str)
        # list[DeviceInfo]
        devices_changed = pyqtSignal(list)
        # midi, freq, cents
        note_detected = pyqtSignal(int, float, float)
        # hot-plug toast: a vendor-class interface just appeared
        interface_appeared = pyqtSignal(object)  # DeviceInfo
else:  # pragma: no cover — only hit in pure unit-test contexts
    class AudioSignals:
        pass


# ---------------------------------------------------------------------------
# Music helpers
# ---------------------------------------------------------------------------
def freq_to_midi(f: float, a4: float = A4_DEFAULT) -> float:
    return 69.0 + 12.0 * math.log2(f / a4)


def cents_dev(f: float, a4: float = A4_DEFAULT) -> tuple[int, float]:
    mf = freq_to_midi(f, a4)
    mr = round(mf)
    return mr, (mf - mr) * 100.0


def yin_pitch(sig: np.ndarray, sr: int,
              fmin: float = MIN_FREQ, fmax: float = MAX_FREQ,
              thr: float = YIN_THRESHOLD) -> tuple[float, float]:
    """Sample-rate-agnostic YIN. Operates in lag space, returns (Hz, ap)."""
    N = len(sig)
    tmin = max(1, int(sr / fmax))
    tmax = min(N // 2, int(sr / fmin))
    if tmax <= tmin:
        return 0.0, 1.0
    diff = np.array(
        [np.dot(d := sig[:N - t] - sig[t:N], d) for t in range(tmax + 1)])
    cmnd = np.ones(tmax + 1)
    run = 0.0
    for t in range(1, tmax + 1):
        run += diff[t]
        cmnd[t] = diff[t] * t / run if run > 0 else 1.0
    tau, mv = -1, 1.0
    for t in range(tmin, tmax):
        if cmnd[t] < thr:
            while t + 1 < tmax and cmnd[t + 1] < cmnd[t]:
                t += 1
            tau, mv = t, cmnd[t]
            break
    if tau == -1:
        tau = tmin + int(np.argmin(cmnd[tmin:tmax]))
        mv = cmnd[tau]
    if 1 < tau < tmax - 1:
        s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        d = 2 * s1 - s0 - s2
        if d:
            tau += 0.5 * (s0 - s2) / d
    return (sr / tau if tau > 0 else 0.0), mv


# ---------------------------------------------------------------------------
# Device enumeration helpers
# ---------------------------------------------------------------------------
def _host_api_name(idx: int) -> str:
    if not AUDIO_OK:
        return ''
    try:
        apis = sd.query_hostapis()
        if 0 <= idx < len(apis):
            return str(apis[idx].get('name', ''))
    except Exception:
        pass
    return ''


def query_input_devices() -> list[DeviceInfo]:
    """Return all input-capable devices. Empty list on any failure —
    callers must not assume PortAudio is healthy.
    """
    if not AUDIO_OK:
        return []
    out: list[DeviceInfo] = []
    try:
        devs = sd.query_devices()
    except Exception:
        return out
    for i, d in enumerate(devs):
        try:
            ch = int(d.get('max_input_channels', 0))
            if ch < 1:
                continue
            out.append(DeviceInfo(
                index=i,
                name=str(d.get('name', f'device #{i}')),
                host_api=_host_api_name(int(d.get('hostapi', -1))),
                max_input_channels=ch,
                default_samplerate=float(d.get('default_samplerate', 44100.0)),
            ))
        except Exception:
            continue
    return out


def _probe_default_input_index() -> Optional[int]:
    """Return the system default input device index, or None.

    Guards against the ``Error querying device -1`` crash documented in
    Aragorn's memo. Never raises.
    """
    if not AUDIO_OK:
        return None
    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        info = sd.query_devices(idx)
        if int(info.get('max_input_channels', 0)) < 1:
            return None
        return int(idx)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AudioEngine
# ---------------------------------------------------------------------------
class AudioEngine:
    """Pitch detection + filtering + device lifecycle.

    Failure is a state, not an exception. ``start()`` and
    ``open_device(spec)`` never raise; check ``self.state`` and
    ``self.last_error`` afterward (or subscribe to
    ``signals.state_changed``).
    """

    def __init__(self):
        self.signals = AudioSignals() if QT_OK else None
        self.state: AudioEngineState = AudioEngineState.INIT
        self.last_error: AudioEngineError = AudioEngineError.NONE
        self.last_error_message: str = ''

        self.a4 = A4_DEFAULT
        self.instr_key = 'eb_alto'
        self.filter_mode = FILTER_MODE_DEFAULT
        self.prefer_wdmks = False

        # Active stream parameters, set on successful open.
        self.samplerate = DEFAULT_SAMPLE_RATE
        self.block_size = DEFAULT_BLOCK_SIZE
        self.hop_size = DEFAULT_HOP_SIZE
        self.active_device: Optional[DeviceInfo] = None

        # Diagnostic scalars. Read snapshot via get_diagnostics().
        self.last_rms_db: float = -120.0
        self.last_ap: float = 1.0
        self.last_freq: float = 0.0
        self.last_locked_midi: Optional[int] = None
        self._overflow_count: int = 0
        self._underflow_count: int = 0

        # Shared mutable state — protected by self._lock.
        self._lock = threading.Lock()
        self._buf = np.zeros(DEFAULT_BLOCK_SIZE, dtype=np.float32)
        self._stream = None
        self._emissions_paused = False
        self._transitioning = False
        self._reset_filter_state()

        # Hot-plug snapshot.
        self._last_device_snapshot: tuple = ()
        # Range gate: see SAX_MIDI from the GUI module. Mirrored here so
        # the engine can drop MIDI values outside the supported range
        # without round-tripping to Qt.
        self._midi_min = 21
        self._midi_max = 91

    # ---- internal state helpers -------------------------------------------
    def _reset_filter_state(self) -> None:
        self._recent: list[tuple[int, float]] = []
        self._locked_midi: Optional[int] = None
        self._hop_in_note: int = 0
        self._pending: list[tuple[int, float, float]] = []

    def _set_state(self, state: AudioEngineState,
                   err: AudioEngineError = AudioEngineError.NONE,
                   msg: str = '') -> None:
        self.state = state
        self.last_error = err
        self.last_error_message = msg
        if self.signals is not None:
            try:
                self.signals.state_changed.emit(state, err, msg)
            except Exception:
                pass

    # ---- public API (never raises) ----------------------------------------
    def set_filter_mode(self, mode: str) -> None:
        if mode not in FILTER_PRESETS:
            return
        with self._lock:
            self.filter_mode = mode
            self._reset_filter_state()

    def set_a4(self, hz: float) -> None:
        with self._lock:
            self.a4 = float(hz)

    def set_prefer_wdmks(self, flag: bool) -> None:
        self.prefer_wdmks = bool(flag)

    def pause_emissions(self) -> None:
        """Drop incoming detections without stopping the stream.

        Used by the GUI during A4 remapping so the callback can't slip a
        measurement into the half-rebuilt ``self.stats``.
        """
        with self._lock:
            self._emissions_paused = True
            self._reset_filter_state()

    def resume_emissions(self) -> None:
        with self._lock:
            self._emissions_paused = False
            self._reset_filter_state()

    def get_buf_snapshot(self) -> np.ndarray:
        """Return a copy of the current ring buffer. Safe for the GUI
        thread to FFT/plot without racing the audio callback."""
        with self._lock:
            return self._buf.copy()

    def get_diagnostics(self) -> AudioEngineDiagnostics:
        with self._lock:
            return AudioEngineDiagnostics(
                rms_db=self.last_rms_db,
                aperiodicity=self.last_ap,
                freq=self.last_freq,
                locked_midi=self.last_locked_midi,
                samplerate=int(self.samplerate),
                block_size=int(self.block_size),
                hop_size=int(self.hop_size),
                overflow_count=self._overflow_count,
                underflow_count=self._underflow_count,
                device_name=self.active_device.name if self.active_device else '',
                host_api=self.active_device.host_api if self.active_device else '',
            )

    def start(self, preferred: Optional[DeviceSelection] = None) -> None:
        """Enumerate, then open the preferred device (or system default).
        Never raises. Sets state + emits signals on completion or failure.
        """
        if not AUDIO_OK:
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.NO_DEVICE,
                            'sounddevice / PortAudio not available')
            return
        if self._transitioning:
            return
        self._transitioning = True
        try:
            self._set_state(AudioEngineState.ENUMERATING)
            devices = query_input_devices()
            if not devices:
                self._set_state(AudioEngineState.FAILED,
                                AudioEngineError.NO_DEVICE,
                                'No audio input devices detected')
                self._last_device_snapshot = ()
                if self.signals is not None:
                    try:
                        self.signals.devices_changed.emit(devices)
                    except Exception:
                        pass
                return
            self._last_device_snapshot = self._snapshot_key(devices)
            if self.signals is not None:
                try:
                    self.signals.devices_changed.emit(devices)
                except Exception:
                    pass
            self._open_with_fallback(preferred, devices)
        finally:
            self._transitioning = False

    def open_device(self, sel: DeviceSelection) -> None:
        """Switch to a user-picked device. Stops the current stream first.
        Never raises.
        """
        if not AUDIO_OK:
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.NO_DEVICE,
                            'sounddevice / PortAudio not available')
            return
        if self._transitioning:
            return
        self._transitioning = True
        try:
            self._teardown_stream()
            devices = query_input_devices()
            if not devices:
                self._set_state(AudioEngineState.FAILED,
                                AudioEngineError.NO_DEVICE,
                                'No audio input devices detected')
                return
            self._open_with_fallback(sel, devices)
        finally:
            self._transitioning = False

    def retry(self) -> None:
        """Re-enumerate and re-open. Tries the last selection first."""
        prev = None
        if self.active_device is not None:
            prev = DeviceSelection(
                name=self.active_device.name,
                host_api=self.active_device.host_api,
                samplerate=int(self.samplerate) if self.samplerate else 0,
            )
        self.start(preferred=prev)

    def stop(self) -> None:
        """Tear down the stream cleanly. Never raises."""
        self._teardown_stream()
        self._set_state(AudioEngineState.STOPPED)

    def refresh_devices(self) -> list[DeviceInfo]:
        """Poll the device list. Emits ``devices_changed`` on a diff.

        If the active device vanished while RUNNING, transitions to
        FAILED(DEVICE_DISCONNECTED). If we're in FAILED(NO_DEVICE) and a
        device just appeared, the GUI's responsibility (we don't auto-
        retry to avoid step-on-toes with the user's banner click).

        Hot-plug toast for vendor-class interfaces is emitted via
        ``interface_appeared`` when applicable.
        """
        if not AUDIO_OK:
            return []
        if self._transitioning:
            # Don't race the open path.
            return []
        devices = query_input_devices()
        key = self._snapshot_key(devices)
        if key == self._last_device_snapshot:
            return devices
        prev_names = {n for (n, _h, _c) in self._last_device_snapshot}
        new_names = {n for (n, _h, _c) in key}
        appeared = new_names - prev_names
        self._last_device_snapshot = key
        if self.signals is not None:
            try:
                self.signals.devices_changed.emit(devices)
            except Exception:
                pass
        # Active device vanished?
        if (self.state == AudioEngineState.RUNNING
                and self.active_device is not None
                and not any(d.name == self.active_device.name
                            and d.host_api == self.active_device.host_api
                            for d in devices)):
            self._teardown_stream()
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.DEVICE_DISCONNECTED,
                            f'Device disconnected: {self.active_device.name}')
        # Vendor-class device appearing? Emit toast.
        if appeared and self.signals is not None:
            import re
            vendor = re.compile(VENDOR_REGEX, re.IGNORECASE)
            for d in devices:
                if d.name in appeared and vendor.search(d.name):
                    try:
                        self.signals.interface_appeared.emit(d)
                    except Exception:
                        pass
                    break
        return devices

    # ---- internals --------------------------------------------------------
    @staticmethod
    def _snapshot_key(devices: list[DeviceInfo]) -> tuple:
        return tuple((d.name, d.host_api, d.max_input_channels)
                     for d in devices)

    def _teardown_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._reset_filter_state()
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _resolve_candidates(self, sel: Optional[DeviceSelection],
                            devices: list[DeviceInfo]) -> list[DeviceInfo]:
        """Return an ordered list of devices to attempt opening.

        First, anything matching the saved selection by (name, host_api),
        then by name alone across the platform's preferred host APIs,
        then the system default device across those host APIs.
        """
        out: list[DeviceInfo] = []
        if sel and sel.name:
            # Exact match first.
            for d in devices:
                if (d.name == sel.name
                        and (not sel.host_api or d.host_api == sel.host_api)):
                    out.append(d)
            # Same name, any host API.
            for d in devices:
                if d.name == sel.name and d not in out:
                    out.append(d)
        # System default, walked through the platform's preferred host APIs.
        default_idx = _probe_default_input_index()
        default_name = ''
        if default_idx is not None:
            for d in devices:
                if d.index == default_idx:
                    default_name = d.name
                    break
        if default_name:
            for d in devices:
                if d.name == default_name and d not in out:
                    out.append(d)
        # Anything else as a last resort.
        for d in devices:
            if d not in out:
                out.append(d)
        # Sort each device's host-API variants per Windows preference.
        return self._reorder_by_host_api(out)

    def _reorder_by_host_api(self,
                             devices: list[DeviceInfo]) -> list[DeviceInfo]:
        """On Windows, push WASAPI to the front and WDM-KS to the back
        unless prefer_wdmks is set."""
        if sys.platform != 'win32':
            return devices
        order = (WIN_HOST_API_ORDER_WITH_KS
                 if self.prefer_wdmks else WIN_HOST_API_ORDER)
        def score(d: DeviceInfo) -> int:
            try:
                return order.index(d.host_api)
            except ValueError:
                # Unknown host API — between WASAPI and WDM-KS.
                return len(order)
        # Stable sort preserves the candidate ordering within each host API.
        return sorted(devices, key=score)

    def _open_with_fallback(self, sel: Optional[DeviceSelection],
                            devices: list[DeviceInfo]) -> None:
        """Walk the candidate list × sample-rate list until one opens.

        On success, sets RUNNING. On exhaustion, sets FAILED with the
        most descriptive error we saw.
        """
        self._set_state(AudioEngineState.OPENING)
        candidates = self._resolve_candidates(sel, devices)
        # Sample rates to try, in order. The saved selection's rate (if
        # any) goes first, then the device default, then a generic list.
        sr_seed = int(sel.samplerate) if sel and sel.samplerate else 0
        last_err: AudioEngineError = AudioEngineError.UNKNOWN
        last_msg: str = ''
        for dev in candidates:
            sr_list: list[int] = []
            if sr_seed:
                sr_list.append(sr_seed)
            sr_list.extend([44100, int(dev.default_samplerate or 0),
                            48000, 96000, 88200, 192000, 32000])
            # Dedup while preserving order; drop zeros.
            seen = set()
            sr_clean = []
            for s in sr_list:
                if s and s not in seen:
                    seen.add(s)
                    sr_clean.append(s)
            for sr in sr_clean:
                ok, err_kind, err_msg = self._try_open(dev, sr)
                if ok:
                    return
                last_err, last_msg = err_kind, err_msg
                # UNSUPPORTED_RATE → try the next sample rate; everything
                # else → move on to the next device.
                if err_kind != AudioEngineError.UNSUPPORTED_RATE:
                    break
        self._set_state(AudioEngineState.FAILED, last_err, last_msg)

    def _try_open(self, dev: DeviceInfo,
                  samplerate: int) -> tuple[bool, AudioEngineError, str]:
        """One open attempt. Returns (ok, error_kind, message).

        Wrapped in a worker thread with HOST_API_OPEN_TIMEOUT_S so a
        wedged driver doesn't freeze the GUI.
        """
        # Recompute block/hop so HOP_MS stays ~46 ms at the new rate.
        hop = max(256, int(round(samplerate * HOP_MS / 1000.0)))
        # BLOCK_SIZE = 8 × HOP_SIZE keeps the ~370 ms YIN window.
        block = hop * 8

        result: dict = {'stream': None, 'err_kind': None, 'err_msg': ''}

        cb = self._make_callback(samplerate)

        def worker() -> None:
            try:
                stream = sd.InputStream(
                    samplerate=samplerate,
                    blocksize=hop,
                    channels=1,
                    dtype='float32',
                    callback=cb,
                    device=dev.index,
                )
            except Exception as exc:
                result['err_kind'] = self._classify_error(exc)
                result['err_msg'] = f'{dev.name} [{dev.host_api}]: {exc}'
                return
            try:
                stream.start()
            except Exception as exc:
                try:
                    stream.close()
                except Exception:
                    pass
                result['err_kind'] = self._classify_error(exc)
                result['err_msg'] = f'{dev.name} [{dev.host_api}]: {exc}'
                return
            result['stream'] = stream

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=HOST_API_OPEN_TIMEOUT_S)
        if t.is_alive():
            # Driver hung. The thread leaks; nothing safe to do with
            # PortAudio in this state besides walking past it. Set a
            # synthetic failure and let the chain advance.
            return False, AudioEngineError.HOSTAPI_FAILURE, (
                f'{dev.name} [{dev.host_api}]: open timed out '
                f'after {HOST_API_OPEN_TIMEOUT_S:.1f}s')
        if result['stream'] is None:
            return False, (result['err_kind'] or AudioEngineError.UNKNOWN), \
                   str(result['err_msg'])

        # Success — install the new stream and reset diagnostics.
        with self._lock:
            self._stream = result['stream']
            self.samplerate = int(samplerate)
            self.block_size = int(block)
            self.hop_size = int(hop)
            self._buf = np.zeros(block, dtype=np.float32)
            self._overflow_count = 0
            self._underflow_count = 0
            self._reset_filter_state()
        self.active_device = dev
        self._set_state(AudioEngineState.RUNNING)
        return True, AudioEngineError.NONE, ''

    @staticmethod
    def _classify_error(exc: BaseException) -> AudioEngineError:
        msg = str(exc).lower()
        # PortAudio numeric codes when present.
        code = getattr(exc, 'args', None)
        text = msg
        if isinstance(code, tuple) and code:
            text = ' '.join(str(c).lower() for c in code)
        if 'invalid sample rate' in text or '-9998' in text:
            return AudioEngineError.UNSUPPORTED_RATE
        if any(s in text for s in (
                'device unavailable', 'busy', 'unanticipated host error',
                '0xaa', '-9999', '-9993', '-9994', '-9996',
                'wdmsyncioctl')):
            return AudioEngineError.DEVICE_BUSY
        if any(s in text for s in (
                'device not found', 'no device', 'invalid device',
                'querying device -1', '-1')):
            return AudioEngineError.NO_DEVICE
        return AudioEngineError.HOSTAPI_FAILURE

    # ---- pitch detection callback -----------------------------------------
    def _make_callback(self, samplerate: int):
        """Build a PortAudio callback bound to a specific sample rate.

        The callback acquires self._lock briefly to mutate shared state
        but releases it before YIN runs. YIN itself reads the local
        ``buf`` copy under the lock, so even a concurrent
        ``get_buf_snapshot()`` sees a consistent frame.
        """
        sr = int(samplerate)
        engine = self

        def cb(indata, frames, ti, st):
            try:
                if st is not None:
                    if getattr(st, 'input_overflow', False):
                        engine._overflow_count += 1
                    if getattr(st, 'input_underflow', False):
                        engine._underflow_count += 1
                mono = indata[:, 0]
                # Update the ring buffer + read filter snapshot under
                # the lock. YIN runs outside the lock.
                with engine._lock:
                    if engine._emissions_paused:
                        return
                    if engine._buf is None or engine._buf.size < frames:
                        # Reallocate if a rebind happened mid-callback.
                        engine._buf = np.zeros(max(DEFAULT_BLOCK_SIZE,
                                                    frames * 8),
                                                dtype=np.float32)
                    engine._buf = np.roll(engine._buf, -frames)
                    engine._buf[-frames:] = mono
                    buf = engine._buf
                    mode = engine.filter_mode
                    a4 = engine.a4
                params = FILTER_PRESETS.get(mode, FILTER_PRESETS['normal'])

                rms = math.sqrt(float(np.mean(buf ** 2)))
                engine.last_rms_db = 20.0 * math.log10(max(rms, 1e-9))

                if rms < params['rms_floor']:
                    engine.last_ap = 1.0
                    engine.last_freq = 0.0
                    with engine._lock:
                        engine.last_locked_midi = engine._locked_midi
                        engine._on_silence(params)
                    return
                sig = buf / (rms + 1e-9)
                freq, ap = yin_pitch(sig, sr)
                engine.last_ap = float(ap)
                engine.last_freq = float(freq)
                if ap > params['yin_thr'] or not (MIN_FREQ < freq < MAX_FREQ):
                    with engine._lock:
                        engine.last_locked_midi = engine._locked_midi
                        engine._on_silence(params)
                    return
                mr, ct = cents_dev(freq, a4)
                if not (engine._midi_min <= mr <= engine._midi_max):
                    with engine._lock:
                        engine.last_locked_midi = engine._locked_midi
                        engine._on_silence(params)
                    return

                # Mutate filter state under the lock.
                emit_payload = None
                with engine._lock:
                    engine.last_locked_midi = (
                        engine._locked_midi if engine._locked_midi is not None
                        else int(mr))
                    engine._recent.append((int(mr), float(freq)))
                    if len(engine._recent) > params['window']:
                        engine._recent.pop(0)
                    if not engine._recent:
                        return  # cleared mid-flight by set_filter_mode
                    latest_midi = engine._recent[-1][0]
                    matches = sum(1 for m, _f in engine._recent
                                  if m == latest_midi)
                    if matches < params['confirm']:
                        return
                    if engine._locked_midi != latest_midi:
                        engine._drop_pending_for_edge(params)
                        # Flush survivors before relocking.
                        while engine._pending:
                            mi, fr, ce = engine._pending.pop(0)
                            engine._enqueue_emit(mi, fr, ce)
                        engine._locked_midi = latest_midi
                        engine._hop_in_note = 0
                    engine._hop_in_note += 1
                    if engine._hop_in_note <= params['edge_hops']:
                        return
                    same = [f for m, f in engine._recent
                            if m == latest_midi]
                    if not same:
                        return  # cleared mid-flight
                    median_freq = float(np.median(same))
                    _mr2, median_cents = cents_dev(median_freq, a4)
                    engine._pending.append(
                        (latest_midi, median_freq, median_cents))
                    if len(engine._pending) > params['edge_hops']:
                        emit_payload = engine._pending.pop(0)
                # Emit outside the lock — Qt signal delivery shouldn't
                # hold the audio mutex.
                if emit_payload is not None:
                    engine._emit(*emit_payload)
            except Exception:
                # Audio callback must never raise back into PortAudio.
                pass

        return cb

    def _drop_pending_for_edge(self, params: dict) -> None:
        keep = max(0, len(self._pending) - params['edge_hops'])
        self._pending = self._pending[:keep]

    def _enqueue_emit(self, midi: int, freq: float, cents: float) -> None:
        """Used while flushing survivors after a relock — delegate to the
        signal emit. Called *under* the lock; emit is cheap (queued Qt
        signal delivery), so we accept the brief hold."""
        self._emit(midi, freq, cents)

    def _emit(self, midi: int, freq: float, cents: float) -> None:
        if self._emissions_paused:
            return
        if self.signals is None:
            return
        try:
            self.signals.note_detected.emit(int(midi), float(freq),
                                            float(cents))
        except Exception:
            pass

    def _on_silence(self, params: dict) -> None:
        """End any held note cleanly. Caller must hold self._lock."""
        if self._locked_midi is not None or self._pending or self._recent:
            self._drop_pending_for_edge(params)
            while self._pending:
                mi, fr, ce = self._pending.pop(0)
                self._enqueue_emit(mi, fr, ce)
        self._recent.clear()
        self._locked_midi = None
        self._hop_in_note = 0

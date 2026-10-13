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
  ``devices_changed``, ``note_detected``, ``note_event``.

The engine *never* raises across its public API. Failures become state:
``self.state = FAILED`` plus ``self.last_error`` + ``self.last_error_message``,
emitted via ``signals.state_changed``. The GUI listens and renders a
banner.

Threading contract:

* The input callback runs on a PortAudio thread and only copies: each block
  goes into the stream's single-producer/single-consumer ``_InputRing`` (and
  the deck tap, under ``self._lock`` for microseconds), then wakes the
  analysis thread. It never runs YIN.
* The analysis thread (one per open input stream) is the ring's consumer and
  the sole owner of the ``sax_detector.Detector``. Once per hop it reads the
  detection window, runs the detector and emits ``note_event`` /
  ``note_detected``; Qt queues those to the GUI thread. ``process_pending()``
  is the same consumer step, callable directly: tests and the headless
  harness run with ``ANALYSIS_IN_THREAD = False`` and analyse inline.
* The GUI thread never touches the detector. Mode, A4 and resets reach it as
  values the consumer reads under ``self._lock`` at the start of each hop.
* GUI thread reads via ``get_buf_snapshot()`` and ``get_diagnostics()``
  which both acquire the lock briefly and return immutable snapshots.
* The hot-plug poll runs on the Qt main thread; ``refresh_devices()``
  does not call into PortAudio from inside the audio callback.
"""

from __future__ import annotations

import datetime
import math
import re
import sys
import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from sax_coordination import OutputCoordinator
from sax_detector import (  # noqa: F401  (re-exported)
    FILTER_MODE_DEFAULT,
    FILTER_PRESETS,
    Detector,
    NoteEvent,
)
from sax_mixer import Mixer, TestToneSource
from sax_pitch import A4_DEFAULT, cents_dev, freq_to_midi  # noqa: F401  (re-exported)
from sax_yin import MAX_FREQ, MIN_FREQ, YIN_THRESHOLD, yin_pitch  # noqa: F401

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
BLOCK_MS = 1000.0 * 16384.0 / 44100.0    # ~372 ms; preserved across rates
DEFAULT_HOP_SIZE = 2048
DEFAULT_BLOCK_SIZE = 16384

# Output-stream callback block. Unlike the input path (which needs a long
# ~370 ms YIN window), the output mixer only has to fill the next playback
# block, so it runs at the hop cadence (~46 ms) for low added latency. The
# mixer's accumulator is sized to exactly this so its zero-allocation fast
# path (frames == max_block) stays armed; recomputed per sample rate to hold
# ~46 ms, mirroring the input hop. See OUT_BLOCK_MS.
OUT_BLOCK_MS = HOP_MS                       # ~46 ms; preserved across rates
DEFAULT_OUT_BLOCK = DEFAULT_HOP_SIZE        # 2048 frames at 44.1k

# Candidate sample rates probed in this order when the user picks "auto".
# 48k and 44.1k come first: pitch detection cost grows with the rate (about
# 0.4 ms per hop at 44.1k/48k and 1.4 ms at 192k, against a 46 ms hop) for no
# visible gain, since the display shows whole cents, which 44.1k already
# resolves. Higher rates remain as fallbacks for devices that
# refuse both, and stay available as explicit choices.
SAMPLERATE_CANDIDATES = (48000, 44100, 96000, 88200, 192000)
SAMPLERATE_PREF_VALUES = ('auto', '44100', '48000', '88200', '96000',
                          '192000')
# MIN_FREQ / MAX_FREQ / YIN_THRESHOLD and yin_pitch (sax_yin), and A4_DEFAULT,
# freq_to_midi and cents_dev (sax_pitch), are re-exported above: callers and
# tests import them from the engine.

# FILTER_PRESETS (the fast / normal / slow response modes) and
# FILTER_MODE_DEFAULT live with the state machine they tune, in sax_detector,
# and are re-exported above.

# Where pitch detection runs. True: a dedicated analysis thread per open input
# stream, so the PortAudio callback only copies samples. False: the callback
# analyses inline after copying. The headless test harness sets False, because
# its fake audio host drives callbacks synchronously and its scenarios need
# each reading delivered before the next block is pumped.
ANALYSIS_IN_THREAD = True

# Per-host-API attempt timeout. Stream open measured at 30–400 ms
# typical; 800 ms is the upper end we'll wait before declaring the host
# API hung and moving on.
HOST_API_OPEN_TIMEOUT_S = 0.8

# Windows host-API preference order. WASAPI first because it's the
# modern shared-mode path that coexists with other apps. WDM-KS last,
# and only if the user explicitly opted in (WDM-KS can fail with Windows
# GetLastError 0xAA, ERROR_BUSY).
WIN_HOST_API_ORDER = ('Windows WASAPI', 'MME', 'Windows DirectSound')
WIN_HOST_API_ORDER_WITH_KS = WIN_HOST_API_ORDER + ('Windows WDM-KS',)

# Vendor regex for ranking external interfaces in the picker.
# v0.5.7: extended with brands a user reported missing (FiiO) plus the
# rest of the common pro/prosumer interface vendors. The ``(?! call)``
# lookahead avoids boosting the Windows "Zoom Video Communications"
# device when that's been installed alongside the Zoom Corp recorder.
# v0.6: wrapped in \b boundaries so short tokens (umc, fiio, uad, evo,
# rme) no longer match inside unrelated words ("UMC202HD" was matching
# the umc prefix; "Studio FIIOX" was matching fiio inside fiiox).  The
# `ur\d` token became `ur\d+` so a string like "UR242" matches the full
# digit run rather than just the first digit.
VENDOR_REGEX = (
    r'\b(?:'
    r'focusrite|scarlett|motu|apollo|universal audio|uad|behringer|umc|'
    r'audient|evo|presonus|rme|babyface|steinberg|ur\d+|tascam|'
    r'zoom(?! call)|'
    r'm-audio|m audio|'
    r'fiio|apogee|roland|native instruments|ni|ssl|'
    r'antelope|ik multimedia'
    r')\b'
)
# Compiled form of VENDOR_REGEX — used by refresh_devices() and
# _auto_recover_after_hotplug(). Keep VENDOR_REGEX (string) for any
# caller that constructs its own pattern (e.g. sax_intonation_gui).
VENDOR_RE = re.compile(VENDOR_REGEX, re.IGNORECASE)


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

    ``samplerate=0`` means auto-negotiate (SAMPLERATE_CANDIDATES order:
    48000, 44100, 96000, 88200, 192000). ``host_api`` may be empty if the saved selection
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
        # midi, freq, cents. Kept for listeners that only need the reading.
        note_detected = pyqtSignal(int, float, float)
        # sax_detector.NoteEvent: the reading plus the A4 (and A4 generation)
        # it was graded against.
        note_event = pyqtSignal(object)
        # hot-plug toast: a vendor-class interface just appeared
        interface_appeared = pyqtSignal(object)  # DeviceInfo
else:  # pragma: no cover — only hit in pure unit-test contexts
    class AudioSignals:
        pass


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

    Guards against sounddevice's ``Error querying device -1`` when the system
    has no default input device. Never raises.
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


def query_output_devices() -> list[DeviceInfo]:
    """Return all output-capable devices. Empty list on any failure — the
    mirror of ``query_input_devices`` for the output picker (D5: the input
    picker pattern, generalized). ``max_input_channels`` in the returned
    DeviceInfo carries the OUTPUT channel count for output devices, so the
    same DeviceInfo/DeviceSelection persistence path is reused unchanged.
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
            ch = int(d.get('max_output_channels', 0))
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


def _probe_default_output_index() -> Optional[int]:
    """Return the system default OUTPUT device index, or None. Mirror of
    ``_probe_default_input_index``; never raises."""
    if not AUDIO_OK:
        return None
    try:
        idx = sd.default.device[1]
        if idx is None or idx < 0:
            return None
        info = sd.query_devices(idx)
        if int(info.get('max_output_channels', 0)) < 1:
            return None
        return int(idx)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Input ring: callback -> analysis hand-off
# ---------------------------------------------------------------------------
class _InputRing:
    """Single-producer/single-consumer sample ring for one input stream.

    The producer (the input callback) appends each delivered block and records
    where it ended as a *mark*; the consumer analyses one detection window
    (``block`` samples) ending at each mark, so there is exactly one analysis
    per delivered block, as when detection ran inside the callback.

    Neither side locks. Positions are absolute frame counts that only grow, and
    the producer publishes ``written`` / ``marks_written`` after its copy, so a
    consumer that sees a mark sees the samples before it. The ring holds
    ``SLACK_HOPS`` hops beyond one window: a consumer that falls further behind
    than that has had its window overwritten, which ``read_window`` detects
    after copying and reports as a dropped hop rather than a torn analysis.

    The ring is created per stream and bound into that stream's callback before
    the stream starts, so the callback can never write one stream's samples
    into another stream's ring or analyse them at another stream's rate.
    """

    MARKS = 64
    SLACK_HOPS = 16

    def __init__(self, block: int, hop: int, samplerate: Optional[int] = None,
                 inline: bool = True) -> None:
        self.block = int(block)
        self.hop = int(hop)
        # None: not bound to a stream; the consumer uses the engine's rate.
        self.samplerate = int(samplerate) if samplerate else None
        # True: the producer analyses inline after each push (no thread).
        self.inline = bool(inline)
        self.capacity = self.block + self.SLACK_HOPS * self.hop
        self.data = np.zeros(self.capacity, dtype=np.float32)
        self.written = 0
        self.marks = [0] * self.MARKS
        self.marks_written = 0
        self.marks_read = 0
        self.window = np.zeros(self.block, dtype=np.float32)
        self.dropped_hops = 0

    # ---- producer ---------------------------------------------------------
    def push(self, mono: np.ndarray) -> None:
        n = int(mono.shape[0])
        if n <= 0:
            return
        cap = self.capacity
        w = self.written
        if n > cap:                      # only the newest cap samples can matter
            mono = mono[n - cap:]
            w += n - cap
            n = cap
        s = w % cap
        e = s + n
        if e <= cap:
            self.data[s:e] = mono
        else:
            first = cap - s
            self.data[s:] = mono[:first]
            self.data[:n - first] = mono[first:]
        self.written = w + n
        self.marks[self.marks_written % self.MARKS] = self.written
        self.marks_written += 1

    # ---- consumer ---------------------------------------------------------
    def pop_mark(self) -> Optional[int]:
        """The end position of the oldest unanalysed block, or None."""
        while True:
            mw = self.marks_written
            if self.marks_read >= mw:
                return None
            if mw - self.marks_read > self.MARKS:
                self.dropped_hops += mw - self.MARKS - self.marks_read
                self.marks_read = mw - self.MARKS
            idx = self.marks_read
            end = self.marks[idx % self.MARKS]
            self.marks_read = idx + 1
            # The producer may have lapped this slot while we read it.
            if self.marks_written - idx <= self.MARKS:
                return end
            self.dropped_hops += 1

    def read_window(self, end: int) -> bool:
        """Copy the ``block`` samples ending at ``end`` into ``self.window``.

        Returns False when the producer has overwritten part of that span (the
        copy may be torn); the caller skips the hop. Positions before the first
        sample read as the ring's initial zeros.
        """
        cap, n = self.capacity, self.block
        s = (end - n) % cap
        if s + n <= cap:
            self.window[:] = self.data[s:s + n]
        else:
            first = cap - s
            self.window[:first] = self.data[s:]
            self.window[first:] = self.data[:n - first]
        # One more hop may be mid-write past `written`; it must not reach the
        # oldest sample of the window.
        return self.written - end <= cap - n - self.hop


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
        # 'auto' (probe SAMPLERATE_CANDIDATES in order) or a specific rate
        # string from SAMPLERATE_PREF_VALUES. The GUI sets this from
        # cfg.audio_samplerate_pref before calling start/open_device.
        self.samplerate_pref: str = 'auto'

        # Active stream parameters, set on successful open.
        self.samplerate = DEFAULT_SAMPLE_RATE
        self.block_size = DEFAULT_BLOCK_SIZE
        self.hop_size = DEFAULT_HOP_SIZE
        # Last successfully-opened INPUT device. Intentionally persisted across
        # stop_output/teardown AND a failed reopen (deliberately NOT cleared
        # there) — it seeds hot-plug recovery and the GUI's device display.
        # LIVENESS is the separate honest signal `state` / `input_running`
        # (_stream is not None), NOT this field being non-None.
        self.active_device: Optional[DeviceInfo] = None

        # Mic input gain (linear; 1.0 = no-op). Scales ONLY the silence-gate
        # decision + the level meter in the detection hop — never the signal
        # buffer — so pitch detection stays scale-invariant and the deck tap
        # records the raw mic. Set from cfg.mic_gain_db via set_mic_gain().
        # Read UNLOCKED by the analysis consumer: a bare-float store is GIL-atomic
        # (like last_rms_db), so the worst case is a one-block-stale gain — no
        # need to pay a lock acquire on the hot path for a single scalar.
        self.mic_gain: float = 1.0

        # Diagnostic scalars. Read snapshot via get_diagnostics().
        self.last_rms_db: float = -120.0
        self.last_ap: float = 1.0
        self.last_freq: float = 0.0
        self.last_locked_midi: Optional[int] = None
        self._overflow_count: int = 0
        self._underflow_count: int = 0

        # Shared mutable state — protected by self._lock.
        self._lock = threading.Lock()
        # The input ring of the open stream (see _InputRing). Before any
        # stream opens, an unbound inline ring at the default size serves
        # feed_input_frames(), analysing at self.samplerate.
        self._ring = _InputRing(DEFAULT_BLOCK_SIZE, DEFAULT_HOP_SIZE)
        # _buf: the detection window of the last analysed hop (the ring's
        # preallocated scratch, never reallocated per hop). The consumer
        # rewrites it under self._lock and get_buf_snapshot copies it under the
        # same lock, so no reader sees it mid-rewrite.
        self._buf: np.ndarray = self._ring.window
        self._stream = None
        self._emissions_paused = False
        self._transitioning = False
        # (error, message) when the last rate probe found the device gone.
        self._probe_error: Optional[tuple[AudioEngineError, str]] = None

        # ---- analysis consumer ---------------------------------------------
        # The detector is owned by whichever thread holds _analysis_lock (the
        # analysis thread, or an inline/test caller of process_pending). It is
        # an RLock so a slot that runs inside an inline emit and feeds frames
        # again cannot deadlock.
        self._analysis_lock = threading.RLock()
        self._detector = Detector(FILTER_MODE_DEFAULT)
        # Reset requests from other threads: bumping _reset_serial (under
        # self._lock) makes the consumer reset the detector before its next hop.
        self._reset_serial = 0
        self._detector_serial = 0
        # Bumped by every set_a4(). Each NoteEvent carries the generation it
        # was graded under; the consumer resets the detector when it changes,
        # so no reading measured against an old A4 is emitted after the change.
        self.a4_generation = 0
        self._detector_a4_generation = 0
        self._analysis_errors = 0
        self._wake = threading.Event()
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_stop: Optional[threading.Event] = None

        # ---- output side (audio-output foundation) -------------------------
        # Separate sd.OutputStream is the PRIMARY path (D5): it matches the
        # common topology (mic on one device, playback on another), keeps the
        # proven input lifecycle untouched, and owns the sample-accurate
        # master clock via the mixer. Duplex stays a same-device opt-in for
        # later. The output stream mirrors the input stream's lifecycle —
        # worker-thread open with timeout, orphan disposal, hot-plug recovery
        # — but on its OWN transitioning guard so it can never corrupt the
        # input guard (both run on the Qt main thread; the guards are
        # independent re-entrancy locks, not cross-thread sync).
        # The mixer owns the output format; every (re)open hands it the new
        # stream's rate and block size (Mixer.set_format), and the sources
        # follow from there.
        self.mixer = Mixer(max_block=DEFAULT_OUT_BLOCK,
                           samplerate=DEFAULT_SAMPLE_RATE)
        self._out_stream = None
        self._out_transitioning = False
        self.output_samplerate = DEFAULT_SAMPLE_RATE
        self.output_block_size = DEFAULT_OUT_BLOCK
        # Last successfully-opened OUTPUT device. Like active_device: persisted
        # across stop/teardown and a failed reopen (seeds hot-plug recovery +
        # GUI display); LIVENESS is `output_running` (_out_stream is not None),
        # not this field being non-None.
        self.active_output_device: Optional[DeviceInfo] = None
        self._last_output_snapshot: tuple = ()
        # Output status — kept separate from the input state machine (which
        # owns the pitch-detection lifecycle) so the two streams never
        # contend for one state field. The GUI polls these / gets the bool
        # back from open_output_device.
        self.output_running: bool = False
        self.last_output_error: AudioEngineError = AudioEngineError.NONE
        self.last_output_error_message: str = ''
        self._out_underflow_count: int = 0

        # ---- D3 mic-coordination --------------------------------------------
        # The coordinator is PURE policy (sax_coordination); the engine is its
        # CONSUMER. Each detection hop the analysis consumer runs one coordination
        # step: vote-exclude the output's sounding MIDIs from the incumbent
        # lock, and push the ducked gain target to the attached duck consumer
        # (the drone source). Inert until something pitched is sounding —
        # get_sounding_output_midis() is empty, so it excludes nothing and the
        # duck stays fully open — so it changes nothing until a drone or
        # pitch pipe plays.
        self.coordinator = OutputCoordinator(samplerate=DEFAULT_SAMPLE_RATE)
        # The source whose gain the duck drives (drone). Attached by the
        # DroneController on enable; duck-typed (anything with set_duck_target).
        self._duck_consumer = None
        # (name, host_api) hint for output hot-plug auto-recovery, mirroring
        # _preferred_hint on the input side. No caller sets it at present, so
        # recovery falls through to the vendor-class / default device choice.
        self._preferred_output_hint: Optional[DeviceSelection] = None
        # Handle for the engine-managed test tone, so
        # start/stop_test_tone are idempotent and don't leak sources.
        self._test_tone_handle = None

        # ---- tape-deck input-recording tap --------------------------------
        # The deck records the MIC (not the mixer output, as in the Android
        # app) to a single take. The engine is pure MECHANISM: a bounded
        # preallocated capture buffer the input hot path slice-assigns into
        # while armed. DeckController (policy) drives start/stop and owns the
        # state machine, WAV encode, and playback source — it never touches
        # the audio callback. A bounded buffer gives ZERO drop until cap and
        # ZERO hot-path alloc (no drain-pump, no chunk queue). The cap is
        # sized at arm time on the CALLING thread, so the callback never
        # allocates. All five fields are protected by self._lock.
        self._deck_armed = False
        self._deck_sink: Optional[np.ndarray] = None
        self._deck_head: int = 0
        self._deck_full: bool = False
        self._deck_sr: int = 0
        # Set when an in-progress take is cut short by an abrupt stream
        # teardown (device switch / stop) rather than a clean stop — the next
        # stop_input_recording() surfaces it via the `truncated` flag so the
        # deck state machine can mark the take partial, never orphan it.
        self._deck_truncated_by_close = False

        # Hot-plug snapshot.
        self._last_device_snapshot: tuple = ()
        # Wall-clock timestamp of the most recent refresh_devices() tick.
        # Diagnostics panel renders this so the user can confirm the
        # poller is alive when a hot-plugged device fails to recover.
        # None until the first poll fires.
        self.last_devices_refresh_at: Optional[datetime.datetime] = None
        # Persisted (name, host_api) hint for hot-plug auto-recovery.
        # Set by the GUI right after load_config so refresh_devices() can
        # resolve "the device the user picked last session" against the
        # current device list, which holds rotated PortAudio indices.
        self._preferred_hint: Optional[DeviceSelection] = None
        # Range gate: see sax_pitch.SAX_MIDI. Mirrored here so
        # the engine can drop MIDI values outside the supported range
        # without round-tripping to Qt.
        self._midi_min = 21
        # v0.6: raised from 91 (G6) to 108 (C8) so piccolo / sopranino /
        # recorder / piano / banjo upper registers aren't silently dropped.
        # Must stay in lock-step with sax_pitch.SAX_MIDI.
        self._midi_max = 108

    # ---- internal state helpers -------------------------------------------
    def _reset_filter_state(self) -> None:
        """Ask the analysis consumer to reset the detector before its next
        hop (held note, vote window and pending readings are discarded).
        Callers hold self._lock."""
        self._reset_serial += 1

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
        """Set the A4 tuning reference (Hz). Ignores non-finite or non-positive
        input: a tuner reference must be a positive frequency, and a4 <= 0 would
        make freq_to_midi()'s log2(f/a4) divide by zero on the detection hot
        path. (Mirrors the isfinite guards on set_mic_gain / input recording.)"""
        try:
            hz = float(hz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(hz) or hz <= 0.0:
            return
        with self._lock:
            self.a4 = hz
            self.a4_generation += 1

    def set_mic_gain(self, linear: float) -> None:
        """Set the mic input gain (linear multiplier). Applied to the
        silence-gate + level meter only (see sax_detector) — the signal
        buffer is never scaled, so the cents readout is unchanged. Clamped to
        roughly +/-24 dB; non-finite / non-numeric input is ignored. Default
        1.0 is a no-op (byte-identical to the pre-feature behaviour)."""
        try:
            g = float(linear)
        except (TypeError, ValueError):
            return
        if not math.isfinite(g):
            return
        self.mic_gain = max(0.05, min(16.0, g))

    def set_prefer_wdmks(self, flag: bool) -> None:
        self.prefer_wdmks = bool(flag)

    def pause_emissions(self) -> None:
        """Drop incoming detections without stopping the stream.

        Used by the GUI during A4 remapping so the analysis consumer can't
        slip a measurement into the half-rebuilt ``self.stats``. Readings
        already queued to the GUI thread are not recalled; their A4
        generation is what marks them stale (see NoteEvent).
        """
        with self._lock:
            self._emissions_paused = True
            self._reset_filter_state()

    def resume_emissions(self) -> None:
        with self._lock:
            self._emissions_paused = False
            self._reset_filter_state()

    def get_buf_snapshot(self) -> np.ndarray:
        """Return a copy of the last analysed detection window. Safe for the
        GUI thread to FFT/plot without racing the analysis consumer."""
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

    def get_active_device(self) -> Optional[DeviceInfo]:
        """Thread-safe snapshot of the currently-active device.

        v0.5.7.1: the GUI used to read ``engine.active_device`` directly
        from the Qt thread while the audio worker rewrote it on a
        successful open. Bare attribute reads are atomic in CPython, but
        the cached DeviceInfo's fields are not — and downstream callers
        (samplerate-changed handler) immediately re-derive
        ``DeviceSelection`` from it. Take the lock for the read and
        return whatever was last installed."""
        with self._lock:
            return self.active_device

    def set_samplerate_pref(self, pref: str) -> None:
        """Update the in-memory rate preference. Does NOT reopen the stream;
        the caller (GUI) decides whether to call ``open_device`` after.

        Anything outside SAMPLERATE_PREF_VALUES degrades to 'auto' so a
        stale config can't wedge the engine on an impossible rate."""
        if pref not in SAMPLERATE_PREF_VALUES:
            pref = 'auto'
        self.samplerate_pref = pref

    def start(self, preferred: Optional[DeviceSelection] = None,
              samplerate_pref: str = 'auto') -> None:
        """Enumerate, then open the preferred device (or system default).
        Never raises. Sets state + emits signals on completion or failure.
        """
        self.set_samplerate_pref(samplerate_pref)
        if not AUDIO_OK:
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.NO_DEVICE,
                            'sounddevice / PortAudio not available')
            return
        if self._transitioning:
            return
        self._transitioning = True
        try:
            # A start while a stream is open (retry() from RUNNING) replaces
            # it: two live callbacks must never feed one engine.
            self._teardown_stream()
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

    def open_device(self, sel: DeviceSelection,
                    samplerate_pref: Optional[str] = None) -> None:
        """Switch to a user-picked device. Stops the current stream first.
        Never raises.

        If ``samplerate_pref`` is None, the engine reuses whatever pref it
        was started with. Pass the new value when the user changes the
        rate combo so the next negotiation runs with the new policy.
        """
        if samplerate_pref is not None:
            self.set_samplerate_pref(samplerate_pref)
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
        self.start(preferred=prev, samplerate_pref=self.samplerate_pref)

    def set_preferred_hint(self, sel: Optional[DeviceSelection]) -> None:
        """Stash the persisted (name, host_api) so refresh_devices() can
        try to auto-recover when a hot-plug introduces a device matching
        the user's saved selection. No-op if ``sel`` is falsy."""
        if sel and sel.name:
            self._preferred_hint = sel
        else:
            self._preferred_hint = None

    def retry_open(self) -> None:
        """Force a fresh PortAudio enumeration and re-resolve the
        persisted device selection against it before re-opening.

        v0.5.7 fix: the previous ``retry()`` path inherited the engine's
        stale snapshot and could not recover when a device was plugged
        in after launch — the saved index was wrong and the saved
        ``(name, host_api)`` was never re-resolved against the fresh
        device list. ``retry_open`` always re-enumerates first.

        Resolution order:
          1. Persisted preferred-hint matched by (name, host_api) in the
             fresh list.
          2. Whatever was active before, if it's still present.
          3. PortAudio's default input device.

        On failure, transitions to FAILED with a clear message
        (``NO_DEVICE`` + "Pinned device not present" when the saved
        selection isn't in the fresh list)."""
        if not AUDIO_OK:
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.NO_DEVICE,
                            'sounddevice / PortAudio not available')
            return
        if self._transitioning:
            return
        # v0.5.7.1: take the _transitioning flag for the entire resolve
        # phase so a concurrent hot-plug auto-recovery (which runs from
        # the refresh_devices() poller thread) can't race us into
        # double-open territory. open_device() and start() set their
        # own _transitioning inside the same flag, so we release the
        # guard BEFORE delegating to either — otherwise the inner call
        # would early-return as a no-op and silently skip the reopen.
        self._transitioning = True
        try:
            # Force re-enumeration; do NOT trust the cached snapshot.
            devices = query_input_devices()
            self.last_devices_refresh_at = datetime.datetime.now()
            self._last_device_snapshot = self._snapshot_key(devices)
            if self.signals is not None:
                try:
                    self.signals.devices_changed.emit(devices)
                except Exception:
                    pass
            if not devices:
                # FAILED must mean no stream: the old one would keep emitting
                # readings under a banner that says the input is down.
                self._teardown_stream()
                self._set_state(AudioEngineState.FAILED,
                                AudioEngineError.NO_DEVICE,
                                'No audio input devices detected')
                return
            # Pick a selection candidate. Always re-resolve by (name,
            # host_api) — never reuse a stored PortAudio index.
            sel: Optional[DeviceSelection] = None
            hint = self._preferred_hint
            if hint and hint.name:
                for d in devices:
                    if (d.name == hint.name
                            and (not hint.host_api
                                 or d.host_api == hint.host_api)):
                        sel = DeviceSelection(
                            name=d.name, host_api=d.host_api, samplerate=0)
                        break
                if sel is None:
                    # Hint device isn't in the fresh list — surface that
                    # instead of silently falling back to a default.
                    self._teardown_stream()
                    self._set_state(
                        AudioEngineState.FAILED,
                        AudioEngineError.NO_DEVICE,
                        f'Pinned device not present: {hint.name}')
                    return
            elif self.active_device is not None:
                sel = DeviceSelection(
                    name=self.active_device.name,
                    host_api=self.active_device.host_api,
                    samplerate=0)
        finally:
            # Drop the guard before delegating; open_device() / start()
            # re-acquire it for their own work.
            self._transitioning = False
        # Reopen with the fresh resolution. open_device handles teardown
        # and state transitions internally.
        self.open_device(sel) if sel else self.start(
            preferred=None, samplerate_pref=self.samplerate_pref)

    def stop(self) -> None:
        """Tear down the stream cleanly. Never raises."""
        self._teardown_stream()
        self._set_state(AudioEngineState.STOPPED)

    @property
    def input_running(self) -> bool:
        """True while the input (mic) stream is open.

        The HONEST pre-click probe the deck's can_record() reads — the
        mirror of ``output_running``, but DERIVED from the live stream
        rather than a stored bool, so it can never drift out of sync if a
        teardown path forgets to clear a flag. start_input_recording()
        shares this same open-check.
        """
        return self._stream is not None

    # ---- tape-deck input recording ----------------------------------------
    # Pure MECHANISM: a bounded preallocated mic-capture buffer the input hot
    # path (_on_input) slice-assigns into while armed. Policy — the state
    # machine, WAV encode, and playback source — lives in
    # sax_deck.DeckController, which drives these and never touches the audio
    # callback. Same mechanism/policy split as D3 (GainGlide vs
    # OutputCoordinator). The buffer gives ZERO drop until cap and ZERO
    # hot-path alloc; the (multi-MB) allocation happens here on the CALLING
    # thread, never in the callback.
    def start_input_recording(self, max_seconds: float) -> bool:
        """Arm the deck tap: capture every mic frame into a bounded buffer
        until stop_input_recording() or the cap (``max_seconds``) is hit.

        Returns False and stays disarmed if the input stream isn't open
        (no false 'recording' with no mic) or ``max_seconds`` is
        non-positive / non-finite. A fresh call discards any prior take.
        """
        try:
            secs = float(max_seconds)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(secs) or secs <= 0.0:
            return False
        with self._lock:
            if self._stream is None:
                return False
            sr = int(self.samplerate)
        # Allocate OUTSIDE the lock — a multi-MB np.zeros must not stall the
        # audio callback. Arm (pointer swap + flags) under the lock briefly.
        # secs is finite + positive (guarded above), but a finite-yet-huge value
        # can overflow secs*sr to inf (int(inf) -> OverflowError) or demand an
        # impossible buffer — reject rather than crash, per the documented
        # "returns False on bad max_seconds" contract.
        try:
            cap = max(1, int(secs * sr))
            sink = np.zeros(cap, dtype=np.float32)
        except (OverflowError, ValueError, MemoryError):
            return False
        with self._lock:
            if self._stream is None:
                return False  # torn down during the allocation
            self._deck_sink = sink
            self._deck_head = 0
            self._deck_full = False
            self._deck_truncated_by_close = False
            self._deck_sr = sr
            self._deck_armed = True
        return True

    def stop_input_recording(self) -> tuple[np.ndarray, int, bool]:
        """Disarm and return ``(take_f32_mono, samplerate, truncated)``.

        ``take`` is a COPY of exactly the frames captured (an empty array
        if none). ``truncated`` is True when the take was cut short — by
        hitting the cap, or by an abrupt stream teardown (device switch /
        stop) rather than this call. The copy runs on the CALLING thread,
        outside the audio hot path.
        """
        with self._lock:
            sink = self._deck_sink
            head = self._deck_head
            sr = self._deck_sr or int(self.samplerate)
            truncated = self._deck_full or self._deck_truncated_by_close
            self._deck_armed = False
            self._deck_sink = None
            self._deck_head = 0
            self._deck_full = False
            self._deck_truncated_by_close = False
        if sink is None or head <= 0:
            return (np.zeros(0, dtype=np.float32), sr, truncated)
        return (sink[:head].copy(), sr, truncated)

    def is_input_recording(self) -> bool:
        """True while the deck tap is armed and capturing mic frames.

        Flips False the instant the cap is hit (auto-disarm) even before
        stop_input_recording() — the deck pump() reads that as 'cap hit'.
        """
        with self._lock:
            return self._deck_armed

    def recorded_frame_count(self) -> int:
        """Frames captured into the current take so far.

        Grows monotonically while recording (the continuous-capture
        observable); equals the cap once a take has hit deck_max_seconds;
        resets to 0 after stop_input_recording().
        """
        with self._lock:
            return self._deck_head

    def refresh_devices(self) -> list[DeviceInfo]:
        """Poll the device list. Emits ``devices_changed`` on a diff.

        If the active device vanished while RUNNING, transitions to
        FAILED(DEVICE_DISCONNECTED). v0.5.7: if we're in
        FAILED(NO_DEVICE) or FAILED(DEVICE_DISCONNECTED) and a NEW
        input device appears, automatically attempt to open it. The
        previous behaviour required the user to click Retry; this hid
        the recovery from anyone who plugged in an interface after
        launch and assumed it would just work.

        Hot-plug toast for vendor-class interfaces is emitted via
        ``interface_appeared`` when applicable.
        """
        if not AUDIO_OK:
            return []
        if self._transitioning:
            # Don't race the open path.
            return []
        devices = query_input_devices()
        # Always advance the refresh timestamp — the diagnostics row
        # uses it to prove the poller is still running even when the
        # device list happens not to have changed.
        self.last_devices_refresh_at = datetime.datetime.now()
        # v0.5.7.8: snapshot active_device once under the lock so the
        # disconnect-detection branch can't observe a different
        # DeviceInfo between the check and the error-message format
        # (a parallel open_stream could swap it out mid-method,
        # producing "Device disconnected: X" when Y actually vanished).
        with self._lock:
            active = self.active_device
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
                and active is not None
                and not any(d.name == active.name
                            and d.host_api == active.host_api
                            for d in devices)):
            self._teardown_stream()
            self._set_state(AudioEngineState.FAILED,
                            AudioEngineError.DEVICE_DISCONNECTED,
                            f'Device disconnected: {active.name}')
        # Vendor-class device appearing? Emit toast.
        if appeared and self.signals is not None:
            for d in devices:
                if d.name in appeared and VENDOR_RE.search(d.name):
                    try:
                        self.signals.interface_appeared.emit(d)
                    except Exception:
                        pass
                    break
        # Hot-plug auto-recovery from FAILED(NO_DEVICE) /
        # FAILED(DEVICE_DISCONNECTED). Only fires when at least one
        # NEW input-capable device appeared; we don't re-attempt on
        # spurious diffs (renames, device-default changes).
        if (self.state == AudioEngineState.FAILED
                and self.last_error in (AudioEngineError.NO_DEVICE,
                                        AudioEngineError.DEVICE_DISCONNECTED)
                and appeared and devices):
            self._auto_recover_after_hotplug(devices)
        return devices

    def _auto_recover_after_hotplug(
            self, devices: list[DeviceInfo]) -> None:
        """Pick a device from the fresh list and try to open it.

        Priority: persisted preferred hint (matched on name + host_api),
        then highest-ranked vendor-regex device, then PortAudio's
        default input device. Quiet if nothing opens — stays in
        FAILED so the banner remains visible."""
        sel: Optional[DeviceSelection] = None
        hint = self._preferred_hint
        if hint and hint.name:
            for d in devices:
                if (d.name == hint.name
                        and (not hint.host_api
                             or d.host_api == hint.host_api)):
                    sel = DeviceSelection(
                        name=d.name, host_api=d.host_api, samplerate=0)
                    break
        if sel is None:
            best: Optional[DeviceInfo] = None
            for d in devices:
                if VENDOR_RE.search(d.name):
                    if (best is None
                            or d.default_samplerate > best.default_samplerate):
                        best = d
            if best is not None:
                sel = DeviceSelection(
                    name=best.name, host_api=best.host_api, samplerate=0)
        if sel is None:
            di = _probe_default_input_index()
            if di is not None:
                for d in devices:
                    if d.index == di:
                        sel = DeviceSelection(
                            name=d.name, host_api=d.host_api, samplerate=0)
                        break
        if sel is None:
            return
        # open_device handles its own transitioning guard, teardown,
        # and FAILED/RUNNING transitions.
        self.open_device(sel)

    # ---- output path ------------------------------------------------------
    # Mirrors the input lifecycle on a separate sd.OutputStream (D5). Never
    # raises across its public surface; failure becomes status
    # (output_running=False + last_output_error*). All entry points run on the
    # Qt main thread and use the independent _out_transitioning guard.
    def open_output_device(self, sel: Optional[DeviceSelection],
                           samplerate_pref: Optional[str] = None) -> bool:
        """Open (or switch to) an output device and start pulling the mixer.

        ``sel`` of None opens the system default output. Tears down any
        current output stream first. Returns True on success, False on
        failure; never raises. The new stream restarts the mixer's sample
        clock so absolute-sample scheduling has a known origin.
        """
        if samplerate_pref is not None:
            self.set_samplerate_pref(samplerate_pref)
        if not AUDIO_OK:
            self._set_output_failed(AudioEngineError.NO_DEVICE,
                                    'sounddevice / PortAudio not available')
            return False
        if self._out_transitioning:
            return False
        self._out_transitioning = True
        try:
            self._teardown_output_stream()
            devices = query_output_devices()
            if not devices:
                self._set_output_failed(AudioEngineError.NO_DEVICE,
                                        'No audio output devices detected')
                self._last_output_snapshot = ()
                return False
            self._last_output_snapshot = self._snapshot_key(devices)
            return self._open_output_with_fallback(sel, devices)
        finally:
            self._out_transitioning = False

    def stop_output(self) -> None:
        """Tear down the output stream cleanly. Never raises. Leaves the mixer
        and its registered sources intact so a re-open resumes them."""
        self._teardown_output_stream()
        self.output_running = False
        self.last_output_error = AudioEngineError.NONE
        self.last_output_error_message = ''

    def refresh_output_devices(self) -> list[DeviceInfo]:
        """Poll the output device list. Mirror of refresh_devices for output.

        If the active output device vanished while running, tears the stream
        down and records DEVICE_DISCONNECTED. If a new output device appears
        while we're stopped after a disconnect / no-device, auto-recovers the
        same way the input path does.
        """
        if not AUDIO_OK:
            return []
        if self._out_transitioning:
            return []
        devices = query_output_devices()
        active = self.active_output_device
        key = self._snapshot_key(devices)
        if key == self._last_output_snapshot:
            return devices
        prev_names = {n for (n, _h, _c) in self._last_output_snapshot}
        new_names = {n for (n, _h, _c) in key}
        appeared = new_names - prev_names
        self._last_output_snapshot = key
        # Active output device vanished?
        if (self.output_running and active is not None
                and not any(d.name == active.name
                            and d.host_api == active.host_api
                            for d in devices)):
            self._teardown_output_stream()
            self._set_output_failed(AudioEngineError.DEVICE_DISCONNECTED,
                                    f'Output device disconnected: {active.name}')
        # New output device appeared while we're down → auto-recover.
        if (not self.output_running
                and self.last_output_error in (
                    AudioEngineError.NO_DEVICE,
                    AudioEngineError.DEVICE_DISCONNECTED)
                and appeared and devices):
            self._auto_recover_output_after_hotplug(devices)
        return devices

    def _auto_recover_output_after_hotplug(
            self, devices: list[DeviceInfo]) -> None:
        """Pick an output device from the fresh list and try to open it.
        Priority: persisted hint (name+host_api) → vendor-class → default."""
        sel: Optional[DeviceSelection] = None
        hint = self._preferred_output_hint
        if hint and hint.name:
            for d in devices:
                if (d.name == hint.name
                        and (not hint.host_api or d.host_api == hint.host_api)):
                    sel = DeviceSelection(name=d.name, host_api=d.host_api,
                                          samplerate=0)
                    break
        if sel is None:
            best: Optional[DeviceInfo] = None
            for d in devices:
                if VENDOR_RE.search(d.name):
                    if (best is None
                            or d.default_samplerate > best.default_samplerate):
                        best = d
            if best is not None:
                sel = DeviceSelection(name=best.name, host_api=best.host_api,
                                      samplerate=0)
        if sel is None:
            di = _probe_default_output_index()
            if di is not None:
                for d in devices:
                    if d.index == di:
                        sel = DeviceSelection(name=d.name, host_api=d.host_api,
                                              samplerate=0)
                        break
        if sel is None:
            return
        self.open_output_device(sel)

    def get_sounding_output_midis(self) -> frozenset[int]:
        """The set of MIDI notes the output is currently sounding (drone /
        pitch pipe). Consumed by the input pitch detector for vote-exclude +
        duck-on-suspicion (D3). Empty when nothing pitched is playing."""
        return self.mixer.sounding_midis()

    def attach_duck_consumer(self, consumer) -> None:
        """Register the source whose gain the D3 duck drives (the drone).
        Duck-typed: ``consumer`` need only expose ``set_duck_target(level)``.
        The DroneController calls this on enable; ``detach_duck_consumer`` on
        disable. A bare attribute store — atomic under the GIL, no lock."""
        self._duck_consumer = consumer

    def detach_duck_consumer(self, consumer=None) -> None:
        """Drop the duck consumer (drone disabled / stopped). With no arg, or
        when ``consumer`` matches the current one, clears it; otherwise no-op
        (so a stale detach can't unhook a newer consumer)."""
        if consumer is None or consumer is self._duck_consumer:
            self._duck_consumer = None

    def coordination_step(self, detected_midi: Optional[int],
                          level: Optional[float] = None) -> frozenset[int]:
        """Run ONE D3 coordination hop and return the MIDIs to vote-exclude.

        Called once per detection hop by the analysis consumer — with the
        detected MIDI, or ``None`` on silence/reject so the coordinator's
        release ramp keeps advancing. ``level`` is the mic level of the newest
        hop, passed while a duck consumer is attached so the coordinator can
        re-admit a note the player holds in unison with the output (see
        sax_coordination). Also the deterministic
        test injection point: a test calls it directly with a synthetic
        ``detected_midi`` (no mic audio needed) and asserts the duck/vote
        behaviour.

        Pushes the (already-ramped) duck level to the attached duck consumer
        (the drone) via ``set_duck_target``; the consumer's own GainGlide does
        the per-sample de-zipper. Inert when nothing pitched sounds: the
        coordinator excludes nothing and the duck target stays 1.0.

        Must be called OUTSIDE ``self._lock`` — it briefly takes the mixer lock
        (via get_sounding_output_midis); never nest the two.
        """
        decision = self.coordinator.update(detected_midi,
                                           self.get_sounding_output_midis(),
                                           level)
        consumer = self._duck_consumer
        if consumer is not None:
            try:
                consumer.set_duck_target(decision.duck_level)
            except Exception:
                # A misbehaving consumer must not break the analysis hop.
                pass
        return decision.excluded_midis

    def start_test_tone(self, freq: float = 440.0) -> Optional[object]:
        """Register a sine TestToneSource on the mixer and return its handle.

        Proves the output path ("a test tone plays through the mixer while
        the tuner still reads the mic"). Idempotent: a second call
        replaces the previous tone rather than stacking. Returns None if no
        output stream is running (nothing would be heard)."""
        if not self.output_running:
            return None
        f = float(freq)
        if not math.isfinite(f) or f <= 0.0:
            return None       # non-finite / non-positive freq -> NaN dphi -> NaN audio
        self.stop_test_tone(self._test_tone_handle)
        tone = TestToneSource(freq=f,
                              samplerate=self.mixer.samplerate,
                              max_block=self.mixer.max_block,
                              gain=0.2, attack_ms=8.0, release_ms=60.0)
        self._test_tone_handle = self.mixer.register(tone)
        return self._test_tone_handle

    def stop_test_tone(self, handle: Optional[object] = None) -> None:
        """Stop the test tone. With no handle, stops the engine-managed one.
        Silent if nothing is registered.

        The enveloped tone is RELEASED (a short fade) rather than hard
        unregistered, so it ends click-free; the Mixer reaps it once the tail
        reaches zero. A handle that predates the envelope (no release()) falls
        back to an immediate unregister."""
        h = handle if handle is not None else self._test_tone_handle
        if h is not None:
            rel = getattr(h, "release", None)
            if callable(rel):
                rel()                     # fade out; Mixer auto-reaps when done
            else:
                self.mixer.unregister(h)  # type: ignore[arg-type]
        if handle is None or handle is self._test_tone_handle:
            self._test_tone_handle = None

    # ---- output internals -------------------------------------------------
    def _set_output_failed(self, err: AudioEngineError, msg: str) -> None:
        self.output_running = False
        self.last_output_error = err
        self.last_output_error_message = msg

    def _resolve_output_candidates(
            self, sel: Optional[DeviceSelection],
            devices: list[DeviceInfo]) -> list[DeviceInfo]:
        """Ordered output devices to attempt: saved selection (name+host_api,
        then name), then system default output, then anything else; each
        reordered by the platform host-API preference. Mirror of
        _resolve_candidates for the output side."""
        out: list[DeviceInfo] = []
        if sel and sel.name:
            for d in devices:
                if (d.name == sel.name
                        and (not sel.host_api or d.host_api == sel.host_api)):
                    out.append(d)
            for d in devices:
                if d.name == sel.name and d not in out:
                    out.append(d)
        default_idx = _probe_default_output_index()
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
        for d in devices:
            if d not in out:
                out.append(d)
        return self._reorder_by_host_api(out)

    def _negotiate_output_sample_rate(self, dev: DeviceInfo) -> list[int]:
        """Output-side mirror of _negotiate_sample_rate, using
        check_output_settings. Honors a pinned rate; otherwise probes
        SAMPLERATE_CANDIDATES in order, then the device default."""
        pref = self.samplerate_pref or 'auto'
        if pref != 'auto':
            try:
                rate = int(pref)
            except ValueError:
                rate = 0
            return [rate] if rate else []
        out: list[int] = []
        self._probe_error = None
        for rate in SAMPLERATE_CANDIDATES:
            try:
                sd.check_output_settings(device=dev.index, channels=1,
                                         dtype='float32', samplerate=rate)
                out.append(rate)
            except Exception as exc:
                if self._probe_vanished(dev, exc):
                    return []
                continue
        dev_default = int(dev.default_samplerate or 0)
        if dev_default and dev_default not in out:
            try:
                sd.check_output_settings(device=dev.index, channels=1,
                                         dtype='float32',
                                         samplerate=dev_default)
                out.append(dev_default)
            except Exception:
                pass
        return out

    def _open_output_with_fallback(self, sel: Optional[DeviceSelection],
                                   devices: list[DeviceInfo]) -> bool:
        """Walk candidate devices × sample rates until one OutputStream opens.
        Mirror of _open_with_fallback. Returns True on success."""
        candidates = self._resolve_output_candidates(sel, devices)
        pinned = (self.samplerate_pref or 'auto') != 'auto'
        last_err: AudioEngineError = AudioEngineError.UNKNOWN
        last_msg = ''
        for dev in candidates:
            sr_clean = self._negotiate_output_sample_rate(dev)
            if not sr_clean and self._probe_error is not None:
                last_err, last_msg = self._probe_error
                continue
            if not sr_clean:
                last_err = AudioEngineError.UNSUPPORTED_RATE
                last_msg = (f'{dev.name} [{dev.host_api}]: output does not '
                            f'accept the requested sample rate')
                continue
            for sr in sr_clean:
                ok, err_kind, err_msg = self._try_open_output(dev, sr)
                if ok:
                    return True
                last_err, last_msg = err_kind, err_msg
                if pinned:
                    break
                if err_kind not in (AudioEngineError.UNSUPPORTED_RATE,
                                    AudioEngineError.HOSTAPI_FAILURE):
                    break
            if pinned:
                break
        self._set_output_failed(last_err, last_msg)
        return False

    def _try_open_output(self, dev: DeviceInfo,
                         samplerate: int) -> tuple[bool, AudioEngineError, str]:
        """One OutputStream open attempt. Mirror of _try_open: worker thread
        with HOST_API_OPEN_TIMEOUT_S join + cancelled-flag orphan disposal so
        a wedged output driver can't freeze the GUI and can't leak a started
        stream."""
        out_block = max(256, int(round(samplerate * OUT_BLOCK_MS / 1000.0)))
        # Hand the mixer this stream's format before it can start pulling:
        # the mixer sizes its accumulator to the block (arming the zero-alloc
        # fast path), restarts its clock for the new stream, and re-rates
        # every source, so a held drone, pad, test tone or deck playback keeps
        # its pitch and a running metronome its beat. No stream is running
        # here (the caller tore the old one down).
        self.mixer.set_format(samplerate, out_block)
        result: dict = {'stream': None, 'err_kind': None, 'err_msg': '',
                        'cancelled': False}
        # Atomic hand-off lock, mirror of _try_open (backlog #7): serialises the
        # worker's check-cancelled+hand-off against the main thread's
        # set-cancelled+claim so a just-handed-off stream can't be orphaned.
        hlock = threading.Lock()
        cb = self._make_output_callback(samplerate)

        def worker() -> None:
            try:
                stream = sd.OutputStream(
                    samplerate=samplerate,
                    blocksize=out_block,
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
            with hlock:
                cancelled = result['cancelled']
                if not cancelled:
                    result['stream'] = stream
            if cancelled:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            return

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=HOST_API_OPEN_TIMEOUT_S)
        if t.is_alive():
            with hlock:
                result['cancelled'] = True
                handoff = result.get('stream')
                result['stream'] = None
            if handoff is not None:
                try:
                    handoff.stop()
                except Exception:
                    pass
                try:
                    handoff.close()
                except Exception:
                    pass
            return False, AudioEngineError.HOSTAPI_FAILURE, (
                f'{dev.name} [{dev.host_api}]: output open timed out '
                f'after {HOST_API_OPEN_TIMEOUT_S:.1f}s')
        if result['stream'] is None:
            return False, (result['err_kind'] or AudioEngineError.UNKNOWN), \
                   str(result['err_msg'])

        # Success — install under the lock, same discipline as the input side.
        with self._lock:
            self._out_stream = result['stream']
            self.output_samplerate = int(samplerate)
            self.output_block_size = int(out_block)
            self._out_underflow_count = 0
            self.active_output_device = dev
            # Publish liveness INSIDE the lock, atomically with the stream
            # install: a concurrent _teardown_output_stream() (which takes the
            # same lock) then cannot slip between the install and the flag and
            # leave output_running=True with _out_stream=None.
            self.output_running = True
        self.last_output_error = AudioEngineError.NONE
        self.last_output_error_message = ''
        return True, AudioEngineError.NONE, ''

    def _make_output_callback(self, samplerate: int):
        """Build the PortAudio output callback. It pulls exactly one block
        from the mixer into the device buffer. No allocation, no locks held
        across numpy work (the mixer enforces both); never raises back into
        PortAudio."""
        engine = self

        def cb(outdata, frames, ti, st):
            try:
                if st is not None and getattr(st, 'output_underflow', False):
                    with engine._lock:
                        engine._out_underflow_count += 1
                # outdata is (frames, channels); the mixer is mono. Pulling
                # into the first channel's view keeps a single source of
                # truth; for a multi-channel device the mixer broadcasts.
                if outdata.ndim == 2 and outdata.shape[1] == 1:
                    engine.mixer.render(outdata[:, 0], frames)
                else:
                    engine.mixer.render(outdata, frames)
            except Exception:
                # Output callback must never raise into PortAudio. On any
                # failure, emit silence for this block rather than garbage.
                try:
                    outdata.fill(0.0)
                except Exception:
                    pass

        return cb

    def _teardown_output_stream(self) -> None:
        """Stop + close the output stream. Mirror of _teardown_stream: null
        the handle under the lock, then stop/close outside it."""
        with self._lock:
            stream = self._out_stream
            self._out_stream = None
            # Clear liveness atomically with the stream (mirror of the install in
            # _try_open_output) so output_running can never be observed True with
            # _out_stream already None. The callers' later output_running=False
            # (stop_output / _set_output_failed) is now a redundant no-op.
            self.output_running = False
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
            # Orphan-disposal for the deck: an abrupt teardown while
            # recording (device switch / stop mid-take) disarms the tap but
            # KEEPS the partial take so the deck's next stop_input_recording()
            # can still return it, flagged truncated — never orphan the take
            # or leave _deck_armed pointing at a closed stream.
            if self._deck_armed:
                self._deck_armed = False
                self._deck_truncated_by_close = True
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        # After the stream: no callback can wake the thread again once the
        # stream has stopped.
        self._stop_analysis_thread()

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

    def _negotiate_sample_rate(self,
                                dev: DeviceInfo) -> list[int]:
        """Return a sample-rate probe list for ``dev`` ordered by
        preference. The caller walks the list and attempts to open each
        rate in turn; the first one that opens cleanly wins.

        Policy (v0.5.6):
        * If ``self.samplerate_pref`` pins a specific rate, the list
          contains only that rate. A failure surfaces as
          UNSUPPORTED_RATE — we do NOT silently fall back when the user
          pinned a value.
        * If pref is ``'auto'``, the list is SAMPLERATE_CANDIDATES
          (48k, 44.1k, then the higher rates) filtered through PortAudio's
          ``check_input_settings`` so unsupported rates are skipped
          before we pay the cost of an InputStream open. The device's
          ``default_samplerate`` is appended at the bottom in case the
          device only accepts exotic rates (cheap webcam mics report
          22050 or 16000).
        """
        pref = self.samplerate_pref or 'auto'
        if pref != 'auto':
            try:
                rate = int(pref)
            except ValueError:
                rate = 0
            return [rate] if rate else []

        out: list[int] = []
        self._probe_error = None
        # Probe each candidate without opening the stream. Lying drivers
        # may pass check_input_settings then fail at .start() — the
        # caller catches that and walks to the next candidate.
        for rate in SAMPLERATE_CANDIDATES:
            try:
                sd.check_input_settings(device=dev.index, channels=1,
                                        dtype='float32', samplerate=rate)
                out.append(rate)
            except Exception as exc:
                if self._probe_vanished(dev, exc):
                    return []
                continue
        # Append device default as a last-ditch fallback for exotic rates.
        dev_default = int(dev.default_samplerate or 0)
        if dev_default and dev_default not in out:
            try:
                sd.check_input_settings(device=dev.index, channels=1,
                                        dtype='float32',
                                        samplerate=dev_default)
                out.append(dev_default)
            except Exception:
                pass
        return out

    def _probe_vanished(self, dev: DeviceInfo, exc: BaseException) -> bool:
        """Record a rate probe that failed because the device is gone.

        A probe is a question about one rate, so a failure normally just
        drops that rate. A device unplugged mid-probe fails every rate,
        though, and reporting that as an unsupported rate hid the unplug
        from the hot-plug recovery, which only reconnects a missing device.
        """
        if self._classify_error(exc) != AudioEngineError.NO_DEVICE:
            return False
        self._probe_error = (AudioEngineError.NO_DEVICE,
                             f'{dev.name} [{dev.host_api}]: {exc}')
        return True

    def _open_with_fallback(self, sel: Optional[DeviceSelection],
                            devices: list[DeviceInfo]) -> None:
        """Walk the candidate list × sample-rate list until one opens.

        On success, sets RUNNING. On exhaustion, sets FAILED with the
        most descriptive error we saw.
        """
        self._set_state(AudioEngineState.OPENING)
        candidates = self._resolve_candidates(sel, devices)
        pinned = (self.samplerate_pref or 'auto') != 'auto'
        last_err: AudioEngineError = AudioEngineError.UNKNOWN
        last_msg: str = ''
        for dev in candidates:
            sr_clean = self._negotiate_sample_rate(dev)
            if not sr_clean and self._probe_error is not None:
                last_err, last_msg = self._probe_error
                continue
            if not sr_clean:
                last_err = AudioEngineError.UNSUPPORTED_RATE
                if pinned:
                    last_msg = (
                        f'{dev.name} [{dev.host_api}]: device does not '
                        f'accept {self.samplerate_pref} Hz')
                else:
                    last_msg = (
                        f'{dev.name} [{dev.host_api}]: device does not '
                        f'accept any standard sample rate '
                        f'(tried 48k / 44.1k / 96k / 88.2k / 192k)')
                continue
            for sr in sr_clean:
                ok, err_kind, err_msg = self._try_open(dev, sr)
                if ok:
                    return
                last_err, last_msg = err_kind, err_msg
                # When the user pinned a rate, surface the failure;
                # never silently fall back to another rate or device.
                if pinned:
                    break
                # In auto mode: UNSUPPORTED_RATE / HOSTAPI_FAILURE on
                # this rate -> try the next rate; everything else means
                # the device itself is wedged, so move to the next dev.
                if err_kind not in (AudioEngineError.UNSUPPORTED_RATE,
                                    AudioEngineError.HOSTAPI_FAILURE):
                    break
            if pinned:
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
        # block = 8 × hop (as DEFAULT_BLOCK_SIZE = 8 × DEFAULT_HOP_SIZE)
        # keeps the ~370 ms YIN window at every sample rate.
        block = hop * 8

        # v0.5.7.7: 'cancelled' lets the main thread tell a slow worker
        # to dispose the stream it eventually obtains. PortAudio offers
        # no cancellation API for InputStream(), so we cannot interrupt
        # the open itself; the best we can do is stop+close whatever
        # stream the worker hands back after we already gave up. Without
        # this, a worker that finishes after the join timeout leaves a
        # fully-started stream owned by nothing -- it keeps the device
        # busy until process exit, and subsequent opens fail.
        result: dict = {'stream': None, 'err_kind': None, 'err_msg': '',
                        'cancelled': False}
        # Serialise the worker's (check-cancelled + hand-off) against the main
        # thread's (set-cancelled + claim-handoff) so a stream finished in the
        # gap between those two steps can never be orphaned (backlog #7).
        hlock = threading.Lock()

        # The stream's ring and rate are bound into its callback BEFORE the
        # stream exists, so a callback that fires between start() and the
        # install below already writes into this stream's own ring, sized and
        # rated for it. Those samples are analysed once the ring is installed.
        ring = _InputRing(block, hop, samplerate, inline=not ANALYSIS_IN_THREAD)
        cb = self._make_callback(samplerate, ring)

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
            # The cancelled-check and the hand-off must be ONE critical section,
            # else the main thread can set cancelled + read stream==None +
            # return in the gap, orphaning this started stream. Dispose OUTSIDE
            # the lock so a blocking stop()/close() can't stall the main thread.
            with hlock:
                cancelled = result['cancelled']
                if not cancelled:
                    result['stream'] = stream
            if cancelled:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            return

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=HOST_API_OPEN_TIMEOUT_S)
        if t.is_alive():
            # Driver is slow (or hung). Tell the worker to dispose any
            # stream it eventually produces. If the worker has already
            # completed by the time we set this flag, the assignment is
            # harmless -- result['stream'] is already populated and we
            # fall through to the success path below.
            # Atomically mark cancelled AND claim any stream the worker already
            # handed off, under the same lock the worker uses, so exactly one
            # side owns the disposal -- no orphan, no double-dispose. Dispose
            # outside the lock.
            with hlock:
                result['cancelled'] = True
                handoff = result.get('stream')
                result['stream'] = None
            if handoff is not None:
                try:
                    handoff.stop()
                except Exception:
                    pass
                try:
                    handoff.close()
                except Exception:
                    pass
            return False, AudioEngineError.HOSTAPI_FAILURE, (
                f'{dev.name} [{dev.host_api}]: open timed out '
                f'after {HOST_API_OPEN_TIMEOUT_S:.1f}s')
        if result['stream'] is None:
            return False, (result['err_kind'] or AudioEngineError.UNKNOWN), \
                   str(result['err_msg'])

        # Success — install the new stream and reset diagnostics.
        # v0.5.7.1: active_device must be assigned under the same lock
        # as everything else readers grab from the engine snapshot.
        # get_diagnostics() reads it inside the lock; the GUI's
        # _on_sr_changed() reads via get_active_device() (also locked);
        # refresh_devices() reads it while comparing against the fresh
        # device list. A bare write here let those readers tear.
        with self._lock:
            self._stream = result['stream']
            self.samplerate = int(samplerate)
            self.block_size = int(block)
            self.hop_size = int(hop)
            self._ring = ring
            self._buf = ring.window
            self._overflow_count = 0
            self._underflow_count = 0
            self._reset_filter_state()
            self.active_device = dev
        if not ring.inline:
            self._start_analysis_thread()
        self._set_state(AudioEngineState.RUNNING)
        return True, AudioEngineError.NONE, ''

    # PortAudio error codes (PaErrorCode in portaudio.h) that pick a
    # category on their own. Every other code falls through to
    # HOSTAPI_FAILURE, which the open loop treats as "try the next rate":
    # that covers paInvalidChannelCount (-9998), paSampleFormatNotSupported
    # (-9994) and paBadIODeviceCombination (-9993).
    _PA_CODE_KINDS = {
        -9997: AudioEngineError.UNSUPPORTED_RATE,   # paInvalidSampleRate
        -9999: AudioEngineError.DEVICE_BUSY,        # paUnanticipatedHostError
        -9985: AudioEngineError.DEVICE_BUSY,        # paDeviceUnavailable
        -9996: AudioEngineError.NO_DEVICE,          # paInvalidDevice
    }

    @staticmethod
    def _classify_error(exc: BaseException) -> AudioEngineError:
        msg = str(exc).lower()
        # PortAudio numeric codes when present.
        code = getattr(exc, 'args', None)
        text = msg
        if isinstance(code, tuple) and code:
            text = ' '.join(str(c).lower() for c in code)
        # Whole-number matches only, so -99970 is not read as -9997.
        codes = {int(m) for m in re.findall(r'(?<!\d)-\d+(?!\d)', text)}
        kinds = {AudioEngine._PA_CODE_KINDS[c] for c in codes
                 if c in AudioEngine._PA_CODE_KINDS}
        if ('invalid sample rate' in text
                or AudioEngineError.UNSUPPORTED_RATE in kinds):
            return AudioEngineError.UNSUPPORTED_RATE
        if (any(s in text for s in (
                'device unavailable', 'busy', 'unanticipated host error',
                '0xaa', 'wdmsyncioctl'))
                or AudioEngineError.DEVICE_BUSY in kinds):
            return AudioEngineError.DEVICE_BUSY
        if (any(s in text for s in (
                'device not found', 'no device', 'invalid device',
                'querying device -1'))
                or AudioEngineError.NO_DEVICE in kinds):
            return AudioEngineError.NO_DEVICE
        # The bare '-1' code: match it as a standalone token only, so
        # '-10', '-12', '-1000' don't get misrouted here (they belong in
        # HOSTAPI_FAILURE below).
        if re.search(r'(?:^|\D)-1(?!\d)', text):
            return AudioEngineError.NO_DEVICE
        return AudioEngineError.HOSTAPI_FAILURE

    # ---- input callback (producer) ----------------------------------------
    def _make_callback(self, samplerate: int, ring: Optional[_InputRing] = None):
        """Build a PortAudio input callback bound to one stream's ring.

        The body lives in :meth:`_on_input` (a real method) so tests can feed
        synthetic frames with no device. ``samplerate`` is carried by the ring;
        the argument is kept for the call sites that name the rate.
        """
        engine = self
        bound = ring if ring is not None else _InputRing(
            DEFAULT_BLOCK_SIZE, DEFAULT_HOP_SIZE, samplerate)

        def cb(indata, frames, ti, st):
            engine._on_input(indata, frames, ti, st, bound)
        return cb

    def _on_input(self, indata, frames, ti, status, ring=None):
        """PortAudio input hot path: count xruns, feed the deck tap, append to
        the ring, wake the analysis consumer. Never raises; never runs YIN
        unless the ring is inline (no analysis thread).

        ``ring`` defaults to the installed one (test feeds). The deck tap sits
        before everything else and ignores the emissions pause, so an armed
        take captures every mic frame even across an A4-remap pause.
        """
        engine = self
        st = status
        try:
            if st is not None:
                # Counter increments under the lock: += is GIL-atomic against
                # torn values but can drop increments against a concurrent
                # get_diagnostics(). Held for microseconds.
                if getattr(st, 'input_overflow', False):
                    with engine._lock:
                        engine._overflow_count += 1
                if getattr(st, 'input_underflow', False):
                    with engine._lock:
                        engine._underflow_count += 1
            mono = indata[:, 0]
            if engine._deck_armed:
                with engine._lock:
                    # ---- deck input-recording tap -----------------------
                    # Bounded prealloc sink + slice-assign: no hot-path alloc.
                    # Overflow clamps at cap, marks the take full, and
                    # auto-disarms — the deck pump() reads
                    # is_input_recording()==False as "cap hit".
                    if engine._deck_armed and engine._deck_sink is not None:
                        _dh = engine._deck_head
                        _cap = engine._deck_sink.size
                        if _dh >= _cap:
                            engine._deck_full = True
                            engine._deck_armed = False
                        else:
                            _dn = mono.shape[0]
                            _dend = _dh + _dn
                            if _dend <= _cap:
                                engine._deck_sink[_dh:_dend] = mono
                                engine._deck_head = _dend
                            else:
                                engine._deck_sink[_dh:_cap] = mono[:_cap - _dh]
                                engine._deck_head = _cap
                                engine._deck_full = True
                                engine._deck_armed = False
            if ring is None:
                ring = engine._ring
            ring.push(mono)
            if not ring.inline:
                engine._wake.set()
            elif ring is engine._ring:
                engine.process_pending()
        except Exception:
            # The audio callback must never raise back into PortAudio.
            pass

    def feed_input_frames(self, mono) -> None:
        """Test/diagnostic entry: feed a 1-D float32 mono block straight into
        the input path, as if PortAudio delivered it, and analyse it before
        returning.

        Shapes the block to the (frames, 1) ``indata`` the callback expects
        and uses a null status. No device needed. Fed a contiguous float32 1-D
        array the copy path allocates nothing, so it is safe in the no-alloc
        gate loop.
        """
        arr = np.asarray(mono, dtype=np.float32).reshape(-1, 1)
        self._on_input(arr, arr.shape[0], None, None)
        self.process_pending()

    # ---- analysis (consumer) ----------------------------------------------
    def process_pending(self) -> int:
        """Analyse every block the input has delivered and not yet analysed,
        emitting the readings each one confirms. Returns the hops analysed.

        This is the analysis thread's whole job, and the synchronous entry for
        tests and the headless harness. Safe from any thread: consumers take
        _analysis_lock; the producer never does.
        """
        hops = 0
        with self._analysis_lock:
            ring = self._ring
            while True:
                end = ring.pop_mark()
                if end is None:
                    break
                with self._lock:
                    intact = ring.read_window(end)
                    self._buf = ring.window
                    paused = self._emissions_paused
                    a4 = self.a4
                    a4_gen = self.a4_generation
                    serial = self._reset_serial
                    mode = self.filter_mode
                if not intact:
                    ring.dropped_hops += 1
                    continue
                if paused:
                    continue
                try:
                    self._analyse_hop(ring, a4, a4_gen, serial, mode)
                except Exception:
                    # A detector bug must not take the consumer down; count it
                    # so it stays visible.
                    self._analysis_errors += 1
                hops += 1
        return hops

    def _analyse_hop(self, ring: _InputRing, a4: float, a4_gen: int,
                     serial: int, mode: str) -> None:
        det = self._detector
        if serial != self._detector_serial or a4_gen != self._detector_a4_generation:
            # A reset request, a mode change or a new A4: nothing the detector
            # holds (vote window, lock, pending readings graded against the
            # old A4) may survive into the new state.
            det.set_mode(mode)
            det.midi_min, det.midi_max = self._midi_min, self._midi_max
            self._detector_serial = serial
            self._detector_a4_generation = a4_gen
        sr = ring.samplerate or int(self.samplerate)

        def coordinate(midi: Optional[int]) -> frozenset:
            # The unison test compares mic levels across the duck, so it needs
            # the newest audio only: the full window still holds ~370 ms from
            # before any gain change. Without a duck consumer the duck changes
            # nothing audible and proves nothing, so no level is passed.
            level = None
            if midi is not None and self._duck_consumer is not None:
                tail = ring.window[-ring.hop:]
                level = math.sqrt(float(np.dot(tail, tail)) / tail.size)
            return self.coordination_step(midi, level)

        events = det.process(ring.window, sr, a4, a4_gen, self.mic_gain,
                             coordinate)
        # Diagnostics: bare float stores are GIL-atomic, so get_diagnostics()
        # sees the old or the new value, never a torn one.
        self.last_rms_db = det.rms_db
        self.last_ap = det.ap
        self.last_freq = det.freq
        with self._lock:
            self.last_locked_midi = det.shown_midi
        for ev in events:
            self._emit(ev)

    def _emit(self, ev: NoteEvent) -> None:
        if self._emissions_paused or self.signals is None:
            return
        try:
            sig = getattr(self.signals, 'note_event', None)
            if sig is not None:
                sig.emit(ev)
            self.signals.note_detected.emit(int(ev.midi), float(ev.freq),
                                            float(ev.cents))
        except Exception:
            pass

    # ---- analysis thread ---------------------------------------------------
    def _start_analysis_thread(self) -> None:
        """Start the consumer thread for the installed ring (idempotent)."""
        t = self._analysis_thread
        if t is not None and t.is_alive():
            return
        stop = threading.Event()
        wake = self._wake
        # The thread holds the engine weakly, so an engine dropped without
        # stop() (tests) is collected and its thread exits on the next wake.
        ref = weakref.ref(self)

        def run() -> None:
            while not stop.is_set():
                # The timeout only bounds how long a dropped engine keeps its
                # thread; real wakes come from the callback once per hop.
                wake.wait(0.5)
                wake.clear()
                if stop.is_set():
                    break
                eng = ref()
                if eng is None:
                    break
                eng.process_pending()
                del eng

        t = threading.Thread(target=run, name='input-analysis', daemon=True)
        self._analysis_stop = stop
        self._analysis_thread = t
        t.start()

    def _stop_analysis_thread(self) -> None:
        t, stop = self._analysis_thread, self._analysis_stop
        self._analysis_thread = None
        self._analysis_stop = None
        if t is None or stop is None:
            return
        stop.set()
        self._wake.set()
        if t is not threading.current_thread():
            t.join(timeout=1.0)

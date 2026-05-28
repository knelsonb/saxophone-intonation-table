#!/usr/bin/env python3
"""
Saxophon-Intonationsanalysator – GUI
=====================================
Abhaengigkeiten (Ubuntu / venv):
    pip install PyQt6 numpy sounddevice reportlab

Starten:
    python3 sax_intonation_gui.py
"""

import os
import sys
import math
import threading
import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QSizePolicy, QFileDialog, QMessageBox,
    QAbstractItemView, QGroupBox, QSplitter,
    QDialog, QLineEdit, QDialogButtonBox, QFormLayout,
    QStyledItemDelegate, QMenu, QCheckBox, QSpinBox, QToolButton,
    QTabWidget, QTabBar, QSlider,
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QRectF, QPointF, QLocale, QByteArray,
)
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QIcon, QGuiApplication

from sax_intonation_log import MeasurementLog
from sax_intonation_chart import render_intonation_chart
import sax_instruments
from sax_instruments import (
    families as instrument_families,
    instruments_in,
    transp_map as build_transp_map,
    display_name as instrument_display_name,
    family_of as instrument_family_of,
    register_custom,
)
import sax_config
from sax_i18n import STRINGS
from sax_session_state import SessionStateController
from sax_export import ExportController
from sax_table import TableController
from sax_instrument_controller import InstrumentController
from sax_device import DeviceController

# Sprint 2: the metronome controller lives in its own module (sax_metronome.py,
# MetronomeController). It may not exist yet while Sprint 2 is in flight — the
# METRO panel is built inert and guards every controller call, so the GUI
# launches and the panel renders whether or not the controller has landed.
try:
    from sax_metronome import MetronomeController as _MetronomeController
except Exception:
    _MetronomeController = None

# Sprint 3: drone (TSF) + pitch pipes (numpy sine) live in their own modules.
# Same inert-until-landed discipline as the metronome. The voice catalog is the
# authoritative source when present; a 5-preset fallback keeps the TUNER preset
# row rendering before sax_drone.py lands (reconcile to the module catalog once
# it's in — do not let these drift from droneVoices.ts).
try:
    from sax_drone import (
        DroneController as _DroneController,
        DRONE_PRESETS as _DRONE_PRESETS,
        DRONE_FULL_GM as _DRONE_FULL_GM,
    )
except Exception:
    _DroneController = None
    # (id, label, GM program) — ids match sax_drone.DRONE_PRESETS (the
    # authoritative desktop catalog), not droneVoices.ts. Fallback only:
    # used when sax_drone hasn't imported, so nothing is persisted against it.
    _DRONE_PRESETS = [
        ('organ', 'Organ', 19), ('strings', 'Strings', 48),
        ('cello', 'Cello', 42), ('tenorsax', 'Tenor Sax', 66),
        ('warmpad', 'Warm Pad', 89),
    ]
    _DRONE_FULL_GM = []
try:
    from sax_pitch_pipes import PitchPipesController as _PitchPipesController
except Exception:
    _PitchPipesController = None

# Sprint 4: the tape deck (record mic → single take → playback/export) lives in
# sax_deck.py (DeckController state machine + DeckPlaybackSource + WAV io). Same
# inert-until-landed discipline as metro/drone — the DECK tab builds and renders
# its transport, and every controller call is guarded, whether or not sax_deck
# has landed yet. The GUI-facing contract (channel-firmed): DeckController(
# mixer, samplerate, *, engine, scratch_dir, on_state_changed); .state ∈
# {idle, recording, have_take, playing} (hyphen/underscore tolerated);
# start_record()->bool, stop(), play(), export(path)->bool.
try:
    from sax_deck import DeckController as _DeckController
except Exception:
    _DeckController = None

APP_NAME = 'Intonation Analyzer'
APP_VERSION = '0.9.0'

# v0.5.4: AudioEngine + pitch detection + filter presets live in their own
# module so the engine has a state machine, host-API fallback chain, and
# hot-plug poller without dragging the GUI through every test path. The
# names we re-export below keep the rest of this file's imports stable.
from sax_audio_engine import (
    AudioEngine,
    AudioEngineState,
    AudioEngineError,
    AudioEngineDiagnostics,
    DeviceInfo,
    DeviceSelection,
    FILTER_PRESETS as _FILTER_PRESETS_EXT,
    FILTER_MODE_DEFAULT as _FILTER_MODE_DEFAULT_EXT,
    HOP_MS as _HOP_MS_EXT,
    BLOCK_MS as _BLOCK_MS_EXT,
    MIN_FREQ as _MIN_FREQ_EXT,
    MAX_FREQ as _MAX_FREQ_EXT,
    A4_DEFAULT as _A4_DEFAULT_EXT,
    DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE_EXT,
    DEFAULT_HOP_SIZE as _DEFAULT_HOP_SIZE_EXT,
    DEFAULT_BLOCK_SIZE as _DEFAULT_BLOCK_SIZE_EXT,
    SAMPLERATE_PREF_VALUES as _SAMPLERATE_PREF_VALUES_EXT,
    SAMPLERATE_CANDIDATES as _SAMPLERATE_CANDIDATES_EXT,
    cents_dev as _cents_dev_ext,
    query_input_devices,
    VENDOR_REGEX,
)


# =============================================================================
# Konstanten & Musik-Logik
# =============================================================================
# Audio constants live in sax_audio_engine. The aliases below preserve
# the existing call sites in this file (spectrum widget, diagnostics
# panel, etc.) without rewriting every line. SAMPLE_RATE / HOP_SIZE /
# BLOCK_SIZE are the *defaults* — the live engine may negotiate a
# different sample rate at startup and rescale block sizes accordingly.
SAMPLE_RATE   = _DEFAULT_SAMPLE_RATE_EXT
HOP_SIZE      = _DEFAULT_HOP_SIZE_EXT
BLOCK_SIZE    = _DEFAULT_BLOCK_SIZE_EXT
MIN_FREQ      = _MIN_FREQ_EXT
MAX_FREQ      = _MAX_FREQ_EXT
A4_DEFAULT    = _A4_DEFAULT_EXT
HOP_MS        = _HOP_MS_EXT
BLOCK_MS      = _BLOCK_MS_EXT
_FILTER_PRESETS = _FILTER_PRESETS_EXT
FILTER_MODE_DEFAULT = _FILTER_MODE_DEFAULT_EXT
SAMPLERATE_PREF_VALUES = _SAMPLERATE_PREF_VALUES_EXT
SAMPLERATE_CANDIDATES = _SAMPLERATE_CANDIDATES_EXT

# v0.5.6: frequency-adaptive cent display precision.
# Parabolic interpolation in YIN pins tau to about 0.1 samples; one cent
# corresponds to tau * (ln(2)/1200) ~= tau * 5.78e-4 samples. The minimum
# resolvable cent step at frequency f and sample rate sr is therefore
# approximately 0.1 / (tau * 5.78e-4) = 173 * f / sr. We snap the
# displayed precision to one of {tenths, halves, wholes} so the readout
# never claims more resolution than the measurement actually delivers.
CENT_PREC_TENTHS_MAX = 0.3
CENT_PREC_HALVES_MAX = 0.7


def cent_precision_floor(freq_hz: float, sample_rate: int) -> float:
    """Return the minimum resolvable cent step at ``freq_hz`` given
    ``sample_rate``. See the module-top derivation."""
    sr = float(sample_rate) if sample_rate else float(_DEFAULT_SAMPLE_RATE_EXT)
    if sr <= 0 or freq_hz <= 0:
        return CENT_PREC_TENTHS_MAX
    return 173.0 * float(freq_hz) / sr


def format_cents(value_cents: float, freq_hz: float,
                 sample_rate: int) -> str:
    """Format a cent value at the precision the measurement supports.

    Always emits a sign so neutral readouts read "+0" / "+0.0" rather
    than the visually-jumpy bare "0". Negative-zero floats coerce to
    "+0..." via the explicit ``>= 0`` test.

    v0.5.7.2: guard against non-finite inputs. NoteStats.mean can be
    NaN after a reset race (empty vals → np.mean), and int(round(NaN))
    raises ValueError. Returning the canonical "–" placeholder matches
    the other no-data cells in the matrix paint path.
    """
    if (not math.isfinite(value_cents) or not math.isfinite(freq_hz)
            or freq_hz <= 0):
        return '–'
    floor_ct = cent_precision_floor(freq_hz, sample_rate)
    if floor_ct <= CENT_PREC_TENTHS_MAX:
        snapped = float(value_cents)
        sign = '+' if snapped >= 0 else '-'
        return f"{sign}{abs(snapped):.1f}"
    if floor_ct <= CENT_PREC_HALVES_MAX:
        snapped = round(float(value_cents) * 2.0) / 2.0
        sign = '+' if snapped >= 0 else '-'
        return f"{sign}{abs(snapped):.1f}"
    snapped = round(float(value_cents))
    sign = '+' if snapped >= 0 else '-'
    return f"{sign}{abs(int(snapped))}"

CHROMA = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F',
          'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

TRANSP     = {'eb': 3, 'bb': 2, 'c': 0}
# MIDI-Bereiche pro Saxophon-Typ (gegriffene Töne, klingende Noten)
# Bass-Sax (Bb):  klingt Bb0 (22) – F#3 (54)   → gegriffen A1–E4  (21–52)
# Bariton (Eb):   klingt Db2 (37) – Ab4 (68)   → gegriffen Bb2–F5 (46–65 + Höhe)
# Tenor (Bb):     klingt Ab2 (44) – Eb5 (75)   → gegriffen G3–D6  (55–74)
# Alt (Eb):       klingt Db3 (49) – Ab5 (80)   → gegriffen Bb3–F6 (58–77)
# Sopran (Bb):    klingt Ab3 (56) – Eb6 (87)   → gegriffen G4–D7  (67–86)
SAX_MIDI   = range(21, 109)  # v0.6: widened from 21..91 to 21..108 to cover
                             # the full catalog (piccolo, recorder, piano,
                             # banjo upper registers).  Must mirror
                             # AudioEngine._midi_max in sax_audio_engine.py.

# Transposition map is now derived from sax_instruments.transp_map(), which
# includes the original six saxophone/C-instrument keys plus everything else
# in the catalog. Rebuilt on demand whenever a custom instrument is added.
TRANSP_MAP: dict[str, int] = build_transp_map()


def _rebuild_transp_map() -> None:
    """Refresh TRANSP_MAP after a custom instrument is registered."""
    global TRANSP_MAP
    TRANSP_MAP = build_transp_map()


def freq_to_midi(f, a4=None):
    ref = a4 if a4 is not None else A4_DEFAULT
    return 69.0 + 12.0 * math.log2(f / ref)

def midi_note_name(m):
    return f"{CHROMA[m % 12]}{m // 12 - 1}"


# Reverse of midi_note_name: accept "G3", "F#5", "Bb2", "B♭2", "g3", etc.
# Returns the integer MIDI number or None if the string can't be parsed.
# Used by the range editor's note-name input field so Frodo doesn't have to
# think in MIDI integers.
_NOTE_PITCH_CLASSES = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}


def note_name_to_midi(text: str) -> int | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Normalise unicode accidentals to ASCII.
    s = s.replace('♯', '#').replace('♭', 'b')
    # Letter (case-insensitive).
    letter = s[0].upper()
    if letter not in _NOTE_PITCH_CLASSES:
        return None
    pc = _NOTE_PITCH_CLASSES[letter]
    rest = s[1:]
    # Optional accidental.
    accidental = 0
    if rest and rest[0] == '#':
        accidental = 1
        rest = rest[1:]
    elif rest and rest[0] == 'b':
        accidental = -1
        rest = rest[1:]
    # Octave digit(s) — allow negative octave like "C-1".
    if not rest:
        return None
    try:
        octave = int(rest)
    except ValueError:
        return None
    midi = (octave + 1) * 12 + pc + accidental
    if midi < 0 or midi > 127:
        return None
    return midi

def cents_dev(f, a4=None):
    mf = freq_to_midi(f, a4)
    mr = round(mf)
    return mr, (mf - mr) * 100.0


# =============================================================================
# YIN Pitch-Detektion — moved to sax_audio_engine.yin_pitch in v0.5.4.
# Local thin wrapper retained for any in-process call sites; the engine
# uses its own implementation directly.
# =============================================================================
from sax_audio_engine import yin_pitch as _yin_pitch_ext


def yin_pitch(sig, sr=SAMPLE_RATE, fmin=MIN_FREQ, fmax=MAX_FREQ,
              thr=None):
    return _yin_pitch_ext(sig, sr, fmin, fmax,
                          thr if thr is not None else 0.12)


# (legacy YIN body removed in v0.5.4 — see sax_audio_engine.yin_pitch)


# =============================================================================
# Messdaten
# =============================================================================
class NoteStats:
    """Welford's online algorithm for mean and population variance.

    Replaces the v0.5.7 list-based accumulator.  O(1) time and O(1) memory
    per note regardless of how many cents readings accumulate.  Population
    std (ddof=0) is preserved to match np.std default and the _agg_stats
    contract locked by Phase-0 tests.

    Thread-safety note: add() and the property readers are both called on
    the Qt main thread (add via the engine.note_detected slot; readers via
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

    def add(self, c: float) -> None:
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


# =============================================================================
# Audio-Engine — moved to sax_audio_engine.AudioEngine in v0.5.4.
# =============================================================================


# =============================================================================
# Tuner-Widget
# =============================================================================
class TunerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.cents  = 0.0
        self.freq   = 0.0
        self.note   = ''
        self.active = False
        self._decay = 0.0
        # v0.5.6: live engine sample rate, set by MainWindow whenever the
        # engine successfully opens. Used to render cents at a precision
        # the measurement can actually support.
        self.sample_rate = _DEFAULT_SAMPLE_RATE_EXT
        t = QTimer(self)
        t.timeout.connect(self._fade)
        t.start(80)
        # Restored to 260 in v0.5.3 after feedback that the 180px tuner
        # cramped the needle / note / cents readout once the diagnostics
        # panel sat beneath it. The spectrum analyzer below has its own
        # min height and the vertical layout pushes diagnostics down the
        # left pane rather than compressing the tuner.
        self.setMinimumHeight(260)

    def set_note(self, note, freq, cents):
        self.note, self.freq, self.cents = note, freq, cents
        self.active = True
        self._decay = 1.0
        self.update()

    def _fade(self):
        if self.active:
            self._decay = max(0.0, self._decay - 0.04)
            if self._decay == 0.0:
                self.active = False
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(18, 18, 24))
        alpha = int(255 * max(0.15, self._decay))

        # Tonname
        if self.note:
            p.setFont(QFont('Monospace', 72, QFont.Weight.Bold))
            p.setPen(QColor(220, 220, 255, alpha))
            p.drawText(QRectF(0, 8, W, H * 0.45),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       self.note)

        # Frequenz
        if self.freq > 0:
            p.setFont(QFont('Monospace', 16))
            p.setPen(QColor(140, 140, 180, alpha))
            p.drawText(QRectF(0, H * 0.44, W, 30),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{self.freq:.1f} Hz")

        # Skala
        sy, sh = H * 0.60, 28
        sw, sx = W * 0.78, (W - W * 0.78) / 2
        mx = 50.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 55))
        p.drawRoundedRect(QRectF(sx, sy, sw, sh), 8, 8)

        zw = sw * (10.0 / mx) / 2
        p.setBrush(QColor(30, 90, 50, 140))
        p.drawRoundedRect(QRectF(sx + sw/2 - zw, sy, zw*2, sh), 6, 6)

        if self.active:
            norm = max(-1.0, min(1.0, self.cents / mx))
            nx   = sx + sw/2 + norm * sw/2
            if   abs(self.cents) <= 5:  nc = QColor(60,  220, 100, alpha)
            elif abs(self.cents) <= 15: nc = QColor(255, 200, 40,  alpha)
            else:                       nc = QColor(240, 70,  70,  alpha)
            p.setBrush(nc)
            p.drawRoundedRect(QRectF(nx - 4, sy - 6, 8, sh + 12), 4, 4)

        p.setPen(QColor(90, 90, 110))
        p.setFont(QFont('Monospace', 9))
        for ct in [-50, -25, 0, 25, 50]:
            nx = sx + sw/2 + (ct / mx) * sw/2
            p.drawLine(QPointF(nx, sy - 2), QPointF(nx, sy + sh + 2))
            p.drawText(QRectF(nx - 18, sy + sh + 4, 36, 16),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{'+' if ct>0 else ''}{ct}")

        if self.active:
            p.setFont(QFont('Monospace', 32, QFont.Weight.Bold))
            if   abs(self.cents) <= 5:  cc = QColor(60,  220, 100, alpha)
            elif abs(self.cents) <= 15: cc = QColor(255, 200, 40,  alpha)
            else:                       cc = QColor(240, 70,  70,  alpha)
            p.setPen(cc)
            p.drawText(QRectF(0, sy + sh + 26, W, 55),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{format_cents(self.cents, self.freq, self.sample_rate)} ct")
        p.end()


# =============================================================================
# Spektrumanalysator-Widget
# =============================================================================
class SpectrumAnalyzerWidget(QWidget):
    """Live spectrum analyzer (FFT magnitude vs. log frequency).

    Pulls slices of `AudioEngine._buf` on its own QTimer tick (~30 fps),
    runs a Hann-windowed rfft, bins magnitudes into log-spaced frequency
    buckets between 27 Hz and 4000 Hz, and paints them as a filled curve
    in dBFS. A peak-hold envelope decays at a fixed rate so transient
    spikes stay readable for a few hundred ms before sliding back down.

    The audio callback is never touched here — the timer reads `_buf`
    cooperatively and tolerates a hop of staleness. Display only, never
    gates or feeds back into pitch detection.
    """

    F_LO = 27.0
    F_HI = 4000.0
    DB_FLOOR = -80.0
    DB_CEIL  = -10.0
    N_BINS = 192          # horizontal resolution (frequency buckets)
    REFRESH_MS = 33       # ~30 fps
    # Peak hold decay in dB per second. 0.3 s "hang" then decay; we
    # approximate the hang implicitly by capping decay to roughly 100
    # dB/s, which means a 70 dB spike takes ~0.7 s to drop fully back
    # to the live curve. Plenty visible without smearing.
    PEAK_DECAY_DB_PER_S = 60.0

    def __init__(self, engine: 'AudioEngine | None'):
        super().__init__()
        self._engine = engine
        # Log-spaced bin edges used to bucket the rfft magnitudes.
        self._edges = np.logspace(math.log10(self.F_LO), math.log10(self.F_HI),
                                   self.N_BINS + 1)
        # Bin center frequencies, used for the X axis mapping.
        self._centers = np.sqrt(self._edges[:-1] * self._edges[1:])
        # Cached Hann window + bucket map. The engine may rebind _buf
        # to a different size when it negotiates a non-44100 sample
        # rate; _rebuild_for() regenerates the cache on the fly.
        self._window = np.hanning(BLOCK_SIZE).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)
        self._bucket = np.searchsorted(self._edges, fft_freqs) - 1
        self._bucket = np.clip(self._bucket, -1, self.N_BINS - 1)
        self._win_norm = float(np.sum(self._window) * 0.5)
        # Live curve and peak-hold envelope, both in dBFS.
        self._levels = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        self._peaks  = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        # Powers-of-two grid lines in Hz, labeled along the bottom.
        self._grid_hz = [32, 64, 128, 256, 512, 1024, 2048, 4096]

        self.setMinimumHeight(150)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.REFRESH_MS)

    def _rebuild_for(self, block_size: int, samplerate: int) -> None:
        """Regenerate window + bucket map for a new buffer size / rate.

        Called by _tick when the engine's buffer size shifts (sample-
        rate renegotiation at startup or device change)."""
        block_size = max(8, int(block_size))
        samplerate = max(1, int(samplerate))
        self._window = np.hanning(block_size).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(block_size, d=1.0 / samplerate)
        self._bucket = np.searchsorted(self._edges, fft_freqs) - 1
        self._bucket = np.clip(self._bucket, -1, self.N_BINS - 1)
        self._win_norm = float(np.sum(self._window) * 0.5)

    def _tick(self) -> None:
        if not AUDIO_OK or self._engine is None:
            return
        # Pull a snapshot copy under the engine's lock so the audio
        # callback can't half-roll the buffer while we FFT it. Fixes
        # the v0.5.3 spectrum-widget data race documented in wave 1.
        try:
            buf = self._engine.get_buf_snapshot()
        except Exception:
            return
        if buf is None or len(buf) < 8:
            return
        # Engine may have reallocated _buf at a different sample rate;
        # if our cached window/bucket maps no longer match, rebuild.
        if buf.size != self._window.size:
            self._rebuild_for(buf.size,
                              int(getattr(self._engine, 'samplerate',
                                           SAMPLE_RATE)))
        # Bucket the rfft magnitudes into log-spaced bins, take the max
        # per bucket so narrow spikes survive the downsampling.
        windowed = buf * self._window
        mag = np.abs(np.fft.rfft(windowed)) / self._win_norm
        col = np.full(self.N_BINS, 1e-12, dtype=np.float32)
        valid = self._bucket >= 0
        # np.maximum.at handles repeated indices; per-bucket peak instead
        # of sum keeps the curve from drifting upward at the low end
        # where many fft bins fall into a single log bucket.
        np.maximum.at(col, self._bucket[valid], mag[valid])
        db = 20.0 * np.log10(col + 1e-12)
        db = np.clip(db, self.DB_FLOOR, 0.0)
        # Light temporal smoothing on the live curve — single-pole IIR
        # with a coefficient picked to settle in ~50 ms at 30 fps. The
        # peak-hold envelope is what makes spikes legible; the live
        # curve just needs to not strobe.
        self._levels = 0.6 * self._levels + 0.4 * db
        # Peak-hold: instantaneous capture, linear decay between ticks.
        decay = self.PEAK_DECAY_DB_PER_S * (self.REFRESH_MS / 1000.0)
        self._peaks = np.maximum(self._peaks - decay, self._levels)
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(10, 10, 16))
        if not AUDIO_OK or self._engine is None:
            p.setPen(QColor(140, 140, 160))
            p.setFont(QFont('Monospace', 11))
            p.drawText(QRectF(0, 0, W, H),
                       Qt.AlignmentFlag.AlignCenter,
                       'audio disabled')
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Plot area leaves room for axis labels along the bottom.
        margin_l = 4
        margin_r = 4
        margin_t = 4
        margin_b = 14
        x0 = margin_l
        y0 = margin_t
        x1 = max(W - margin_r, x0 + 1)
        y1 = max(H - margin_b, y0 + 1)
        plot_w = x1 - x0
        plot_h = y1 - y0

        log_lo = math.log10(self.F_LO)
        log_hi = math.log10(self.F_HI)
        log_span = log_hi - log_lo

        def x_of(freq: float) -> float:
            return x0 + (math.log10(freq) - log_lo) / log_span * plot_w

        def y_of(db: float) -> float:
            # DB_CEIL maps to top (y0), DB_FLOOR to bottom (y1).
            t = (db - self.DB_FLOOR) / (self.DB_CEIL - self.DB_FLOOR)
            t = max(0.0, min(1.0, t))
            return y1 - t * plot_h

        # Vertical gridlines at powers of two, with thin labels.
        p.setFont(QFont('Monospace', 8))
        grid_pen = QPen(QColor(40, 44, 60))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            p.drawLine(QPointF(gx, y0), QPointF(gx, y1))
        p.setPen(QColor(120, 130, 150))
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            label = f'{hz}' if hz < 1000 else f'{hz // 1000}k'
            p.drawText(QRectF(gx - 20, y1 + 1, 40, margin_b),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       label)

        # Horizontal dB gridline at -40 dBFS for visual reference.
        p.setPen(QPen(QColor(40, 44, 60), 1, Qt.PenStyle.DashLine))
        y_mid = y_of(-40.0)
        p.drawLine(QPointF(x0, y_mid), QPointF(x1, y_mid))

        # Live curve as a filled polygon under the line.
        from PyQt6.QtGui import QPolygonF
        poly = QPolygonF()
        poly.append(QPointF(x_of(self._centers[0]), y1))
        for i, f in enumerate(self._centers):
            poly.append(QPointF(x_of(float(f)), y_of(float(self._levels[i]))))
        poly.append(QPointF(x_of(self._centers[-1]), y1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(58, 156, 220, 110))
        p.drawPolygon(poly)

        # Bright line on top of the fill.
        line_pen = QPen(QColor(120, 200, 255))
        line_pen.setWidth(2)
        p.setPen(line_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._levels[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        # Peak-hold envelope as a thinner, paler line on top.
        peak_pen = QPen(QColor(240, 220, 140))
        peak_pen.setWidth(1)
        p.setPen(peak_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._peaks[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        p.end()


# Backwards-compatible alias. v0.5.2 referenced SpectrogramWidget; the
# rest of the file still uses the old name where harmless, but new
# constructions in the splitter use the analyzer class directly.
SpectrogramWidget = SpectrumAnalyzerWidget


# =============================================================================
# Diagnose-Panel
# =============================================================================
class DataPanelWidget(QWidget):
    """Read-only key:value readout of audio + engine state.

    The values come straight from constants and from the AudioEngine's
    last_* attributes, refreshed every 250 ms. Diagnostic only; never
    influences playback or detection."""

    REFRESH_MS = 250

    def __init__(self, engine: 'AudioEngine | None',
                 t_func, get_notes_count, get_cfg):
        super().__init__()
        self._engine = engine
        self._t = t_func
        self._get_notes_count = get_notes_count
        self._get_cfg = get_cfg
        self._device_label = self._resolve_device_label()

        from PyQt6.QtWidgets import QGridLayout
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 6, 8, 6)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(2)
        self._rows: dict[str, QLabel] = {}
        # Order matters — top-to-bottom layout of the panel.
        self._row_keys = [
            'data_device', 'data_rate_requested', 'data_rate_negotiated',
            'data_samplerate', 'data_block_hop',
            'data_blocksize', 'data_hopsize',
            'data_halfcent_floor',
            'data_freqrange', 'data_filter_mode', 'data_filter_params',
            'data_a4', 'data_rms', 'data_aperiodicity', 'data_freq',
            'data_midi', 'data_notes_count',
            'data_hotplug_poller',
        ]
        for r, key in enumerate(self._row_keys):
            k_lbl = QLabel(self._t(key) + ':')
            k_lbl.setStyleSheet('color:#8a8aa0;font-family:Monospace;font-size:11px;')
            v_lbl = QLabel('—')
            v_lbl.setStyleSheet('color:#d8d8ee;font-family:Monospace;font-size:11px;')
            v_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(k_lbl, r, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(v_lbl, r, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self._rows[key] = v_lbl
        self._grid.setColumnStretch(1, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self.REFRESH_MS)
        self._refresh_static()
        self._refresh()

    def retranslate(self, t_func) -> None:
        self._t = t_func
        # Re-label each row's key column. The grid stores key labels at
        # column 0 in the same order as self._row_keys.
        for r, key in enumerate(self._row_keys):
            item = self._grid.itemAtPosition(r, 0)
            if item is not None and item.widget() is not None:
                item.widget().setText(self._t(key) + ':')
        self._refresh_static()
        self._refresh()

    def _resolve_device_label(self) -> str:
        """Mirror what the engine reports if it's running. Falls back to
        the PortAudio default *only* through the safe probe — never the
        bare query_devices(kind='input') path that crashes when no
        input device exists (the v0.5.3 silent-crash bug)."""
        if not AUDIO_OK:
            return '—'
        eng = self._engine
        if eng is not None and getattr(eng, 'active_device', None) is not None:
            d = eng.active_device
            return f'{d.name}  [{d.host_api}]' if d.host_api else d.name
        return '—'

    def _refresh_static(self) -> None:
        """Update fields that only change on instrument/config changes."""
        self._rows['data_device'].setText(self._resolve_device_label())
        self._rows['data_freqrange'].setText(
            f'{MIN_FREQ:.1f} – {MAX_FREQ:.1f} Hz')

    def _refresh(self) -> None:
        if not AUDIO_OK or self._engine is None:
            for key in ('data_a4', 'data_rms', 'data_aperiodicity',
                        'data_freq', 'data_midi', 'data_filter_mode',
                        'data_filter_params'):
                self._rows[key].setText('—')
            for key in ('data_samplerate', 'data_blocksize', 'data_hopsize',
                        'data_rate_requested', 'data_rate_negotiated',
                        'data_block_hop', 'data_halfcent_floor'):
                self._rows[key].setText('—')
            self._rows['data_notes_count'].setText(
                str(int(self._get_notes_count())))
            self._rows['data_hotplug_poller'].setText('—')
            return
        # Pull a single atomic snapshot of the engine's diagnostic
        # scalars; avoids reading half-updated last_* values mid-callback.
        diag = self._engine.get_diagnostics()
        mode = self._engine.filter_mode
        params = _FILTER_PRESETS.get(mode, {})
        self._rows['data_device'].setText(self._resolve_device_label())
        sr = max(1, diag.samplerate)
        hop = max(1, diag.hop_size)
        hop_ms = 1000.0 * hop / sr
        self._rows['data_samplerate'].setText(f'{sr} Hz')
        self._rows['data_blocksize'].setText(f'{diag.block_size} samples')
        self._rows['data_hopsize'].setText(
            f'{hop} samples ({hop_ms:.1f} ms)')
        # v0.5.6 audio negotiation + precision rows.
        cfg = self._get_cfg() if self._get_cfg else None
        pref = (str(getattr(cfg, 'audio_samplerate_pref', 'auto'))
                if cfg else 'auto')
        if pref == 'auto':
            self._rows['data_rate_requested'].setText('Auto')
        else:
            self._rows['data_rate_requested'].setText(f'{pref} Hz')
        self._rows['data_rate_negotiated'].setText(f'{sr} Hz')
        block_ms = 1000.0 * diag.block_size / sr
        self._rows['data_block_hop'].setText(
            f'{diag.block_size} / {hop} ({block_ms:.0f} ms / {hop_ms:.0f} ms)')
        floor_a4 = cent_precision_floor(440.0, sr)
        self._rows['data_halfcent_floor'].setText(f'{floor_a4:.2f} ¢')
        self._rows['data_filter_mode'].setText(mode)
        if params:
            self._rows['data_filter_params'].setText(
                'win={window}  conf={confirm}  yin_thr={yin_thr}  '
                'rms_floor={rms_floor:g}  edge={edge_hops}'.format(**params))
        else:
            self._rows['data_filter_params'].setText('—')
        self._rows['data_a4'].setText(f'{self._engine.a4:.2f} Hz')
        self._rows['data_rms'].setText(f'{diag.rms_db:+.1f} dBFS')
        self._rows['data_aperiodicity'].setText(f'{diag.aperiodicity:.3f}')
        if diag.freq > 0:
            self._rows['data_freq'].setText(f'{diag.freq:.2f} Hz')
        else:
            self._rows['data_freq'].setText('—')
        m = diag.locked_midi
        if m is not None:
            self._rows['data_midi'].setText(
                f'{m}  ({midi_note_name(int(m))})')
        else:
            self._rows['data_midi'].setText('—')
        self._rows['data_notes_count'].setText(
            str(int(self._get_notes_count())))
        # Hot-plug poller status. The poller is a 1 Hz QTimer in
        # MainWindow that calls engine.refresh_devices(); the engine
        # stamps last_devices_refresh_at on every tick. Surfacing the
        # last-refresh time here lets the user confirm at a glance
        # that hot-plug detection is alive when a new interface fails
        # to recover.
        last = getattr(self._engine, 'last_devices_refresh_at', None)
        if last is None:
            self._rows['data_hotplug_poller'].setText(
                self._t('data_hotplug_value_never', interval=1))
        else:
            try:
                hhmmss = last.strftime('%H:%M:%S')
            except Exception:
                hhmmss = '—'
            self._rows['data_hotplug_poller'].setText(
                self._t('data_hotplug_value', interval=1, when=hhmmss))


# =============================================================================
# Delegate: grafischer Intonationsbalken in der Tabelle
# =============================================================================
class CentBarDelegate(QStyledItemDelegate):
    """Zeichnet einen zentrierten, farbcodierten Balken für Cent-Abweichungen.
    Cell payload is a dict {'cents': float, 'freq': float}; legacy bare
    floats are still accepted so older callers keep working."""

    MAX_CENT = 50.0   # ±50 ct = volle Balkenhälfte

    def __init__(self, parent=None, sample_rate_getter=None):
        super().__init__(parent)
        # v0.5.6: live sample rate for adaptive cent precision in the
        # printed value beside each bar. The bar geometry itself uses
        # the raw float so the visual position stays smooth.
        self._sr_get = sample_rate_getter or (
            lambda: _DEFAULT_SAMPLE_RATE_EXT)

    def paint(self, painter, option, index):
        raw = index.data(Qt.ItemDataRole.UserRole)
        freq = 0.0
        if isinstance(raw, dict):
            try:
                cents = float(raw.get('cents'))
                freq = float(raw.get('freq') or 0.0)
            except (TypeError, ValueError):
                super().paint(painter, option, index)
                return
        else:
            try:
                cents = float(raw)
            except (TypeError, ValueError):
                super().paint(painter, option, index)
                return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = option.rect
        # Hintergrund (Auswahl berücksichtigen)
        from PyQt6.QtWidgets import QStyle
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(r, QColor('#2d4a7a'))
        else:
            painter.fillRect(r, QColor(0, 0, 0, 0))

        # Farbe nach Betrag
        abw = abs(cents)
        if abw < 10:
            bar_col = QColor('#3a9e5f')   # grün
        elif abw < 20:
            bar_col = QColor('#c8a020')   # gelb
        else:
            bar_col = QColor('#c03030')   # rot

        # Dimensionen
        pad_x, pad_y = 8, 5
        w = r.width() - 2 * pad_x
        h = r.height() - 2 * pad_y
        cx = r.left() + pad_x + w // 2   # Mittellinie x

        # Hintergrundleiste
        bg_h = max(4, h // 3)
        bg_y = r.top() + pad_y + (h - bg_h) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(45, 45, 60))
        painter.drawRoundedRect(r.left() + pad_x, bg_y, w, bg_h, 3, 3)

        # Grüne Mitte-Zone (±5 ct)
        zone_w = max(2, int(w / 2 * 5.0 / self.MAX_CENT))
        painter.setBrush(QColor(30, 80, 40, 160))
        painter.drawRect(cx - zone_w, bg_y, zone_w * 2, bg_h)

        # Füllbalken
        norm = max(-1.0, min(1.0, cents / self.MAX_CENT))
        fill_w = max(2, int(abs(norm) * w / 2))
        bar_h = bg_h + 2
        bar_y = bg_y - 1
        painter.setBrush(bar_col)
        if cents >= 0:
            painter.drawRoundedRect(cx, bar_y, fill_w, bar_h, 2, 2)
        else:
            painter.drawRoundedRect(cx - fill_w, bar_y, fill_w, bar_h, 2, 2)

        # Mittellinie
        painter.setPen(QPen(QColor(180, 180, 200), 1))
        painter.drawLine(cx, bg_y - 2, cx, bg_y + bg_h + 2)

        # Cent-Wert als Text rechts — adaptive precision (v0.5.6)
        sr = self._sr_get() or _DEFAULT_SAMPLE_RATE_EXT
        txt = f"{format_cents(cents, freq, sr)} ct"
        painter.setPen(bar_col)
        painter.setFont(QFont('Monospace', 9))
        txt_rect = option.rect.adjusted(0, 0, -4, 0)
        painter.drawText(txt_rect,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         txt)

        painter.restore()

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        return sh.__class__(max(sh.width(), 160), max(sh.height(), 28))


# =============================================================================
# Delegate: matrix-mode cell with mean number + bar + std whiskers + live arrow
# =============================================================================
class MatrixCellDelegate(QStyledItemDelegate):
    """Paints each piano-roll cell with the same data the single-column
    table exposes per row, just stacked vertically:

    * Top strip: fingered note name on the left, sounding note name on
      the right (always shown for in-range cells so the user can read
      the cell's identity without consulting the row/column headers).
    * Mid strip: mean cents (color-coded by magnitude) and ±std as a
      smaller adjacent value.
    * Lower strip: horizontal scale centered on 0 ct with the filled
      bar from center to mean, plus ±1σ whiskers below.
    * Bottom-right corner: small N counter.

    Active cell (currently-played note within the last 1.5s) gets the
    blue background tint. Out-of-range cells render only the dimmed
    note label (so the player still knows what note that row+column
    represents) on a darker background.

    Data layout per cell (set on the QTableWidgetItem):
        ItemDataRole.UserRole → dict {
            'mean':          float | None,
            'std':           float | None,
            'n':             int,
            'in_range':      bool,
            'active':        bool,
            'fingered_name': str,
            'sounding_name': str,
        }
    """

    MAX_CENT = 50.0   # ±50 ct = bar saturated

    def __init__(self, parent=None, sample_rate_getter=None):
        super().__init__(parent)
        # v0.5.6: live sample rate for adaptive cent precision.
        self._sr_get = sample_rate_getter or (
            lambda: _DEFAULT_SAMPLE_RATE_EXT)

    def paint(self, painter, option, index):
        data = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            super().paint(painter, option, index)
            return

        r = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        in_range = data.get('in_range', True)
        active   = data.get('active', False)
        mean     = data.get('mean')
        std      = data.get('std') or 0.0
        n        = int(data.get('n') or 0)
        fingered = data.get('fingered_name', '')
        sounding = data.get('sounding_name', '')
        freq     = float(data.get('freq') or 0.0)
        sr       = self._sr_get() or _DEFAULT_SAMPLE_RATE_EXT

        if not in_range:
            # Out-of-range cells render with NO border and NO text — they
            # blend into the table background so the user only sees the
            # cells that correspond to physically playable notes. The
            # surrounding grid structure stays so column alignment is
            # preserved.
            painter.fillRect(r, QColor('#12121a'))
            painter.restore()
            return

        # In-range background — active blue tint or default panel color.
        if active:
            painter.fillRect(r, QColor('#2c5a8a'))
        else:
            painter.fillRect(r, QColor('#1a1a24'))

        # Subtle cell border — only on in-range cells, so the playable
        # area visually pops as a grid against the unbordered out-of-
        # range background.
        painter.setPen(QPen(QColor(55, 55, 75), 1))
        painter.drawRect(r.adjusted(0, 0, -1, -1))

        pad_x = 4
        # ----- Top strip: note names ----------------------------------
        top_h = 13
        top_y = r.top() + 1
        painter.setFont(QFont('Monospace', 7))
        painter.setPen(QColor(150, 150, 175))
        if fingered:
            painter.drawText(
                QRectF(r.left() + pad_x, top_y, r.width() / 2 - pad_x, top_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                fingered)
        if sounding and sounding != fingered:
            painter.setPen(QColor(120, 130, 160))
            painter.drawText(
                QRectF(r.left() + r.width() / 2, top_y,
                       r.width() / 2 - pad_x, top_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                sounding)

        # v0.5.7.3: belt-and-suspenders against NaN sneaking through
        # from NoteStats (despite the source-side snapshot). NaN in
        # `mean` makes the bar width / fill width NaN and Qt's
        # drawRoundedRect with a NaN width is undefined. Treat NaN/inf
        # the same as "no measurement yet" — show the seeded dot.
        if mean is None or not math.isfinite(float(mean)):
            # In-range but no measurement yet — show a centered dot to
            # acknowledge the seeded slot.
            painter.setPen(QColor(80, 80, 100))
            painter.setFont(QFont('Monospace', 10))
            painter.drawText(
                QRectF(r.left(), r.top() + top_h, r.width(), r.height() - top_h),
                Qt.AlignmentFlag.AlignCenter, '·')
            painter.restore()
            return

        # ----- Mid strip: mean (color) + ±std ------------------------
        col = (QColor('#3a9e5f') if abs(mean) <= 5 else
               QColor('#c8a020') if abs(mean) <= 12 else QColor('#c03030'))
        mid_h = 18
        mid_y = r.top() + top_h
        painter.setPen(col)
        painter.setFont(QFont('Monospace', 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(r.left() + pad_x, mid_y,
                   r.width() * 0.62 - pad_x, mid_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            format_cents(mean, freq, sr))
        if n > 1 and std > 0:
            painter.setPen(QColor(170, 170, 200))
            painter.setFont(QFont('Monospace', 8))
            # Strip sign from format_cents and prepend "±".
            std_txt = format_cents(std, freq, sr).lstrip('+-')
            painter.drawText(
                QRectF(r.left() + r.width() * 0.55, mid_y,
                       r.width() * 0.45 - pad_x, mid_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"±{std_txt}")

        # ----- Bar with whiskers --------------------------------------
        scale_w = r.width() - 2 * pad_x
        scale_h = 4
        cx = r.left() + r.width() / 2
        scale_y = r.top() + top_h + mid_h + 3
        # Background trough.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(45, 45, 60))
        painter.drawRoundedRect(QRectF(r.left() + pad_x, scale_y,
                                        scale_w, scale_h), 2, 2)
        # ±5 ct in-tune band.
        zone_w = max(2.0, scale_w / 2 * 5.0 / self.MAX_CENT)
        painter.setBrush(QColor(40, 110, 60, 160))
        painter.drawRect(QRectF(cx - zone_w, scale_y, zone_w * 2, scale_h))
        # Filled bar.
        norm = max(-1.0, min(1.0, mean / self.MAX_CENT))
        fill_w = abs(norm) * scale_w / 2
        painter.setBrush(col)
        if mean >= 0:
            painter.drawRoundedRect(QRectF(cx, scale_y - 1,
                                            fill_w, scale_h + 2), 2, 2)
        else:
            painter.drawRoundedRect(QRectF(cx - fill_w, scale_y - 1,
                                            fill_w, scale_h + 2), 2, 2)
        painter.setPen(QPen(QColor(200, 200, 220), 1))
        painter.drawLine(QPointF(cx, scale_y - 2),
                          QPointF(cx, scale_y + scale_h + 2))

        # Whiskers.
        if n > 1 and std > 0:
            std_lo = max(-1.0, min(1.0, (mean - std) / self.MAX_CENT))
            std_hi = max(-1.0, min(1.0, (mean + std) / self.MAX_CENT))
            x_lo = cx + std_lo * scale_w / 2
            x_hi = cx + std_hi * scale_w / 2
            w_y = scale_y + scale_h + 4
            painter.setPen(QPen(QColor(220, 220, 235, 200), 1.4))
            painter.drawLine(QPointF(x_lo, w_y), QPointF(x_hi, w_y))
            painter.drawLine(QPointF(x_lo, w_y - 2),
                              QPointF(x_lo, w_y + 2))
            painter.drawLine(QPointF(x_hi, w_y - 2),
                              QPointF(x_hi, w_y + 2))

        # N counter, bottom-right.
        painter.setPen(QColor(140, 140, 165))
        painter.setFont(QFont('Monospace', 7))
        painter.drawText(
            QRectF(r.right() - 28, r.bottom() - 12, 26, 10),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"n={n}")

        painter.restore()

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        # Need vertical room: top (13) + mid (18) + bar (4) + whiskers (8)
        # + N counter (8) = ~55px minimum.
        return sh.__class__(max(sh.width(), 110), max(sh.height(), 60))


# =============================================================================
# Audio-Eingang Chip + Banner + Picker (v0.5.4)
# =============================================================================
# Status-dot colour palette — matches the existing dark theme.
_AUDIO_DOT_COLORS = {
    AudioEngineState.RUNNING: '#2ecc71',
    AudioEngineState.OPENING: '#b7770d',
    AudioEngineState.ENUMERATING: '#b7770d',
    AudioEngineState.FAILED:  '#c0392b',
    AudioEngineState.STOPPED: '#666',
    AudioEngineState.INIT:    '#666',
}


class _StatusDot(QWidget):
    """8 px solid circle. Recoloured by the chip whenever engine state
    changes — gives the user a glanceable health indicator without
    needing to read text."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor('#666')
        self.setFixedSize(10, 10)

    def set_color(self, hex_color: str) -> None:
        c = QColor(hex_color)
        if c != self._color:
            self._color = c
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, self.width(), self.height())
        p.end()


class AudioChip(QPushButton):
    """Toolbar chip that shows current audio-input device + state.

    Clicking opens the picker. State updates are driven by the engine's
    ``state_changed`` signal — the chip never queries PortAudio
    directly, which keeps the GUI thread off the cold-init path
    documented in Legolas's perf memo."""

    def __init__(self, t_func):
        super().__init__()
        self._t = t_func
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet("""
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:4px 10px;font-size:12px;
                         text-align:left;}
            QPushButton:hover{background:#3d566e;}
            QPushButton:pressed{background:#2c3e50;}
        """)
        # Layout: dot · label · device name · chevron.
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        self._dot = _StatusDot(self)
        lay.addWidget(self._dot)
        # v0.6.2: explicit transparent backgrounds on the inner labels.
        # The MainWindow-level stylesheet sets `QWidget{background:#12121a}`
        # which is a type selector that matches QLabel too. Without a
        # local `background:transparent` override, the labels paint dark
        # navy rectangles on top of the chip's slate-blue button bg —
        # rendering as vertical color bands across the chip.
        self._label = QLabel(self._t('audio_chip_label'))
        self._label.setStyleSheet('color:#bdc3c7;font-size:10px;'
                                  'font-weight:bold;background:transparent;')
        lay.addWidget(self._label)
        self._name = QLabel(self._t('audio_chip_none'))
        self._name.setStyleSheet('color:#eee;font-size:12px;'
                                 'background:transparent;')
        lay.addWidget(self._name, 1)
        self._chevron = QLabel('▾')
        self._chevron.setStyleSheet('color:#888;font-size:12px;'
                                    'background:transparent;')
        lay.addWidget(self._chevron)

    def retranslate(self, t_func) -> None:
        self._t = t_func
        self._label.setText(self._t('audio_chip_label'))
        self.setToolTip(self._t('audio_chip_tip'))

    def update_from_state(self, state: AudioEngineState,
                          device_name: str, host_api: str,
                          samplerate: int) -> None:
        color = _AUDIO_DOT_COLORS.get(state, '#666')
        self._dot.set_color(color)
        if state == AudioEngineState.RUNNING and device_name:
            short = device_name if len(device_name) <= 24 else device_name[:23] + '…'
            suffix = ''
            # Only surface non-44.1k rates in the chip — keeps the chip
            # clean when nothing unusual is happening.
            if samplerate and samplerate != 44100:
                suffix = f' · {samplerate / 1000:g} kHz'
            self._name.setText(f'{short}{suffix}')
            tip = device_name
            if host_api:
                tip += f'  [{host_api}]'
            if samplerate:
                tip += f'  · {samplerate} Hz'
            self.setToolTip(tip)
        elif state in (AudioEngineState.OPENING,
                        AudioEngineState.ENUMERATING):
            self._name.setText(self._t('audio_chip_opening'))
            self.setToolTip(self._t('audio_chip_opening'))
        else:
            self._name.setText(self._t('audio_chip_none'))
            self.setToolTip(self._t('audio_chip_tip'))


class AudioRecoveryBanner(QWidget):
    """Inline banner that appears when the engine is in FAILED state.

    Sits above the tuner; carries two buttons (Retry, pick a different
    device). The copy is selected by ``AudioEngineError`` per Frodo's
    UX memo — never displays raw PortAudio error codes to the user."""

    def __init__(self, t_func, on_retry, on_pick):
        super().__init__()
        self._t = t_func
        self._on_retry = on_retry
        self._on_pick = on_pick
        self.setStyleSheet("""
            QWidget{background:#1e1e2e;border:1px solid #444;border-left:4px solid #c0392b;border-radius:5px;}
            QLabel{color:#eee;font-size:12px;}
            QPushButton{background:#34495e;color:#eee;border:none;border-radius:4px;
                         padding:5px 12px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
        """)
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        self._msg = QLabel('')
        self._msg.setWordWrap(True)
        lay.addWidget(self._msg, 1)
        self._btn_retry = QPushButton(self._t('audio_banner_retry'))
        self._btn_retry.clicked.connect(lambda: self._on_retry())
        lay.addWidget(self._btn_retry)
        self._btn_pick = QPushButton(self._t('audio_banner_pick'))
        self._btn_pick.clicked.connect(lambda: self._on_pick())
        lay.addWidget(self._btn_pick)
        self.hide()

    def retranslate(self, t_func) -> None:
        self._t = t_func
        self._btn_retry.setText(self._t('audio_banner_retry'))
        self._btn_pick.setText(self._t('audio_banner_pick'))

    def show_for(self, err: AudioEngineError, device_name: str,
                 raw_msg: str) -> None:
        if err == AudioEngineError.NO_DEVICE:
            self._msg.setText('⚠  ' + self._t('audio_banner_no_device'))
        elif err == AudioEngineError.DEVICE_DISCONNECTED:
            self._msg.setText('⚠  ' + self._t('audio_banner_disconnect',
                                              name=device_name or '?'))
        elif err == AudioEngineError.DEVICE_BUSY:
            self._msg.setText('⚠  ' + self._t('audio_banner_busy',
                                              name=device_name or '?'))
        elif err == AudioEngineError.UNSUPPORTED_RATE:
            self._msg.setText('⚠  '
                              + self._t('audio_banner_unsupported_rate'))
        else:
            # HOSTAPI_FAILURE / UNKNOWN — surface a generic line. The
            # raw PortAudio text goes to diagnostics, not here.
            self._msg.setText('⚠  '
                              + self._t('audio_banner_unknown', msg=raw_msg or '—'))
        self.show()


class InfoBanner(QWidget):
    """Non-modal info banner with one optional action + a Dismiss button.

    Sister widget to ``AudioRecoveryBanner``; used for hot-plug
    notifications and wrong-instrument hints — anything that previously
    interrupted Frodo with a modal QMessageBox while he was playing.

    Use ``show_message(text, action_label=None, action_callback=None)`` to
    surface a banner; Dismiss always hides it. If no action label is
    provided, only the Dismiss button is shown."""

    def __init__(self, t_func):
        super().__init__()
        self._t = t_func
        self._action_callback = None
        self.setStyleSheet("""
            QWidget{background:#1e2a3a;border:1px solid #444;border-left:4px solid #3498db;border-radius:5px;}
            QLabel{color:#eee;font-size:12px;}
            QPushButton{background:#34495e;color:#eee;border:none;border-radius:4px;
                         padding:5px 12px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
        """)
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        self._msg = QLabel('')
        self._msg.setWordWrap(True)
        lay.addWidget(self._msg, 1)
        self._btn_action = QPushButton('')
        self._btn_action.clicked.connect(self._on_action)
        lay.addWidget(self._btn_action)
        self._btn_dismiss = QPushButton(self._t('audio_toast_dismiss'))
        self._btn_dismiss.clicked.connect(self.hide)
        lay.addWidget(self._btn_dismiss)
        self.hide()

    def retranslate(self, t_func) -> None:
        self._t = t_func
        self._btn_dismiss.setText(self._t('audio_toast_dismiss'))

    def _on_action(self) -> None:
        cb = self._action_callback
        self.hide()
        if cb is not None:
            try:
                cb()
            except Exception:
                # The action handler is user code; never let an exception
                # in the action propagate up to the Qt event loop and
                # tear the window down on Frodo.
                pass

    def show_message(self, text: str, action_label: str | None = None,
                     action_callback=None) -> None:
        self._msg.setText(text)
        if action_label and action_callback is not None:
            self._action_callback = action_callback
            self._btn_action.setText(action_label)
            self._btn_action.show()
        else:
            self._action_callback = None
            self._btn_action.hide()
        self.show()


def _promote_vendor_prefix(name: str) -> str:
    """If ``name`` contains a known vendor brand mid-string (typical of
    Windows naming like "Headset (FIIO DSP Audio)" or "Microphone
    (2- Scarlett Solo)"), return a ``"{VENDOR} · {rest}"`` form with
    the matched vendor lifted to the front and the parenthesised vendor
    fragment stripped from the body.

    Returns the original ``name`` unchanged if no vendor regex matches
    or if the match already sits at position 0 (no promotion needed)."""
    import re
    if not name:
        return name
    m = re.search(VENDOR_REGEX, name, re.IGNORECASE)
    if m is None:
        return name
    vendor = m.group(0).upper()
    # Step 1: for each parenthesised fragment that contains a vendor
    # token, strip only the vendor occurrence from the paren interior and
    # keep the rest.  Examples:
    #   "Microphone (2- Scarlett Solo)" -> "Microphone (2- Solo)"
    #   "Line In (FIIO) - ASUS"         -> "Line In  - ASUS"  (parens empty, dropped)
    # Non-vendor parens are preserved verbatim.  v0.6 change: previously
    # the WHOLE paren was dropped when it contained a vendor, losing
    # meaningful non-vendor content like "(2- Solo)".  Empty parens
    # produced by full-vendor stripping collapse to a single space.
    paren_re = re.compile(r'\([^)]*\)')
    cleaned_parts: list[str] = []
    pos = 0
    for pm in paren_re.finditer(name):
        cleaned_parts.append(name[pos:pm.start()])
        inner = pm.group(0)
        if re.search(VENDOR_REGEX, inner, re.IGNORECASE) is None:
            cleaned_parts.append(inner)
        else:
            stripped = re.sub(VENDOR_REGEX, '', inner,
                              count=1, flags=re.IGNORECASE)
            # Tidy paren whitespace produced by the strip.
            stripped = re.sub(r'\(\s+', '(', stripped)
            stripped = re.sub(r'\s+\)', ')', stripped)
            stripped = re.sub(r'\s+', ' ', stripped)
            # If only the vendor was inside, collapse the now-empty
            # parens to a single space (Step 3 collapses runs of space).
            if stripped in ('()', '( )'):
                cleaned_parts.append(' ')
            else:
                cleaned_parts.append(stripped)
        pos = pm.end()
    cleaned_parts.append(name[pos:])
    body = ''.join(cleaned_parts)
    # Step 2 (v0.5.7.4): strip bare occurrences of the matched vendor
    # token still left in the body. Without this, "FIIO Q3" produced
    # "FIIO · FIIO Q3" because step 1 only touches paren-wrapped vendor
    # text. We only strip the SAME vendor that was hoisted to the prefix
    # — other vendor-list tokens in the body are part of the device
    # name proper (e.g. "Universal Audio Apollo Twin X" should keep
    # "Apollo" in the body even though Apollo is a known brand).
    body = re.sub(
        rf'\b{re.escape(m.group(0))}\b', ' ', body, count=1, flags=re.IGNORECASE
    )
    # Step 3: clean up whitespace and orphan separators left by removal.
    # v0.6 dropped a `re.sub(r'\s*-\s*', ' - ', body)` step that used to
    # normalise dash spacing; with the new paren-content-preservation
    # logic, dashes inside parens (e.g. "(2- Solo)") must stay intact.
    body = re.sub(r'\s*\(\s*\)\s*', ' ', body)  # empty parens
    body = re.sub(r'\s+', ' ', body).strip(' -·:|')
    if not body:
        # Input was effectively just the vendor token (possibly wrapped
        # in parens) — return it bare rather than echoing back "VENDOR · ".
        return vendor
    return f'{vendor} · {body}'


class AudioPickerDialog(QDialog):
    """Modal device picker. Two-line rows, dedup-by-name with an
    expandable host-API sublist. Vendor regex ranks external interfaces
    to the top. The dialog never opens a stream itself — accepting the
    selection just calls ``engine.open_device(spec)`` and lets the
    state machine do the rest."""

    # v0.5.7.2: sample-rate re-open runs on a worker thread (open_device
    # internally walks candidates with t.join(timeout=0.8) per attempt,
    # which would freeze the UI for multiple seconds on a slow device).
    # The worker emits this signal carrying (prev_pref, new_pref) and a
    # QueuedConnection on the receiving slot marshals back to the GUI
    # thread for the UI updates.
    _sr_reopen_done = pyqtSignal(str, str)

    def __init__(self, parent, t_func, engine: AudioEngine,
                 cfg: sax_config.AppConfig, current: 'DeviceInfo | None'):
        super().__init__(parent)
        self._t = t_func
        self._engine = engine
        self._cfg = cfg
        self._current = current
        self._chosen: 'DeviceInfo | None' = None
        # v0.5.7.2: debounce flag — ignore further combo changes while
        # an open is in flight. The combo is also disabled, but the
        # flag guards against programmatic edits / queued signals that
        # could slip in around the worker boundary.
        self._sr_switch_in_flight: bool = False
        self._sr_prev_pref: str = 'auto'
        # v0.5.7.3: close-during-worker UAF guard. If the user closes the
        # dialog while an sr-reopen worker is still running, the worker's
        # finally block used to emit `_sr_reopen_done` on a QObject that
        # Qt had already destroyed (segfault / undefined behaviour). The
        # threading.Event lets the worker bail out of the emit, and we
        # also disconnect the signal in closeEvent as a second line of
        # defence.
        self._closing_event: threading.Event = threading.Event()
        self._sr_reopen_done.connect(
            self._on_sr_reopen_done, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle(self._t('audio_picker_title'))
        self.setModal(True)
        self.setMinimumWidth(560)
        # v0.5.7.1: dialog used to open at Qt's minimum-content height,
        # which fit ~3 rows and forced the user to scroll past the most
        # interesting devices (vendor interfaces below the system mics).
        # Open at a size where the full typical list fits unscrolled.
        self.setMinimumSize(480, 480)
        self.setStyleSheet("""
            QDialog{background:#1e1e2e;color:#eee;}
            QLabel{color:#eee;font-size:12px;}
            QListWidget{background:#15151f;border:1px solid #444;
                         color:#eee;font-size:12px;}
            QListWidget::item{padding:6px;}
            QListWidget::item:selected{background:#34495e;}
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:6px 14px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
            QCheckBox{color:#bbb;font-size:11px;}
        """)
        self._build()
        self._refill()
        # v0.5.7.1: explicit default size so the picker opens at a
        # comfortable height for the typical Windows device count.
        self.resize(560, 600)

    def _build(self) -> None:
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # v0.5.6: sample-rate preference row. Sits ABOVE the device list
        # so the user reads the policy first and the chosen device second.
        sr_row = QHBoxLayout()
        sr_lbl = QLabel(self._t('samplerate_label') + ':')
        sr_lbl.setStyleSheet('color:#bbb;font-size:12px;')
        sr_row.addWidget(sr_lbl)
        self._sr_combo = QComboBox()
        self._sr_combo.addItem(self._t('samplerate_auto'), 'auto')
        for hz in (192000, 96000, 88200, 48000, 44100):
            # Number-with-thin-space formatting reads as "192 000 Hz" in
            # both DE and EN locales without dragging in locale plumbing.
            label = f'{hz:,}'.replace(',', ' ') + ' Hz'
            self._sr_combo.addItem(label, str(hz))
        cur_pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                'auto') or 'auto')
        idx = self._sr_combo.findData(cur_pref)
        if idx < 0:
            idx = 0
        self._sr_combo.setCurrentIndex(idx)
        # v0.5.7.2: snapshot for revert-on-failure path.
        self._sr_prev_pref = cur_pref
        self._sr_combo.currentIndexChanged.connect(self._on_sr_changed)
        self._sr_combo.setStyleSheet(
            'QComboBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:4px;padding:3px 8px;font-size:12px;}'
            'QComboBox::drop-down{border:none;width:18px;background:#1e1e2e;}'
            'QComboBox QAbstractItemView{background:#1e1e2e;color:#ddd;'
            'border:1px solid #444;outline:0;'
            'selection-background-color:#34495e;selection-color:#fff;}'
            'QComboBox QAbstractItemView::item{background:#1e1e2e;color:#ddd;'
            'padding:4px 8px;border:none;}'
            'QComboBox QAbstractItemView::item:selected{background:#34495e;color:#fff;}')
        sr_row.addWidget(self._sr_combo, 1)
        root.addLayout(sr_row)

        self._sr_error_lbl = QLabel('')
        self._sr_error_lbl.setStyleSheet(
            'color:#e07070;font-size:11px;padding:0 2px;')
        self._sr_error_lbl.setVisible(False)
        self._sr_error_lbl.setWordWrap(True)
        root.addWidget(self._sr_error_lbl)

        top = QHBoxLayout()
        self._btn_rescan = QPushButton(self._t('audio_picker_rescan'))
        self._btn_rescan.clicked.connect(self._refill)
        top.addWidget(self._btn_rescan)
        top.addStretch()
        root.addLayout(top)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _i: self._accept())
        root.addWidget(self._list, 1)

        opts = QHBoxLayout()
        self._cb_show_all = QCheckBox(self._t('audio_picker_show_all'))
        self._cb_show_all.setChecked(bool(getattr(self._cfg,
                                                    'show_all_host_apis',
                                                    False)))
        self._cb_show_all.toggled.connect(self._on_toggle_show_all)
        opts.addWidget(self._cb_show_all)
        self._cb_ks = QCheckBox(self._t('audio_picker_prefer_ks'))
        self._cb_ks.setChecked(bool(getattr(self._cfg, 'prefer_wdmks', False)))
        self._cb_ks.toggled.connect(self._on_toggle_ks)
        opts.addWidget(self._cb_ks)
        opts.addStretch()
        root.addLayout(opts)

        btns = QHBoxLayout()
        self._btn_use = QPushButton(self._t('audio_picker_use'))
        self._btn_use.clicked.connect(self._accept)
        self._btn_cancel = QPushButton(self._t('audio_picker_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self._btn_use)
        btns.addWidget(self._btn_cancel)
        root.addLayout(btns)

    def _on_sr_changed(self, _idx: int) -> None:
        """User picked a new sample-rate policy. Persist immediately,
        then reopen the active device with the new pref on a worker
        thread.

        v0.5.7.2: open_device walks candidates and runs
        ``t.join(timeout=0.8)`` per attempt, which on Qt's main thread
        froze the dialog for multiple seconds. We now spawn a worker,
        disable the combo while it runs, and marshal the UI updates
        back through ``_sr_reopen_done`` (QueuedConnection) so all
        widget mutation happens on the GUI thread again. The engine's
        internal lock (v0.5.4) makes the off-thread call safe.
        """
        if self._sr_switch_in_flight:
            # Debounce: ignore further changes until the current worker
            # finishes. The combo is disabled below, so this only fires
            # for queued signals that slipped past the disable.
            return
        new_pref = str(self._sr_combo.currentData() or 'auto')
        if new_pref not in SAMPLERATE_PREF_VALUES:
            new_pref = 'auto'
        prev_pref = self._sr_prev_pref
        if new_pref == prev_pref:
            return
        self._cfg.audio_samplerate_pref = new_pref
        sax_config.save_config(self._cfg)
        self._sr_error_lbl.setVisible(False)
        dev = self._engine.get_active_device()
        if dev is None:
            self._sr_prev_pref = new_pref
            return
        sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                              samplerate=0)

        # Block the GUI from issuing more re-opens until this one lands.
        self._sr_switch_in_flight = True
        self._sr_combo.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # v0.5.7.3: capture the event by closure (not self.) so the
        # worker can check it even if the dialog is mid-teardown. The
        # event lives independently of the QObject's lifetime.
        closing_event = self._closing_event

        def _worker() -> None:
            try:
                self._engine.stop()
                self._engine.open_device(sel, samplerate_pref=new_pref)
            finally:
                # Even if the engine raises, hand control back to the
                # GUI thread so the combo gets re-enabled — UNLESS the
                # dialog is being closed, in which case the QObject
                # may already be destroyed and emitting on a dead
                # receiver is undefined behaviour.
                if not closing_event.is_set():
                    try:
                        self._sr_reopen_done.emit(prev_pref, new_pref)
                    except RuntimeError:
                        # "wrapped C/C++ object has been deleted" —
                        # raced past our event check. Swallow; there's
                        # no receiver to hand off to anyway.
                        pass

        t = threading.Thread(target=_worker, name='sr-reopen',
                             daemon=True)
        t.start()

    def _on_sr_reopen_done(self, prev_pref: str, new_pref: str) -> None:
        """Runs on the GUI thread (QueuedConnection). Re-enables the
        combo, surfaces the inline error on UNSUPPORTED_RATE, and
        reverts the selection if the new pref was refused."""
        try:
            QGuiApplication.restoreOverrideCursor()
        except Exception:
            pass
        failed_rate = (
            self._engine.state == AudioEngineState.FAILED
            and self._engine.last_error == AudioEngineError.UNSUPPORTED_RATE
        )
        if failed_rate:
            shown_rate = new_pref if new_pref != 'auto' else '–'
            self._sr_error_lbl.setText(
                self._t('samplerate_unsupported', rate=shown_rate))
            self._sr_error_lbl.setVisible(True)
            # Revert to previous pref in combo + cfg, then retry the
            # open with the previous pref so the user is left in the
            # same audio state they started with.
            self._cfg.audio_samplerate_pref = prev_pref
            sax_config.save_config(self._cfg)
            self._sr_combo.blockSignals(True)
            revert_idx = self._sr_combo.findData(prev_pref)
            if revert_idx < 0:
                revert_idx = 0
            self._sr_combo.setCurrentIndex(revert_idx)
            self._sr_combo.blockSignals(False)
            dev = self._engine.get_active_device()
            if dev is not None:
                sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                                      samplerate=0)
                # Retry on a worker too so we don't re-introduce the
                # main-thread block we just removed. Best-effort: fire
                # and forget — failure here just leaves the engine in
                # FAILED, which the banner handles separately.
                def _retry() -> None:
                    try:
                        self._engine.open_device(sel,
                                                 samplerate_pref=prev_pref)
                    except Exception:
                        pass
                threading.Thread(target=_retry, name='sr-revert',
                                 daemon=True).start()
            self._sr_prev_pref = prev_pref
        else:
            self._sr_prev_pref = new_pref
        self._sr_combo.setEnabled(True)
        self._sr_switch_in_flight = False

    def closeEvent(self, event) -> None:
        """v0.5.7.3: signal any in-flight sr-reopen worker that we're
        going away so it skips its emit (which would otherwise land on a
        deleted QObject). Also disconnect the signal as belt-and-
        suspenders — if a queued emission is already sitting in the
        event loop, the disconnect makes it a no-op."""
        self._closing_event.set()
        try:
            self._sr_reopen_done.disconnect(self._on_sr_reopen_done)
        except (TypeError, RuntimeError):
            # Already disconnected or signal/slot already torn down.
            pass
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """``accept()`` / ``reject()`` route through here, so set the
        closing flag here too. closeEvent isn't always called on
        programmatic accept/reject paths."""
        self._closing_event.set()
        super().done(result)

    def _on_toggle_show_all(self, checked: bool) -> None:
        self._cfg.show_all_host_apis = bool(checked)
        sax_config.save_config(self._cfg)
        self._refill()

    def _on_toggle_ks(self, checked: bool) -> None:
        self._cfg.prefer_wdmks = bool(checked)
        sax_config.save_config(self._cfg)
        self._engine.set_prefer_wdmks(bool(checked))

    def _rank(self, d: DeviceInfo) -> int:
        """Higher = sorts earlier. Vendor regex dominates."""
        import re
        score = 0
        if re.search(VENDOR_REGEX, d.name, re.IGNORECASE):
            score += 100
        api = d.host_api.lower()
        if 'wasapi' in api:
            score += 20
        elif 'wdm-ks' in api:
            score += 10
        if d.default_samplerate >= 48000:
            score += 5
        low = d.name.lower()
        if any(x in low for x in ('webcam', 'hdmi', 'nvidia', 'amd')):
            score -= 50
        if any(x in low for x in ('microphone array', 'stereo mix')):
            score -= 20
        return score

    def _refill(self) -> None:
        from PyQt6.QtWidgets import QListWidgetItem
        self._list.clear()
        devices = self._engine.refresh_devices()
        if not devices:
            it = QListWidgetItem(self._t('audio_picker_no_devices'))
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(it)
            return
        show_all = self._cb_show_all.isChecked()
        # Group by canonical name unless show-all is on.
        groups: dict[str, list[DeviceInfo]] = {}
        order: list[str] = []
        for d in devices:
            key = d.name if not show_all else f'{d.name}\0{d.host_api}'
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(d)
        # Pick a representative for each group (WASAPI > WDM-KS > MME > rest).
        def api_pref(d: DeviceInfo) -> int:
            api = d.host_api.lower()
            if 'wasapi' in api:
                return 0
            if 'wdm-ks' in api:
                return 1
            if api == 'mme':
                return 2
            if 'directsound' in api:
                return 3
            return 9
        rows: list[tuple[DeviceInfo, list[DeviceInfo]]] = []
        for key in order:
            members = sorted(groups[key], key=api_pref)
            rows.append((members[0], members))
        # Sort rows by rank, current device pinned to the top.
        def sort_key(row):
            primary = row[0]
            is_current = (
                self._current is not None
                and primary.name == self._current.name
                and primary.host_api == self._current.host_api)
            return (0 if is_current else 1, -self._rank(primary),
                    primary.name.lower())
        rows.sort(key=sort_key)
        for primary, members in rows:
            self._add_row(primary, members)

    def _add_row(self, primary: DeviceInfo,
                 members: list[DeviceInfo]) -> None:
        from PyQt6.QtWidgets import QListWidgetItem
        is_current = (
            self._current is not None
            and primary.name == self._current.name
            and primary.host_api == self._current.host_api)
        meta = self._t('audio_picker_row_meta',
                       api=primary.host_api or '—',
                       ch=primary.max_input_channels,
                       sr=primary.default_samplerate)
        badge = ''
        if is_current:
            badge = f'  ◀ {self._t("audio_picker_current")}'
        elif len(members) > 1:
            badge = f'  ▸ {self._t("audio_picker_apis_more", n=len(members))}'
        # v0.5.7: if a vendor brand appears mid-string in the Windows
        # device name (e.g. "Headset (FIIO DSP Audio)"), promote the
        # brand to the front of the row label so saxophone players can
        # scan by brand instead of by Windows device-naming convention.
        # The full original name remains in the row's tooltip.
        display_name = _promote_vendor_prefix(primary.name)
        text = f'{display_name}{badge}\n    {meta}'
        it = QListWidgetItem(text)
        it.setData(Qt.ItemDataRole.UserRole, primary)
        it.setToolTip(primary.name)
        self._list.addItem(it)
        if is_current:
            self._list.setCurrentItem(it)

    def _accept(self) -> None:
        it = self._list.currentItem()
        if it is None:
            self.reject()
            return
        dev = it.data(Qt.ItemDataRole.UserRole)
        if not isinstance(dev, DeviceInfo):
            self.reject()
            return
        self._chosen = dev
        self.accept()

    def chosen(self) -> 'DeviceInfo | None':
        return self._chosen


# =============================================================================
# Per-instrument range editor (v0.5.5)
# =============================================================================
class RangeEditorDialog(QDialog):
    """Modal editor for the (lo, hi) fingered-MIDI range of one
    instrument. Persists to ~/.intonation_analyzer/instrument_ranges.json
    via sax_instruments.save_range_override / clear_range_override.

    The dialog owns no policy: the caller (MainWindow) refreshes its
    table after accept() so the new range takes effect immediately."""

    def __init__(self, parent, t, instrument_key: str,
                 display_name: str, current_lo: int, current_hi: int,
                 baked_lo: int, baked_hi: int, has_baked: bool,
                 display: str = 'griff', transp: int = 0):
        """``current_lo`` / ``current_hi`` / ``baked_lo`` / ``baked_hi``
        are ALWAYS fingered MIDI (canonical disk format). The dialog
        renders them as sounding MIDI when ``display == 'klingend'`` by
        adding ``transp`` semitones, and reverses the offset on save —
        the JSON on disk never changes format regardless of which user
        opens the editor.
        """
        super().__init__(parent)
        self._t = t
        self._key = instrument_key
        self._baked = (baked_lo, baked_hi)  # canonical fingered MIDI
        self._has_baked = has_baked
        self._display = 'klingend' if display == 'klingend' else 'griff'
        self._transp = int(transp) if self._display == 'klingend' else 0
        self.setWindowTitle(self._t('range_editor_title', name=display_name))
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)
        self._lo_spin = QSpinBox()
        self._lo_spin.setRange(0, 127)
        self._lo_spin.setValue(int(current_lo) + self._transp)
        self._hi_spin = QSpinBox()
        self._hi_spin.setRange(0, 127)
        self._hi_spin.setValue(int(current_hi) + self._transp)
        # v0.6.2: explicit sub-control geometry for the up/down buttons.
        # Setting bg + border + padding on QSpinBox without styling
        # ::up-button / ::down-button leaves the buttons with effectively
        # zero clickable width on Windows — the up arrow visibly draws
        # but the hit-rect collapses. Reserving 18px of right padding +
        # positioning the two sub-controls top/bottom right restores
        # clickability and keeps the dark theme.
        spin_css = self._spin_css_ok()
        self._lo_spin.setStyleSheet(spin_css)
        self._hi_spin.setStyleSheet(spin_css)
        # v0.6 Phase-4 (Item 4): primary input is the note-name QLineEdit;
        # the MIDI spinbox is kept visible for power users. Wire-up:
        # the name edit drives the spin (parses on editingFinished), and
        # the spin drives the edit (formats on valueChanged). Cycle guard
        # via _suppress_sync so we don't bounce between the two.
        self._suppress_sync = False
        self._lo_name = QLineEdit()
        self._hi_name = QLineEdit()
        name_css_ok = (
            'QLineEdit{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:3px 6px;font-size:13px;min-width:80px;}'
        )
        self._lo_name.setStyleSheet(name_css_ok)
        self._hi_name.setStyleSheet(name_css_ok)
        self._lo_name.setText(midi_note_name(int(current_lo) + self._transp))
        self._hi_name.setText(midi_note_name(int(current_hi) + self._transp))
        self._lo_name.setPlaceholderText(self._t('range_note_name_hint'))
        self._hi_name.setPlaceholderText(self._t('range_note_name_hint'))
        self._lo_name.setToolTip(self._t('range_note_name_hint'))
        self._hi_name.setToolTip(self._t('range_note_name_hint'))
        if self._display == 'klingend':
            lo_label = self._t('range_lo_label_sounding')
            hi_label = self._t('range_hi_label_sounding')
        else:
            lo_label = self._t('range_lo_label')
            hi_label = self._t('range_hi_label')
        # Each row hosts name-edit + spinbox so power users still see the
        # MIDI integer. Name-edit is first because it's the primary input.
        from PyQt6.QtWidgets import QWidget as _QW
        def _make_row(name_edit, spin) -> _QW:
            w = _QW()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(name_edit, 1)
            h.addWidget(spin, 0)
            return w
        form.addRow(lo_label, _make_row(self._lo_name, self._lo_spin))
        form.addRow(hi_label, _make_row(self._hi_name, self._hi_spin))
        layout.addLayout(form)

        # Mode hint so the user knows what the spinboxes mean and that
        # the file format is stable across users with different display
        # preferences.
        mode_note_key = ('range_mode_note_sound'
                        if self._display == 'klingend'
                        else 'range_mode_note_griff')
        self._mode_lbl = QLabel(self._t(mode_note_key))
        self._mode_lbl.setStyleSheet('color:#888;font-size:11px;'
                                     'font-style:italic;')
        self._mode_lbl.setWordWrap(True)
        layout.addWidget(self._mode_lbl)

        self._preview_lbl = QLabel('')
        self._preview_lbl.setStyleSheet('color:#bbb;font-size:12px;')
        self._preview_lbl.setWordWrap(True)
        layout.addWidget(self._preview_lbl)

        self._error_lbl = QLabel('')
        self._error_lbl.setStyleSheet('color:#e07070;font-size:12px;')
        self._error_lbl.setWordWrap(True)
        self._error_lbl.setVisible(False)
        layout.addWidget(self._error_lbl)

        # Buttons row.
        btn_row = QHBoxLayout()
        if has_baked:
            restore_label = self._t('range_restore_default')
        else:
            # Show baked numbers in the current display mode so they
            # match what the spinboxes display.
            disp_lo = baked_lo + self._transp
            disp_hi = baked_hi + self._transp
            restore_label = self._t('range_restore_fallback',
                                    lo=disp_lo, hi=disp_hi)
        self._btn_restore = QPushButton(restore_label)
        self._btn_restore.clicked.connect(self._on_restore)
        self._btn_cancel = QPushButton(self._t('range_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_save = QPushButton(self._t('range_save'))
        self._btn_save.clicked.connect(self._on_save)
        btn_css = (
            'QPushButton{background:#34495e;color:#eee;border:none;'
            'border-radius:5px;padding:6px 12px;font-size:12px;}'
            'QPushButton:hover{background:#3d566e;}'
            'QPushButton:disabled{background:#2a2a3a;color:#666;}'
        )
        self._btn_restore.setStyleSheet(btn_css)
        self._btn_cancel.setStyleSheet(btn_css)
        self._btn_save.setStyleSheet(btn_css)
        btn_row.addWidget(self._btn_restore)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_save)
        layout.addLayout(btn_row)

        # Live preview wiring.
        self._lo_spin.valueChanged.connect(self._on_spin_changed)
        self._hi_spin.valueChanged.connect(self._on_spin_changed)
        self._lo_name.editingFinished.connect(
            lambda: self._on_name_edited(self._lo_name, self._lo_spin))
        self._hi_name.editingFinished.connect(
            lambda: self._on_name_edited(self._hi_name, self._hi_spin))
        self._update_preview()

        # Dark theme to match the rest of the app.
        self.setStyleSheet('QDialog{background:#12121a;color:#ddd;} '
                           'QLabel{color:#ccc;}')

    def _on_spin_changed(self) -> None:
        # Spin changed → sync the matching note-name edit, then refresh
        # the preview. Guarded so we don't recurse when the name-edit
        # slot is the one mutating the spin.
        if not self._suppress_sync:
            self._suppress_sync = True
            try:
                self._lo_name.setText(midi_note_name(self._lo_spin.value()))
                self._hi_name.setText(midi_note_name(self._hi_spin.value()))
                # Clear red border if the user previously typed a bad name.
                ok_css = (
                    'QLineEdit{background:#1e1e2e;color:#ddd;border:1px solid #444;'
                    'border-radius:5px;padding:3px 6px;font-size:13px;min-width:80px;}'
                )
                self._lo_name.setStyleSheet(ok_css)
                self._hi_name.setStyleSheet(ok_css)
            finally:
                self._suppress_sync = False
        self._update_preview()

    def _on_name_edited(self, edit, spin) -> None:
        # Parse the user's text. On success, push the MIDI into the spin
        # (and let _on_spin_changed reformat). On failure, paint the edit
        # red and leave the spin alone — the brief is strict about not
        # silently mutating the underlying number when parsing fails.
        if self._suppress_sync:
            return
        text = edit.text()
        midi = note_name_to_midi(text)
        if midi is None:
            bad_css = (
                'QLineEdit{background:#3a1e1e;color:#fdd;border:1px solid #c0392b;'
                'border-radius:5px;padding:3px 6px;font-size:13px;min-width:80px;}'
            )
            edit.setStyleSheet(bad_css)
            self._error_lbl.setText(self._t('range_note_name_invalid'))
            self._error_lbl.setVisible(True)
            return
        # Valid name → clear red border, push to spin (which re-formats
        # the edit canonically via _on_spin_changed).
        ok_css = (
            'QLineEdit{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:3px 6px;font-size:13px;min-width:80px;}'
        )
        edit.setStyleSheet(ok_css)
        spin.setValue(midi)

    # v0.6.2: spin-button sub-controls. Same styling re-used at
    # construction and on every _update_preview repaint so the up/down
    # buttons remain clickable in both the ok and bad-range states.
    _SPIN_SUBCONTROLS = (
        'QSpinBox::up-button{subcontrol-origin:border;'
        'subcontrol-position:top right;width:16px;'
        'background:#2a2a3a;border:none;'
        'border-top-right-radius:5px;border-left:1px solid #444;}'
        'QSpinBox::up-button:hover{background:#3d566e;}'
        'QSpinBox::up-button:pressed{background:#16161e;}'
        'QSpinBox::down-button{subcontrol-origin:border;'
        'subcontrol-position:bottom right;width:16px;'
        'background:#2a2a3a;border:none;'
        'border-bottom-right-radius:5px;border-left:1px solid #444;'
        'border-top:1px solid #444;}'
        'QSpinBox::down-button:hover{background:#3d566e;}'
        'QSpinBox::down-button:pressed{background:#16161e;}'
    )

    def _spin_css_ok(self) -> str:
        # Right-padding reserves 22px for the two stacked sub-controls
        # (16px button + 1px border + 5px breathing room).
        return (
            'QSpinBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:3px 22px 3px 6px;font-size:13px;'
            'min-width:80px;}'
            + self._SPIN_SUBCONTROLS
        )

    def _spin_css_bad(self) -> str:
        return (
            'QSpinBox{background:#3a1e1e;color:#fdd;border:1px solid #c0392b;'
            'border-radius:5px;padding:3px 22px 3px 6px;font-size:13px;'
            'min-width:80px;}'
            + self._SPIN_SUBCONTROLS
        )

    def _update_preview(self) -> None:
        lo = self._lo_spin.value()
        hi = self._hi_spin.value()
        invalid = lo > hi
        # Visual cue on the offending field.
        bad_css = self._spin_css_bad()
        ok_css = self._spin_css_ok()
        self._lo_spin.setStyleSheet(bad_css if invalid else ok_css)
        self._hi_spin.setStyleSheet(bad_css if invalid else ok_css)
        if invalid:
            self._error_lbl.setText(self._t('range_invalid'))
            self._error_lbl.setVisible(True)
            self._preview_lbl.setText('')
            self._btn_save.setEnabled(False)
            return
        self._error_lbl.setVisible(False)
        self._btn_save.setEnabled(True)
        semis = hi - lo
        octs = semis / 12.0
        self._preview_lbl.setText(self._t(
            'range_preview_fmt',
            lo_name=midi_note_name(lo), hi_name=midi_note_name(hi),
            semis=semis, octs=octs))

    def _on_restore(self) -> None:
        # Baked is canonical fingered; render in current display mode.
        lo, hi = self._baked
        self._lo_spin.setValue(lo + self._transp)
        self._hi_spin.setValue(hi + self._transp)

    def _on_save(self) -> None:
        # Spinbox values are in the current display mode. Convert back
        # to fingered (canonical) before persisting so the file format
        # is stable across users with different display preferences.
        disp_lo = self._lo_spin.value()
        disp_hi = self._hi_spin.value()
        if disp_lo > disp_hi:
            return
        lo = disp_lo - self._transp
        hi = disp_hi - self._transp
        # If the user dialed back to the baked default AND a baked entry
        # exists, clear the override instead of writing a redundant
        # entry. Keeps the overrides file lean and lets future baked
        # changes flow through.
        if self._has_baked and (lo, hi) == self._baked:
            sax_instruments.clear_range_override(self._key)
        else:
            sax_instruments.save_range_override(self._key, lo, hi)
        self.accept()


# =============================================================================
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Pick the user's system language as the default. German-speaking
        # locales get the original DE strings; anyone else gets English.
        self.lang       = 'de' if QLocale.system().name().startswith('de') else 'en'
        # v0.5.7.1: default to Bb tenor — the most common community-band
        # sax. Existing users with a persisted ``last_instrument_key``
        # keep their own choice (restored a few lines below); this only
        # affects fresh installs and first-launches.
        self.instrument = 'bb_tenor'
        self.display    = 'griff'
        self.stats: dict[int, NoteStats] = {}
        self._lock = threading.Lock()
        self._recording = True   # Aufnahme läuft beim Start
        self._active_midi: int | None = None
        self._active_midi_at: datetime.datetime | None = None
        self._layout_mode: str = 'single'   # 'single' | 'matrix'
        # Cumulative count of distinct note emissions since launch (each
        # _on_note increments). Diagnostics readout consumes this; the
        # table-level note count comes from `len(self.stats)`.
        self._notes_count: int = 0
        # v0.6 Phase-4 (Item 3): wrong-instrument detector. Counts
        # consecutive _on_note calls whose fingered MIDI lands outside the
        # currently selected instrument's range. Reset when a note lands
        # inside the range (false alarm) or when the user changes
        # instrument/A4. ``_oor_banner_shown`` keeps the prompt from
        # re-firing for the same instrument selection after the user has
        # dismissed it once.
        self._oor_count: int = 0
        self._oor_banner_shown: bool = False
        # Threshold: ~12 notes feels like one phrase of all-wrong notes.
        self._oor_threshold: int = 12

        # Load user config + previously-registered custom instruments before
        # building the UI so the catalog reflects them at first paint.
        self._cfg = sax_config.load_config()
        for c in sax_config.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)
        _rebuild_transp_map()

        # v0.5.5: restore last-session preferences that influence the
        # initial UI build. Geometry, splitter sizes, and nickname text
        # are widget-bound and get applied after _build_ui returns.
        # First-launch (empty config) falls back to the locale-derived
        # default already set above.
        if getattr(self._cfg, 'last_lang', '') in ('de', 'en'):
            self.lang = self._cfg.last_lang
        if getattr(self._cfg, 'last_instrument_key', ''):
            # Trust the saved key only if the catalog still knows it; a
            # stale custom instrument that was deleted should fall back
            # rather than crash the combo population.
            if self._cfg.last_instrument_key in TRANSP_MAP:
                self.instrument = self._cfg.last_instrument_key
        if getattr(self._cfg, 'last_display_mode', '') in ('griff', 'klingend'):
            self.display = self._cfg.last_display_mode

        self._engine = AudioEngine()
        self._engine.set_filter_mode(
            getattr(self._cfg, 'filter_mode', FILTER_MODE_DEFAULT))
        self._engine.set_prefer_wdmks(
            bool(getattr(self._cfg, 'prefer_wdmks', False)))
        # Persistence comes from config (welcome dialog), with the env var
        # SAX_INTONATION_LOG_PATH as a power-user override that always wins.
        env_path = os.environ.get('SAX_INTONATION_LOG_PATH')
        log_path = env_path if env_path else self._cfg.effective_log_path()
        self._log = MeasurementLog(path=log_path or None)
        if AUDIO_OK:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4)
            self._engine.signals.note_detected.connect(self._on_note)
            self._engine.signals.state_changed.connect(self._on_engine_state)
            self._engine.signals.devices_changed.connect(
                self._on_devices_changed)
            self._engine.signals.interface_appeared.connect(
                self._on_interface_appeared)

        self._build_ui()
        # Phase-5 extraction: persistable session state (geometry,
        # splitter, instrument/nickname/display/A4/lang) lives on the
        # SessionStateController. Instantiated here, AFTER _build_ui,
        # because every widget reference it captures must already
        # exist. The two thin MainWindow methods below delegate to it.
        self._session_state = SessionStateController(
            self, self._cfg,
            engine=self._engine,
            splitter=self._splitter,
            instr_combo=self._instr_combo,
            nick_edit=self._nick_edit,
            disp_combo=self._disp_combo,
            a4_combo=self._a4_combo,
            lang_combo=self._lang_combo,
        )
        # Phase-5 extraction: TXT / PDF / CSV / PNG-chart export flows and
        # their helper dialogs live on ExportController. Live values
        # (instrument, stats, lang, maker, model, nickname) reach the
        # controller via getter-callables so they're read at export time,
        # not frozen at construction.
        self._export = ExportController(
            self,
            log=self._log,
            cfg=self._cfg,
            engine=self._engine,
            t_func=self._t,
            get_instrument_key=lambda: self.instrument,
            get_stats=lambda: self.stats,
            get_lang=lambda: self.lang,
            get_maker=lambda: getattr(self, '_last_maker', ''),
            get_model=lambda: getattr(self, '_last_model', ''),
            get_nickname=lambda: (self._nick_edit.text()
                                  if hasattr(self, '_nick_edit') else ''),
        )
        # Phase-5 extraction: intonation-table refresh + matrix paint
        # pipeline lives on TableController.  The QTimer that drives the
        # 300 ms refresh stays wired to MainWindow._refresh_table (just
        # below) so resizeEvent's deferred refresh path also works
        # unchanged.  The controller's configure_for_mode() also lands
        # Legolas W8: QTableWidgetItem instances are allocated once per
        # layout change and reused across refresh ticks.
        self._table_ctrl = TableController(
            self,
            table=self._table,
            cfg=self._cfg,
            engine=self._engine,
            t_func=self._t,
            get_instrument_key=lambda: self.instrument,
            get_stats=lambda: self.stats,
            get_display_mode=lambda: self.display,
            get_a4=lambda: float(self._a4_combo.currentText()),
        )
        # Phase-5 extraction: family/instrument selection, nickname editing,
        # per-instrument range editor, custom-instrument registration, and
        # the autotune-from-history flow live on InstrumentController.
        # The controller itself is instantiated INSIDE _build_ui (after
        # every widget it references exists) — see the end of _build_ui
        # for the construction site. MainWindow keeps thin slot-shape
        # wrappers so Qt signal/slot wiring is unchanged; the on_*
        # callbacks let MainWindow do its own post-change refresh
        # (OOR counter reset, seed expected notes, refresh table)
        # without the controller reaching into table or stats territory.
        self._restore_session_state()
        self._seed_expected_notes()
        self._update_record_btn_style()

        # Phase-5 extraction (fifth and final): audio-device picker open,
        # retry, hot-plug poll, engine-state / devices-changed /
        # interface-appeared Qt slot bodies, and the persist-active-device
        # write-back live on DeviceController. The Qt signal-slot
        # connect() lines above and the QTimer wiring below stay in
        # MainWindow for grep-ability — only the slot bodies move.
        # Instantiated here, AFTER _build_ui, because the two banner
        # references it captures must already exist.
        self._device_ctrl = DeviceController(
            self,
            engine=self._engine,
            cfg=self._cfg,
            t_func=self._t,
            info_banner=getattr(self, '_info_banner', None),
            audio_banner=getattr(self, '_audio_banner', None),
            get_lang=lambda: self.lang,
            on_active_device_changed=None,
        )

        # Start the engine AFTER the UI exists so a startup PortAudio
        # failure paints a banner instead of crashing __init__. The
        # engine's start() never raises — it sets state and emits
        # state_changed. This is the v0.5.4 headline fix.
        if AUDIO_OK:
            saved = self._device_selection_from_cfg()
            pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                'auto') or 'auto')
            # v0.5.7: the engine needs the saved (name, host_api) so its
            # hot-plug poller and retry_open path can re-resolve the
            # user's preferred device against a fresh device list
            # without round-tripping through MainWindow.
            self._engine.set_preferred_hint(saved)
            self._engine.start(preferred=saved, samplerate_pref=pref)

        # Hot-plug poller: 1 Hz per Legolas's measurements (cached
        # query_devices is ~0 ms; only re-init is expensive, and we
        # only re-init when the device set actually changed).
        if AUDIO_OK:
            self._device_poll = QTimer(self)
            self._device_poll.timeout.connect(self._poll_devices)
            self._device_poll.start(1000)

        if not AUDIO_OK:
            QMessageBox.information(self, self._t('audio_error_title'),
                                     self._t('audio_error'))

        # First-boot welcome dialog asks about persistence. Skipped on every
        # subsequent launch.
        if not self._cfg.welcome_shown:
            self._show_welcome_dialog()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_table)
        self._refresh_timer.start(300)

    # ── Übersetzungs-Helfer ──────────────────────────────────────────────────
    def _t(self, key, **kwargs):
        s = STRINGS[self.lang].get(key, STRINGS['de'].get(key, f'[{key}]'))
        return s.format(**kwargs) if kwargs else s

    # ── UI aufbauen ──────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 10, 14, 10)

        # ── Toolbar ───────────────────────────────────────────────────────────
        # Always two predictable rows: inputs on top (instrument selection,
        # display, A4, language, import), actions on bottom (autotune, record,
        # reset, exports). When the window is wide enough, both rows breathe
        # easily; when it's narrow, content stays grouped by intent instead
        # of wrapping at arbitrary positions.
        toolbar_container = QWidget()
        toolbar_v = QVBoxLayout(toolbar_container)
        toolbar_v.setContentsMargins(0, 0, 0, 0)
        toolbar_v.setSpacing(6)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar_actions = QHBoxLayout()
        toolbar_actions.setSpacing(8)
        toolbar_v.addLayout(toolbar)
        toolbar_v.addLayout(toolbar_actions)

        # Instrument: family combo + sub-instrument combo + Custom + nickname.
        self._grp_instr = QGroupBox(self._t('grp_instrument'))
        il = QHBoxLayout(self._grp_instr)
        il.setContentsMargins(8, 4, 8, 4)
        il.setSpacing(6)

        self._family_combo = QComboBox()
        for family_key, name_de, name_en in instrument_families():
            label = name_de if self.lang == 'de' else name_en
            self._family_combo.addItem(label, family_key)
        self._family_combo.setMinimumWidth(130)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        il.addWidget(self._family_combo)

        self._instr_combo = QComboBox()
        self._instr_combo.setMinimumWidth(180)
        self._instr_combo.currentIndexChanged.connect(self._on_instr_changed)
        il.addWidget(self._instr_combo)

        # v0.5.5: gear button opens the per-instrument range editor. Sits
        # immediately to the right of the instrument combo so the user
        # finds it without hunting through menus. Unicode glyph instead of
        # an SVG asset — the project ships no settings icon and we keep
        # the dependency surface minimal.
        self._btn_range = QToolButton()
        self._btn_range.setText('⚙')   # U+2699 GEAR
        self._btn_range.setToolTip(self._t('gear_tip'))
        self._btn_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_range.setStyleSheet("""
            QToolButton{background:#1e1e2e;color:#ddd;border:1px solid #444;
                         border-radius:5px;padding:2px 6px;font-size:15px;
                         min-height:28px;min-width:28px;}
            QToolButton:hover{background:#2a2a3a;border:1px solid #6699cc;}
            QToolButton:pressed{background:#16161e;}
        """)
        self._btn_range.clicked.connect(self._open_range_editor)
        il.addWidget(self._btn_range)

        self._btn_custom = QPushButton(self._t('custom_label'))
        self._btn_custom.setToolTip(self._t('custom_dlg_title'))
        self._btn_custom.setMaximumWidth(110)
        self._btn_custom.clicked.connect(self._on_add_custom)
        self._btn_custom.setStyleSheet("""
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:6px 10px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
            QPushButton:pressed{background:#2c3e50;}
        """)
        il.addWidget(self._btn_custom)

        self._nick_edit = QLineEdit()
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))
        # v0.5.7.1: 160px capped at ~12 characters, which truncated real
        # horn names like "Selmer Reference 54 Tenor #2" before the user
        # could see what they'd typed. Bump to 320 max / 280 min.
        self._nick_edit.setMinimumWidth(280)
        self._nick_edit.setMaximumWidth(320)
        self._nick_edit.editingFinished.connect(self._on_nickname_changed)
        self._nick_edit.setStyleSheet("""
            QLineEdit{background:#1e1e2e;border:1px solid #444;
                       border-radius:5px;color:#ddd;padding:4px 8px;font-size:12px;}
            QLineEdit:focus{border:1px solid #6699cc;}
        """)
        il.addWidget(self._nick_edit)

        # v0.5.7: the "Filter to instrument range" checkbox lives
        # alongside the min-N spinbox under the table (both gate what
        # the table shows). The "Show spectrum analyzer & diagnostics"
        # checkbox lives in a footer under the TunerWidget (next to
        # the widgets it controls). The checkboxes themselves are
        # constructed below in _build_ui after the splitter, with the
        # same handlers + persistence — the only thing that changed
        # is the parent layout.

        # Select the saxophone family + the default instrument (Bb tenor
        # since v0.5.7.1). DEFERRED to the end of _build_ui so the
        # InstrumentController exists by the time _on_instr_changed fires
        # — see the construction site near the bottom of this method.

        # Anzeige
        self._grp_disp = QGroupBox(self._t('grp_display'))
        dl = QHBoxLayout(self._grp_disp)
        dl.setContentsMargins(8, 4, 8, 4)
        self._disp_combo = QComboBox()
        self._disp_combo.addItems([self._t('disp_griff'), self._t('disp_klingend')])
        self._disp_combo.currentIndexChanged.connect(self._on_disp_changed)
        dl.addWidget(self._disp_combo)

        # Layout mode override: Auto / List / Grid. Auto picks based on
        # window width; the explicit choices let the user pin the layout
        # they want regardless of how wide the window is.
        self._layout_combo = QComboBox()
        self._layout_combo.addItem(self._t('layout_auto'), 'auto')
        self._layout_combo.addItem(self._t('layout_single'), 'single')
        self._layout_combo.addItem(self._t('layout_matrix'), 'matrix')
        pref = getattr(self._cfg, 'layout_mode_preference', 'auto')
        for i in range(self._layout_combo.count()):
            if self._layout_combo.itemData(i) == pref:
                self._layout_combo.setCurrentIndex(i)
                break
        self._layout_combo.currentIndexChanged.connect(
            self._on_layout_pref_changed)
        dl.addWidget(self._layout_combo)

        # Kammerton
        self._grp_a4 = QGroupBox(self._t('grp_a4'))
        al = QHBoxLayout(self._grp_a4)
        al.setContentsMargins(8, 4, 8, 4)
        self._a4_combo = QComboBox()
        for hz in range(430, 451):
            self._a4_combo.addItem(f'{hz} Hz', hz)
        self._a4_combo.setCurrentIndex(10)   # 440 Hz
        self._a4_combo.setMinimumWidth(100)
        self._a4_combo.currentIndexChanged.connect(self._on_a4_changed)
        al.addWidget(self._a4_combo)

        # Response: pitch-detection smoothing preset. Mirrors a guitar-
        # tuner "Fast/Normal/Slow" toggle. Routed straight into the
        # engine; persisted in cfg.filter_mode.
        self._grp_filter = QGroupBox(self._t('grp_filter'))
        fl = QHBoxLayout(self._grp_filter)
        fl.setContentsMargins(8, 4, 8, 4)
        self._filter_combo = QComboBox()
        self._filter_combo.addItem(self._t('filter_fast'),   'fast')
        self._filter_combo.addItem(self._t('filter_normal'), 'normal')
        self._filter_combo.addItem(self._t('filter_slow'),   'slow')
        cur_mode = getattr(self._cfg, 'filter_mode', FILTER_MODE_DEFAULT)
        for i in range(self._filter_combo.count()):
            if self._filter_combo.itemData(i) == cur_mode:
                self._filter_combo.setCurrentIndex(i)
                break
        self._filter_combo.setToolTip(self._t('filter_tip'))
        self._filter_combo.currentIndexChanged.connect(
            self._on_filter_mode_changed)
        self._filter_combo.setMinimumWidth(100)
        fl.addWidget(self._filter_combo)

        # Sprache
        self._grp_lang = QGroupBox(self._t('grp_language'))
        ll2 = QHBoxLayout(self._grp_lang)
        ll2.setContentsMargins(8, 4, 8, 4)
        self._lang_combo = QComboBox()
        self._lang_combo.addItem('Deutsch', 'de')
        self._lang_combo.addItem('English', 'en')
        self._lang_combo.setMinimumWidth(100)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        ll2.addWidget(self._lang_combo)

        # Buttons
        self._btn_autotune = self._make_btn(self._t('btn_autotune'), '#1a6b3a', self._on_autotune)
        self._btn_record   = self._make_btn(self._t('btn_stop'),     '#b7770d', self._on_record_toggle)
        self._btn_reset    = self._make_btn(self._t('btn_reset'),    '#c0392b', self._on_reset)
        self._btn_txt      = self._make_btn(self._t('btn_txt'),      '#2980b9', self._export_txt)
        self._btn_pdf      = self._make_btn(self._t('btn_pdf'),      '#8e44ad', self._export_pdf)
        self._btn_csv      = self._make_btn(self._t('btn_csv'),      '#16a085', self._export_csv)
        self._btn_chart    = self._make_btn(self._t('btn_chart'),    '#d35400', self._export_chart)
        self._btn_import   = self._make_btn(self._t('btn_import'),   '#7f8c8d', self._import_csv)

        # AUDIO IN chip — Frodo-UX memo: between Language and Import.
        # The chip is the user's at-a-glance health indicator when the
        # tuner goes silent. Clicking opens the picker modal.
        self._audio_chip = AudioChip(self._t)
        self._audio_chip.clicked.connect(self._open_audio_picker)
        self._audio_chip.setMinimumWidth(220)
        # Reflect whatever state the engine has at this point. The
        # engine may not have run start() yet (we wait for the UI to
        # finish building); the chip will repaint when state_changed
        # fires.
        self._audio_chip.update_from_state(
            self._engine.state, '', '', 0)

        # Inputs row: instrument config + audio chip + import.
        toolbar.addWidget(self._grp_instr)
        toolbar.addWidget(self._grp_disp)
        toolbar.addWidget(self._grp_a4)
        toolbar.addWidget(self._grp_filter)
        toolbar.addWidget(self._grp_lang)
        toolbar.addWidget(self._audio_chip)
        toolbar.addWidget(self._btn_import)
        toolbar.addStretch()
        # Actions row: autotune + recording controls + exports.
        toolbar_actions.addWidget(self._btn_autotune)
        toolbar_actions.addWidget(self._btn_record)
        toolbar_actions.addWidget(self._btn_reset)
        toolbar_actions.addStretch()
        toolbar_actions.addWidget(self._btn_txt)
        toolbar_actions.addWidget(self._btn_pdf)
        toolbar_actions.addWidget(self._btn_chart)
        toolbar_actions.addWidget(self._btn_csv)
        root.addWidget(toolbar_container)

        # ── Splitter ──────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Links: Tuner (+ optional spectrogram + diagnostics panel)
        left = QWidget()
        ll3 = QVBoxLayout(left)
        ll3.setContentsMargins(0, 0, 6, 0)
        # Recovery banner — appears only when the engine is in FAILED.
        # Hidden by default; show_for() makes it visible with the right copy.
        self._audio_banner = AudioRecoveryBanner(
            self._t, self._retry_audio, self._open_audio_picker)
        ll3.addWidget(self._audio_banner)
        # v0.6 Phase-4: non-modal info banner for hot-plug (Item 1) and
        # wrong-instrument hint (Item 3). Hidden by default; the GUI calls
        # show_message() to surface it with an optional Switch action and
        # an always-present Dismiss button.
        self._info_banner = InfoBanner(self._t)
        ll3.addWidget(self._info_banner)

        self._tuner = TunerWidget()
        # Tuner gets a fixed-ish vertical slot now that it shares the
        # pane with optional panels. Expanding horizontally but only
        # Preferred vertically lets the spectrogram claim leftover space
        # when the panels are visible.
        self._tuner.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Preferred)
        # Tuner takes its full minimum height (260px) at the top of the
        # left pane. The spectrum analyzer + diagnostics panel sit below
        # it and only consume space when diagnostics are enabled.
        ll3.addWidget(self._tuner)
        self._status_lbl = QLabel(self._t('no_signal'))
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet('color:#888;font-size:13px;padding:4px;')
        ll3.addWidget(self._status_lbl)

        # Sprint 3: drone controls live on the TUNER tab (per the S3 lock),
        # directly under the readout — quick access while tuning against the
        # drone. The 128-GM voice picker lives in SETUP; the compact 5-preset
        # row + volume + semitone are here.
        ll3.addWidget(self._build_drone_bar())

        # v0.5.7: "Show spectrum analyzer & diagnostics" footer row.
        # Lives directly below the tuner so the toggle is adjacent to
        # the widgets it controls. Same handler, same persistence as
        # the old toolbar-resident checkbox — only the parent moved.
        diag_footer = QHBoxLayout()
        diag_footer.setContentsMargins(4, 0, 4, 0)
        self._cb_diag = QCheckBox(self._t('show_diagnostics'))
        self._cb_diag.setChecked(
            bool(getattr(self._cfg, 'show_diagnostics', False)))
        self._cb_diag.setToolTip(self._t('show_diagnostics_tip'))
        self._cb_diag.setStyleSheet("""
            QCheckBox { color: #bbb; font-size: 12px; padding: 2px 4px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
        """)
        self._cb_diag.toggled.connect(self._on_diagnostics_toggled)
        diag_footer.addWidget(self._cb_diag)
        diag_footer.addStretch()
        ll3.addLayout(diag_footer)

        # Spectrum analyzer + diagnostics panels. Always constructed so
        # the toggle is just show/hide — keeps the timer threading
        # consistent and avoids reconstruction cost when flipped
        # repeatedly. Attribute name kept as _spectro to minimize churn
        # against the rest of the file's references.
        self._spectro_grp = QGroupBox(self._t('spectro_title'))
        sg_l = QVBoxLayout(self._spectro_grp)
        sg_l.setContentsMargins(6, 6, 6, 6)
        self._spectro = SpectrumAnalyzerWidget(self._engine if AUDIO_OK else None)
        sg_l.addWidget(self._spectro)
        ll3.addWidget(self._spectro_grp, 1)

        self._data_grp = QGroupBox(self._t('data_panel_title'))
        dg_l = QVBoxLayout(self._data_grp)
        dg_l.setContentsMargins(6, 6, 6, 6)
        self._data_panel = DataPanelWidget(
            self._engine if AUDIO_OK else None,
            self._t,
            lambda: self._notes_count,
            lambda: self._cfg,
        )
        dg_l.addWidget(self._data_panel)
        ll3.addWidget(self._data_grp)

        show_diag = bool(getattr(self._cfg, 'show_diagnostics', False))
        self._spectro_grp.setVisible(show_diag)
        self._data_grp.setVisible(show_diag)

        # Rechts: Tabelle
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        self._table_lbl = QLabel(self._t('table_empty_hint'))
        self._table_lbl.setStyleSheet('font-size:14px;font-weight:bold;color:#ccc;padding:2px 0 6px 0;')
        self._table_lbl.setWordWrap(True)
        rl.addWidget(self._table_lbl)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(self._table_headers())
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget{background:#1a1a24;color:#ddd;gridline-color:#333;
                         font-size:14px;border:none;}
            QTableWidget::item{padding:5px 8px;}
            QTableWidget::item:alternate{background:#1f1f2e;}
            QTableWidget::item:selected{background:#2d4a7a;color:#fff;}
            QHeaderView::section{background:#252535;color:#aaa;font-size:12px;
                                  padding:6px;border:none;border-bottom:1px solid #444;}
        """)
        rl.addWidget(self._table)
        self._bar_delegate = CentBarDelegate(
            self._table, sample_rate_getter=self._engine_sample_rate)
        self._table.setItemDelegateForColumn(5, self._bar_delegate)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Min-N filter under the table: hide notes with fewer measurements
        # than the threshold so single accidental blips don't pollute the
        # view. Default 5 = "actually held". Mirrors the autotune
        # min-n requirement.
        min_n_row = QHBoxLayout()
        min_n_row.setContentsMargins(0, 6, 0, 0)
        self._min_n_lbl = QLabel(self._t('min_n_label'))
        self._min_n_lbl.setStyleSheet('color:#aaa;font-size:12px;')
        self._min_n_spin = QSpinBox()
        self._min_n_spin.setRange(0, 999)
        self._min_n_spin.setValue(int(getattr(self._cfg, 'min_n_visible', 5)))
        self._min_n_spin.setToolTip(self._t('min_n_tip'))
        self._min_n_spin.setMinimumWidth(70)
        self._min_n_spin.setStyleSheet(
            'QSpinBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:2px 6px;font-size:12px;}')
        self._min_n_spin.valueChanged.connect(self._on_min_n_changed)
        min_n_row.addWidget(self._min_n_lbl)
        min_n_row.addWidget(self._min_n_spin)
        # v0.5.7: "Filter to instrument range" checkbox lives in this
        # row alongside the min-N spinbox. Both controls gate what the
        # table shows; co-locating them lets the user adjust scope in
        # one place instead of hunting in the top toolbar. UI logic
        # preserved verbatim: checked = filter ON, unchecked = show all
        # (stored as the inverted cfg.allow_out_of_range).
        self._cb_oor = QCheckBox(self._t('allow_oor'))
        self._cb_oor.setChecked(not self._cfg.allow_out_of_range)
        self._cb_oor.setToolTip(self._t('allow_oor_tip'))
        self._cb_oor.setStyleSheet("""
            QCheckBox { color: #bbb; font-size: 12px; padding: 2px 4px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
        """)
        self._cb_oor.toggled.connect(self._on_oor_toggled)
        min_n_row.addSpacing(16)
        min_n_row.addWidget(self._cb_oor)
        min_n_row.addStretch()
        rl.addLayout(min_n_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 620])
        # Held for closeEvent persistence and restore-on-launch (v0.5.5).
        self._splitter = splitter

        # ── Navigation shell (Sprint 1) ───────────────────────────────────────
        # D4: the global toolbar (built above) stays ABOVE a QTabWidget. The
        # TUNER tab hosts the existing tuner/table splitter VERBATIM — it's the
        # desktop's centerpiece and is more capable than Android's tuner, so it
        # stays untouched rather than shrinking. METRO / DECK / SETUP are homes
        # for later sprints; METRO and DECK are placeholders now but already
        # carry the status-dot indicator hook (green=running on METRO, pulsing
        # red=recording on DECK, mirroring Android's TabBar) so Sprint 2/4 only
        # flips a signal instead of rebuilding the nav. SETUP carries the
        # audio-output device picker + test-tone control this sprint delivers.
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tab_keys = ['tuner', 'metro', 'deck', 'setup']
        self._tabs.addTab(splitter, self._t('nav_tab_tuner'))
        self._tabs.addTab(self._build_metro_tab(), self._t('nav_tab_metro'))
        self._tabs.addTab(self._build_deck_tab(), self._t('nav_tab_deck'))
        self._tabs.addTab(self._build_setup_tab(), self._t('nav_tab_setup'))
        for i, key in enumerate(self._tab_keys):
            self._tabs.setTabToolTip(i, self._t(f'nav_tab_{key}_tip'))
        self._install_tab_indicators()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)
        # Restore the last-active tab (Treebeard's cfg.last_active_tab,
        # allowlist-coerced upstream). Unknown / fresh-install → TUNER.
        self._restore_active_tab()

        # Fensterstil
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#12121a;color:#ddd;}
            QGroupBox{border:1px solid #333;border-radius:6px;font-size:12px;
                      color:#aaa;margin-top:6px;padding-top:4px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;top:-2px;}
            QComboBox{background:#1e1e2e;border:1px solid #444;border-radius:5px;
                      color:#ddd;padding:4px 8px;font-size:13px;min-height:28px;}
            QComboBox:hover{background:#252535;border:1px solid #6699cc;}
            QComboBox::drop-down{border:none;width:20px;background:#1e1e2e;}
            QComboBox QAbstractItemView{background:#1e1e2e;color:#ddd;
                      border:1px solid #444;outline:0;
                      selection-background-color:#34495e;
                      selection-color:#fff;}
            QComboBox QAbstractItemView::item{background:#1e1e2e;color:#ddd;
                      padding:4px 8px;border:none;}
            QComboBox QAbstractItemView::item:selected{background:#34495e;
                      color:#fff;}
            QComboBox QAbstractItemView::item:hover{background:#2a2a3a;
                      color:#fff;}
            QSplitter::handle{background:#333;}
            QTabWidget::pane{border:none;border-top:1px solid #333;top:-1px;}
            QTabBar::tab{background:#161620;color:#999;font-size:12px;
                         font-weight:bold;letter-spacing:1px;
                         padding:8px 18px;margin-right:2px;
                         border:1px solid #333;border-bottom:none;
                         border-top-left-radius:6px;border-top-right-radius:6px;}
            QTabBar::tab:hover{background:#1f1f2e;color:#ccc;}
            QTabBar::tab:selected{background:#12121a;color:#6699cc;
                         border-bottom:2px solid #6699cc;}
        """)

        self.setWindowTitle(self._t('window_title'))

        # Phase-5 extraction: InstrumentController owns family/instr/nickname
        # work, the range editor, custom-instrument registration, and
        # autotune. Instantiated here at the END of _build_ui because every
        # widget it captures (family_combo, instr_combo, nick_edit,
        # a4_combo) now exists. The wrappers on MainWindow (defined
        # further down) delegate to this controller; they MUST NOT be
        # invoked before this point.
        self._instr_ctrl = InstrumentController(
            self,
            cfg=self._cfg, engine=self._engine, log=self._log, t_func=self._t,
            family_combo=self._family_combo,
            instr_combo=self._instr_combo,
            nick_edit=self._nick_edit,
            a4_combo=self._a4_combo,
            get_stats=lambda: self.stats,
            get_display_mode=lambda: self.display,
            get_lang=lambda: self.lang,
            set_instrument=lambda key: setattr(self, 'instrument', key),
            set_a4=self._instr_set_a4,
            on_instrument_changed=self._on_instrument_changed_external,
            on_range_changed=self._on_range_changed_external,
            on_a4_changed=self._on_a4_changed_external,
        )
        # Now safe to populate the family + sub-instrument combos via the
        # controller (their _on_* wrappers delegate here).
        self._select_family_for_instrument(self.instrument)
        self._populate_instrument_combo(select_key=self.instrument)

    # ── Navigation shell helpers (Sprint 1) ──────────────────────────────────
    def _build_setup_tab(self) -> QWidget:
        """SETUP tab. Sprint 1 delivers the audio-OUTPUT controls: an
        output-device picker, the duplex-preference toggle (D5), and a
        test-tone button that proves the output path — Sprint-1 acceptance
        is a tone playing through the mixer while the tuner keeps reading
        the mic (D3: readout stays live). Later sprints grow this tab with
        theme switching and the rest of SETUP parity."""
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(14)

        # Audio output device + duplex preference.
        out_grp = QGroupBox(self._t('setup_output_group'))
        og = QFormLayout(out_grp)
        og.setContentsMargins(12, 10, 12, 10)
        og.setSpacing(8)
        self._out_device_combo = QComboBox()
        self._out_device_combo.setMinimumWidth(320)
        self._out_device_combo.currentIndexChanged.connect(
            self._on_output_device_changed)
        # Handle for the active test-tone source (None while silent).
        self._test_tone_handle = None
        # Cache of the output enumeration from the last combo population, so
        # _on_output_device_changed can resolve a chosen name's host_api
        # without re-querying the engine (N3).
        self._output_devices_cache: list = []
        self._refresh_output_device_combo()
        og.addRow(self._t('setup_output_device'), self._out_device_combo)
        self._cb_prefer_duplex = QCheckBox(self._t('setup_prefer_duplex'))
        self._cb_prefer_duplex.setToolTip(self._t('setup_prefer_duplex_tip'))
        self._cb_prefer_duplex.setChecked(
            bool(getattr(self._cfg, 'output_prefer_duplex', False)))
        self._cb_prefer_duplex.setStyleSheet(
            'QCheckBox{color:#bbb;font-size:12px;padding:2px 4px;}'
            'QCheckBox::indicator{width:14px;height:14px;}')
        self._cb_prefer_duplex.toggled.connect(self._on_prefer_duplex_toggled)
        og.addRow('', self._cb_prefer_duplex)
        outer.addWidget(out_grp)

        # Drone voice — the full 128-GM picker lives here (per the S3 lock); the
        # TUNER drone bar carries the 5-preset quick row. Both drive the same
        # voice and stay in sync via DroneController.on_state_changed.
        voice_grp = QGroupBox(self._t('setup_drone_voice_group'))
        vg = QFormLayout(voice_grp)
        vg.setContentsMargins(12, 10, 12, 10)
        self._drone_voice_combo = QComboBox()
        self._drone_voice_combo.setMinimumWidth(260)
        # Presets first (labelled), then the full GM catalog. Each item's data
        # is the voice id the controller resolves.
        seen_ids = set()
        for v in list(_DRONE_PRESETS) + list(_DRONE_FULL_GM):
            vid, label, program = self._drone_voice_fields(v)
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            self._drone_voice_combo.addItem(label, vid)
        sel_id = str(getattr(self._cfg, 'drone_voice_id', 'organ') or 'organ')
        for i in range(self._drone_voice_combo.count()):
            if self._drone_voice_combo.itemData(i) == sel_id:
                self._drone_voice_combo.setCurrentIndex(i)
                break
        self._drone_voice_combo.currentIndexChanged.connect(
            self._on_drone_voice_changed)
        vg.addRow(self._t('setup_drone_voice'), self._drone_voice_combo)
        outer.addWidget(voice_grp)

        # Test tone — proves the output path end to end.
        tone_grp = QGroupBox(self._t('setup_testtone_group'))
        tg = QVBoxLayout(tone_grp)
        tg.setContentsMargins(12, 10, 12, 10)
        self._btn_test_tone = QPushButton(self._t('setup_testtone_play'))
        self._btn_test_tone.setCheckable(True)
        self._btn_test_tone.setToolTip(self._t('setup_testtone_tip'))
        self._btn_test_tone.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_test_tone.setStyleSheet("""
            QPushButton{background:#16a085;color:#fff;border:none;
                         border-radius:5px;padding:8px 14px;font-size:13px;}
            QPushButton:hover{background:#1abc9c;}
            QPushButton:checked{background:#b7770d;}
        """)
        self._btn_test_tone.toggled.connect(self._on_test_tone_toggled)
        tg.addWidget(self._btn_test_tone)
        # Inline feedback when the tone can't start (no output device yet, or
        # the engine output mirror isn't available). Hidden until needed so
        # the button never silently lies about playing (N2).
        self._test_tone_status = QLabel('')
        self._test_tone_status.setWordWrap(True)
        self._test_tone_status.setStyleSheet('color:#c0392b;font-size:12px;'
                                              'padding:4px 2px 0 2px;')
        self._test_tone_status.setVisible(False)
        tg.addWidget(self._test_tone_status)
        outer.addWidget(tone_grp)
        outer.addStretch()
        return w

    def _current_output_selection(self) -> 'DeviceSelection':
        """The persisted output device as a DeviceSelection (mirrors the
        input path's _device_selection_from_cfg). Empty name = system
        default; samplerate 0 = auto-negotiate."""
        return DeviceSelection(
            name=str(getattr(self._cfg, 'output_device_name', '') or ''),
            host_api=str(getattr(self._cfg, 'output_device_host_api', '') or ''),
            samplerate=int(getattr(self._cfg, 'output_device_samplerate', 0) or 0),
        )

    def _ensure_output_open(self) -> bool:
        """Make sure an output stream is running, opening the configured device
        only if it ISN'T already. Returns True if output is running afterward.
        Deliberately does NOT reopen an already-running stream — a reopen runs
        mixer.reset_clock(), which would wipe a running metronome's scheduled
        beat (the rolling-horizon scheduler only reschedules from a fired beat).
        New sources register on the existing stream fine."""
        if getattr(self._engine, 'output_running', False):
            return True
        if hasattr(self._engine, 'open_output_device'):
            try:
                self._engine.open_output_device(self._current_output_selection())
            except Exception:
                pass
        return bool(getattr(self._engine, 'output_running', False))

    def _reopen_output_bracketed(self, sel: 'DeviceSelection') -> None:
        """Open a (different) output device, bracketing a running metronome with
        stop()/start() so the mixer.reset_clock() inside the reopen can't
        silently kill its beat chain (Option A — the metronome pauses across a
        hardware switch, then resumes on the new device). The GUI orchestrates
        the device change, so the bracket lives here, not in the engine (stays
        client-ignorant) or the controller (stable contract). NOTE: covers only
        GUI-orchestrated switches; engine-internal hot-plug auto-recovery is out
        of reach here (Option B / backlog)."""
        ctrl = getattr(self, '_metro_ctrl', None)
        was_running = ctrl is not None and ctrl.is_running()
        if was_running:
            ctrl.stop()
        if hasattr(self._engine, 'open_output_device'):
            try:
                self._engine.open_output_device(sel)
            except Exception:
                pass
        if was_running and getattr(self._engine, 'output_running', False):
            sr = int(getattr(self._engine, 'output_samplerate', 0) or 0)
            if sr and hasattr(ctrl, 'set_samplerate'):
                try:
                    ctrl.set_samplerate(sr)
                except Exception:
                    pass
            ctrl.start()

    def _refresh_output_device_combo(self) -> None:
        """Fill the SETUP output combo: System default first, then the
        engine's output enumeration. hasattr-guarded so the GUI still
        launches if the engine output mirror hasn't landed yet (the combo
        then shows only System default)."""
        combo = self._out_device_combo
        was_blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._t('setup_output_default'), '')
        devices = []
        if hasattr(self._engine, 'refresh_output_devices'):
            try:
                devices = self._engine.refresh_output_devices() or []
            except Exception:
                devices = []
        self._output_devices_cache = list(devices)
        saved_name = str(getattr(self._cfg, 'output_device_name', '') or '')
        select_idx = 0
        for d in devices:
            combo.addItem(f'{d.name} · {d.host_api}', d.name)
            if saved_name and d.name == saved_name:
                select_idx = combo.count() - 1
        combo.setCurrentIndex(select_idx)
        combo.blockSignals(was_blocked)

    def _on_output_device_changed(self, _idx: int) -> None:
        name = self._out_device_combo.currentData() or ''
        host_api = ''
        # Resolve host_api from the cache captured at combo population (N3) —
        # no re-query of the engine.
        if name:
            for d in self._output_devices_cache:
                if d.name == name:
                    host_api = d.host_api
                    break
        self._cfg.output_device_name = name
        self._cfg.output_device_host_api = host_api
        self._cfg.output_device_samplerate = 0   # auto-negotiate
        try:
            sax_config.save_config(self._cfg)
        except Exception:
            pass
        # Re-open on the new device. Brackets a running metronome (Option A) so
        # the reopen's mixer.reset_clock() can't silently kill its beat chain.
        self._reopen_output_bracketed(self._current_output_selection())

    def _on_prefer_duplex_toggled(self, checked: bool) -> None:
        # Persist the preference; it's consumed the next time the output
        # device opens. We don't re-open mid-tone — that would chop a
        # playing test tone for a setting that costs nothing to defer.
        self._cfg.output_prefer_duplex = bool(checked)
        try:
            sax_config.save_config(self._cfg)
        except Exception:
            pass

    def _on_test_tone_toggled(self, on: bool) -> None:
        """Start/stop a 440 Hz test tone through the mixer. The tuner is
        deliberately NOT muted — D3 says the readout stays live while output
        sounds; this is the manual proof of that path."""
        if on:
            self._test_tone_status.setVisible(False)
            # Make sure an output stream exists before sounding — without
            # reopening an already-running one (that would reset the mixer
            # clock and kill a running metronome).
            self._ensure_output_open()
            if hasattr(self._engine, 'start_test_tone'):
                try:
                    self._test_tone_handle = self._engine.start_test_tone(440.0)
                except Exception:
                    self._test_tone_handle = None
            else:
                self._test_tone_handle = None
            if self._test_tone_handle is None:
                # Nothing is sounding — don't let the button claim "Stop".
                # Revert the toggle (blocking its signal to avoid re-entrancy)
                # and tell the user why (N2).
                blocked = self._btn_test_tone.blockSignals(True)
                self._btn_test_tone.setChecked(False)
                self._btn_test_tone.blockSignals(blocked)
                self._btn_test_tone.setText(self._t('setup_testtone_play'))
                self._test_tone_status.setText(self._t('setup_testtone_failed'))
                self._test_tone_status.setVisible(True)
                return
            self._btn_test_tone.setText(self._t('setup_testtone_stop'))
        else:
            self._btn_test_tone.setText(self._t('setup_testtone_play'))
            self._test_tone_status.setVisible(False)
            if (self._test_tone_handle is not None
                    and hasattr(self._engine, 'stop_test_tone')):
                try:
                    self._engine.stop_test_tone(self._test_tone_handle)
                except Exception:
                    pass
            self._test_tone_handle = None

    def _install_tab_indicators(self) -> None:
        """Attach status-dot widgets to the METRO and DECK tabs, hidden
        until set_metro_running / set_deck_recording flip them. Mirrors
        Android's TabBar (TabBar.tsx): solid green when the metronome runs,
        pulsing red while the deck records. Building the hook now (Sprint 1)
        means Sprints 2/4 only emit a signal, not rebuild the nav."""
        bar = self._tabs.tabBar()
        self._metro_dot = _StatusDot()
        self._metro_dot.set_color('#2ecc71')   # green
        bar.setTabButton(self._tab_keys.index('metro'),
                         QTabBar.ButtonPosition.RightSide, self._metro_dot)
        self._deck_dot = _StatusDot()
        self._deck_dot.set_color('#e74c3c')     # red
        bar.setTabButton(self._tab_keys.index('deck'),
                         QTabBar.ButtonPosition.RightSide, self._deck_dot)
        # Hide AFTER setTabButton: setTabButton shows the button widget, so a
        # setVisible(False) before it gets overridden — the dots would then
        # show at launch with nothing running/recording. Hide them here.
        self._metro_dot.setVisible(False)
        self._deck_dot.setVisible(False)
        # 1 Hz pulse for the DECK recording dot (Android fades at ~1 Hz).
        # Dormant until set_deck_recording(True) starts it.
        self._deck_pulse_bright = True
        self._deck_pulse_timer = QTimer(self)
        self._deck_pulse_timer.setInterval(500)
        self._deck_pulse_timer.timeout.connect(self._pulse_deck_dot)

    def _pulse_deck_dot(self) -> None:
        self._deck_pulse_bright = not self._deck_pulse_bright
        self._deck_dot.set_color('#e74c3c' if self._deck_pulse_bright
                                 else '#6e2420')

    def set_metro_running(self, running: bool) -> None:
        """Public hook for the (future) MetronomeController: show a solid
        green dot on the METRO tab while the metronome is running."""
        self._metro_dot.setVisible(bool(running))

    def set_deck_recording(self, recording: bool) -> None:
        """Public hook for the DeckController: show a pulsing red dot on the
        DECK tab while a take is recording. No-ops if the indicator hasn't been
        built yet (the deck tab is built before _install_tab_indicators runs)."""
        if not hasattr(self, '_deck_dot'):
            return
        if recording:
            self._deck_pulse_bright = True
            self._deck_dot.set_color('#e74c3c')
            self._deck_dot.setVisible(True)
            if not self._deck_pulse_timer.isActive():
                self._deck_pulse_timer.start()
        else:
            self._deck_pulse_timer.stop()
            self._deck_dot.setVisible(False)

    def _on_tab_changed(self, idx: int) -> None:
        # Persist the active tab live (belt); closeEvent is the suspenders.
        if 0 <= idx < len(self._tab_keys):
            self._cfg.last_active_tab = self._tab_keys[idx]

    def _restore_active_tab(self) -> None:
        key = str(getattr(self._cfg, 'last_active_tab', 'tuner') or 'tuner')
        if key in self._tab_keys:
            self._tabs.setCurrentIndex(self._tab_keys.index(key))

    # ── METRO tab (Sprint 2) ──────────────────────────────────────────────────
    # These mirror Android useMetronome (BPM_MIN/MAX, the four parity presets).
    # When sax_metronome.py lands, reconcile by importing its authoritative
    # constants rather than duplicating them here.
    METRO_TIME_SIGS = ('2/4', '3/4', '4/4', '6/8')
    METRO_BPM_MIN = 30
    METRO_BPM_MAX = 300

    def _build_metro_tab(self) -> QWidget:
        """The metronome panel. The static UI is built now (Sprint 2); audio is
        driven by MetronomeController (sax_metronome.py) once it lands. Every
        controller call is guarded, so the panel renders and the pure-UI
        controls (BPM display, time-sig, volume) respond even while the
        controller is absent. Audio actions (start, tap) surface an inline
        'not available yet' status rather than lying about playing — the same
        honesty discipline as the test-tone button. The GUI-facing controller
        API is the channel-blessed contract: bpm/time_sig/volume/running,
        set_bpm/nudge_bpm/set_time_sig/set_volume, register_tap()->Optional[int],
        start/stop/toggle/is_running, on_state_changed."""
        # Local UI state, seeded from persisted config (Treebeard's Sprint-2
        # fields, read defensively).
        self._metro_bpm = self._clamp_bpm(int(getattr(self._cfg, 'last_bpm', 100)))
        self._metro_time_sig = str(getattr(self._cfg, 'last_time_sig', '4/4'))
        if self._metro_time_sig not in self.METRO_TIME_SIGS:
            self._metro_time_sig = '4/4'
        self._metro_running = False
        # The metronome controller (sax_metronome.py). Constructed here when the
        # module is present AND the engine exposes a mixer; stays None (an inert
        # panel) otherwise — e.g. headless with audio disabled. on_state_changed
        # is the zero-arg callback that re-syncs this panel from controller
        # state (the controller is the single source of truth for run/bpm/sig).
        self._metro_ctrl = None
        if (_MetronomeController is not None
                and getattr(self._engine, 'mixer', None) is not None):
            try:
                sr = int(getattr(self._engine, 'output_samplerate', 0)
                         or self._engine_sample_rate() or 44100)
                self._metro_ctrl = _MetronomeController(
                    self._engine.mixer, sr,
                    bpm=self._metro_bpm,
                    time_sig=self._metro_time_sig,
                    volume=float(getattr(self._cfg, 'click_volume', 1.0)),
                    on_state_changed=self._on_metro_state_changed)
            except Exception:
                self._metro_ctrl = None

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Big BPM readout.
        self._metro_bpm_lbl = QLabel(str(self._metro_bpm))
        self._metro_bpm_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._metro_bpm_lbl.setStyleSheet(
            'color:#6699cc;font-size:64px;font-weight:bold;')
        outer.addWidget(self._metro_bpm_lbl)
        unit = QLabel(self._t('metro_bpm'))
        unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit.setStyleSheet('color:#888;font-size:14px;letter-spacing:3px;')
        outer.addWidget(unit)

        # BPM controls: −  [slider 30..300]  +   |  Tap.
        bpm_row = QHBoxLayout()
        btn_down = QToolButton()
        btn_down.setText('−')
        btn_down.setToolTip(self._t('metro_nudge_down_tip'))
        btn_down.clicked.connect(lambda: self._metro_nudge(-1))
        btn_up = QToolButton()
        btn_up.setText('+')
        btn_up.setToolTip(self._t('metro_nudge_up_tip'))
        btn_up.clicked.connect(lambda: self._metro_nudge(1))
        for b in (btn_down, btn_up):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                'QToolButton{background:#1e1e2e;color:#ddd;border:1px solid '
                '#444;border-radius:5px;padding:4px 12px;font-size:18px;'
                'min-width:30px;}QToolButton:hover{border:1px solid #6699cc;}')
        self._metro_slider = QSlider(Qt.Orientation.Horizontal)
        self._metro_slider.setRange(self.METRO_BPM_MIN, self.METRO_BPM_MAX)
        self._metro_slider.setValue(self._metro_bpm)
        self._metro_slider.valueChanged.connect(self._on_metro_slider)
        self._btn_metro_tap = QPushButton(self._t('metro_tap'))
        self._btn_metro_tap.setToolTip(self._t('metro_tap_tip'))
        self._btn_metro_tap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_metro_tap.clicked.connect(self._on_metro_tap)
        self._btn_metro_tap.setStyleSheet(
            'QPushButton{background:#34495e;color:#eee;border:none;'
            'border-radius:5px;padding:6px 16px;font-size:13px;}'
            'QPushButton:hover{background:#3d566e;}')
        bpm_row.addWidget(btn_down)
        bpm_row.addWidget(self._metro_slider, 1)
        bpm_row.addWidget(btn_up)
        bpm_row.addSpacing(12)
        bpm_row.addWidget(self._btn_metro_tap)
        outer.addLayout(bpm_row)

        # Time-signature selector (exclusive).
        ts_grp = QGroupBox(self._t('metro_timesig'))
        ts_l = QHBoxLayout(ts_grp)
        ts_l.setContentsMargins(12, 8, 12, 8)
        self._metro_ts_btns: dict = {}
        for ts in self.METRO_TIME_SIGS:
            b = QPushButton(ts)
            b.setCheckable(True)
            b.setChecked(ts == self._metro_time_sig)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _checked, t=ts: self._on_metro_timesig(t))
            b.setStyleSheet(
                'QPushButton{background:#1e1e2e;color:#ccc;border:1px solid '
                '#444;border-radius:5px;padding:6px 14px;font-size:14px;}'
                'QPushButton:checked{background:#2d4a7a;color:#fff;border:1px '
                'solid #6699cc;}QPushButton:hover{border:1px solid #6699cc;}')
            self._metro_ts_btns[ts] = b
            ts_l.addWidget(b)
        outer.addWidget(ts_grp)

        # Click volume.
        vol_grp = QGroupBox(self._t('metro_volume'))
        vol_l = QHBoxLayout(vol_grp)
        vol_l.setContentsMargins(12, 8, 12, 8)
        self._metro_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._metro_vol_slider.setRange(0, 100)
        vol0 = int(round(float(getattr(self._cfg, 'click_volume', 1.0)) * 100))
        self._metro_vol_slider.setValue(max(0, min(100, vol0)))
        self._metro_vol_slider.valueChanged.connect(self._on_metro_volume)
        self._metro_vol_lbl = QLabel(f'{self._metro_vol_slider.value()}%')
        self._metro_vol_lbl.setStyleSheet(
            'color:#aaa;font-size:12px;min-width:44px;')
        vol_l.addWidget(self._metro_vol_slider, 1)
        vol_l.addWidget(self._metro_vol_lbl)
        outer.addWidget(vol_grp)

        # Start / stop (the run toggle; green→red on run).
        self._btn_metro_start = QPushButton(self._t('metro_start'))
        self._btn_metro_start.setCheckable(True)
        self._btn_metro_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_metro_start.toggled.connect(self._on_metro_toggle)
        self._btn_metro_start.setStyleSheet(
            'QPushButton{background:#1a6b3a;color:#fff;border:none;'
            'border-radius:6px;padding:12px;font-size:16px;font-weight:bold;}'
            'QPushButton:hover{background:#218a4b;}'
            'QPushButton:checked{background:#c0392b;}')
        outer.addWidget(self._btn_metro_start)

        # Inline status — shown only when an audio action can't run yet, so the
        # controls never silently lie.
        self._metro_status = QLabel('')
        self._metro_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._metro_status.setWordWrap(True)
        self._metro_status.setStyleSheet(
            'color:#c0392b;font-size:12px;padding-top:4px;')
        self._metro_status.setVisible(False)
        outer.addWidget(self._metro_status)
        outer.addStretch()
        return w

    def _clamp_bpm(self, v: int) -> int:
        return max(self.METRO_BPM_MIN, min(self.METRO_BPM_MAX, int(v)))

    def _set_metro_bpm(self, bpm: int, *, from_slider: bool = False) -> None:
        bpm = self._clamp_bpm(bpm)
        self._metro_bpm = bpm
        self._metro_bpm_lbl.setText(str(bpm))
        if not from_slider and self._metro_slider.value() != bpm:
            blocked = self._metro_slider.blockSignals(True)
            self._metro_slider.setValue(bpm)
            self._metro_slider.blockSignals(blocked)
        self._cfg.last_bpm = bpm
        if self._metro_ctrl is not None:
            self._metro_ctrl.set_bpm(bpm)

    def _metro_nudge(self, delta: int) -> None:
        self._set_metro_bpm(self._metro_bpm + delta)

    def _on_metro_slider(self, value: int) -> None:
        self._set_metro_bpm(value, from_slider=True)

    def _on_metro_timesig(self, ts: str) -> None:
        self._metro_time_sig = ts
        for key, btn in self._metro_ts_btns.items():
            blocked = btn.blockSignals(True)
            btn.setChecked(key == ts)
            btn.blockSignals(blocked)
        self._cfg.last_time_sig = ts
        if self._metro_ctrl is not None:
            self._metro_ctrl.set_time_sig(ts)

    def _on_metro_volume(self, value: int) -> None:
        self._metro_vol_lbl.setText(f'{value}%')
        vol = value / 100.0
        self._cfg.click_volume = vol
        if self._metro_ctrl is not None:
            self._metro_ctrl.set_volume(vol)

    def _on_metro_tap(self) -> None:
        if self._metro_ctrl is None:
            self._metro_status.setText(self._t('metro_unavailable'))
            self._metro_status.setVisible(True)
            return
        self._metro_status.setVisible(False)
        # The controller updates its own bpm and fires on_state_changed, which
        # re-syncs the readout here — no manual set needed.
        self._metro_ctrl.register_tap()

    def _metro_revert_start(self) -> None:
        """Un-check the Start button (without re-firing the toggle) and show
        the unavailable status — used whenever the metronome can't actually
        sound, so the button never claims to be running."""
        blocked = self._btn_metro_start.blockSignals(True)
        self._btn_metro_start.setChecked(False)
        self._btn_metro_start.blockSignals(blocked)
        self._btn_metro_start.setText(self._t('metro_start'))
        self._metro_status.setText(self._t('metro_unavailable'))
        self._metro_status.setVisible(True)

    def _on_metro_toggle(self, on: bool) -> None:
        if self._metro_ctrl is None:
            self._metro_revert_start()
            return
        if not on:
            self._metro_ctrl.stop()   # fires on_state_changed
            return
        self._metro_status.setVisible(False)
        # The metronome sounds through the output mixer — ensure an output
        # stream is open before claiming to run (same honesty as the test
        # tone: don't show 'running' if nothing can be heard). _ensure_output_open
        # won't reopen an already-running stream, so starting the metronome
        # while a test tone plays doesn't reset the clock.
        if not self._ensure_output_open():
            self._metro_revert_start()
            return
        # Match the controller's clock to the (possibly re-negotiated) output
        # rate before it schedules beats.
        sr = int(getattr(self._engine, 'output_samplerate', 0) or 0)
        if sr and hasattr(self._metro_ctrl, 'set_samplerate'):
            try:
                self._metro_ctrl.set_samplerate(sr)
            except Exception:
                pass
        self._metro_ctrl.start()      # fires on_state_changed -> dot + label

    def _on_metro_state_changed(self) -> None:
        """Zero-arg slot for MetronomeController.on_state_changed (GUI thread),
        fired on every bpm/time-sig/volume/running change. The controller is
        the single source of truth: pull state from it and sync the panel so
        the run dot, Start/Stop label, BPM readout, and time-sig can't drift."""
        ctrl = self._metro_ctrl
        if ctrl is None:
            return
        running = bool(ctrl.is_running())
        self._metro_running = running
        blocked = self._btn_metro_start.blockSignals(True)
        self._btn_metro_start.setChecked(running)
        self._btn_metro_start.blockSignals(blocked)
        self._btn_metro_start.setText(
            self._t('metro_stop') if running else self._t('metro_start'))
        self.set_metro_running(running)
        # Sync BPM (e.g. tap-tempo changed it inside the controller) + persist.
        bpm = self._clamp_bpm(int(ctrl.bpm))
        if bpm != self._metro_bpm:
            self._metro_bpm = bpm
            self._metro_bpm_lbl.setText(str(bpm))
            b2 = self._metro_slider.blockSignals(True)
            self._metro_slider.setValue(bpm)
            self._metro_slider.blockSignals(b2)
            self._cfg.last_bpm = bpm
        # Sync time-signature + persist.
        ts = str(ctrl.time_sig)
        if ts in self._metro_ts_btns and ts != self._metro_time_sig:
            self._metro_time_sig = ts
            for k, b in self._metro_ts_btns.items():
                bb = b.blockSignals(True)
                b.setChecked(k == ts)
                b.blockSignals(bb)
            self._cfg.last_time_sig = ts

    # ── Tape deck (Sprint 4) ──────────────────────────────────────────────────
    # Android useDeck parity: record the MIC to a single take, play it back
    # through the mixer, export it to a WAV. DeckController (sax_deck.py) owns the
    # state machine + the engine input-recording tap + WAV io; this panel is its
    # transport and is inert+guarded until the module lands. Honesty discipline:
    # the pulsing-red dot + a 'recording' state appear ONLY when the controller
    # actually armed the mic input — never on a bare button press.
    #
    # The one wire beyond the drone/metro pattern: deck.pump() runs on a GUI-side
    # timer. Playback-end and cap-hit transitions are detected there (the audio
    # thread only flips a lock-free .finished flag); pump() performs the
    # transition and fires on_state_changed ON THE GUI THREAD, so buttons are
    # always relabelled on-thread (Sauron's contract, 3550).
    DECK_STATES = ('idle', 'recording', 'have_take', 'playing')

    def _build_deck_tab(self) -> QWidget:
        """The DECK tab: Record / Stop / Play / Export transport with
        state-driven enable + labels, an inline status line, and the pulsing-red
        recording dot (set_deck_recording, built Sprint 1). Wired to
        DeckController once sax_deck lands; inert + guarded until then."""
        self._deck_played_once = False
        self._deck_last_state = None
        self._deck_last_can_record = None
        # Construct the controller (None → inert transport). Mirrors drone/metro,
        # plus max_seconds (Sauron's contract) from the clamped config field.
        self._deck_ctrl = None
        if (_DeckController is not None
                and getattr(self._engine, 'mixer', None) is not None):
            try:
                sr = int(getattr(self._engine, 'output_samplerate', 0)
                         or self._engine_sample_rate() or 44100)
                self._deck_ctrl = _DeckController(
                    self._engine.mixer, sr,
                    engine=self._engine,           # arms the input-recording tap
                    max_seconds=float(getattr(self._cfg, 'deck_max_seconds',
                                              300.0) or 300.0),
                    scratch_dir=(str(getattr(self._cfg, 'deck_scratch_dir', '')
                                     or '') or None),
                    on_state_changed=self._on_deck_state_changed)
            except Exception:
                self._deck_ctrl = None

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel(self._t('deck_group'))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color:#888;font-size:18px;font-weight:bold;'
                            'letter-spacing:3px;')
        outer.addWidget(title)

        # Big state readout: empty / recording / take ready / playing.
        self._deck_state_lbl = QLabel('')
        self._deck_state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._deck_state_lbl.setStyleSheet('color:#6699cc;font-size:20px;'
                                           'font-weight:bold;padding:8px;')
        outer.addWidget(self._deck_state_lbl)

        # Transport: Record (toggle, green→red) · Stop · Play/Replay · Export.
        row = QHBoxLayout()
        self._btn_deck_record = QPushButton(self._t('deck_record'))
        self._btn_deck_record.setCheckable(True)
        self._btn_deck_record.setToolTip(self._t('deck_record_tip'))
        self._btn_deck_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_deck_record.toggled.connect(self._on_deck_record_toggle)
        self._btn_deck_record.setStyleSheet(
            'QPushButton{background:#1a6b3a;color:#fff;border:none;border-radius:'
            '6px;padding:12px 18px;font-size:15px;font-weight:bold;}'
            'QPushButton:hover{background:#218a4b;}'
            'QPushButton:checked{background:#c0392b;}'
            'QPushButton:disabled{background:#222;color:#666;}')
        self._btn_deck_stop = QPushButton(self._t('deck_stop'))
        self._btn_deck_play = QPushButton(self._t('deck_play'))
        self._btn_deck_play.setToolTip(self._t('deck_play_tip'))
        self._btn_deck_export = QPushButton(self._t('deck_export'))
        self._btn_deck_export.setToolTip(self._t('deck_export_tip'))
        self._btn_deck_stop.clicked.connect(self._on_deck_stop)
        self._btn_deck_play.clicked.connect(self._on_deck_play)
        self._btn_deck_export.clicked.connect(self._on_deck_export)
        for b in (self._btn_deck_stop, self._btn_deck_play,
                  self._btn_deck_export):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                'QPushButton{background:#34495e;color:#eee;border:none;'
                'border-radius:6px;padding:12px 18px;font-size:15px;}'
                'QPushButton:hover{background:#3d566e;}'
                'QPushButton:disabled{background:#222;color:#666;}')
        row.addWidget(self._btn_deck_record)
        row.addWidget(self._btn_deck_stop)
        row.addWidget(self._btn_deck_play)
        row.addWidget(self._btn_deck_export)
        outer.addLayout(row)

        # Parity note: records the mic only, a single take.
        hint = QLabel(self._t('deck_hint'))
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#666;font-size:11px;padding-top:2px;')
        outer.addWidget(hint)

        # Inline status — shown only when an action can't run / fails or to
        # confirm an export, so the transport never silently lies.
        self._deck_status = QLabel('')
        self._deck_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._deck_status.setWordWrap(True)
        self._deck_status.setVisible(False)
        outer.addWidget(self._deck_status)
        outer.addStretch()

        # GUI-thread pump tick: drives playback-end + cap-hit transitions and
        # keeps the can_record() button-enable fresh as the mic opens/closes.
        self._deck_pump_timer = QTimer(self)
        self._deck_pump_timer.setInterval(100)
        self._deck_pump_timer.timeout.connect(self._deck_pump)
        if self._deck_ctrl is not None:
            self._deck_pump_timer.start()

        # Paint the initial (idle/empty, or controller-driven) transport.
        self._sync_deck_transport()
        return w

    def _deck_state(self) -> str:
        """Current deck state normalized into DECK_STATES. Tolerates a str or an
        enum, and hyphen OR underscore ('have-take'/'have_take'), so the GUI
        can't drift from sax_deck's exact spelling (str-enum, value 'have-take')."""
        ctrl = self._deck_ctrl
        if ctrl is None:
            return 'idle'
        raw = getattr(ctrl, 'state', 'idle')
        val = getattr(raw, 'value', None) or getattr(raw, 'name', None) or raw
        s = str(val).replace('-', '_').lower()
        return s if s in self.DECK_STATES else 'idle'

    def _deck_can_record(self) -> bool:
        """Honest pre-click probe: can a take be armed right now? Backed by the
        controller's can_record() (engine.input_running). Optimistic only when
        the controller lacks the probe — start_record()'s False-return is the
        authoritative post-click check either way."""
        ctrl = self._deck_ctrl
        if ctrl is None:
            return False
        if hasattr(ctrl, 'can_record'):
            try:
                return bool(ctrl.can_record())
            except Exception:
                return False
        return True

    def _sync_deck_transport(self) -> None:
        """Single source of truth: paint every transport control, the recording
        dot, and the state readout from the controller's state. Called at build,
        on every on_state_changed, and on the pump tick (for can_record changes)."""
        state = self._deck_state()
        can_record = self._deck_can_record()
        recording = state == 'recording'
        playing = state == 'playing'
        have_take = state in ('have_take', 'playing')

        # Record toggle: checked iff recording; enabled while recording (so the
        # untoggle can stop it) or when a fresh arm is honestly possible.
        blocked = self._btn_deck_record.blockSignals(True)
        self._btn_deck_record.setChecked(recording)
        self._btn_deck_record.blockSignals(blocked)
        self._btn_deck_record.setEnabled(
            recording or (state in ('idle', 'have_take') and can_record))
        # Stop: only while something is sounding/capturing.
        self._btn_deck_stop.setEnabled(recording or playing)
        # Play/Replay: a take exists (have-take → play; playing → restart@0).
        self._btn_deck_play.setEnabled(have_take)
        self._btn_deck_play.setText(
            self._t('deck_replay') if (self._deck_played_once or playing)
            else self._t('deck_play'))
        # Export: whenever a take exists.
        self._btn_deck_export.setEnabled(have_take)

        # The pulsing-red dot is driven ONLY by a real 'recording' state — never
        # by a bare button press (the honesty seam Treebeard locks).
        self.set_deck_recording(recording)

        key = {'idle': 'deck_status_empty', 'recording': 'deck_status_recording',
               'have_take': 'deck_status_have_take',
               'playing': 'deck_status_playing'}[state]
        self._deck_state_lbl.setText(self._t(key))

        self._deck_last_state = state
        self._deck_last_can_record = can_record

    def _deck_show_status(self, key: str, *, error: bool = True) -> None:
        """Inline transport status. Red for can't/failed, green for confirmation
        (export done) — never leave a stale message of the wrong colour."""
        self._deck_status.setStyleSheet(
            'color:%s;font-size:12px;padding-top:4px;'
            % ('#c0392b' if error else '#2ecc71'))
        self._deck_status.setText(self._t(key))
        self._deck_status.setVisible(True)

    def _on_deck_state_changed(self) -> None:
        """Zero-arg slot for DeckController.on_state_changed — always fires on
        the GUI thread (Sauron marshals playback-end/cap-hit through pump()).
        Clears stale status and repaints the transport from the new state."""
        self._deck_status.setVisible(False)
        self._sync_deck_transport()

    def _deck_pump(self) -> None:
        """GUI-thread tick: let the controller detect playback-end / cap-hit
        (it fires on_state_changed when a transition happens), then re-sync if
        the state or the can_record() probe changed (e.g. the mic opened)."""
        ctrl = self._deck_ctrl
        if ctrl is None:
            return
        try:
            if hasattr(ctrl, 'pump'):
                ctrl.pump()          # may fire on_state_changed → _sync
        except Exception:
            pass
        if (self._deck_state() != self._deck_last_state
                or self._deck_can_record() != self._deck_last_can_record):
            self._sync_deck_transport()

    def _deck_revert_record(self) -> None:
        """Un-check Record (without re-firing the toggle) and show 'unavailable'
        — used whenever recording can't actually arm, so the button never claims
        to be recording and the dot stays hidden."""
        blocked = self._btn_deck_record.blockSignals(True)
        self._btn_deck_record.setChecked(False)
        self._btn_deck_record.blockSignals(blocked)
        self.set_deck_recording(False)
        self._deck_show_status('deck_unavailable')

    def _on_deck_record_toggle(self, on: bool) -> None:
        if self._deck_ctrl is None:
            self._deck_revert_record()
            return
        if not on:
            # Un-toggling Record while recording = stop the take.
            try:
                self._deck_ctrl.stop()      # fires on_state_changed
            except Exception:
                self._sync_deck_transport()
            return
        self._deck_status.setVisible(False)
        # Recording arms the MIC INPUT (not the output). The controller opens it
        # and returns False if it can't — then revert + status, so there is no
        # false 'recording' state when the input isn't open.
        ok = False
        try:
            ok = bool(self._deck_ctrl.start_record())
        except Exception:
            ok = False
        if ok:
            self._deck_played_once = False   # a fresh take replaces the old one
            # success → controller fires on_state_changed → recording paint.
        else:
            self._deck_revert_record()
            self._sync_deck_transport()

    def _on_deck_stop(self) -> None:
        if self._deck_ctrl is None:
            return
        try:
            self._deck_ctrl.stop()           # fires on_state_changed
        except Exception:
            self._sync_deck_transport()

    def _on_deck_play(self) -> None:
        if self._deck_ctrl is None:
            self._deck_show_status('deck_play_unavailable')
            return
        self._deck_status.setVisible(False)
        # Playback sounds through the output mixer — ensure an output stream is
        # open before claiming to play (don't show 'playing' if nothing can be
        # heard). Mirrors drone/metro.
        if not self._ensure_output_open():
            self._deck_show_status('deck_play_unavailable')
            return
        sr = int(getattr(self._engine, 'output_samplerate', 0) or 0)
        if sr and hasattr(self._deck_ctrl, 'set_samplerate'):
            try:
                self._deck_ctrl.set_samplerate(sr)
            except Exception:
                pass
        ok = False
        try:
            ok = bool(self._deck_ctrl.play())   # fires on_state_changed
        except Exception:
            ok = False
        if ok:
            self._deck_played_once = True
        else:
            self._deck_show_status('deck_play_unavailable')
            self._sync_deck_transport()

    def _on_deck_export(self) -> None:
        if self._deck_ctrl is None:
            self._deck_show_status('deck_export_fail')
            return
        # File dialog — reuse sax_export's getSaveFileName pattern, .wav filter.
        start = str(getattr(self._cfg, 'last_take_path', '') or '')
        path, _sel = QFileDialog.getSaveFileName(
            self, self._t('deck_export_title'), start,
            self._t('deck_export_filter'))
        if not path:
            return
        if not path.lower().endswith('.wav'):
            path += '.wav'
        ok = False
        try:
            ok = bool(self._deck_ctrl.export(path))
        except Exception:
            ok = False
        if ok:
            self._cfg.last_take_path = path
            self._deck_show_status('deck_export_done', error=False)
        else:
            self._deck_show_status('deck_export_fail')

    # ── Drone + pitch pipes (Sprint 3) ────────────────────────────────────────
    @staticmethod
    def _drone_voice_fields(v) -> tuple:
        """(id, label, program) for a drone-voice entry, accepting either the
        sax_drone DroneVoice object (.id/.label/.program) or the fallback
        (id, label, program) tuple."""
        if isinstance(v, tuple):
            return v[0], v[1], v[2]
        return v.id, v.label, v.program

    @staticmethod
    def _fmt_semitone(n: int) -> str:
        return f'{n:+d}' if n else '0'

    def _build_drone_bar(self) -> QWidget:
        """The drone strip on the TUNER tab: on/off + 5-preset row + volume +
        semitone steppers + a pitch-pipes launcher. Wired to DroneController
        (sax_drone.py) once it lands; inert + guarded until then. Turning the
        drone on requires an output stream (honesty discipline — no 'on' state
        if nothing can sound). The 128-GM picker lives in SETUP and stays in
        sync via the controller's on_state_changed (single source of truth)."""
        # Seed persisted state (Treebeard's Sprint-3 config fields, defensive).
        self._drone_voice_id = str(getattr(self._cfg, 'drone_voice_id', 'organ')
                                   or 'organ')
        self._drone_volume = float(getattr(self._cfg, 'drone_volume', 0.5) or 0.5)
        self._drone_semitones = max(-12, min(12, int(
            getattr(self._cfg, 'drone_semitones', 0) or 0)))
        self._pipes_ctrl = None
        self._pipes_dlg = None
        # Construct the drone controller (None → inert panel).
        self._drone_ctrl = None
        if (_DroneController is not None
                and getattr(self._engine, 'mixer', None) is not None):
            try:
                sr = int(getattr(self._engine, 'output_samplerate', 0)
                         or self._engine_sample_rate() or 44100)
                self._drone_ctrl = _DroneController(
                    self._engine.mixer, sr,
                    voice_id=self._drone_voice_id,
                    volume=self._drone_volume,
                    semitones=self._drone_semitones,
                    a4=float(getattr(self._engine, 'a4', 440.0) or 440.0),
                    enabled=False,
                    engine=self._engine,   # enables D3 duck-attach on enable
                    on_state_changed=self._on_drone_state_changed)
            except Exception:
                self._drone_ctrl = None

        grp = QGroupBox(self._t('drone_group'))
        g = QVBoxLayout(grp)
        g.setContentsMargins(8, 4, 8, 6)
        g.setSpacing(6)

        # Row 1: on/off + 5-preset row.
        row1 = QHBoxLayout()
        self._btn_drone_on = QPushButton(self._t('drone_group'))
        self._btn_drone_on.setCheckable(True)
        self._btn_drone_on.setToolTip(self._t('drone_on_tip'))
        self._btn_drone_on.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_drone_on.toggled.connect(self._on_drone_toggle)
        self._btn_drone_on.setStyleSheet(
            'QPushButton{background:#1a6b3a;color:#fff;border:none;'
            'border-radius:5px;padding:6px 12px;font-size:13px;font-weight:bold;}'
            'QPushButton:hover{background:#218a4b;}'
            'QPushButton:checked{background:#c0392b;}')
        row1.addWidget(self._btn_drone_on)
        self._drone_preset_btns: dict = {}
        for v in _DRONE_PRESETS:
            vid, label, _program = self._drone_voice_fields(v)
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(vid == self._drone_voice_id)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, i=vid: self._on_drone_preset(i))
            b.setStyleSheet(
                'QPushButton{background:#1e1e2e;color:#ccc;border:1px solid '
                '#444;border-radius:5px;padding:5px 10px;font-size:12px;}'
                'QPushButton:checked{background:#2d4a7a;color:#fff;border:1px '
                'solid #6699cc;}QPushButton:hover{border:1px solid #6699cc;}')
            self._drone_preset_btns[vid] = b
            row1.addWidget(b)
        row1.addStretch()
        g.addLayout(row1)

        # Row 2: volume · semitone steppers · pitch-pipes launcher.
        row2 = QHBoxLayout()
        vlbl = QLabel(self._t('drone_volume'))
        vlbl.setStyleSheet('color:#aaa;font-size:12px;')
        row2.addWidget(vlbl)
        self._drone_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._drone_vol_slider.setRange(0, 100)
        self._drone_vol_slider.setValue(max(0, min(100,
            int(round(self._drone_volume * 100)))))
        self._drone_vol_slider.valueChanged.connect(self._on_drone_volume)
        row2.addWidget(self._drone_vol_slider, 1)
        row2.addSpacing(10)
        slbl = QLabel(self._t('drone_semitone'))
        slbl.setStyleSheet('color:#aaa;font-size:12px;')
        row2.addWidget(slbl)
        btn_sd = QToolButton()
        btn_sd.setText('−')
        btn_sd.setToolTip(self._t('drone_semitone_down_tip'))
        btn_sd.clicked.connect(lambda: self._drone_semitone_nudge(-1))
        self._drone_semi_lbl = QLabel(self._fmt_semitone(self._drone_semitones))
        self._drone_semi_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drone_semi_lbl.setStyleSheet(
            'color:#ddd;font-size:13px;min-width:28px;')
        btn_su = QToolButton()
        btn_su.setText('+')
        btn_su.setToolTip(self._t('drone_semitone_up_tip'))
        btn_su.clicked.connect(lambda: self._drone_semitone_nudge(1))
        for b in (btn_sd, btn_su):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                'QToolButton{background:#1e1e2e;color:#ddd;border:1px solid '
                '#444;border-radius:5px;padding:2px 9px;font-size:15px;}'
                'QToolButton:hover{border:1px solid #6699cc;}')
        row2.addWidget(btn_sd)
        row2.addWidget(self._drone_semi_lbl)
        row2.addWidget(btn_su)
        row2.addSpacing(12)
        self._btn_pipes = QPushButton(self._t('pipes_launch'))
        self._btn_pipes.setToolTip(self._t('pipes_tip'))
        self._btn_pipes.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pipes.clicked.connect(self._open_pitch_pipes)
        self._btn_pipes.setStyleSheet(
            'QPushButton{background:#34495e;color:#eee;border:none;'
            'border-radius:5px;padding:5px 12px;font-size:12px;}'
            'QPushButton:hover{background:#3d566e;}')
        row2.addWidget(self._btn_pipes)
        g.addLayout(row2)

        # Inline status — shown when an audio action can't run yet.
        self._drone_status = QLabel('')
        self._drone_status.setWordWrap(True)
        self._drone_status.setStyleSheet(
            'color:#c0392b;font-size:11px;padding:2px;')
        self._drone_status.setVisible(False)
        g.addWidget(self._drone_status)
        return grp

    def _on_drone_toggle(self, on: bool) -> None:
        if self._drone_ctrl is None:
            self._drone_revert_toggle()
            return
        if not on:
            self._drone_ctrl.set_enabled(False)
            return
        self._drone_status.setVisible(False)
        if not self._ensure_output_open():
            self._drone_revert_toggle()
            return
        sr = int(getattr(self._engine, 'output_samplerate', 0) or 0)
        if sr and hasattr(self._drone_ctrl, 'set_samplerate'):
            try:
                self._drone_ctrl.set_samplerate(sr)
            except Exception:
                pass
        self._drone_ctrl.set_enabled(True)   # fires on_state_changed

    def _drone_revert_toggle(self) -> None:
        blocked = self._btn_drone_on.blockSignals(True)
        self._btn_drone_on.setChecked(False)
        self._btn_drone_on.blockSignals(blocked)
        self._drone_status.setText(self._t('drone_unavailable'))
        self._drone_status.setVisible(True)

    def _on_drone_preset(self, voice_id: str) -> None:
        self._drone_voice_id = voice_id
        self._cfg.drone_voice_id = voice_id
        if self._drone_ctrl is not None:
            self._drone_ctrl.set_voice(voice_id)   # fires on_state_changed
        else:
            self._refresh_drone_voice_widgets()

    def _on_drone_volume(self, value: int) -> None:
        self._drone_volume = value / 100.0
        self._cfg.drone_volume = self._drone_volume
        if self._drone_ctrl is not None:
            self._drone_ctrl.set_volume(self._drone_volume)

    def _drone_semitone_nudge(self, delta: int) -> None:
        n = max(-12, min(12, self._drone_semitones + delta))
        self._drone_semitones = n
        self._drone_semi_lbl.setText(self._fmt_semitone(n))
        self._cfg.drone_semitones = n
        if self._drone_ctrl is not None:
            self._drone_ctrl.set_semitones(n)

    def _refresh_drone_voice_widgets(self) -> None:
        """Sync the TUNER preset row + the SETUP voice combo to the current
        voice id (both drive the same drone voice)."""
        for vid, b in self._drone_preset_btns.items():
            blocked = b.blockSignals(True)
            b.setChecked(vid == self._drone_voice_id)
            b.blockSignals(blocked)
        combo = getattr(self, '_drone_voice_combo', None)
        if combo is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == self._drone_voice_id:
                    blocked = combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(blocked)
                    break

    def _on_drone_state_changed(self) -> None:
        """Zero-arg slot for DroneController.on_state_changed. The controller is
        the single source of truth: sync the on/off button, voice selection
        (preset row + SETUP combo), volume, and semitone from it + persist."""
        ctrl = self._drone_ctrl
        if ctrl is None:
            return
        enabled = bool(ctrl.is_enabled()) if hasattr(ctrl, 'is_enabled') \
            else bool(getattr(ctrl, 'enabled', False))
        blocked = self._btn_drone_on.blockSignals(True)
        self._btn_drone_on.setChecked(enabled)
        self._btn_drone_on.blockSignals(blocked)
        vid = str(getattr(ctrl, 'voice_id', self._drone_voice_id))
        if vid != self._drone_voice_id:
            self._drone_voice_id = vid
            self._cfg.drone_voice_id = vid
        self._refresh_drone_voice_widgets()

    def _on_drone_voice_changed(self, _idx: int) -> None:
        """SETUP 128-GM picker handler — routes through the same path as the
        TUNER preset row."""
        combo = self._drone_voice_combo
        vid = combo.currentData()
        if vid:
            self._on_drone_preset(str(vid))

    def _open_pitch_pipes(self) -> None:
        """Pitch-pipes modal: 12 chromatic pads (C4–B4). Pads route to
        PitchPipesController (numpy sine); tap toggles a sustained reference
        tone. Needs an output stream — honest status if it can't sound."""
        if self._pipes_dlg is not None and self._pipes_dlg.isVisible():
            self._pipes_dlg.raise_()
            return
        # Lazily construct the controller.
        if (self._pipes_ctrl is None and _PitchPipesController is not None
                and getattr(self._engine, 'mixer', None) is not None):
            try:
                sr = int(getattr(self._engine, 'output_samplerate', 0)
                         or self._engine_sample_rate() or 44100)
                self._pipes_ctrl = _PitchPipesController(
                    self._engine.mixer, sr,
                    a4=float(getattr(self._engine, 'a4', 440.0) or 440.0),
                    on_state_changed=self._on_pipes_state_changed)
            except Exception:
                self._pipes_ctrl = None

        from PyQt6.QtWidgets import QGridLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(self._t('pipes_title'))
        dlg.setStyleSheet('QDialog{background:#12121a;}')
        v = QVBoxLayout(dlg)
        tip = QLabel(self._t('pipes_tip'))
        tip.setWordWrap(True)
        tip.setStyleSheet('color:#aaa;font-size:12px;padding:2px 2px 6px 2px;')
        v.addWidget(tip)
        grid = QGridLayout()
        grid.setSpacing(6)
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self._pipe_btns: dict = {}
        for i in range(12):
            midi = 60 + i                      # C4..B4
            b = QPushButton(names[i])
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, m=midi: self._on_pipe_tapped(m))
            b.setStyleSheet(
                'QPushButton{background:#1e1e2e;color:#ddd;border:1px solid '
                '#444;border-radius:6px;padding:14px;font-size:15px;'
                'font-weight:bold;}QPushButton:checked{background:#2d4a7a;'
                'color:#fff;border:1px solid #6699cc;}'
                'QPushButton:hover{border:1px solid #6699cc;}')
            self._pipe_btns[midi] = b
            grid.addWidget(b, i // 4, i % 4)
        v.addLayout(grid)
        self._pipes_status = QLabel('')
        self._pipes_status.setStyleSheet('color:#c0392b;font-size:11px;')
        self._pipes_status.setVisible(False)
        v.addWidget(self._pipes_status)
        # Stop any sustaining pipe when the modal closes.
        dlg.finished.connect(lambda _r: self._stop_pitch_pipes())
        self._pipes_dlg = dlg
        self._on_pipes_state_changed()         # reflect any active pad
        dlg.show()

    def _on_pipe_tapped(self, midi: int) -> None:
        if self._pipes_ctrl is None:
            self._pipes_status.setText(self._t('pipes_unavailable'))
            self._pipes_status.setVisible(True)
            self._sync_pipe_btns()
            return
        if not self._ensure_output_open():
            self._pipes_status.setText(self._t('pipes_unavailable'))
            self._pipes_status.setVisible(True)
            self._sync_pipe_btns()
            return
        self._pipes_status.setVisible(False)
        self._pipes_ctrl.toggle(midi)          # fires on_pipes_state_changed

    def _stop_pitch_pipes(self) -> None:
        if self._pipes_ctrl is not None:
            try:
                self._pipes_ctrl.release_all()
            except Exception:
                pass

    def _sync_pipe_btns(self) -> None:
        # Desktop pitch pipes can sustain multiple pads (controller exposes
        # active_midis() — chord-capable reference), so highlight every active.
        active = set()
        if self._pipes_ctrl is not None:
            try:
                active = set(self._pipes_ctrl.active_midis())
            except Exception:
                active = set()
        for midi, b in getattr(self, '_pipe_btns', {}).items():
            blocked = b.blockSignals(True)
            b.setChecked(midi in active)
            b.blockSignals(blocked)

    def _on_pipes_state_changed(self) -> None:
        """Zero-arg slot for PitchPipesController.on_state_changed — highlight
        the sustaining pad (if the modal is open)."""
        if getattr(self, '_pipe_btns', None):
            self._sync_pipe_btns()

    def _table_headers(self):
        return [self._t('col_fingered'), self._t('col_sounding'),
                self._t('col_mean'), self._t('col_std'),
                self._t('col_n'), self._t('col_tendency')]

    def _make_btn(self, text, color, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        b.setMinimumHeight(36)
        b.setMinimumWidth(150)
        b.setStyleSheet(f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """)
        return b

    # ── Sprache wechseln ─────────────────────────────────────────────────────
    def _on_lang_changed(self, idx):
        self.lang = self._lang_combo.itemData(idx)
        self._retranslate()

    def _retranslate(self):
        """Aktualisiert alle beschrifteten Widgets ohne Neuaufbau."""
        self.setWindowTitle(self._t('window_title'))
        self._grp_instr.setTitle(self._t('grp_instrument'))
        self._grp_disp.setTitle(self._t('grp_display'))
        self._grp_a4.setTitle(self._t('grp_a4'))
        self._grp_filter.setTitle(self._t('grp_filter'))
        self._grp_lang.setTitle(self._t('grp_language'))

        # Family + sub-instrument combos re-populated in current language.
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for family_key, name_de, name_en in instrument_families():
            label = name_de if self.lang == 'de' else name_en
            self._family_combo.addItem(label, family_key)
        self._family_combo.blockSignals(False)
        self._select_family_for_instrument(self.instrument)
        self._populate_instrument_combo(select_key=self.instrument)
        self._btn_custom.setText(self._t('custom_label'))
        self._btn_custom.setToolTip(self._t('custom_dlg_title'))
        if hasattr(self, '_btn_range'):
            self._btn_range.setToolTip(self._t('gear_tip'))
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))

        # Display-Combo
        idx_d = self._disp_combo.currentIndex()
        self._disp_combo.blockSignals(True)
        self._disp_combo.clear()
        self._disp_combo.addItems([self._t('disp_griff'), self._t('disp_klingend')])
        self._disp_combo.setCurrentIndex(idx_d)
        self._disp_combo.blockSignals(False)

        # Layout-Combo — re-label the Auto/List/Grid items in the new lang.
        if hasattr(self, '_layout_combo'):
            cur = self._layout_combo.currentData()
            self._layout_combo.blockSignals(True)
            self._layout_combo.clear()
            self._layout_combo.addItem(self._t('layout_auto'), 'auto')
            self._layout_combo.addItem(self._t('layout_single'), 'single')
            self._layout_combo.addItem(self._t('layout_matrix'), 'matrix')
            for i in range(self._layout_combo.count()):
                if self._layout_combo.itemData(i) == cur:
                    self._layout_combo.setCurrentIndex(i)
                    break
            self._layout_combo.blockSignals(False)

        # Filter-mode combo — re-label Fast/Normal/Slow.
        if hasattr(self, '_filter_combo'):
            cur = self._filter_combo.currentData()
            self._filter_combo.blockSignals(True)
            self._filter_combo.clear()
            self._filter_combo.addItem(self._t('filter_fast'),   'fast')
            self._filter_combo.addItem(self._t('filter_normal'), 'normal')
            self._filter_combo.addItem(self._t('filter_slow'),   'slow')
            for i in range(self._filter_combo.count()):
                if self._filter_combo.itemData(i) == cur:
                    self._filter_combo.setCurrentIndex(i)
                    break
            self._filter_combo.setToolTip(self._t('filter_tip'))
            self._filter_combo.blockSignals(False)
        if hasattr(self, '_min_n_lbl'):
            self._min_n_lbl.setText(self._t('min_n_label'))
            self._min_n_spin.setToolTip(self._t('min_n_tip'))
        # v0.5.7: "Filter to instrument range" checkbox relocated next
        # to the min-N spinbox; same retranslate behaviour as before.
        if hasattr(self, '_cb_oor'):
            self._cb_oor.setText(self._t('allow_oor'))
            self._cb_oor.setToolTip(self._t('allow_oor_tip'))

        # Buttons
        self._btn_autotune.setText(self._t('btn_autotune'))
        self._btn_record.setText(self._t('btn_stop' if self._recording else 'btn_start'))
        self._btn_reset.setText(self._t('btn_reset'))
        self._btn_txt.setText(self._t('btn_txt'))
        self._btn_pdf.setText(self._t('btn_pdf'))
        self._btn_csv.setText(self._t('btn_csv'))
        self._btn_chart.setText(self._t('btn_chart'))
        self._btn_import.setText(self._t('btn_import'))

        # Tabellen-Header
        self._table.setHorizontalHeaderLabels(self._table_headers())

        # Status & Tabellenlabel
        self._status_lbl.setText(self._t('no_signal'))

        # Diagnose-Panel + Spektrogramm.
        if hasattr(self, '_cb_diag'):
            self._cb_diag.setText(self._t('show_diagnostics'))
            self._cb_diag.setToolTip(self._t('show_diagnostics_tip'))
        if hasattr(self, '_spectro_grp'):
            self._spectro_grp.setTitle(self._t('spectro_title'))
        if hasattr(self, '_data_grp'):
            self._data_grp.setTitle(self._t('data_panel_title'))
        if hasattr(self, '_data_panel'):
            self._data_panel.retranslate(self._t)
        if hasattr(self, '_audio_chip'):
            self._audio_chip.retranslate(self._t)
        if hasattr(self, '_audio_banner'):
            self._audio_banner.retranslate(self._t)
        if hasattr(self, '_info_banner'):
            self._info_banner.retranslate(self._t)

        self._refresh_table()

    # ── Audio-Callback ────────────────────────────────────────────────────────
    def _on_note(self, midi_kl: int, freq: float, cents: float):
        if not self._recording:
            return
        # v0.6 Phase-4 (Item 3): wrong-instrument detector. Run BEFORE the
        # OOR drop below so the counter still ticks when the user has
        # "filter to range" enabled — otherwise we'd never count notes
        # outside the range. Compute the fingered MIDI relative to the
        # currently selected instrument and bump/reset the counter.
        try:
            _t_oor = TRANSP_MAP.get(self.instrument, 0)
            _lo_f, _hi_f = sax_instruments.fingered_range(self.instrument)
            _midi_fingered_oor = midi_kl - _t_oor
            _in_range = (_lo_f <= _midi_fingered_oor <= _hi_f)
        except Exception:
            _in_range = True
        if _in_range:
            self._oor_count = 0
        else:
            self._oor_count += 1
            if (not self._oor_banner_shown
                    and self._oor_count >= self._oor_threshold
                    and hasattr(self, '_info_banner')):
                self._oor_banner_shown = True
                instr_label = self._instr_label(self.instrument)
                self._info_banner.show_message(
                    self._t('wrong_instrument_banner_body',
                            instrument=instr_label))

        # Drop out-of-range notes when the toggle is off — keeps the table
        # bounded to the instrument's nominal range.
        if not self._cfg.allow_out_of_range:
            transp = TRANSP_MAP.get(self.instrument, 0)
            lo_f, hi_f = sax_instruments.fingered_range(self.instrument)
            midi_fingered = midi_kl - transp
            if not (lo_f <= midi_fingered <= hi_f):
                return
        with self._lock:
            if midi_kl not in self.stats:
                self.stats[midi_kl] = NoteStats()
            self.stats[midi_kl].add(cents)
        # Bump the diagnostics counter outside the lock — single thread
        # writes it, single thread reads via the timer; integer add is
        # atomic enough for a display-only counter.
        self._notes_count += 1
        # Per-measurement log. Instrument/A4 are read off the active run
        # inside the log, not from `self`, so a callback firing during a UI
        # change still attributes to the run that was active when it fired.
        midi_gr = midi_kl - TRANSP_MAP.get(self.instrument, 0)
        self._log.add_measurement(midi_sounding=midi_kl,
                                   midi_fingered=midi_gr,
                                   cents=cents, freq_hz=freq)

        transp     = TRANSP_MAP.get(self.instrument, 0)
        midi_gr    = midi_kl - transp
        kl_name    = midi_note_name(midi_kl)
        gr_name    = midi_note_name(midi_gr)
        disp_name  = gr_name if self.display == 'griff' else kl_name
        sr_now = self._engine_sample_rate()
        self._tuner.sample_rate = sr_now
        self._tuner.set_note(disp_name, freq, cents)
        cents_str = format_cents(cents, freq, sr_now)
        self._status_lbl.setText(self._t(
            'status_fmt', fingered=gr_name, sounding=kl_name,
            freq=freq, cents_str=cents_str, a4=self._engine.a4))
        # Highlight the row currently being played so the user can see
        # which entry in a long table just ticked. _refresh_table reads
        # this on its next tick (every 300ms via _refresh_timer).
        self._active_midi = midi_kl
        self._active_midi_at = datetime.datetime.now()

    # ── Tabelle ──────────────────────────────────────────────────────────────────────
    # Two layout modes:
    #   'single' — single-column-of-notes table (1 row per played note +
    #              1 row per seeded expected note).
    #   'matrix' — piano-roll: 12 chromatic rows × N octave columns, where
    #              N covers the instrument's range padded ±1 octave. Cells
    #              outside the instrument's range render greyed out.
    # The mode is chosen automatically based on the available width.
    _MATRIX_COL_WIDTH = 112    # min usable width per octave column (px)
    _MATRIX_HEADER_W  = 64     # row-header width (note names)
    _MATRIX_ROW_HEIGHT = 60    # tall enough for: names + mean+std + bar + N
    # Hysteresis: enter matrix at the calculated threshold, drop back to
    # single only when noticeably narrower. Prevents thrash near the edge.
    _MATRIX_HYSTERESIS = 48    # px

    def _refresh_table(self):
        """QTimer entry point — fires every 300 ms.  Decides the layout
        mode (single vs. matrix) here on MainWindow because the choice
        depends on the live viewport width, then hands off to
        TableController for the actual paint.  v0.6 Phase-5: the per-
        cell paint pipeline and cell-cache lives on the controller."""
        if not hasattr(self, '_table') or not hasattr(self, '_table_ctrl'):
            return
        desired = self._desired_layout_mode()
        if desired != self._layout_mode or self._table_ctrl._current_mode is None:
            self._layout_mode = desired
            self._table_ctrl.configure_for_mode(desired)
        self._table_ctrl.refresh()

    def _matrix_octave_range(self) -> tuple[int, int]:
        """Wrapper kept so the table context-menu code keeps a single
        call site for resolving the matrix's current octave range.
        Delegates to TableController, where the real implementation
        and the half-step-beyond logic now live."""
        return self._table_ctrl._matrix_octave_range()

    def _desired_layout_mode(self) -> str:
        """User-preference first; falls back to a width-driven auto pick.
        In auto mode, matrix is used as long as the viewport can fit at
        least two octave columns; below that, single-column for
        readability. Matrix never truncates a playable note — wider
        instruments overflow into a horizontal scrollbar instead of
        dropping back to single."""
        pref = getattr(self._cfg, 'layout_mode_preference', 'auto')
        if pref == 'single':
            return 'single'
        if pref == 'matrix':
            return 'matrix'
        if not hasattr(self, '_table'):
            return 'single'
        w = self._table.viewport().width()
        if w <= 0:
            w = self._table.width()
        if w <= 0:
            return 'single'
        floor = self._MATRIX_HEADER_W + 2 * self._MATRIX_COL_WIDTH
        if self._layout_mode == 'matrix':
            return 'matrix' if w >= floor - self._MATRIX_HYSTERESIS else 'single'
        return 'matrix' if w >= floor else 'single'

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Defer the mode-decision one event-loop tick so the splitter has
        # time to propagate the new viewport size into the table widget
        # before _desired_layout_mode reads it. Without the singleShot,
        # rapid window shrinks sometimes leave the table in matrix mode
        # because viewport().width() still reflects the pre-resize size.
        if hasattr(self, '_table'):
            QTimer.singleShot(0, self._refresh_table)

    # ── Instrument / Anzeige / A4 ─────────────────────────────────────────────
    # Phase-5 extraction: the family/instr/nickname slots, range editor,
    # custom-instrument registration, and autotune flow now live on
    # InstrumentController. The MainWindow methods below remain as thin
    # delegating wrappers so the Qt signal/slot wiring in _build_ui is
    # untouched. on_*_external below are the post-change refresh hooks
    # the controller calls back so MainWindow updates OOR counter,
    # info banner, expected-note seed, and table.

    def _on_family_changed(self, _idx):
        self._instr_ctrl.on_family_changed(_idx)

    def _populate_instrument_combo(self, select_key: str | None) -> None:
        self._instr_ctrl.populate_instrument_combo(select_key)

    def _select_family_for_instrument(self, instrument_key: str) -> None:
        self._instr_ctrl.select_family_for_instrument(instrument_key)

    def _on_instr_changed(self, idx):
        self._instr_ctrl.on_instr_changed(idx)

    def _open_range_editor(self) -> None:
        self._instr_ctrl.open_range_editor()

    def _on_instrument_changed_external(self) -> None:
        """Called by InstrumentController after it has changed the active
        instrument key (combo select). Owns the side effects that don't
        belong inside the controller: wrong-instrument detector reset,
        info-banner hide, expected-note seed, table refresh."""
        # v0.6 Phase-4 (Item 3): the wrong-instrument detector is anchored
        # to the *selected* instrument. New selection ⇒ counter resets and
        # the one-shot prompt becomes eligible to fire again.
        self._oor_count = 0
        self._oor_banner_shown = False
        if hasattr(self, '_info_banner'):
            self._info_banner.hide()
        # Seed the stats with empty slots for every expected fingered note so
        # the table immediately shows the player what the instrument's range
        # looks like. Real measurements fill in as the player plays;
        # out-of-range notes (overtones, accidentals) appear automatically
        # via the existing _on_note path.
        self._seed_expected_notes()
        self._refresh_table()

    def _on_range_changed_external(self) -> None:
        """Called by InstrumentController after the per-instrument range
        editor accepts new bounds. The instrument *key* hasn't changed,
        so we do NOT reset the wrong-instrument detector — just re-seed
        expected notes against the new bounds and repaint."""
        self._seed_expected_notes()
        self._refresh_table()

    def _on_a4_changed_external(self) -> None:
        """Called by InstrumentController after the autotune flow has
        applied a new A4. The autotune body already cleared self.stats
        under the lock; here we just refresh the table so the new
        baseline paints."""
        self._refresh_table()

    def _instr_set_a4(self, hz: float) -> None:
        """Setter callback used by InstrumentController's autotune flow.
        Mirrors the v0.5.x in-line sequence: block the combo's signal so
        _on_a4_changed does NOT fire (the autotune flow does its own
        stats reset + refresh via on_a4_changed), set the index, set the
        engine A4."""
        hz_int = int(hz)
        self._a4_combo.blockSignals(True)
        self._a4_combo.setCurrentIndex(hz_int - 430)
        self._a4_combo.blockSignals(False)
        self._engine.a4 = float(hz_int)

    def _seed_expected_notes(self) -> None:
        """Populate self.stats with empty NoteStats for the current
        instrument's expected fingered range. Existing measurements are
        preserved."""
        transp = TRANSP_MAP.get(self.instrument, 0)
        lo, hi = sax_instruments.fingered_range(self.instrument)
        with self._lock:
            for fingered in range(lo, hi + 1):
                sounding = fingered + transp
                if sounding in SAX_MIDI and sounding not in self.stats:
                    self.stats[sounding] = NoteStats()

    def _on_oor_toggled(self, checked: bool) -> None:
        """Persist the choice and refresh the table so the matrix can
        shrink back to nominal range when the filter goes on.

        UI semantic: `checked` means "filter to range ON" (restrict).
        Internally we store the inverse as `allow_out_of_range`."""
        self._cfg.allow_out_of_range = not checked
        sax_config.save_config(self._cfg)
        # Wave-1 bug #7: the docstring above promises a refresh; the
        # v0.5.3 implementation forgot to actually call _refresh_table,
        # so toggling "Filter to instrument range" only took effect on
        # the next other refresh.
        self._refresh_table()

    # ── Audio-Geräteverwaltung (v0.5.4) ───────────────────────────────────
    def _engine_sample_rate(self) -> int:
        """Best-known live sample rate for adaptive cent precision.
        Falls back to the constant default when the engine is FAILED or
        audio is disabled — defensive; nothing should call the formatter
        in those states, but a stale delegate paint can race teardown."""
        try:
            sr = int(getattr(self._engine, 'samplerate', 0) or 0)
        except Exception:
            sr = 0
        return sr if sr > 0 else int(_DEFAULT_SAMPLE_RATE_EXT)

    def _device_selection_from_cfg(self) -> DeviceSelection:
        return DeviceSelection(
            name=str(getattr(self._cfg, 'audio_device_name', '') or ''),
            host_api=str(getattr(self._cfg, 'audio_device_host_api', '') or ''),
            samplerate=int(getattr(self._cfg, 'audio_device_samplerate', 0) or 0),
        )

    # Phase-5 extraction: audio device handling lives on DeviceController.
    # The methods below are one-line wrappers — Qt signal/slot connect()
    # bindings (lines ~2156-2160) and the QTimer wiring (~2249) point at
    # MainWindow methods, so we preserve them at MainWindow's surface
    # so PyQt's runtime slot type-check sees the exact bound-method
    # signature it saw before the extraction.
    def _persist_active_device(self) -> None:
        self._device_ctrl.persist_active_device()

    def _open_audio_picker(self) -> None:
        self._device_ctrl.open_audio_picker()

    def _retry_audio(self) -> None:
        self._device_ctrl.retry_audio()

    def _poll_devices(self) -> None:
        self._device_ctrl.poll_devices()

    def _on_engine_state(self, state, err, msg) -> None:
        self._device_ctrl.on_engine_state(state, err, msg)

    def _on_devices_changed(self, _devices) -> None:
        self._device_ctrl.on_devices_changed(_devices)

    def _on_interface_appeared(self, device: 'DeviceInfo') -> None:
        self._device_ctrl.on_interface_appeared(device)

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        """Show or hide the spectrogram + diagnostics panels and persist
        the choice. Widgets remain constructed either way so flipping
        the toggle has no allocation cost on the audio thread."""
        self._cfg.show_diagnostics = bool(checked)
        sax_config.save_config(self._cfg)
        if hasattr(self, '_spectro_grp'):
            self._spectro_grp.setVisible(bool(checked))
        if hasattr(self, '_data_grp'):
            self._data_grp.setVisible(bool(checked))
        self._refresh_table()

    def _on_layout_pref_changed(self, _idx: int) -> None:
        """User picked Auto / List / Grid. Persist and refresh the table."""
        pref = self._layout_combo.currentData()
        if pref in ('auto', 'single', 'matrix'):
            self._cfg.layout_mode_preference = pref
            sax_config.save_config(self._cfg)
            self._refresh_table()

    def _on_filter_mode_changed(self, _idx: int) -> None:
        """User picked Fast / Normal / Slow. Reroute live audio through
        the new preset and persist."""
        mode = self._filter_combo.currentData()
        if mode in _FILTER_PRESETS:
            self._cfg.filter_mode = mode
            sax_config.save_config(self._cfg)
            if AUDIO_OK:
                self._engine.set_filter_mode(mode)

    def _on_min_n_changed(self, value: int) -> None:
        """User adjusted the min-N filter. Persist + redraw the table.
        The filter is purely a display gate — measurements are still
        collected; rows just hide until they accumulate enough hits."""
        self._cfg.min_n_visible = max(0, int(value))
        sax_config.save_config(self._cfg)
        self._refresh_table()

    def _on_nickname_changed(self) -> None:
        self._instr_ctrl.on_nickname_changed()

    def _on_add_custom(self) -> None:
        self._instr_ctrl.on_add_custom()


    def _add_dialog_lang_toggle(self, dlg, layout, on_change=None):
        """Drop a tiny DE | EN toggle at the top of any modal dialog so the
        user can switch language without having to cancel out and find the
        main-window combo. `on_change`, if given, is called after the lang
        switch so the dialog can repaint its own labels."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addStretch()

        def make_btn(label, code):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(self.lang == code)
            btn.setFixedWidth(46)
            btn.setStyleSheet("""
                QPushButton{background:#2a2a3e;color:#bbb;border:1px solid #444;
                             border-radius:4px;padding:3px 8px;font-size:11px;}
                QPushButton:checked{background:#2c5282;color:white;border:1px solid #6699cc;}
                QPushButton:hover:!checked{background:#34344a;}
            """)
            return btn

        btn_de = make_btn('DE', 'de')
        btn_en = make_btn('EN', 'en')

        def switch(target):
            if self.lang == target:
                return
            # Flip via the main window's combo so all subscribers (the main
            # toolbar, labels, etc.) update too.
            for i in range(self._lang_combo.count()):
                if self._lang_combo.itemData(i) == target:
                    self._lang_combo.setCurrentIndex(i)
                    break
            btn_de.setChecked(self.lang == 'de')
            btn_en.setChecked(self.lang == 'en')
            # Update the dialog's own window title.
            dlg.setWindowTitle(self._t(dlg.property('_lang_title_key')
                                        or 'window_title'))
            if on_change is not None:
                on_change()

        btn_de.clicked.connect(lambda: switch('de'))
        btn_en.clicked.connect(lambda: switch('en'))
        row.addWidget(btn_de)
        row.addWidget(btn_en)
        layout.addLayout(row)

    def _show_welcome_dialog(self) -> None:
        """First-boot dialog. Offers opt-in for persistent JSONL log."""
        dlg = QDialog(self)
        dlg.setProperty('_lang_title_key', 'welcome_title')
        dlg.setWindowTitle(self._t('welcome_title'))
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #ccc; font-size: 13px; }
            QCheckBox { color: #ddd; font-size: 13px; padding: 6px 0; }
            QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 8px 18px; font-size: 13px;
            }
            QPushButton:hover  { background: #3a6da8; }
        """)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 14, 24, 18)

        # Language toggle at the top — load-bearing on first launch, since
        # this dialog fires before the user has touched the main combo.
        info = QLabel(self._t('welcome_info'))
        info.setWordWrap(True)
        cb = QCheckBox(self._t('welcome_persist'))
        cb.setChecked(False)
        default_path = sax_config.CONFIG_DIR / sax_config.DEFAULT_LOG_FILENAME
        path_lbl = QLabel(self._t('welcome_path', path=str(default_path)))
        path_lbl.setStyleSheet('color: #888; font-size: 11px;')
        path_lbl.setWordWrap(True)
        btns = QDialogButtonBox()
        btn_go = btns.addButton(self._t('welcome_continue'),
                                 QDialogButtonBox.ButtonRole.AcceptRole)
        btn_later = btns.addButton(self._t('welcome_skip'),
                                    QDialogButtonBox.ButtonRole.RejectRole)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('welcome_info'))
            cb.setText(self._t('welcome_persist'))
            path_lbl.setText(self._t('welcome_path', path=str(default_path)))
            btn_go.setText(self._t('welcome_continue'))
            btn_later.setText(self._t('welcome_skip'))

        self._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addWidget(cb)
        layout.addWidget(path_lbl)
        layout.addWidget(btns)
        accepted = (dlg.exec() == QDialog.DialogCode.Accepted)
        # Whether they accepted or skipped, mark welcome as shown so we don't
        # ask again. Persistence is only enabled if they ticked the box.
        self._cfg.welcome_shown = True
        if accepted and cb.isChecked():
            self._cfg.persistence_enabled = True
            # Rebuild the log on the chosen path. Existing in-memory entries
            # would be lost otherwise — but at this point the user has only
            # been in the app a few seconds at most, so that's fine.
            new_path = self._cfg.effective_log_path()
            if new_path:
                self._log = MeasurementLog(path=new_path)
                if AUDIO_OK:
                    self._log.start_run(instrument=self.instrument,
                                        a4_hz=self._engine.a4,
                                        label=self._nick_edit.text().strip())
        sax_config.save_config(self._cfg)

    # _ask_custom_instrument moved to InstrumentController in Phase 5.
    # It was only invoked by _on_add_custom (also extracted), so no
    # MainWindow wrapper is retained — the controller calls itself.

    def _on_disp_changed(self, idx):
        self.display = ['griff', 'klingend'][idx]
        self._refresh_table()

    def _on_a4_changed(self, idx):
        """Concert pitch changed. Cents are a function of frequency + A4,
        so the cents values stored in self.stats are invalidated — but the
        underlying frequencies are immutable and still live in the log.
        Re-derive the table by walking the log's measurements at the new
        A4 instead of throwing away everything the user just recorded."""
        new_a4 = float(self._a4_combo.itemData(idx))
        # v0.6 Phase-4 (Item 3): A4 change recalibrates midi assignments,
        # so the wrong-instrument counter resets too.
        self._oor_count = 0
        self._oor_banner_shown = False
        # Wave-1 bug #5: A4-change race. The audio callback can fire
        # _on_note between the moment we build `remapped` and the moment
        # we assign it to self.stats, dropping a measurement into the
        # old dict that's about to be discarded — or worse, into a
        # half-built dict. pause_emissions() tells the engine to drop
        # incoming detections while we rebuild; we hold self._lock
        # across the swap so any future emission seeing the new stats
        # never partially-overlaps with the rebuild.
        self._engine.pause_emissions()
        try:
            self._engine.set_a4(new_a4)
            remapped: dict[int, NoteStats] = {}
            for m in self._log.measurements():
                # v0.5.7.3: skip non-positive / non-finite freqs. An
                # imported CSV row with freq_hz=0 (or NaN/inf) would
                # otherwise call freq_to_midi -> math.log2(0) -> ValueError
                # and crash the A4 combo handler. Such rows carry no
                # tuning information; drop them silently.
                f = float(m.freq_hz)
                if not math.isfinite(f) or f <= 0.0:
                    continue
                midi_round, cents = cents_dev(f, new_a4)
                if midi_round in SAX_MIDI:
                    ns = remapped.setdefault(midi_round, NoteStats())
                    ns.add(cents)
            with self._lock:
                self.stats = remapped
        finally:
            self._engine.resume_emissions()
        # Re-seed expected blank rows on top of the remapped data so the
        # instrument's range still shows even after a clean re-derive.
        self._seed_expected_notes()

        # Open a new run at the new A4 so subsequent measurements are
        # tagged correctly; coalescing in `start_run` keeps empty runs out
        # of the log when the user is just scrubbing the combo.
        if AUDIO_OK and self._recording:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=new_a4,
                                label=self._nick_edit.text().strip())
        self._refresh_table()
        # Sprint 3: keep the drone + pitch-pipe reference frequencies tuned to
        # the new concert pitch (acceptance: pads sound correct at current A4).
        for _c in (getattr(self, '_drone_ctrl', None),
                   getattr(self, '_pipes_ctrl', None)):
            if _c is not None and hasattr(_c, 'set_a4'):
                try:
                    _c.set_a4(new_a4)
                except Exception:
                    pass

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _on_record_toggle(self):
        self._recording = not self._recording
        if self._recording:
            # Resumed recording ⇒ open a fresh run for the log.
            if AUDIO_OK:
                self._log.start_run(instrument=self.instrument,
                                    a4_hz=self._engine.a4,
                                    label=self._nick_edit.text().strip())
            self._btn_record.setText(self._t('btn_stop'))
            self._btn_record.setStyleSheet(self._btn_record.styleSheet().replace('#2ecc71', '#b7770d').replace('#27ae60', '#b7770d'))
            self._update_record_btn_style()
            self._status_lbl.setText(self._t('no_signal'))
        else:
            self._log.end_run()
            self._update_record_btn_style()
            self._btn_record.setText(self._t('btn_start'))

    def _update_record_btn_style(self):
        if self._recording:
            color = '#b7770d'   # orange = läuft
        else:
            color = '#27ae60'   # grün = pausiert, klicken zum Starten
        self._btn_record.setStyleSheet(f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """)
        self._btn_record.setText(
            self._t('btn_stop') if self._recording else self._t('btn_start'))

    # ── Tabellen-Kontextmenü ──────────────────────────────────────────────────
    def _on_table_context_menu(self, pos):
        row = self._table.rowAt(pos.y())
        if row < 0:
            return

        transp = TRANSP_MAP.get(self.instrument, 0)
        midi_kl: int | None = None

        if self._layout_mode == 'matrix':
            # Matrix mode: map (row, col) to the sounding MIDI for the
            # clicked cell. Row = chroma index (0..11), col = octave from
            # the current octave range.
            col = self._table.columnAt(pos.x())
            if col < 0:
                return
            lo_oct, _hi_oct = self._matrix_octave_range()
            oct_ = lo_oct + col
            midi_visible = (oct_ + 1) * 12 + row
            if self.display == 'griff':
                midi_kl = midi_visible + transp
            else:
                midi_kl = midi_visible
            # Only offer the action if this cell actually holds played data.
            with self._lock:
                st = self.stats.get(midi_kl)
            if st is None or st.n == 0:
                return
        else:
            # Single-column mode: row directly indexes into sorted stats.
            with self._lock:
                keys = sorted(self.stats.keys())
            if row >= len(keys):
                return
            midi_kl = keys[row]

        midi_gr  = midi_kl - transp
        note_str = f"{midi_note_name(midi_gr)} / {midi_note_name(midi_kl)}"

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#1e1e2e; color:#ddd; border:1px solid #444;
                    font-size:13px; padding:4px; }
            QMenu::item { padding:6px 20px; border-radius:4px; }
            QMenu::item:selected { background:#c03030; color:white; }
        """)
        action = menu.addAction(f"\U0001f5d1  {self._t('ctx_discard')}")
        action.setData(midi_kl)

        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen and chosen.data() == midi_kl:
            if QMessageBox.question(
                self, self._t('reset_title'),
                self._t('ctx_discard_confirm', note=note_str),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                with self._lock:
                    self.stats.pop(midi_kl, None)
                self._refresh_table()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _on_reset(self):
        if QMessageBox.question(
            self, self._t('reset_title'), self._t('reset_msg'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            with self._lock:
                self.stats.clear()
            # Reset implies "I might be switching instruments" — clear the
            # cached maker/model so the next export re-prompts.
            self._instr_info_asked = False
            self._refresh_table()

    # ── Kammerton ermitteln ───────────────────────────────────────────────────
    def _on_autotune(self):
        self._instr_ctrl.on_autotune()

    # ── Instrument-Modell-Dialog ──────────────────────────────────────────────
    # The maker/model prompt was extracted to ExportController in Phase 5;
    # it's invoked internally by each export_* flow on the controller.
    # MainWindow no longer needs a wrapper — nothing outside the export
    # flow calls it.

    # ── Export TXT ────────────────────────────────────────────────────────────
    def _export_txt(self):
        self._export.export_txt()

    # ── Export PDF ────────────────────────────────────────────────────────────
    def _export_pdf(self):
        self._export.export_pdf()

    # ── Export CSV ────────────────────────────────────────────────────────────
    def _export_csv(self):
        self._export.export_csv()

    def _instr_label(self, key: str) -> str:
        long_key = f'instr_long_{key}'
        if long_key in STRINGS[self.lang]:
            return self._t(long_key)
        return instrument_display_name(key, self.lang)

    # ── Import CSV ────────────────────────────────────────────────────────────
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self._t('csv_import_title'), '',
            self._t('csv_filter'))
        if not path:
            return
        # v0.6 Phase-4: confirm before merging into the current session.
        # The user picked a file but might not realise the import merges
        # into the live log; Frodo wants a "are you sure?" beat before
        # potentially mixing somebody else's CSV with his own data.
        from pathlib import Path as _Path
        filename = _Path(path).name
        confirm = QMessageBox.question(
            self, self._t('csv_import_confirm_title'),
            self._t('csv_import_confirm_body', filename=filename),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok)
        if confirm != QMessageBox.StandardButton.Ok:
            return
        try:
            runs, meas, skipped = self._log.import_raw_csv(path)
        except ValueError:
            QMessageBox.warning(self, self._t('err_title'),
                                self._t('csv_import_badhdr'))
            return
        except OSError as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))
            return

        if runs == 0 and meas == 0 and skipped == 0:
            QMessageBox.information(self, self._t('csv_import_title'),
                                    self._t('csv_import_empty'))
            return
        QMessageBox.information(
            self, self._t('csv_import_title'),
            self._t('csv_import_saved',
                    runs=runs, meas=meas, skipped=skipped))

    # ── Export Chart (PNG) ────────────────────────────────────────────────────
    def _export_chart(self):
        self._export.export_chart()

    def _restore_session_state(self) -> None:
        """Thin delegator — the implementation lives on
        SessionStateController (sax_session_state.py). Kept as a method
        so external/internal call sites that reference this name keep
        working without change."""
        self._session_state.restore()

    def closeEvent(self, ev):
        # v0.5.5: snapshot the full session state to the config file so the
        # next launch lands the user back where they were. Wrapped in
        # try/except because save-on-exit must never block the engine
        # from stopping cleanly — a corrupt config is recoverable, a
        # dangling audio stream is not.
        try:
            self._save_session_state()
        except Exception:
            pass
        # Tear the tape deck down BEFORE the engine stops, so a recording/playing
        # take disarms the input tap and unregisters its mixer source — close()
        # is idempotent and never raises (Sauron's contract), never orphans.
        try:
            if getattr(self, '_deck_pump_timer', None) is not None:
                self._deck_pump_timer.stop()
            if getattr(self, '_deck_ctrl', None) is not None:
                self._deck_ctrl.close()
        except Exception:
            pass
        self._engine.stop()
        ev.accept()

    def _save_session_state(self) -> None:
        """Thin delegator + persistence. The snapshot work lives on
        SessionStateController; the save_config call stays here because
        the controller is intentionally agnostic about *when* to
        persist. Called from closeEvent. The per-setting saves
        scattered through the GUI remain as a belt; this is the
        suspenders."""
        self._session_state.save()
        sax_config.save_config(self._cfg)


def _today():
    return datetime.date.today().strftime('%Y%m%d')


# =============================================================================
def _load_app_icon() -> QIcon | None:
    """Locate the bundled application icon. Returns None if it can't be
    found — never crashes startup over a missing asset."""
    candidates = [
        Path(__file__).parent / 'assets' / 'icon.png',
        Path(__file__).parent / 'assets' / 'icon.ico',
    ]
    # PyInstaller drops resources into sys._MEIPASS at runtime.
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.insert(0, Path(meipass) / 'assets' / 'icon.png')
        candidates.insert(0, Path(meipass) / 'assets' / 'icon.ico')
    for p in candidates:
        if p.exists():
            icon = QIcon(str(p))
            if not icon.isNull():
                return icon
    return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = MainWindow()
    if icon is not None:
        win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec())

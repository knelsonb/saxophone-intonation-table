"""Custom widgets and item delegates for the main window: the tuner readout,
the spectrum analyzer, the diagnostics panel, the table bar and matrix-cell
delegates, the audio-input chip and the two inline banners.

Each widget paints from the active sax_theme palette on every paint, so a
theme switch only needs a repaint. None of them owns application state: the
window passes in the engine, a translation function and getters, and calls
``retranslate`` on a language switch.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QWidget,
)

import sax_theme
from sax_audio_engine import (
    AUDIO_OK,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_RATE,
    FILTER_PRESETS,
    MAX_FREQ,
    MIN_FREQ,
    AudioEngineError,
    AudioEngineState,
)
from sax_pitch import format_cents, midi_note_name
from sax_theme import IN_TUNE_CENTS, cents_band

if TYPE_CHECKING:
    from sax_audio_engine import AudioEngine


# Palette role for each cents_band() result.
_BAND_ROLE = {'in': 'ok', 'close': 'warn', 'off': 'bad'}


def _themed(role: str, alpha: int = 255) -> QColor:
    """A QColor for the named role in the ACTIVE theme palette (sax_theme).

    The custom-painted widgets call this each paintEvent, so a theme switch
    repaints them in the new palette. ``alpha`` (0-255) applies translucency
    for fills/overlays without minting per-theme rgba roles."""
    c = QColor(getattr(sax_theme.active(), role))
    if alpha != 255:
        c.setAlpha(alpha)
    return c


# Diagnostics: the smallest cent step the measurement can resolve.
# Parabolic interpolation in YIN pins tau to about 0.1 samples; one cent
# corresponds to tau * (ln(2)/1200) ~= tau * 5.78e-4 samples. The minimum
# resolvable cent step at frequency f and sample rate sr is therefore
# approximately 0.1 / (tau * 5.78e-4) = 173 * f / sr.
CENT_PREC_TENTHS_MAX = 0.3


def cent_precision_floor(freq_hz: float, sample_rate: int) -> float:
    """Return the minimum resolvable cent step at ``freq_hz`` given
    ``sample_rate``. See the derivation above."""
    sr = float(sample_rate) if sample_rate else float(DEFAULT_SAMPLE_RATE)
    if sr <= 0 or freq_hz <= 0:
        return CENT_PREC_TENTHS_MAX
    return 173.0 * float(freq_hz) / sr


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
        self.sample_rate = DEFAULT_SAMPLE_RATE
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
        p.fillRect(0, 0, W, H, _themed('window_bg'))
        alpha = int(255 * max(0.15, self._decay))

        # Tonname
        if self.note:
            p.setFont(QFont('Monospace', 72, QFont.Weight.Bold))
            p.setPen(_themed('text_bright', alpha))
            p.drawText(QRectF(0, 8, W, H * 0.45),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       self.note)

        # Frequenz
        if self.freq > 0:
            p.setFont(QFont('Monospace', 16))
            p.setPen(_themed('text_dim', alpha))
            p.drawText(QRectF(0, H * 0.44, W, 30),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{self.freq:.1f} Hz")

        # Skala
        sy, sh = H * 0.60, 28
        sw, sx = W * 0.78, (W - W * 0.78) / 2
        mx = 50.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_themed('grid'))
        p.drawRoundedRect(QRectF(sx, sy, sw, sh), 8, 8)

        # Half-width of IN_TUNE_CENTS on a scale spanning -mx..+mx over sw.
        zw = sw * (IN_TUNE_CENTS / mx) / 2
        p.setBrush(_themed('ok', 140))
        p.drawRoundedRect(QRectF(sx + sw/2 - zw, sy, zw*2, sh), 6, 6)

        if self.active:
            norm = max(-1.0, min(1.0, self.cents / mx))
            nx   = sx + sw/2 + norm * sw/2
            nc = _themed(_BAND_ROLE[cents_band(self.cents)], alpha)
            p.setBrush(nc)
            p.drawRoundedRect(QRectF(nx - 4, sy - 6, 8, sh + 12), 4, 4)

        p.setPen(_themed('text_dim'))
        p.setFont(QFont('Monospace', 9))
        for ct in [-50, -25, 0, 25, 50]:
            nx = sx + sw/2 + (ct / mx) * sw/2
            p.drawLine(QPointF(nx, sy - 2), QPointF(nx, sy + sh + 2))
            p.drawText(QRectF(nx - 18, sy + sh + 4, 36, 16),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{'+' if ct>0 else ''}{ct}")

        if self.active:
            p.setFont(QFont('Monospace', 32, QFont.Weight.Bold))
            cc = _themed(_BAND_ROLE[cents_band(self.cents)], alpha)
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
        self._window = np.hanning(DEFAULT_BLOCK_SIZE).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(DEFAULT_BLOCK_SIZE, d=1.0 / DEFAULT_SAMPLE_RATE)
        self._bucket = np.searchsorted(self._edges, fft_freqs) - 1
        self._bucket = np.clip(self._bucket, -1, self.N_BINS - 1)
        self._win_norm = float(np.sum(self._window) * 0.5)
        # Live curve and peak-hold envelope, both in dBFS.
        self._levels = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        self._peaks  = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        # Powers-of-two grid lines in Hz, labeled along the bottom.
        self._grid_hz = [32, 64, 128, 256, 512, 1024, 2048, 4096]

        self.setMinimumHeight(150)

        # Runs only while the widget is visible (showEvent/hideEvent): each
        # tick is a full FFT, and the diagnostics panel is hidden by default.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start(self.REFRESH_MS)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

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
            self._rebuild_for(buf.size, int(self._engine.samplerate))
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
        p.fillRect(0, 0, W, H, _themed('window_bg'))
        if not AUDIO_OK or self._engine is None:
            p.setPen(_themed('text_dim'))
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
        grid_pen = QPen(_themed('grid'))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            p.drawLine(QPointF(gx, y0), QPointF(gx, y1))
        p.setPen(_themed('text_dim'))
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            label = f'{hz}' if hz < 1000 else f'{hz // 1000}k'
            p.drawText(QRectF(gx - 20, y1 + 1, 40, margin_b),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       label)

        # Horizontal dB gridline at -40 dBFS for visual reference.
        p.setPen(QPen(_themed('grid'), 1, Qt.PenStyle.DashLine))
        y_mid = y_of(-40.0)
        p.drawLine(QPointF(x0, y_mid), QPointF(x1, y_mid))

        # Live curve as a filled polygon under the line.
        poly = QPolygonF()
        poly.append(QPointF(x_of(self._centers[0]), y1))
        for i, f in enumerate(self._centers):
            poly.append(QPointF(x_of(float(f)), y_of(float(self._levels[i]))))
        poly.append(QPointF(x_of(self._centers[-1]), y1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_themed('accent', 110))
        p.drawPolygon(poly)

        # Bright line on top of the fill.
        line_pen = QPen(_themed('accent'))
        line_pen.setWidth(2)
        p.setPen(line_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._levels[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        # Peak-hold envelope as a thinner, paler line on top.
        peak_pen = QPen(_themed('warn'))
        peak_pen.setWidth(1)
        p.setPen(peak_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._peaks[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        p.end()




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
            top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            self._grid.addWidget(k_lbl, r, 0, top_left)
            self._grid.addWidget(v_lbl, r, 1, top_left)
            self._rows[key] = v_lbl
        self._grid.setColumnStretch(1, 1)

        # Polls only while visible; see showEvent/hideEvent.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._refresh_static()
        self._refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh()
        self._timer.start(self.REFRESH_MS)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

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
        d = eng.active_device if eng is not None else None
        if d is not None:
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
        params = FILTER_PRESETS.get(mode, {})
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
        last = self._engine.last_devices_refresh_at
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
            lambda: DEFAULT_SAMPLE_RATE)

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
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(r, _themed('accent_muted'))
        else:
            painter.fillRect(r, QColor(0, 0, 0, 0))

        # Colour from the shared thresholds, on the rounded value shown.
        bar_col = _themed(_BAND_ROLE[cents_band(cents)])

        # Dimensionen
        pad_x, pad_y = 8, 5
        w = r.width() - 2 * pad_x
        h = r.height() - 2 * pad_y
        cx = r.left() + pad_x + w // 2   # Mittellinie x

        # Hintergrundleiste
        bg_h = max(4, h // 3)
        bg_y = r.top() + pad_y + (h - bg_h) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_themed('grid'))
        painter.drawRoundedRect(r.left() + pad_x, bg_y, w, bg_h, 3, 3)

        # Green in-tune zone (±IN_TUNE_CENTS)
        zone_w = max(2, int(w / 2 * IN_TUNE_CENTS / self.MAX_CENT))
        painter.setBrush(_themed('ok', 160))
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
        painter.setPen(QPen(_themed('text'), 1))
        painter.drawLine(cx, bg_y - 2, cx, bg_y + bg_h + 2)

        # Cent-Wert als Text rechts — adaptive precision (v0.5.6)
        sr = self._sr_get() or DEFAULT_SAMPLE_RATE
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
            lambda: DEFAULT_SAMPLE_RATE)

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
        sr       = self._sr_get() or DEFAULT_SAMPLE_RATE

        if not in_range:
            # Out-of-range cells render with NO border and NO text — they
            # blend into the table background so the user only sees the
            # cells that correspond to physically playable notes. The
            # surrounding grid structure stays so column alignment is
            # preserved.
            painter.fillRect(r, _themed('window_bg'))
            painter.restore()
            return

        # In-range background — active blue tint or default panel color.
        if active:
            painter.fillRect(r, _themed('accent_muted'))
        else:
            painter.fillRect(r, _themed('base_bg'))

        # Subtle cell border — only on in-range cells, so the playable
        # area visually pops as a grid against the unbordered out-of-
        # range background.
        painter.setPen(QPen(_themed('grid'), 1))
        painter.drawRect(r.adjusted(0, 0, -1, -1))

        pad_x = 4
        # ----- Top strip: note names ----------------------------------
        top_h = 13
        top_y = r.top() + 1
        painter.setFont(QFont('Monospace', 7))
        painter.setPen(_themed('text_dim'))
        if fingered:
            painter.drawText(
                QRectF(r.left() + pad_x, top_y, r.width() / 2 - pad_x, top_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                fingered)
        if sounding and sounding != fingered:
            painter.setPen(_themed('text_dim'))
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
            painter.setPen(_themed('text_dim'))
            painter.setFont(QFont('Monospace', 10))
            painter.drawText(
                QRectF(r.left(), r.top() + top_h, r.width(), r.height() - top_h),
                Qt.AlignmentFlag.AlignCenter, '·')
            painter.restore()
            return

        # ----- Mid strip: mean (color) + ±std ------------------------
        col = _themed(_BAND_ROLE[cents_band(mean)])
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
            painter.setPen(_themed('text_dim'))
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
        painter.setBrush(_themed('grid'))
        painter.drawRoundedRect(QRectF(r.left() + pad_x, scale_y,
                                        scale_w, scale_h), 2, 2)
        # In-tune band (±IN_TUNE_CENTS).
        zone_w = max(2.0, scale_w / 2 * IN_TUNE_CENTS / self.MAX_CENT)
        painter.setBrush(_themed('ok', 160))
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
        painter.setPen(QPen(_themed('text'), 1))
        painter.drawLine(QPointF(cx, scale_y - 2),
                          QPointF(cx, scale_y + scale_h + 2))

        # Whiskers.
        if n > 1 and std > 0:
            std_lo = max(-1.0, min(1.0, (mean - std) / self.MAX_CENT))
            std_hi = max(-1.0, min(1.0, (mean + std) / self.MAX_CENT))
            x_lo = cx + std_lo * scale_w / 2
            x_hi = cx + std_hi * scale_w / 2
            w_y = scale_y + scale_h + 4
            painter.setPen(QPen(_themed('text_bright', 200), 1.4))
            painter.drawLine(QPointF(x_lo, w_y), QPointF(x_hi, w_y))
            painter.drawLine(QPointF(x_lo, w_y - 2),
                              QPointF(x_lo, w_y + 2))
            painter.drawLine(QPointF(x_hi, w_y - 2),
                              QPointF(x_hi, w_y + 2))

        # N counter, bottom-right.
        painter.setPen(_themed('text_dim'))
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
# Audio-Eingang Chip + Banner (v0.5.4)
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


class StatusDot(QWidget):
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
    directly, which keeps the GUI thread off the slow PortAudio
    cold-init path."""

    def __init__(self, t_func):
        super().__init__()
        self._t = t_func
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setObjectName('audioChip')   # themed via QPushButton#audioChip
        # Layout: dot · label · device name · chevron.
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        self._dot = StatusDot(self)
        lay.addWidget(self._dot)
        # v0.6.2: explicit transparent backgrounds on the inner labels.
        # The MainWindow-level stylesheet sets `QWidget{background:#12121a}`
        # which is a type selector that matches QLabel too. Without a
        # local `background:transparent` override, the labels paint dark
        # navy rectangles on top of the chip's slate-blue button bg —
        # rendering as vertical color bands across the chip.
        # Colour + transparent bg come from QPushButton#audioChip QLabel; keep
        # only the per-label font here.
        self._label = QLabel(self._t('audio_chip_label'))
        self._label.setStyleSheet('font-size:10px;font-weight:bold;')
        lay.addWidget(self._label)
        self._name = QLabel(self._t('audio_chip_none'))
        self._name.setStyleSheet('font-size:12px;')
        lay.addWidget(self._name, 1)
        self._chevron = QLabel('▾')
        self._chevron.setStyleSheet('font-size:12px;')
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
    device). The copy is selected by ``AudioEngineError``, so the
    user never sees raw PortAudio error codes."""

    def __init__(self, t_func, on_retry, on_pick):
        super().__init__()
        self._t = t_func
        self._on_retry = on_retry
        self._on_pick = on_pick
        # Themed via QWidget#audioBanner in build_app_qss (was an inline dark box
        # that stayed dark-on-light). The red left-border keeps the error cue.
        self.setObjectName('audioBanner')
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
    notifications and wrong-instrument hints — anything that would otherwise
    interrupt the player with a modal QMessageBox.

    Use ``show_message(text, action_label=None, action_callback=None)`` to
    surface a banner; Dismiss always hides it. If no action label is
    provided, only the Dismiss button is shown."""

    def __init__(self, t_func):
        super().__init__()
        self._t = t_func
        self._action_callback = None
        self.setObjectName('infoBanner')   # themed via QWidget#infoBanner
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
                # tear the window down.
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

# =============================================================================
# sax_tab_metro.py — the METRO tab
#
# The metronome panel: BPM readout, slider, nudge and tap, time signature,
# click volume and Start/Stop. Sound comes from MetronomeController
# (sax_metronome.py), which is the single source of truth for tempo, time
# signature and whether it runs; this panel repaints from it on every
# on_state_changed.
#
# Honesty discipline: Start asks for the output through OutputUsers and only
# shows "running" once the metronome actually sounds; when no output can
# open, the button goes back off with an inline reason.
#
# Tempo, time signature and volume are written to the settings object as
# they change; the window saves it (on close, and with other settings).
# ``running_changed`` drives the green dot on the METRO tab.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sax_metronome import MetronomeController
from sax_output_status import show_with_reason

if TYPE_CHECKING:
    from sax_audio_engine import AudioEngine
    from sax_config import AppConfig
    from sax_output_users import OutputUsers
    from sax_translator import Translator

# These mirror Android useMetronome (BPM_MIN/MAX, the four presets).
# sax_metronome.py holds the authoritative constants; keep these in step.
TIME_SIGS = ('2/4', '3/4', '4/4', '6/8')
BPM_MIN = 30
BPM_MAX = 300


def clamp_bpm(v: int) -> int:
    return max(BPM_MIN, min(BPM_MAX, int(v)))


class MetroTab(QWidget):
    """The METRO tab; see the module comment."""

    running_changed = pyqtSignal(bool)

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 engine: 'AudioEngine', output_users: 'OutputUsers',
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._cfg = cfg
        self._output_users = output_users
        # Seeded from the settings (already coerced and clamped on load).
        self._bpm = clamp_bpm(cfg.last_bpm)
        self._time_sig = cfg.last_time_sig if cfg.last_time_sig in TIME_SIGS else '4/4'
        self._ctrl = MetronomeController(
            engine.mixer, int(engine.mixer.samplerate),
            bpm=self._bpm, time_sig=self._time_sig, volume=cfg.click_volume,
            on_state_changed=self._on_state_changed)
        self._build()

    @property
    def controller(self) -> MetronomeController:
        return self._ctrl

    @property
    def bpm(self) -> int:
        return self._bpm

    @property
    def time_sig(self) -> str:
        return self._time_sig

    # ── Building ─────────────────────────────────────────────────────────────
    def _build(self) -> None:
        tr = self._tr
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Big BPM readout.
        self._bpm_lbl = QLabel(str(self._bpm))
        self._bpm_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bpm_lbl.setObjectName('bpmReadout')   # themed accent number
        outer.addWidget(self._bpm_lbl)
        unit = QLabel()
        tr.bind(unit.setText, 'metro_bpm')
        unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit.setObjectName('bpmUnit')
        outer.addWidget(unit)

        # BPM controls: −  [slider 30..300]  +   |  Tap.
        bpm_row = QHBoxLayout()
        btn_down = QToolButton()
        btn_down.setText('−')
        tr.bind(btn_down.setToolTip, 'metro_nudge_down_tip')
        btn_down.clicked.connect(lambda: self.nudge(-1))
        btn_up = QToolButton()
        btn_up.setText('+')
        tr.bind(btn_up.setToolTip, 'metro_nudge_up_tip')
        btn_up.clicked.connect(lambda: self.nudge(1))
        for b in (btn_down, btn_up):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName('stepperLg')   # themed via QToolButton#stepperLg
        self._slider = QSlider(Qt.Orientation.Horizontal)
        tr.bind(self._slider.setAccessibleName, 'metro_bpm')
        self._slider.setRange(BPM_MIN, BPM_MAX)
        self._slider.setValue(self._bpm)
        self._slider.valueChanged.connect(
            lambda value: self._set_bpm(value, from_slider=True))
        self._btn_tap = QPushButton()
        tr.bind(self._btn_tap.setText, 'metro_tap')
        tr.bind(self._btn_tap.setToolTip, 'metro_tap_tip')
        self._btn_tap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_tap.clicked.connect(self._on_tap)
        self._btn_tap.setObjectName('chrome')   # themed via QPushButton#chrome
        bpm_row.addWidget(btn_down)
        bpm_row.addWidget(self._slider, 1)
        bpm_row.addWidget(btn_up)
        bpm_row.addSpacing(12)
        bpm_row.addWidget(self._btn_tap)
        outer.addLayout(bpm_row)

        # Time-signature selector (exclusive).
        ts_grp = QGroupBox()
        tr.bind(ts_grp.setTitle, 'metro_timesig')
        ts_l = QHBoxLayout(ts_grp)
        ts_l.setContentsMargins(12, 8, 12, 8)
        self._ts_btns: dict[str, QPushButton] = {}
        for ts in TIME_SIGS:
            b = QPushButton(ts)
            b.setCheckable(True)
            b.setChecked(ts == self._time_sig)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _checked, t=ts: self.set_time_sig(t))
            b.setObjectName('chip')   # themed via QPushButton#chip
            self._ts_btns[ts] = b
            ts_l.addWidget(b)
        outer.addWidget(ts_grp)

        # Click volume.
        vol_grp = QGroupBox()
        tr.bind(vol_grp.setTitle, 'metro_volume')
        vol_l = QHBoxLayout(vol_grp)
        vol_l.setContentsMargins(12, 8, 12, 8)
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        tr.bind(self._vol_slider.setAccessibleName, 'metro_volume')
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(max(0, min(100, int(round(self._cfg.click_volume * 100)))))
        self._vol_slider.valueChanged.connect(self._on_volume)
        self._vol_lbl = QLabel(f'{self._vol_slider.value()}%')
        self._vol_lbl.setObjectName('dimLabel')
        self._vol_lbl.setMinimumWidth(44)
        vol_l.addWidget(self._vol_slider, 1)
        vol_l.addWidget(self._vol_lbl)
        outer.addWidget(vol_grp)

        # Start / stop (the run toggle; green→red on run).
        self._btn_start = QPushButton()
        tr.bind(self._btn_start.setText, 'metro_start')
        self._btn_start.setCheckable(True)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.toggled.connect(self._on_toggle)
        self._btn_start.setStyleSheet(
            'QPushButton{background:#1a6b3a;color:#fff;border:none;'
            'border-radius:6px;padding:12px;font-size:16px;font-weight:bold;}'
            'QPushButton:hover{background:#218a4b;}'
            'QPushButton:checked{background:#c0392b;}')
        outer.addWidget(self._btn_start)

        # Inline status — shown only when an audio action can't run yet, so the
        # controls never silently lie.
        self._status = QLabel('')
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setObjectName('errorLabel')   # themed via QLabel#errorLabel
        self._status.setVisible(False)
        outer.addWidget(self._status)
        outer.addStretch()

    # ── Tempo, metre, volume ─────────────────────────────────────────────────
    def nudge(self, delta: int) -> None:
        self._set_bpm(self._bpm + delta)

    def _set_bpm(self, bpm: int, *, from_slider: bool = False) -> None:
        bpm = clamp_bpm(bpm)
        self._bpm = bpm
        self._bpm_lbl.setText(str(bpm))
        if not from_slider and self._slider.value() != bpm:
            blocked = self._slider.blockSignals(True)
            self._slider.setValue(bpm)
            self._slider.blockSignals(blocked)
        self._cfg.last_bpm = bpm
        self._ctrl.set_bpm(bpm)

    def set_time_sig(self, ts: str) -> None:
        self._show_time_sig(ts)
        self._cfg.last_time_sig = ts
        self._ctrl.set_time_sig(ts)

    def _show_time_sig(self, ts: str) -> None:
        self._time_sig = ts
        for key, btn in self._ts_btns.items():
            blocked = btn.blockSignals(True)
            btn.setChecked(key == ts)
            btn.blockSignals(blocked)

    def _on_volume(self, value: int) -> None:
        self._vol_lbl.setText(f'{value}%')
        vol = value / 100.0
        self._cfg.click_volume = vol
        self._ctrl.set_volume(vol)

    def _on_tap(self) -> None:
        self._status.setVisible(False)
        # The controller updates its own bpm and fires on_state_changed, which
        # re-syncs the readout here — no manual set needed.
        self._ctrl.register_tap()

    # ── Start / stop ─────────────────────────────────────────────────────────
    def _on_toggle(self, on: bool) -> None:
        if not on:
            self._output_users.drop('metro')
            self._ctrl.stop()   # fires on_state_changed
            return
        self._status.setVisible(False)
        # The metronome sounds through the output mixer: it starts once an
        # output runs (so 'running' is never shown with nothing to hear), or
        # reverts with the reason if none can open. Starting it on an output
        # that already runs leaves the stream (and anything sounding) alone.
        self._output_users.want('metro', self._ctrl.start, self._revert_start)

    def _revert_start(self, status=None) -> None:
        """Un-check Start (without re-firing the toggle) and show why the
        metronome cannot sound, so the button never claims to be running."""
        blocked = self._btn_start.blockSignals(True)
        self._btn_start.setChecked(False)
        self._btn_start.blockSignals(blocked)
        self._tr.bind(self._btn_start.setText, 'metro_start')
        show_with_reason(self._tr, self._status, 'metro_unavailable', status)
        self._status.setVisible(True)

    def _on_state_changed(self) -> None:
        """Zero-arg slot for MetronomeController.on_state_changed (GUI thread),
        fired on every bpm/time-sig/volume/running change: repaint from the
        controller so the run dot, Start/Stop label, BPM readout and time
        signature can't drift."""
        ctrl = self._ctrl
        running = bool(ctrl.is_running())
        blocked = self._btn_start.blockSignals(True)
        self._btn_start.setChecked(running)
        self._btn_start.blockSignals(blocked)
        self._tr.bind(self._btn_start.setText, 'metro_stop' if running else 'metro_start')
        self.running_changed.emit(running)
        # Tap tempo changes the BPM inside the controller.
        bpm = clamp_bpm(int(ctrl.bpm))
        if bpm != self._bpm:
            self._bpm = bpm
            self._bpm_lbl.setText(str(bpm))
            blocked = self._slider.blockSignals(True)
            self._slider.setValue(bpm)
            self._slider.blockSignals(blocked)
            self._cfg.last_bpm = bpm
        ts = str(ctrl.time_sig)
        if ts in self._ts_btns and ts != self._time_sig:
            self._show_time_sig(ts)
            self._cfg.last_time_sig = ts

    # ── Output lifecycle ─────────────────────────────────────────────────────
    def park_output(self) -> None:
        """The output is gone: stop, and start again when it is back."""
        if self._ctrl.is_running():
            self._ctrl.stop()
            self._output_users.park('metro', self._ctrl.start, self._revert_start)

    def shut_down(self) -> None:
        self._ctrl.stop()

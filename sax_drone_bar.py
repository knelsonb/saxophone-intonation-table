# =============================================================================
# sax_drone_bar.py — the drone strip on the TUNER tab, and the pitch pipes
#
# DroneBar: on/off, the five preset voices, volume, the ±12 semitone
# stepper and the pitch-pipes launcher, wired to DroneController
# (sax_drone.py). The controller is the single source of truth for the
# voice and whether the drone sounds; the bar repaints from it on every
# on_state_changed and announces the voice with ``voice_changed`` (the SETUP
# tab's full voice picker shows it). ``select_voice`` is how that picker
# changes it.
#
# PitchPipesWindow: twelve chromatic pads (C4–B4) that sustain a reference
# tone each (PitchPipesController). Built on first open and reused, so the
# texts bound to the Translator stay valid while it is open.
#
# Both sound through the output mixer and follow the honesty discipline:
# nothing is shown as sounding unless an output runs; when none can open,
# the toggle goes back off with an inline reason. Both follow the session's
# A4, so they always sound against the reference the tuner measures with.
# =============================================================================
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sax_drone import DRONE_PRESETS, DroneController
from sax_output_status import show_with_reason
from sax_pitch_pipes import PitchPipesController

if TYPE_CHECKING:
    from sax_audio_engine import AudioEngine
    from sax_config import AppConfig
    from sax_output_users import OutputUsers
    from sax_session import SessionModel
    from sax_translator import Translator

SEMITONE_RANGE = 12
PIPE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
PIPE_LOW_MIDI = 60            # C4; the pads run C4..B4


def format_semitones(n: int) -> str:
    return f'{n:+d}' if n else '0'


class PitchPipesWindow(QDialog):
    """The pads; the drone bar decides what a tap does."""

    def __init__(self, tr: 'Translator', on_tap: Callable[[int], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        tr.bind(self.setWindowTitle, 'pipes_title')
        # bg inherits the global QWidget/window_bg rule (themed); pads + tip +
        # status carry their own themed objectName rules.
        box = QVBoxLayout(self)
        tip = QLabel()
        tr.bind(tip.setText, 'pipes_tip')
        tip.setWordWrap(True)
        tip.setObjectName('dimLabel')   # themed via QLabel#dimLabel
        box.addWidget(tip)
        grid = QGridLayout()
        grid.setSpacing(6)
        self._pads: dict[int, QPushButton] = {}
        for i, name in enumerate(PIPE_NAMES):
            midi = PIPE_LOW_MIDI + i
            b = QPushButton(name)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, m=midi: on_tap(m))
            b.setObjectName('padChip')   # themed via QPushButton#padChip
            self._pads[midi] = b
            grid.addWidget(b, i // 4, i % 4)
        box.addLayout(grid)
        self._status = QLabel('')
        self._status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._status.setVisible(False)
        box.addWidget(self._status)

    @property
    def status_label(self) -> QLabel:
        return self._status

    def show_active(self, midis) -> None:
        """Light every sustaining pad (the pipes can hold a chord)."""
        active = set(midis)
        for midi, b in self._pads.items():
            blocked = b.blockSignals(True)
            b.setChecked(midi in active)
            b.blockSignals(blocked)


class DroneBar(QGroupBox):
    """The drone strip; see the module comment."""

    # The drone's voice changed (a preset click, the SETUP picker, or the
    # controller falling back to the pure sine).
    voice_changed = pyqtSignal(str)

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 engine: 'AudioEngine', session: 'SessionModel',
                 output_users: 'OutputUsers',
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._cfg = cfg
        self._engine = engine
        self._session = session
        self._output_users = output_users
        # Seeded from the settings (coerced and clamped on load); a saved
        # volume of 0.0 is a real choice and is kept.
        self._voice_id = cfg.drone_voice_id
        self._volume = cfg.drone_volume
        self._semitones = cfg.drone_semitones
        # The pitch pipes, their window and controller are built on first open.
        self._pipes_ctrl: Optional[PitchPipesController] = None
        self._pipes_window: Optional[PitchPipesWindow] = None
        self._ctrl = DroneController(
            engine.mixer, int(engine.mixer.samplerate),
            voice_id=self._voice_id,
            volume=self._volume,
            semitones=self._semitones,
            a4=session.a4,
            enabled=False,
            engine=engine,   # enables D3 duck-attach on enable
            on_state_changed=self._on_state_changed)
        self._build()
        session.a4_changed.connect(self._on_a4_changed)

    # ── Read access ──────────────────────────────────────────────────────────
    @property
    def controller(self) -> DroneController:
        return self._ctrl

    @property
    def pipes_controller(self) -> Optional[PitchPipesController]:
        return self._pipes_ctrl

    @property
    def pipes_window(self) -> Optional[PitchPipesWindow]:
        return self._pipes_window

    @property
    def voice_id(self) -> str:
        return self._voice_id

    @property
    def semitones(self) -> int:
        return self._semitones

    # ── Building ─────────────────────────────────────────────────────────────
    def _build(self) -> None:
        tr = self._tr
        tr.bind(self.setTitle, 'drone_group')
        box = QVBoxLayout(self)
        box.setContentsMargins(8, 4, 8, 6)
        box.setSpacing(6)

        # Row 1: on/off + 5-preset row.
        row1 = QHBoxLayout()
        self._btn_on = QPushButton()
        tr.bind(self._btn_on.setText, 'drone_group')
        self._btn_on.setCheckable(True)
        tr.bind(self._btn_on.setToolTip, 'drone_on_tip')
        self._btn_on.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_on.toggled.connect(self._on_toggle)
        self._btn_on.setStyleSheet(
            'QPushButton{background:#1a6b3a;color:#fff;border:none;'
            'border-radius:5px;padding:6px 12px;font-size:13px;font-weight:bold;}'
            'QPushButton:hover{background:#218a4b;}'
            'QPushButton:checked{background:#c0392b;}')
        row1.addWidget(self._btn_on)
        self._preset_btns: dict[str, QPushButton] = {}
        for vid, _label, _program in DRONE_PRESETS:
            b = QPushButton()
            tr.bind(b.setText, f'drone_voice_{vid}')
            b.setCheckable(True)
            b.setChecked(vid == self._voice_id)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, i=vid: self.select_voice(i))
            b.setObjectName('chip')   # themed via QPushButton#chip
            self._preset_btns[vid] = b
            row1.addWidget(b)
        row1.addStretch()
        box.addLayout(row1)

        # Row 2: volume · semitone steppers · pitch-pipes launcher.
        row2 = QHBoxLayout()
        vol_lbl = QLabel()
        tr.bind(vol_lbl.setText, 'drone_volume')
        vol_lbl.setObjectName('dimLabel')
        row2.addWidget(vol_lbl)
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        tr.bind(self._vol_slider.setAccessibleName, 'drone_volume')
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(max(0, min(100, int(round(self._volume * 100)))))
        self._vol_slider.valueChanged.connect(self._on_volume)
        row2.addWidget(self._vol_slider, 1)
        row2.addSpacing(10)
        semi_lbl = QLabel()
        tr.bind(semi_lbl.setText, 'drone_semitone')
        semi_lbl.setObjectName('dimLabel')
        row2.addWidget(semi_lbl)
        btn_down = QToolButton()
        btn_down.setText('−')
        tr.bind(btn_down.setToolTip, 'drone_semitone_down_tip')
        btn_down.clicked.connect(lambda: self.nudge_semitones(-1))
        self._semi_lbl = QLabel(format_semitones(self._semitones))
        self._semi_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # No colour: it inherits the themed QWidget text colour; keep the size.
        self._semi_lbl.setStyleSheet('font-size:13px;min-width:28px;')
        btn_up = QToolButton()
        btn_up.setText('+')
        tr.bind(btn_up.setToolTip, 'drone_semitone_up_tip')
        btn_up.clicked.connect(lambda: self.nudge_semitones(1))
        for b in (btn_down, btn_up):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName('stepper')   # themed via QToolButton#stepper
        row2.addWidget(btn_down)
        row2.addWidget(self._semi_lbl)
        row2.addWidget(btn_up)
        row2.addSpacing(12)
        self._btn_pipes = QPushButton()
        tr.bind(self._btn_pipes.setText, 'pipes_launch')
        tr.bind(self._btn_pipes.setToolTip, 'pipes_tip')
        self._btn_pipes.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pipes.clicked.connect(self.open_pitch_pipes)
        self._btn_pipes.setObjectName('chrome')   # themed via QPushButton#chrome
        row2.addWidget(self._btn_pipes)
        box.addLayout(row2)

        # Inline status — shown when an audio action can't run yet.
        self._status = QLabel('')
        self._status.setWordWrap(True)
        self._status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._status.setVisible(False)
        box.addWidget(self._status)

    # ── Drone ────────────────────────────────────────────────────────────────
    def _on_toggle(self, on: bool) -> None:
        if not on:
            self._output_users.drop('drone')
            self._ctrl.set_enabled(False)
            return
        self._status.setVisible(False)
        self._output_users.want('drone', self._drone_on, self._revert_toggle)

    def _drone_on(self) -> None:
        """The output runs: sound the drone (built at the output's format)."""
        self._ctrl.set_enabled(True)   # fires on_state_changed
        if self._ctrl.synth_fallback:
            # The SoundFont synth could not load and the drone switched to
            # the pure-sine voice: say so (the voice itself was announced).
            self._tr.bind(self._status.setText, 'drone_sine_fallback')
            self._status.setVisible(True)

    def _revert_toggle(self, status=None) -> None:
        blocked = self._btn_on.blockSignals(True)
        self._btn_on.setChecked(False)
        self._btn_on.blockSignals(blocked)
        show_with_reason(self._tr, self._status, 'drone_unavailable', status)
        self._status.setVisible(True)

    def select_voice(self, voice_id: str) -> None:
        self._voice_id = voice_id
        self._cfg.drone_voice_id = voice_id
        self._ctrl.set_voice(voice_id)   # fires on_state_changed

    def _on_volume(self, value: int) -> None:
        self._volume = value / 100.0
        self._cfg.drone_volume = self._volume
        self._ctrl.set_volume(self._volume)

    def nudge_semitones(self, delta: int) -> None:
        n = max(-SEMITONE_RANGE, min(SEMITONE_RANGE, self._semitones + delta))
        self._semitones = n
        self._semi_lbl.setText(format_semitones(n))
        self._cfg.drone_semitones = n
        self._ctrl.set_semitones(n)

    def _on_state_changed(self) -> None:
        """Zero-arg slot for DroneController.on_state_changed: repaint the
        on/off button and the voice (preset row, and the SETUP picker through
        voice_changed) from the controller, and keep the setting."""
        ctrl = self._ctrl
        blocked = self._btn_on.blockSignals(True)
        self._btn_on.setChecked(ctrl.is_enabled())
        self._btn_on.blockSignals(blocked)
        vid = str(ctrl.voice_id)
        if vid != self._voice_id:
            self._voice_id = vid
            self._cfg.drone_voice_id = vid
        for preset, b in self._preset_btns.items():
            blocked = b.blockSignals(True)
            b.setChecked(preset == self._voice_id)
            b.blockSignals(blocked)
        self.voice_changed.emit(self._voice_id)

    def _on_a4_changed(self, a4: float) -> None:
        self._ctrl.set_a4(a4)
        if self._pipes_ctrl is not None:      # built on first open
            self._pipes_ctrl.set_a4(a4)

    # ── Pitch pipes ──────────────────────────────────────────────────────────
    def open_pitch_pipes(self) -> None:
        """Show the pitch pipes (raise them when already open). Pads route to
        PitchPipesController (numpy sine); a tap toggles a sustained
        reference tone, which needs an output — honest status if it can't
        sound."""
        window = self._pipes_window
        if window is not None:
            if not window.isVisible():
                window.status_label.setVisible(False)
                self._sync_pads()
                window.show()
            window.raise_()
            return
        if self._pipes_ctrl is None:
            self._pipes_ctrl = PitchPipesController(
                self._engine.mixer, int(self._engine.mixer.samplerate),
                a4=self._session.a4,
                engine=self._engine,   # held pads are D3 duck consumers
                on_state_changed=self._sync_pads)
        # Parented to the main window, as the app's other dialogs are.
        window = self._pipes_window = PitchPipesWindow(
            self._tr, self._on_pad_tapped, self.window())
        # Stop any sustaining pipe when the window closes.
        window.finished.connect(lambda _r: self._stop_pipes())
        self._sync_pads()         # reflect any active pad
        window.show()

    def _on_pad_tapped(self, midi: int) -> None:
        key = f'pipe:{midi}'
        if self._pipes_ctrl is not None and midi in self._pipes_ctrl.active_midis():
            self._pipes_ctrl.toggle(midi)      # release: needs no output
            return
        self._pipes_window.status_label.setVisible(False)
        if self._output_users.waiting(key):
            # Tapped again before the output came: the player changed their mind.
            self._output_users.drop(key)
        else:
            self._output_users.want(key, partial(self._pipe_on, midi), self._pipes_failed)
        self._sync_pads()

    def _pipe_on(self, midi: int) -> None:
        """The output runs: sound the pad (it fires on_state_changed)."""
        if midi not in self._pipes_ctrl.active_midis():
            self._pipes_ctrl.toggle(midi)

    def _pipes_failed(self, status) -> None:
        label = self._pipes_window.status_label
        show_with_reason(self._tr, label, 'pipes_unavailable', status)
        label.setVisible(True)
        self._sync_pads()

    def _stop_pipes(self) -> None:
        self._output_users.drop_prefix('pipe:')
        if self._pipes_ctrl is not None:
            try:
                self._pipes_ctrl.release_all()
            except Exception:
                pass

    def _sync_pads(self) -> None:
        if self._pipes_window is None:
            return
        try:
            active = self._pipes_ctrl.active_midis() if self._pipes_ctrl else ()
        except Exception:
            active = ()
        self._pipes_window.show_active(active)

    # ── Output lifecycle ─────────────────────────────────────────────────────
    def park_output(self) -> None:
        """The output is gone: switch the drone and any held pads off at once
        and wait to sound them again when it is back."""
        users = self._output_users
        if self._ctrl.is_enabled():
            self._ctrl.set_enabled(False)
            users.park('drone', self._drone_on, self._revert_toggle)
        if self._pipes_ctrl is not None:
            for midi in sorted(self._pipes_ctrl.active_midis()):
                users.park(f'pipe:{midi}', partial(self._pipe_on, midi), self._pipes_failed)
            self._pipes_ctrl.silence()

    def shut_down(self) -> None:
        """Release the drone and the pipes; each step runs even if the other
        fails, so nothing keeps sounding into a closing stream."""
        for step in (self._ctrl.shutdown, self._stop_pipes):
            try:
                step()
            except Exception:
                pass

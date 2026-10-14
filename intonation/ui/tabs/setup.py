# =============================================================================
# intonation/ui/tabs/setup.py — the SETUP tab
#
# Audio output (device, duplex preference, a test tone that proves the path
# end to end), the drone's full voice picker, the theme, and second copies of
# the nickname, concert-pitch and response controls from the toolbar, plus
# the mic gain.
#
# What it changes itself: the output device choice, the duplex preference,
# the mic gain (settings + engine) and the test tone. Everything shared with
# the rest of the window goes through what it is given: the session (A4 and
# nickname, which it also follows), and callbacks for the A4 recalculation,
# the response mode, the theme and the range editor. The drone voice it
# offers is announced with ``drone_voice_picked`` and shown with
# ``show_drone_voice``, so the drone bar stays the voice's only owner.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from intonation.audio.engine import query_output_devices
from intonation.audio.sources.drone import DRONE_FULL_GM, DRONE_PRESETS, DRONE_SYNTH_VOICES
from intonation.ui import theme
from intonation.ui.controls import (
    MIC_GAIN_DB_STEPS,
    a4_combo,
    db_to_linear,
    response_combo,
    show_data,
)
from intonation.ui.output_status import show_with_reason

if TYPE_CHECKING:
    from intonation.audio.engine import AudioEngine
    from intonation.audio.output_users import OutputUsers
    from intonation.core.session import SessionModel
    from intonation.i18n.translator import Translator
    from intonation.persistence.config import AppConfig, ConfigStore


class SetupTab(QWidget):
    """The SETUP tab; see the module comment."""

    # The player picked a drone voice in the full picker.
    drone_voice_picked = pyqtSignal(str)

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 store: 'ConfigStore', engine: 'AudioEngine',
                 session: 'SessionModel', output_users: 'OutputUsers',
                 request_output_open: Callable[[], None],
                 apply_a4: Callable[[float], None],
                 apply_response: Callable[[str], None],
                 apply_theme: Callable[[str], None],
                 open_range_editor: Callable[[], None],
                 nickname_committed: Callable[[], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._cfg = cfg
        self._store = store
        self._engine = engine
        self._session = session
        self._output_users = output_users
        self._request_output_open = request_output_open
        self._apply_a4 = apply_a4
        self._apply_response = apply_response
        self._apply_theme = apply_theme
        # Handle for the active test-tone source (None while silent).
        self._test_tone_handle = None
        # The output enumeration behind the device combo, so a chosen name's
        # host API is known without asking the engine again.
        self._output_devices: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(14)
        outer.addWidget(self._build_output_group())
        outer.addWidget(self._build_voice_group())
        outer.addWidget(self._build_appearance_group())
        outer.addWidget(self._build_instrument_group(open_range_editor,
                                                     nickname_committed))
        outer.addWidget(self._build_a4_group())
        outer.addWidget(self._build_response_group())
        outer.addWidget(self._build_mic_gain_group())
        outer.addWidget(self._build_test_tone_group())
        outer.addStretch()

        session.a4_changed.connect(lambda hz: show_data(self._a4_combo, int(hz)))
        session.nickname_changed.connect(self._show_nickname)

    # ── Building ─────────────────────────────────────────────────────────────
    def _build_output_group(self) -> QGroupBox:
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_output_group')
        form = QFormLayout(grp)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(8)
        self._output_combo = QComboBox()
        self._output_combo.setMinimumWidth(320)
        tr.bind(self._output_combo.setAccessibleName, 'setup_output_device')
        # Filled once here; later changes to the device list arrive from the
        # engine's hot-plug poll (show_output_devices). Only enumerates.
        try:
            devices = query_output_devices()
        except Exception:
            devices = []
        self.show_output_devices(devices)
        tr.bind_item(self._output_combo, 0, 'setup_output_default')
        self._output_combo.currentIndexChanged.connect(self._on_output_device_changed)
        label = QLabel()
        tr.bind(label.setText, 'setup_output_device')
        form.addRow(label, self._output_combo)
        self._cb_prefer_duplex = QCheckBox()
        tr.bind(self._cb_prefer_duplex.setText, 'setup_prefer_duplex')
        tr.bind(self._cb_prefer_duplex.setToolTip, 'setup_prefer_duplex_tip')
        self._cb_prefer_duplex.setChecked(self._cfg.output_prefer_duplex)
        self._cb_prefer_duplex.toggled.connect(self._on_prefer_duplex_toggled)
        form.addRow('', self._cb_prefer_duplex)
        return grp

    def _build_voice_group(self) -> QGroupBox:
        # The full voice picker lives here; the TUNER drone bar carries the
        # 5-preset quick row. Both show the drone's one voice.
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_drone_voice_group')
        form = QFormLayout(grp)
        form.setContentsMargins(12, 10, 12, 10)
        combo = self._voice_combo = QComboBox()
        combo.setMinimumWidth(260)
        tr.bind(combo.setAccessibleName, 'setup_drone_voice')
        # Presets, then the desktop-only synth voices (pure sine), then the
        # full GM catalog. Each item's data is the voice id the controller
        # resolves. Presets and synth voices have translated names. The GM
        # voices keep their standard General MIDI names, numbered as in the
        # GM table: two presets share a GM voice's name (Cello, Tenor Sax),
        # and the number tells the preset and the raw GM voice apart.
        named = DRONE_PRESETS + DRONE_SYNTH_VOICES
        for vid, _label, _program in named:
            combo.addItem('', vid)
            tr.bind_item(combo, combo.count() - 1, f'drone_voice_{vid}')
        named_ids = {vid for vid, _label, _program in named}
        for vid, gm_name, program in DRONE_FULL_GM:
            if vid not in named_ids:
                combo.addItem(f'GM {program + 1} · {gm_name}', vid)
        show_data(combo, self._cfg.drone_voice_id)
        combo.currentIndexChanged.connect(self._on_voice_changed)
        label = QLabel()
        tr.bind(label.setText, 'setup_drone_voice')
        form.addRow(label, combo)
        return grp

    def _build_appearance_group(self) -> QGroupBox:
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_appearance_group')
        form = QFormLayout(grp)
        form.setContentsMargins(12, 10, 12, 10)
        self._theme_combo = QComboBox()
        tr.bind(self._theme_combo.setAccessibleName, 'setup_theme')
        tr.bind_items(self._theme_combo,
                      [(f'theme_{name}', name) for name in theme.THEME_ORDER])
        show_data(self._theme_combo, theme.active_name())
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        label = QLabel()
        tr.bind(label.setText, 'setup_theme')
        form.addRow(label, self._theme_combo)
        return grp

    def _build_instrument_group(self, open_range_editor, nickname_committed) -> QGroupBox:
        # The range editor is also reachable via the toolbar gear. No inline
        # stylesheet: the themed QPushButton/QLineEdit QSS re-themes them.
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_instrument_group')
        box = QVBoxLayout(grp)
        box.setContentsMargins(12, 10, 12, 10)
        # Horn nickname (cfg.last_nickname, used by CSV per-nickname
        # slicing): a second editor for the session's nickname.
        row = QHBoxLayout()
        label = QLabel()
        tr.bind(label.setText, 'grp_nickname')
        row.addWidget(label)
        self._nick_edit = QLineEdit()
        self._nick_edit.setObjectName('setupNickname')
        tr.bind(self._nick_edit.setPlaceholderText, 'nickname_tip')
        tr.bind(self._nick_edit.setAccessibleName, 'grp_nickname')
        self._nick_edit.setText(self._session.nickname)
        self._nick_edit.textChanged.connect(self._session.set_nickname)
        self._nick_edit.editingFinished.connect(nickname_committed)
        row.addWidget(self._nick_edit)
        box.addLayout(row)
        self._btn_range = QPushButton()
        tr.bind(self._btn_range.setText, 'setup_range_edit')
        self._btn_range.setObjectName('setupRangeEdit')
        self._btn_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_range.clicked.connect(open_range_editor)
        box.addWidget(self._btn_range)
        return grp

    def _build_a4_group(self) -> QGroupBox:
        # A second copy of the toolbar's concert-pitch selector. A pick goes
        # through the window's A4 recalculation; both copies then follow the
        # session's A4 without emitting, so the recalculation runs once.
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'grp_a4')
        box = QVBoxLayout(grp)
        box.setContentsMargins(12, 10, 12, 10)
        self._a4_combo = a4_combo(self._session.a4)
        self._a4_combo.setObjectName('setupA4')
        self._a4_combo.setMinimumWidth(160)
        tr.bind(self._a4_combo.setAccessibleName, 'grp_a4')
        self._a4_combo.currentIndexChanged.connect(
            lambda _i: self._apply_a4(float(self._a4_combo.currentData())))
        box.addWidget(self._a4_combo)
        return grp

    def _build_response_group(self) -> QGroupBox:
        # A second copy of the toolbar's Response selector (Fast / Normal /
        # Slow); the window applies a pick and shows it in both.
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'grp_filter')
        box = QVBoxLayout(grp)
        box.setContentsMargins(12, 10, 12, 10)
        self._response_combo = response_combo(tr, self._engine.filter_mode)
        self._response_combo.setObjectName('setupResponse')
        self._response_combo.setMinimumWidth(220)
        tr.bind(self._response_combo.setAccessibleName, 'grp_filter')
        self._response_combo.currentIndexChanged.connect(
            lambda _i: self._apply_response(self._response_combo.currentData()))
        box.addWidget(self._response_combo)
        return grp

    def _build_mic_gain_group(self) -> QGroupBox:
        # Boost a quiet mic (in dB) so it clears the detection floor; 0 dB is
        # a no-op. The engine scales ONLY the silence gate + level meter (not
        # the signal), so the cents readout is unchanged.
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_mic_gain')
        box = QVBoxLayout(grp)
        box.setContentsMargins(12, 10, 12, 10)
        combo = self._mic_gain_combo = QComboBox()
        combo.setObjectName('setupMicGain')
        combo.setMinimumWidth(160)
        tr.bind(combo.setAccessibleName, 'setup_mic_gain')
        tr.bind(combo.setToolTip, 'setup_mic_gain_tip')
        for db in MIC_GAIN_DB_STEPS:
            combo.addItem('0 dB' if db == 0 else f'{db:+d} dB', db)
        # Show the nearest offered step (the engine already uses the exact
        # saved dB; the player can only pick a discrete step).
        saved = self._cfg.mic_gain_db
        combo.setCurrentIndex(min(range(combo.count()),
                                  key=lambda i: abs(combo.itemData(i) - saved)))
        combo.currentIndexChanged.connect(self._on_mic_gain_changed)
        box.addWidget(combo)
        # A plain-language line saying what the gain does, so the control has
        # a VISIBLE result although the cents readout is, by design, unaffected.
        self._mic_gain_status = QLabel('')
        self._mic_gain_status.setWordWrap(True)
        self._mic_gain_status.setObjectName('dimLabel')   # themed dim hint
        self._show_mic_gain_status()
        box.addWidget(self._mic_gain_status)
        return grp

    def _build_test_tone_group(self) -> QGroupBox:
        tr = self._tr
        grp = QGroupBox()
        tr.bind(grp.setTitle, 'setup_testtone_group')
        box = QVBoxLayout(grp)
        box.setContentsMargins(12, 10, 12, 10)
        btn = self._btn_test_tone = QPushButton()
        tr.bind(btn.setText, 'setup_testtone_play')
        btn.setCheckable(True)
        tr.bind(btn.setToolTip, 'setup_testtone_tip')
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton{background:#16a085;color:#fff;border:none;
                         border-radius:5px;padding:8px 14px;font-size:13px;}
            QPushButton:hover{background:#1abc9c;}
            QPushButton:checked{background:#b7770d;}
        """)
        btn.toggled.connect(self._on_test_tone_toggled)
        box.addWidget(btn)
        # Inline feedback when the tone can't start (no output device), so
        # the button never silently lies about playing (N2).
        self._test_tone_status = QLabel('')
        self._test_tone_status.setWordWrap(True)
        self._test_tone_status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._test_tone_status.setVisible(False)
        box.addWidget(self._test_tone_status)
        return grp

    # ── Following values set elsewhere ───────────────────────────────────────
    def show_output_devices(self, devices) -> None:
        """Offer ``devices`` (system default first), with the saved output
        selected when it is present. Called again when outputs come or go."""
        combo = self._output_combo
        blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._tr.t('setup_output_default'), '')
        self._output_devices = list(devices)
        saved = self._cfg.output_device_name
        select = 0
        for d in devices:
            combo.addItem(f'{d.name} · {d.host_api}', d.name)
            if saved and d.name == saved:
                select = combo.count() - 1
        combo.setCurrentIndex(select)
        combo.blockSignals(blocked)

    def show_drone_voice(self, voice_id: str) -> None:
        show_data(self._voice_combo, voice_id)

    def show_response(self, mode: str) -> None:
        show_data(self._response_combo, mode)

    def _show_nickname(self, text: str) -> None:
        # The editor being typed in already shows it; touching it would move
        # its cursor.
        if self._nick_edit.text() != text:
            blocked = self._nick_edit.blockSignals(True)
            self._nick_edit.setText(text)
            self._nick_edit.blockSignals(blocked)

    # ── Settings ─────────────────────────────────────────────────────────────
    def _on_output_device_changed(self, _idx: int) -> None:
        name = self._output_combo.currentData() or ''
        host_api = next((d.host_api for d in self._output_devices if d.name == name), '')
        cfg = self._cfg
        cfg.output_device_name = name
        cfg.output_device_host_api = host_api if name else ''
        cfg.output_device_samplerate = 0   # auto-negotiate
        self._store.save(cfg)
        # Re-open on the new device. Whatever is sounding carries over: the
        # mixer re-rates every source for the new stream and keeps the
        # metronome's next beat where it was due.
        self._request_output_open()

    def _on_prefer_duplex_toggled(self, checked: bool) -> None:
        # Used the next time the output opens. No re-open mid-tone: that would
        # chop a playing test tone for a setting that costs nothing to defer.
        self._cfg.output_prefer_duplex = bool(checked)
        self._store.save(self._cfg)

    def _on_voice_changed(self, _idx: int) -> None:
        voice_id = self._voice_combo.currentData()
        if voice_id:
            self.drone_voice_picked.emit(str(voice_id))

    def _on_theme_changed(self, _idx: int) -> None:
        self._apply_theme(str(self._theme_combo.currentData() or theme.DEFAULT_THEME))

    def _on_mic_gain_changed(self, _idx: int) -> None:
        db = self._mic_gain_combo.currentData()
        if db is None:
            return
        self._cfg.mic_gain_db = float(db)
        self._store.save(self._cfg)
        self._engine.set_mic_gain(db_to_linear(db))
        self._show_mic_gain_status()

    def _show_mic_gain_status(self) -> None:
        db = int(round(self._cfg.mic_gain_db))
        label = self._mic_gain_status
        if db == 0:
            self._tr.bind(label.setText, 'setup_mic_gain_off')
        elif db > 0:
            self._tr.bind(label.setText, 'setup_mic_gain_boost', db=db)
        else:
            self._tr.bind(label.setText, 'setup_mic_gain_cut', db=db)

    # ── Test tone ────────────────────────────────────────────────────────────
    def _on_test_tone_toggled(self, on: bool) -> None:
        """Start/stop a 440 Hz test tone through the mixer. The tuner is
        deliberately NOT muted — the readout stays live while output sounds;
        this is the manual proof of that path."""
        self._test_tone_status.setVisible(False)
        if on:
            self._output_users.want('tone', self._test_tone_on, self._test_tone_failed)
        else:
            self._output_users.drop('tone')
            self._stop_test_tone()
            self._show_test_tone_on(False)

    def _test_tone_on(self) -> None:
        """The output runs: sound the tone (the engine builds it at the
        output's format; the mixer keeps it in tune across later switches)."""
        try:
            handle = self._engine.start_test_tone(440.0)
        except Exception:
            handle = None
        if handle is None:
            self._test_tone_failed(None)
            return
        self._test_tone_handle = handle
        self._show_test_tone_on(True)

    def _test_tone_failed(self, status) -> None:
        """Nothing is sounding — don't let the button claim "Stop"; say why
        (N2), naming the output when an open failed."""
        self._show_test_tone_on(False)
        show_with_reason(self._tr, self._test_tone_status, 'setup_testtone_failed', status)
        self._test_tone_status.setVisible(True)

    def _stop_test_tone(self, fade: bool = True) -> None:
        if self._test_tone_handle is not None:
            try:
                self._engine.stop_test_tone(self._test_tone_handle, fade=fade)
            except Exception:
                pass
        self._test_tone_handle = None

    def _show_test_tone_on(self, on: bool) -> None:
        """Paint the button without re-entering its toggled handler."""
        btn = self._btn_test_tone
        blocked = btn.blockSignals(True)
        btn.setChecked(on)
        btn.blockSignals(blocked)
        self._tr.bind(btn.setText, 'setup_testtone_stop' if on else 'setup_testtone_play')

    # ── Output lifecycle ─────────────────────────────────────────────────────
    def park_output(self) -> None:
        """The output is gone: stop the tone at once (a fade would only play
        when the output is back) and wait to sound it again."""
        if self._test_tone_handle is not None:
            self._stop_test_tone(fade=False)
            self._show_test_tone_on(False)
            self._output_users.park('tone', self._test_tone_on, self._test_tone_failed)

    def shut_down(self) -> None:
        self._stop_test_tone()

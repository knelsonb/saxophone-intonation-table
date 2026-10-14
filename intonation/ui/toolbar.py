# =============================================================================
# intonation/ui/toolbar.py — the toolbar above the tabs
#
# Two predictable rows: inputs on top (instrument, display and layout,
# concert pitch, response, language, the audio-input chip, CSV import),
# actions below (autotune, record on/off, reset, the exports). When the
# window is narrow, content stays grouped by intent instead of wrapping at
# arbitrary positions.
#
# The toolbar shows the session (A4, display, nickname, recording) and the
# language by following their signals, and reports what the player does
# through its own signals; the window decides what happens. The two
# instrument combos are handed to the InstrumentController, which fills
# them.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from intonation.ui.controls import a4_combo, response_combo, show_data
from intonation.ui.widgets import AudioChip

if TYPE_CHECKING:
    from intonation.audio.engine import AudioEngine
    from intonation.core.session import SessionModel
    from intonation.i18n.translator import Translator
    from intonation.persistence.config import AppConfig

LANGUAGE_NAMES = (('de', 'Deutsch'), ('en', 'English'))

# Background of the record button: orange while recording, green while paused.
_RECORD_COLOURS = {True: '#b7770d', False: '#27ae60'}


def _button_qss(color: str) -> str:
    return f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """


class Toolbar(QWidget):
    """The toolbar; see the module comment."""

    a4_picked = pyqtSignal(float)
    response_picked = pyqtSignal(str)
    layout_picked = pyqtSignal(str)
    nickname_committed = pyqtSignal()
    range_editor_requested = pyqtSignal()
    custom_instrument_requested = pyqtSignal()
    audio_picker_requested = pyqtSignal()
    autotune_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    import_requested = pyqtSignal()
    # 'txt', 'pdf', 'csv' or 'chart'.
    export_requested = pyqtSignal(str)

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 engine: 'AudioEngine', session: 'SessionModel',
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._session = session
        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        inputs = QHBoxLayout()
        inputs.setSpacing(8)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        rows.addLayout(inputs)
        rows.addLayout(actions)

        inputs.addWidget(self._build_instrument_group())
        inputs.addWidget(self._build_display_group(cfg.layout_mode_preference))
        inputs.addWidget(self._build_a4_group())
        inputs.addWidget(self._build_response_group(engine.filter_mode))
        inputs.addWidget(self._build_language_group())
        # The audio-input chip: the at-a-glance health indicator when the
        # tuner goes silent. Clicking opens the device picker. It shows the
        # engine's state from start-up and repaints on every state change.
        self._audio_chip = AudioChip(tr)
        self._audio_chip.clicked.connect(self.audio_picker_requested)
        self._audio_chip.setMinimumWidth(220)
        self._audio_chip.update_from_state(engine.state, '', '', 0)
        inputs.addWidget(self._audio_chip)
        self._btn_import = self._action_button('btn_import', '#7f8c8d', self.import_requested.emit)
        inputs.addWidget(self._btn_import)
        inputs.addStretch()

        self._btn_autotune = self._action_button('btn_autotune', '#1a6b3a',
                                          self.autotune_requested.emit)
        self._btn_record = self._action_button('btn_stop', '#b7770d',
                                        lambda: session.set_recording(not session.recording))
        self._btn_reset = self._action_button('btn_reset', '#c0392b', self.reset_requested.emit)
        self._btn_txt = self._action_button('btn_txt', '#2980b9',
                                     lambda: self.export_requested.emit('txt'))
        self._btn_pdf = self._action_button('btn_pdf', '#8e44ad',
                                     lambda: self.export_requested.emit('pdf'))
        self._btn_csv = self._action_button('btn_csv', '#16a085',
                                     lambda: self.export_requested.emit('csv'))
        self._btn_chart = self._action_button('btn_chart', '#d35400',
                                       lambda: self.export_requested.emit('chart'))
        for b in (self._btn_autotune, self._btn_record, self._btn_reset):
            actions.addWidget(b)
        actions.addStretch()
        for b in (self._btn_txt, self._btn_pdf, self._btn_chart, self._btn_csv):
            actions.addWidget(b)
        self._show_recording(session.recording)

        session.a4_changed.connect(lambda hz: show_data(self._a4_combo, int(hz)))
        session.display_changed.connect(lambda mode: show_data(self._disp_combo, mode))
        session.nickname_changed.connect(self._show_nickname)
        session.recording_changed.connect(self._show_recording)
        tr.language_changed.connect(lambda lang: show_data(self._lang_combo, lang))

    # ── Shared with the window ───────────────────────────────────────────────
    @property
    def family_combo(self) -> QComboBox:
        return self._family_combo

    @property
    def instr_combo(self) -> QComboBox:
        return self._instr_combo

    @property
    def audio_chip(self) -> AudioChip:
        return self._audio_chip

    def show_response(self, mode: str) -> None:
        show_data(self._filter_combo, mode)

    # ── Building ─────────────────────────────────────────────────────────────
    def _titled_group(self, key: str) -> tuple[QGroupBox, QHBoxLayout]:
        box = QGroupBox()
        self._tr.bind(box.setTitle, key)
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 4, 8, 4)
        return box, row

    def _action_button(self, key: str, color: str, slot) -> QPushButton:
        b = QPushButton()
        self._tr.bind(b.setText, key)
        b.clicked.connect(slot)
        b.setMinimumHeight(36)
        b.setMinimumWidth(150)
        b.setStyleSheet(_button_qss(color))
        return b

    def _build_instrument_group(self) -> QGroupBox:
        tr = self._tr
        grp, row = self._titled_group('grp_instrument')
        row.setSpacing(6)
        # Family and instrument, filled by the InstrumentController.
        self._family_combo = QComboBox()
        self._family_combo.setMinimumWidth(130)
        row.addWidget(self._family_combo)
        self._instr_combo = QComboBox()
        self._instr_combo.setMinimumWidth(180)
        row.addWidget(self._instr_combo)
        # The per-instrument range editor, right beside the instrument combo
        # so it is found without hunting through menus. A Unicode glyph: the
        # project ships no settings icon.
        btn_range = QToolButton()
        btn_range.setText('⚙')   # U+2699 GEAR
        tr.bind(btn_range.setToolTip, 'gear_tip')
        btn_range.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_range.setObjectName('gear')   # themed via QToolButton#gear
        btn_range.clicked.connect(self.range_editor_requested)
        row.addWidget(btn_range)
        self._btn_custom = QPushButton()
        tr.bind(self._btn_custom.setText, 'custom_label')
        tr.bind(self._btn_custom.setToolTip, 'custom_dlg_title')
        self._btn_custom.setMaximumWidth(110)
        self._btn_custom.clicked.connect(self.custom_instrument_requested)
        self._btn_custom.setObjectName('chrome')   # themed via QPushButton#chrome
        row.addWidget(self._btn_custom)
        # The horn's nickname (also editable in SETUP). 280..320 px keeps real
        # horn names like "Selmer Reference 54 Tenor #2" readable.
        self._nick_edit = QLineEdit(self._session.nickname)
        tr.bind(self._nick_edit.setPlaceholderText, 'nickname_tip')
        self._nick_edit.setMinimumWidth(280)
        self._nick_edit.setMaximumWidth(320)
        self._nick_edit.textChanged.connect(self._session.set_nickname)
        self._nick_edit.editingFinished.connect(self.nickname_committed)
        self._nick_edit.setObjectName('nickEdit')   # themed via QLineEdit#nickEdit
        row.addWidget(self._nick_edit)
        return grp

    def _build_display_group(self, layout_pref: str) -> QGroupBox:
        tr = self._tr
        grp, row = self._titled_group('grp_display')
        self._disp_combo = QComboBox()
        tr.bind_items(self._disp_combo,
                      [('disp_griff', 'griff'), ('disp_klingend', 'klingend')])
        show_data(self._disp_combo, self._session.display)
        self._disp_combo.currentIndexChanged.connect(
            lambda _i: self._session.set_display(self._disp_combo.currentData()))
        row.addWidget(self._disp_combo)
        # Table layout: Auto picks from the window width; List and Grid pin
        # the layout whatever the width.
        self._layout_combo = QComboBox()
        tr.bind_items(self._layout_combo, [('layout_auto', 'auto'),
                                           ('layout_single', 'single'),
                                           ('layout_matrix', 'matrix')])
        show_data(self._layout_combo, layout_pref)
        self._layout_combo.currentIndexChanged.connect(
            lambda _i: self.layout_picked.emit(self._layout_combo.currentData()))
        row.addWidget(self._layout_combo)
        return grp

    def _build_a4_group(self) -> QGroupBox:
        grp, row = self._titled_group('grp_a4')
        self._a4_combo = a4_combo(self._session.a4)
        self._a4_combo.setMinimumWidth(100)
        self._a4_combo.currentIndexChanged.connect(
            lambda _i: self.a4_picked.emit(float(self._a4_combo.currentData())))
        row.addWidget(self._a4_combo)
        return grp

    def _build_response_group(self, mode: str) -> QGroupBox:
        # Pitch-detection smoothing, like a guitar tuner's Fast/Normal/Slow.
        grp, row = self._titled_group('grp_filter')
        self._filter_combo = response_combo(self._tr, mode)
        self._filter_combo.setMinimumWidth(100)
        self._filter_combo.currentIndexChanged.connect(
            lambda _i: self.response_picked.emit(self._filter_combo.currentData()))
        row.addWidget(self._filter_combo)
        return grp

    def _build_language_group(self) -> QGroupBox:
        grp, row = self._titled_group('grp_language')
        self._lang_combo = QComboBox()
        for code, name in LANGUAGE_NAMES:   # each language in its own name
            self._lang_combo.addItem(name, code)
        show_data(self._lang_combo, self._tr.lang)
        self._lang_combo.setMinimumWidth(100)
        self._lang_combo.currentIndexChanged.connect(
            lambda _i: self._tr.set_language(self._lang_combo.currentData()))
        row.addWidget(self._lang_combo)
        return grp

    # ── Following ────────────────────────────────────────────────────────────
    def _show_nickname(self, text: str) -> None:
        # The editor being typed in already shows it; touching it would move
        # its cursor.
        if self._nick_edit.text() != text:
            blocked = self._nick_edit.blockSignals(True)
            self._nick_edit.setText(text)
            self._nick_edit.blockSignals(blocked)

    def _show_recording(self, recording: bool) -> None:
        self._btn_record.setStyleSheet(_button_qss(_RECORD_COLOURS[recording]))
        self._tr.bind(self._btn_record.setText, 'btn_stop' if recording else 'btn_start')

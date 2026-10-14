#!/usr/bin/env python3
"""
Saxophon-Intonationsanalysator – GUI
=====================================
The main window: it builds the toolbar, the TUNER / METRO / DECK / SETUP
tabs and wires the controllers (sax_table, sax_export, sax_device,
sax_instrument_controller, sax_session_state) and the audio engine
together. Widgets live in sax_widgets, dialogs in sax_dialogs.

Abhaengigkeiten (Ubuntu / venv):
    pip install PyQt6 numpy sounddevice reportlab

Starten:
    python3 main.py
"""

import sys

# Launched as a script (`python sax_intonation_gui.py`): hand over to the
# launcher before importing anything else, so its crash handler is installed
# first and this module is loaded once, under its own name.
if __name__ == '__main__':
    import main
    sys.exit(main.main())

import csv
from pathlib import Path

from PyQt6.QtCore import QLocale, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import sax_config
import sax_crash
import sax_instruments
import sax_theme
from sax_audio_engine import (
    AUDIO_OK,
    DEFAULT_SAMPLE_RATE,
    FILTER_PRESETS,
    AudioEngine,
    DeviceInfo,
    DeviceSelection,
    OutputState,
)
from sax_controls import db_to_linear, response_combo, show_data
from sax_device import DeviceController
from sax_dialogs import (  # noqa: F401  (AudioPickerDialog, RangeEditorDialog,
    AudioPickerDialog,  # _promote_vendor_prefix re-exported for older importers)
    RangeEditorDialog,
    WelcomeDialog,
    _promote_vendor_prefix,
    add_lang_toggle,
)
from sax_drone_bar import DroneBar
from sax_export import ExportController
from sax_i18n import STRINGS
from sax_instrument_controller import InstrumentController
from sax_instruments import (
    display_name as instrument_display_name,
    families as instrument_families,
    register_custom,
    transp_map,
)
from sax_intonation_log import MeasurementLog
from sax_output_users import OutputUsers
from sax_paths import AppPaths
from sax_pitch import (  # noqa: F401  (re-exported for older importers)
    A4_DEFAULT,
    CHROMA,
    SAX_MIDI,
    cents_dev,
    format_cents,
    freq_to_midi,
    midi_note_name,
    note_name_to_midi,
)
from sax_session import RunTracker, SessionModel
from sax_session_state import SessionStateController
from sax_tab_deck import DeckTab
from sax_tab_metro import MetroTab
from sax_tab_setup import SetupTab
from sax_tab_tuner import TunerTab
from sax_translator import Translator
from sax_widgets import (  # noqa: F401  (MatrixCellDelegate re-exported)
    AudioChip,
    AudioRecoveryBanner,
    CentBarDelegate,
    DataPanelWidget,
    InfoBanner,
    MatrixCellDelegate,
    SpectrumAnalyzerWidget,
    StatusDot,
    TunerWidget,
)

APP_NAME = 'Intonation Analyzer'
APP_VERSION = '1.4.0'

# =============================================================================
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths | None = None):
        super().__init__()
        # Pick the user's system language as the default. German-speaking
        # locales get the original DE strings; anyone else gets English.
        # The saved language replaces it below, before any widget is built.
        lang = 'de' if QLocale.system().name().startswith('de') else 'en'
        # v0.5.7.1: default to Bb tenor — the most common community-band
        # sax. Existing users with a persisted ``last_instrument_key``
        # keep their own choice (restored a few lines below); this only
        # affects fresh installs and first-launches.
        instrument = 'bb_tenor'
        display = 'griff'
        # Cumulative count of distinct note emissions since launch (each
        # _on_note increments). Diagnostics readout consumes this; the
        # table-level note count comes from the session's stats.
        self._notes_count: int = 0
        # Wrong-instrument detector. Counts
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
        # Every file the app keeps is resolved once, here, and handed to the
        # store that owns it.
        self._paths = paths if paths is not None else AppPaths.from_environment()
        self._store = sax_config.ConfigStore(self._paths)
        sax_instruments.use_range_overrides(
            sax_instruments.RangeOverrides(self._paths.ranges))
        sax_crash.set_log_path(self._paths.crash_log)
        self._cfg = self._store.load()
        # Apply the saved theme before any widget is built, so the global
        # stylesheet (in _build_ui) and the custom-painted widgets all read the
        # right palette from first paint. Live switches go through _apply_theme.
        sax_theme.set_active(self._cfg.theme)
        for c in self._store.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)

        # v0.5.5: restore last-session preferences that influence the
        # initial UI build. Geometry, splitter sizes, and nickname text
        # are widget-bound and get applied after _build_ui returns.
        # First-launch (empty config) falls back to the locale-derived
        # default already set above.
        if self._cfg.last_lang in ('de', 'en'):
            lang = self._cfg.last_lang
        # Every translated text in the window is bound to this translator and
        # follows a language switch by itself (see sax_translator).
        self._tr = Translator(lang, self)
        self._tr.language_changed.connect(self._on_language_changed)
        if self._cfg.last_instrument_key:
            # Trust the saved key only if the catalog still knows it; a
            # stale custom instrument that was deleted should fall back
            # rather than crash the combo population.
            if self._cfg.last_instrument_key in transp_map():
                instrument = self._cfg.last_instrument_key
        # A saved custom instrument cannot be resolved while its file is
        # unreadable. The session runs on the default instrument, but the
        # saved choice is kept unless the player picks another one.
        keep_saved_instrument = (self._store.customs_unreadable
                                 and self._cfg.last_instrument_key not in transp_map())
        if self._cfg.last_display_mode in ('griff', 'klingend'):
            display = self._cfg.last_display_mode
        # What is being measured: A4, instrument, display, nickname,
        # recording on/off and the per-note statistics. The combos, the
        # engine, the drone, the pitch pipes and the history log follow its
        # signals (wired below and in _build_ui); the table and the exports
        # read it. Readings arrive through the engine's queued note signal,
        # so it is only ever touched on the Qt thread.
        self._session = SessionModel(instrument=instrument, display=display,
                                     parent=self)

        self._engine = AudioEngine()
        self._engine.set_filter_mode(self._cfg.filter_mode)
        self._engine.set_mic_gain(db_to_linear(self._cfg.mic_gain_db))
        self._engine.set_prefer_wdmks(self._cfg.prefer_wdmks)
        # Persistence comes from config (welcome dialog), with the env var
        # SAX_INTONATION_LOG_PATH as a power-user override that always wins.
        self._log = MeasurementLog(
            path=self._paths.history(self._cfg.effective_log_path(self._paths)))
        # Runs follow the session. The first one opens after
        # _restore_session_state below, so it is stamped with the restored A4
        # and nickname.
        self._runs = RunTracker(self._session, self._log, enabled=AUDIO_OK)
        if AUDIO_OK:
            self._engine.signals.note_event.connect(self._on_note_event)
            self._engine.signals.state_changed.connect(self._on_engine_state)
            self._engine.signals.devices_changed.connect(
                self._on_devices_changed)
            self._engine.signals.interface_appeared.connect(
                self._on_interface_appeared)
        # Wired even without PortAudio: an output request then fails, and
        # the failure is what turns the asking toggle back off.
        self._engine.signals.output_state_changed.connect(self._on_output_status)
        # Sound sources the player switched on, waiting for (or parked
        # until) the output runs. See sax_output_users.
        self._output_users = OutputUsers(
            lambda: bool(self._engine.output_running), self._request_output_open)
        # The "output disconnected" notice while it is shown.
        self._output_notice = None

        self._build_ui()
        # Persistable session state (geometry,
        # splitter, instrument/nickname/display/A4/lang) lives on the
        # SessionStateController. Instantiated here, AFTER _build_ui,
        # because every widget reference it captures must already
        # exist. The two thin MainWindow methods below delegate to it.
        self._session_state = SessionStateController(
            self, self._cfg,
            session=self._session,
            translator=self._tr,
            splitter=self._tuner_tab.splitter,
            apply_a4=self._apply_a4,
            keep_saved_instrument=keep_saved_instrument,
        )
        # TXT / PDF / CSV / PNG-chart export flows and their helper dialogs
        # live on ExportController; it reads the session at export time.
        self._export = ExportController(
            self,
            log=self._log,
            session=self._session,
            tr=self._tr,
            instrument_label=self._instr_label,
            sample_rate=self._engine_sample_rate,
            add_lang_toggle=self._add_dialog_lang_toggle,
        )
        # Family/instrument selection, nickname editing,
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
        self._session.seed_expected_notes()
        self._runs.start_run()
        self._update_record_btn_style()

        # Audio-device picker open,
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
            store=self._store,
            tr=self._tr,
            chip=self._audio_chip,
            info_banner=self._info_banner,
            audio_banner=self._audio_banner,
            tuner_tab=self._tuner_tab,
        )

        # Start the engine AFTER the UI exists so a startup PortAudio
        # failure paints a banner instead of crashing __init__. The
        # engine's start() never raises — it sets state and emits
        # state_changed. This is the v0.5.4 headline fix.
        if AUDIO_OK:
            saved = self._device_selection_from_cfg()
            pref = self._cfg.audio_samplerate_pref
            # v0.5.7: the engine needs the saved (name, host_api) so its
            # hot-plug poller and retry_open path can re-resolve the
            # user's preferred device against a fresh device list
            # without round-tripping through MainWindow.
            self._engine.set_preferred_hint(saved)
            self._engine.request_start(preferred=saved, samplerate_pref=pref)

        # Hot-plug poller: 1 Hz (cached
        # query_devices is ~0 ms; only re-init is expensive, and we
        # only re-init when the device set actually changed).
        self._device_poll = QTimer(self)
        self._device_poll.timeout.connect(self._poll_devices)
        if AUDIO_OK:
            self._device_poll.start(1000)

        if not AUDIO_OK:
            QMessageBox.information(self, self._t('audio_error_title'),
                                     self._t('audio_error'))

        self._report_settings_problem()
        # First-boot welcome dialog asks about persistence. Skipped on every
        # subsequent launch, and when the settings file could not be read:
        # that player is not new, their file is only locked.
        if not self._cfg.welcome_shown and not self._store.unreadable:
            self._show_welcome_dialog()

        self._tuner_tab.start_refreshing()

    # ── Übersetzungs-Helfer ──────────────────────────────────────────────────
    @property
    def lang(self) -> str:
        return self._tr.lang

    def _t(self, key, **kwargs):
        """A string in the current language, for text shown once (message
        boxes, file dialogs). Text that stays on screen is bound instead."""
        return self._tr.t(key, **kwargs)

    # ── UI aufbauen ──────────────────────────────────────────────────────────
    def _build_ui(self):
        tr = self._tr
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
        self._grp_instr = QGroupBox()
        tr.bind(self._grp_instr.setTitle, 'grp_instrument')
        il = QHBoxLayout(self._grp_instr)
        il.setContentsMargins(8, 4, 8, 4)
        il.setSpacing(6)

        self._family_combo = QComboBox()
        self._fill_family_combo()
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
        tr.bind(self._btn_range.setToolTip, 'gear_tip')
        self._btn_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_range.setObjectName('gear')   # themed via QToolButton#gear
        self._btn_range.clicked.connect(self._open_range_editor)
        il.addWidget(self._btn_range)

        self._btn_custom = QPushButton()
        tr.bind(self._btn_custom.setText, 'custom_label')
        tr.bind(self._btn_custom.setToolTip, 'custom_dlg_title')
        self._btn_custom.setMaximumWidth(110)
        self._btn_custom.clicked.connect(self._on_add_custom)
        self._btn_custom.setObjectName('chrome')   # themed via QPushButton#chrome
        il.addWidget(self._btn_custom)

        self._nick_edit = QLineEdit()
        tr.bind(self._nick_edit.setPlaceholderText, 'nickname_tip')
        # v0.5.7.1: 160px capped at ~12 characters, which truncated real
        # horn names like "Selmer Reference 54 Tenor #2" before the user
        # could see what they'd typed. Bump to 320 max / 280 min.
        self._nick_edit.setMinimumWidth(280)
        self._nick_edit.setMaximumWidth(320)
        self._nick_edit.textChanged.connect(self._session.set_nickname)
        self._nick_edit.editingFinished.connect(self._runs.nickname_committed)
        # De-inlined (THEME-INLINE): styled via QLineEdit#nickEdit in
        # build_app_qss so it re-themes on a theme switch. DARK is byte-identical
        # (the rule's roles equal the old inline hardcodes).
        self._nick_edit.setObjectName('nickEdit')
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
        self._grp_disp = QGroupBox()
        tr.bind(self._grp_disp.setTitle, 'grp_display')
        dl = QHBoxLayout(self._grp_disp)
        dl.setContentsMargins(8, 4, 8, 4)
        self._disp_combo = QComboBox()
        tr.bind_items(self._disp_combo,
                      [('disp_griff', 'griff'), ('disp_klingend', 'klingend')])
        self._disp_combo.setCurrentIndex(
            self._disp_combo.findData(self._session.display))
        self._disp_combo.currentIndexChanged.connect(self._on_disp_changed)
        dl.addWidget(self._disp_combo)

        # Layout mode override: Auto / List / Grid. Auto picks based on
        # window width; the explicit choices let the user pin the layout
        # they want regardless of how wide the window is.
        self._layout_combo = QComboBox()
        tr.bind_items(self._layout_combo, [('layout_auto', 'auto'),
                                           ('layout_single', 'single'),
                                           ('layout_matrix', 'matrix')])
        pref = self._cfg.layout_mode_preference
        for i in range(self._layout_combo.count()):
            if self._layout_combo.itemData(i) == pref:
                self._layout_combo.setCurrentIndex(i)
                break
        self._layout_combo.currentIndexChanged.connect(
            self._on_layout_pref_changed)
        dl.addWidget(self._layout_combo)

        # Kammerton
        self._grp_a4 = QGroupBox()
        tr.bind(self._grp_a4.setTitle, 'grp_a4')
        al = QHBoxLayout(self._grp_a4)
        al.setContentsMargins(8, 4, 8, 4)
        self._a4_combo = QComboBox()
        for hz in range(430, 451):
            self._a4_combo.addItem(f'{hz} Hz', hz)
        self._a4_combo.setCurrentIndex(self._a4_combo.findData(int(self._session.a4)))
        self._a4_combo.setMinimumWidth(100)
        self._a4_combo.currentIndexChanged.connect(self._on_a4_changed)
        al.addWidget(self._a4_combo)

        # Response: pitch-detection smoothing preset. Mirrors a guitar-
        # tuner "Fast/Normal/Slow" toggle. Routed straight into the
        # engine; persisted in cfg.filter_mode.
        self._grp_filter = QGroupBox()
        tr.bind(self._grp_filter.setTitle, 'grp_filter')
        fl = QHBoxLayout(self._grp_filter)
        fl.setContentsMargins(8, 4, 8, 4)
        self._filter_combo = response_combo(tr, self._engine.filter_mode)
        self._filter_combo.setMinimumWidth(100)
        self._filter_combo.currentIndexChanged.connect(
            lambda _i: self._apply_response(self._filter_combo.currentData()))
        fl.addWidget(self._filter_combo)

        # Sprache
        self._grp_lang = QGroupBox()
        tr.bind(self._grp_lang.setTitle, 'grp_language')
        ll2 = QHBoxLayout(self._grp_lang)
        ll2.setContentsMargins(8, 4, 8, 4)
        self._lang_combo = QComboBox()
        self._lang_combo.addItem('Deutsch', 'de')
        self._lang_combo.addItem('English', 'en')
        self._lang_combo.setCurrentIndex(self._lang_combo.findData(self.lang))
        self._lang_combo.setMinimumWidth(100)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        ll2.addWidget(self._lang_combo)

        # Buttons
        self._btn_autotune = self._make_btn('btn_autotune', '#1a6b3a', self._on_autotune)
        self._btn_record   = self._make_btn('btn_stop',     '#b7770d', self._on_record_toggle)
        self._btn_reset    = self._make_btn('btn_reset',    '#c0392b', self._on_reset)
        self._btn_txt      = self._make_btn('btn_txt',      '#2980b9', self._export_txt)
        self._btn_pdf      = self._make_btn('btn_pdf',      '#8e44ad', self._export_pdf)
        self._btn_csv      = self._make_btn('btn_csv',      '#16a085', self._export_csv)
        self._btn_chart    = self._make_btn('btn_chart',    '#d35400', self._export_chart)
        self._btn_import   = self._make_btn('btn_import',   '#7f8c8d', self._import_csv)

        # AUDIO IN chip, placed between Language and Import.
        # The chip is the user's at-a-glance health indicator when the
        # tuner goes silent. Clicking opens the picker modal.
        self._audio_chip = AudioChip(tr)
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

        # ── Banners ───────────────────────────────────────────────────────────
        # Recovery banner — appears only when the engine is in FAILED.
        # Hidden by default; show_for() makes it visible with the right copy.
        self._audio_banner = AudioRecoveryBanner(
            tr, self._retry_audio, self._open_audio_picker)
        # Non-modal info banner for hot-plug, the wrong-instrument hint and
        # settings problems: an optional action and an always-present Dismiss.
        self._info_banner = InfoBanner(tr)

        # ── TUNER tab ─────────────────────────────────────────────────────────
        # The drone strip sits under the tuner readout; its full voice picker
        # is in SETUP.
        self._drone_bar = DroneBar(tr=tr, cfg=self._cfg, engine=self._engine,
                                   session=self._session,
                                   output_users=self._output_users)
        self._tuner_tab = TunerTab(
            tr=tr, cfg=self._cfg, store=self._store, engine=self._engine,
            session=self._session,
            banners=(self._audio_banner, self._info_banner),
            drone_bar=self._drone_bar,
            sample_rate=self._engine_sample_rate,
            notes_count=lambda: self._notes_count)

        # ── Navigation shell ──────────────────────────────────────────────────
        # The global toolbar (built above) stays ABOVE a QTabWidget. The
        # TUNER tab hosts the tuner/table splitter as-is — it's the desktop's
        # centerpiece and is more capable than Android's tuner, so it keeps
        # its full size. METRO / DECK / SETUP are the other three tabs; METRO
        # and DECK carry a status-dot indicator (green=running on METRO,
        # pulsing red=recording on DECK, mirroring Android's TabBar) that the
        # controllers flip with a signal. SETUP carries the audio-output
        # device picker, test-tone control and the other settings.
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tab_keys = ['tuner', 'metro', 'deck', 'setup']
        self._tabs.addTab(self._tuner_tab, '')
        self._metro_tab = MetroTab(tr=tr, cfg=self._cfg, engine=self._engine,
                                   output_users=self._output_users)
        self._metro_tab.running_changed.connect(self.set_metro_running)
        self._tabs.addTab(self._metro_tab, '')
        self._deck_tab = DeckTab(tr=tr, cfg=self._cfg, engine=self._engine,
                                 output_users=self._output_users)
        self._tabs.addTab(self._deck_tab, '')
        self._setup_tab = SetupTab(
            tr=tr, cfg=self._cfg, store=self._store, engine=self._engine,
            session=self._session, output_users=self._output_users,
            request_output_open=self._request_output_open,
            apply_a4=self._apply_a4,
            apply_response=self._apply_response,
            apply_theme=self._apply_theme,
            open_range_editor=self._open_range_editor,
            nickname_committed=self._runs.nickname_committed)
        self._setup_tab.drone_voice_picked.connect(self._drone_bar.select_voice)
        self._drone_bar.voice_changed.connect(self._setup_tab.show_drone_voice)
        self._engine.signals.output_devices_changed.connect(
            self._setup_tab.show_output_devices)
        self._tabs.addTab(self._setup_tab, '')
        for i, key in enumerate(self._tab_keys):
            tr.bind_tab(self._tabs, i, f'nav_tab_{key}', f'nav_tab_{key}_tip')
        self._install_tab_indicators()
        self._deck_tab.recording_changed.connect(self.set_deck_recording)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)
        # Restore the last-active tab (cfg.last_active_tab,
        # allowlist-coerced upstream). Unknown / fresh-install → TUNER.
        self._restore_active_tab()

        # Fensterstil
        # Themed via sax_theme. The active palette was set from
        # cfg.theme in __init__ before any widget was built; build_app_qss(DARK)
        # reproduces this exact stylesheet, so dark is byte-for-byte unchanged
        # while night/light re-render from their palettes. Live theme switches
        # re-call setStyleSheet via _apply_theme.
        self.setStyleSheet(sax_theme.build_app_qss(sax_theme.active()))
        self._sync_qpalette()

        tr.bind_text(self.setWindowTitle,
                     lambda: f"{tr.t('window_title')} {APP_VERSION}")

        # InstrumentController owns family/instr/nickname
        # work, the range editor, custom-instrument registration, and
        # autotune. Instantiated here at the END of _build_ui because every
        # widget it captures (family_combo, instr_combo, nick_edit,
        # a4_combo) now exists. The wrappers on MainWindow (defined
        # further down) delegate to this controller; they MUST NOT be
        # invoked before this point.
        self._instr_ctrl = InstrumentController(
            self,
            session=self._session, store=self._store, engine=self._engine,
            tr=self._tr,
            family_combo=self._family_combo,
            instr_combo=self._instr_combo,
            add_lang_toggle=self._add_dialog_lang_toggle,
            on_instrument_changed=self._on_instrument_changed_external,
            on_range_changed=self._on_range_changed_external,
            on_a4_changed=self._on_a4_changed_external,
        )
        # Now safe to populate the family + sub-instrument combos via the
        # controller (their _on_* wrappers delegate here).
        self._select_family_for_instrument(self._session.instrument)
        self._populate_instrument_combo(select_key=self._session.instrument)
        # Everything that shows a session value follows it from here on.
        self._session.a4_changed.connect(self._on_session_a4)
        self._session.display_changed.connect(self._on_session_display)
        self._session.nickname_changed.connect(self._on_session_nickname)
        self._session.recording_changed.connect(self._on_session_recording)

    # ── Navigation shell helpers ─────────────────────────────────────────────
    def _current_output_selection(self) -> 'DeviceSelection':
        """The persisted output device as a DeviceSelection (mirrors the
        input path's _device_selection_from_cfg). Empty name = system
        default; samplerate 0 = auto-negotiate."""
        return DeviceSelection(
            name=self._cfg.output_device_name,
            host_api=self._cfg.output_device_host_api,
            samplerate=self._cfg.output_device_samplerate,
        )

    def _request_output_open(self) -> None:
        """Ask for the chosen output to be opened. The answer comes back as
        an OutputStatus (_on_output_status)."""
        self._engine.request_open_output(self._current_output_selection())

    # ── Output state ──────────────────────────────────────────────────────────
    def _on_output_status(self, status) -> None:
        """The output stream changed state (engine signal, GUI thread)."""
        users = self._output_users
        if status.state in (OutputState.DISCONNECTED, OutputState.FAILED):
            # The stream is gone (unplugged, or a reopen failed): whatever
            # was sounding is switched off and parked; on_status below starts
            # it again or tells it why it cannot.
            self._park_output_users()
        elif status.state == OutputState.RUNNING and self._output_notice:
            self._info_banner.hide_if_showing(self._output_notice)
            self._output_notice = None
        users.on_status(status)
        if status.state == OutputState.DISCONNECTED or (
                status.state == OutputState.FAILED and users.any_parked()):
            # Something waits for the output to come back: say which one.
            if status.device:
                self._output_notice = self._tr.msg('audio_output_disconnected',
                                                   name=status.device)
            else:
                self._output_notice = self._tr.msg('output_err_none')
            self._info_banner.show_message(self._output_notice)

    def _park_output_users(self) -> None:
        """The output is gone (unplugged, or a reopen failed): switch off
        everything that was sounding, so no toggle claims to play into
        nothing, and park it to come back on when the output is back (or be
        told why it cannot). Nothing renders now, so sources are dropped at
        once rather than faded: a fade would only play when the output is
        back, over the source started afresh."""
        self._setup_tab.park_output()
        self._metro_tab.park_output()
        self._drone_bar.park_output()
        self._deck_tab.park_output()

    def _apply_theme(self, name: str) -> None:
        """Switch the live UI theme: set the active palette, re-render the
        global stylesheet, repaint the custom-painted widgets (which read
        sax_theme.active() each paint), and persist the choice. Coerces any
        value; never raises into the signal."""
        pal = sax_theme.set_active(name)
        self.setStyleSheet(sax_theme.build_app_qss(pal))
        self._sync_qpalette()
        # The stylesheet re-polish repaints QSS-styled widgets; the QPainter-
        # based widgets (tuner, table, chart) need an explicit nudge to re-read
        # the palette.
        self.update()
        for _w in self.findChildren(QWidget):
            _w.update()
        self._cfg.theme = pal.name
        self._save_settings()

    def _sync_qpalette(self) -> None:
        """Set the QApplication palette from the active theme, so palette-based
        rendering the global QSS doesn't reach — QTableView/QListView viewport
        (Base), header sections, scrollbars, tooltips — also follows the theme.
        Without this they default to the dark Base and read as black blocks on
        the light theme (the matrix viewport + header row)."""
        from PyQt6.QtGui import QPalette
        app = QApplication.instance()
        if app is None:
            return
        p = sax_theme.active()
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(p.window_bg))
        pal.setColor(QPalette.ColorRole.Base, QColor(p.window_bg))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p.base_bg))
        pal.setColor(QPalette.ColorRole.Text, QColor(p.text))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(p.text))
        pal.setColor(QPalette.ColorRole.Button, QColor(p.button_bg))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(p.text))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(p.accent))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p.text_bright))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.base_bg))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p.text))
        app.setPalette(pal)

    def _install_tab_indicators(self) -> None:
        """Attach status-dot widgets to the METRO and DECK tabs, hidden
        until set_metro_running / set_deck_recording flip them. Mirrors
        Android's TabBar (TabBar.tsx): solid green when the metronome runs,
        pulsing red while the deck records. The controllers only emit a
        signal to flip them."""
        bar = self._tabs.tabBar()
        self._metro_dot = StatusDot()
        self._metro_dot.set_color('#2ecc71')   # green
        bar.setTabButton(self._tab_keys.index('metro'),
                         QTabBar.ButtonPosition.RightSide, self._metro_dot)
        self._deck_dot = StatusDot()
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
        """Slot for the DECK tab's recording_changed: show a pulsing red dot on the
        DECK tab while a take is recording."""
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
        key = self._cfg.last_active_tab
        if key in self._tab_keys:
            self._tabs.setCurrentIndex(self._tab_keys.index(key))

    def _make_btn(self, key, color, slot):
        b = QPushButton()
        self._tr.bind(b.setText, key)
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
        self._tr.set_language(self._lang_combo.itemData(idx))

    def _on_language_changed(self, _lang: str) -> None:
        """After the translator re-rendered every bound text: rebuild what is
        not a translation key (instrument names come from the catalog in each
        language) and repaint the table, whose captions are drawn per refresh."""
        idx = self._lang_combo.findData(self.lang)
        if idx != self._lang_combo.currentIndex():
            blocked = self._lang_combo.blockSignals(True)
            self._lang_combo.setCurrentIndex(idx)
            self._lang_combo.blockSignals(blocked)
        self._fill_family_combo()
        self._select_family_for_instrument(self._session.instrument)
        self._populate_instrument_combo(select_key=self._session.instrument)

    def _fill_family_combo(self) -> None:
        combo = self._family_combo
        blocked = combo.blockSignals(True)
        combo.clear()
        for family_key, name_de, name_en in instrument_families():
            combo.addItem(name_de if self.lang == 'de' else name_en, family_key)
        combo.blockSignals(blocked)

    # ── Audio-Callback ────────────────────────────────────────────────────────
    def _on_note_event(self, ev) -> None:
        """A reading from the analysis thread, queued to this thread.

        A reading graded against an A4 that has since been replaced is
        dropped, not re-graded: it was measured before the change, so it
        belongs to the run that ended with that A4, not to the run the change
        opened. The generation counter (not the A4 value) decides, so a round
        trip 440 -> 442 -> 440 still drops readings from the first 440."""
        if ev.a4_generation != self._engine.a4_generation:
            return
        self._on_note(ev.midi, ev.freq, ev.cents)

    def _on_note(self, midi_kl: int, freq: float, cents: float):
        session = self._session
        if not session.recording:
            return
        # Wrong-instrument detector. Run BEFORE the
        # OOR drop below so the counter still ticks when the user has
        # "filter to range" enabled — otherwise we'd never count notes
        # outside the range.
        try:
            in_range = session.in_range(midi_kl)
        except Exception:
            in_range = True
        if in_range:
            self._oor_count = 0
        else:
            self._oor_count += 1
            if (not self._oor_banner_shown
                    and self._oor_count >= self._oor_threshold):
                self._oor_banner_shown = True
                instrument = session.instrument
                self._info_banner.show_message(
                    lambda: self._t('wrong_instrument_banner_body',
                                    instrument=self._instr_label(instrument)))

        # Drop out-of-range notes when the toggle is off — keeps the table
        # bounded to the instrument's nominal range.
        if not self._cfg.allow_out_of_range and not in_range:
            return
        # The note also becomes the table's highlighted row.
        session.add_reading(midi_kl, cents, freq)
        # Diagnostics counter: every emitted note, including ones the table
        # later discards.
        self._notes_count += 1
        # Per-measurement log. Instrument/A4 are read off the active run
        # inside the log, not from the session, so a callback firing during
        # a UI change still attributes to the run that was active when it
        # fired.
        midi_gr = midi_kl - session.transposition
        self._log.add_measurement(midi_sounding=midi_kl,
                                   midi_fingered=midi_gr,
                                   cents=cents, freq_hz=freq)

        self._tuner_tab.show_reading(midi_kl, freq, cents)

    # ── Instrument / Anzeige / A4 ─────────────────────────────────────────────
    # The family/instr/nickname slots, range editor,
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
        # The wrong-instrument detector is anchored
        # to the *selected* instrument. New selection ⇒ counter resets and
        # the one-shot prompt becomes eligible to fire again.
        self._oor_count = 0
        self._oor_banner_shown = False
        self._info_banner.hide()
        # Seed the stats with empty slots for every expected fingered note so
        # the table immediately shows the player what the instrument's range
        # looks like. Real measurements fill in as the player plays;
        # out-of-range notes (overtones, accidentals) appear automatically
        # via the existing _on_note path.
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_range_changed_external(self) -> None:
        """Called by InstrumentController after the per-instrument range
        editor accepts new bounds. The instrument *key* hasn't changed,
        so we do NOT reset the wrong-instrument detector — just re-seed
        expected notes against the new bounds and repaint."""
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_a4_changed_external(self) -> None:
        """Called by InstrumentController after the autotune flow has
        applied a new A4 and cleared the table: show the instrument's range
        again and repaint. The session already moved everything that
        follows A4 (combos, engine, drone, pipes, the history run)."""
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_session_a4(self, a4: float) -> None:
        """The session's A4 changed (toolbar, SETUP, autotune or start-up
        restore): show it in the toolbar combo and hand it to the detection
        engine. The SETUP combo, the drone and the pitch pipes follow the
        same signal, so nothing can disagree about the reference."""
        show_data(self._a4_combo, int(a4))
        self._engine.set_a4(a4)

    def _on_session_display(self, mode: str) -> None:
        show_data(self._disp_combo, mode)

    def _on_session_nickname(self, text: str) -> None:
        """Both nickname editors show the session's nickname. The one being
        typed in already does; touching it would move its cursor."""
        edit = self._nick_edit
        if edit.text() != text:
            blocked = edit.blockSignals(True)
            edit.setText(text)
            edit.blockSignals(blocked)

    def _on_session_recording(self, on: bool) -> None:
        self._update_record_btn_style()
        if on:
            self._tuner_tab.show_status('no_signal')

    # ── Audio-Geräteverwaltung (v0.5.4) ───────────────────────────────────
    def _engine_sample_rate(self) -> int:
        """Best-known live sample rate for adaptive cent precision.
        Falls back to the constant default when the engine is FAILED or
        audio is disabled — defensive; nothing should call the formatter
        in those states, but a stale delegate paint can race teardown."""
        sr = int(self._engine.samplerate or 0)
        return sr if sr > 0 else int(DEFAULT_SAMPLE_RATE)

    def _output_rate(self) -> int:
        """Rate the output controllers start from. The mixer owns the output
        format and moves their sources along when it changes."""
        return int(self._engine.mixer.samplerate)

    def _device_selection_from_cfg(self) -> DeviceSelection:
        return DeviceSelection(
            name=self._cfg.audio_device_name,
            host_api=self._cfg.audio_device_host_api,
            samplerate=self._cfg.audio_device_samplerate,
        )

    # Audio device handling lives on DeviceController.
    # The methods below are one-line wrappers — Qt signal/slot connect()
    # bindings and the hot-plug QTimer wiring in __init__ point at
    # MainWindow methods, so we preserve them at MainWindow's surface
    # so PyQt's runtime slot type-check sees the exact bound-method
    # signature it saw before the extraction.
    def _open_audio_picker(self) -> None:
        self._device_ctrl.open_audio_picker()

    def _retry_audio(self) -> None:
        self._device_ctrl.retry_audio()

    def _poll_devices(self) -> None:
        self._device_ctrl.poll_devices()

    def _on_engine_state(self, state, err, msg, device) -> None:
        self._device_ctrl.on_engine_state(state, err, msg, device)

    def _on_devices_changed(self, _devices) -> None:
        self._device_ctrl.on_devices_changed(_devices)

    def _on_interface_appeared(self, device: 'DeviceInfo') -> None:
        self._device_ctrl.on_interface_appeared(device)

    def _on_layout_pref_changed(self, _idx: int) -> None:
        """User picked Auto / List / Grid. Persist and refresh the table."""
        pref = self._layout_combo.currentData()
        if pref in ('auto', 'single', 'matrix'):
            self._cfg.layout_mode_preference = pref
            self._save_settings()
            self._tuner_tab.refresh_table()

    def _apply_response(self, mode) -> None:
        """A Response (filter mode) pick from the toolbar or the SETUP combo:
        show it in both (without emitting, so it is applied once), save it
        and route the live engine. Unknown modes are ignored."""
        if mode not in FILTER_PRESETS:
            return
        show_data(self._filter_combo, mode)
        self._setup_tab.show_response(mode)
        self._cfg.filter_mode = mode
        self._save_settings()
        if AUDIO_OK:
            self._engine.set_filter_mode(mode)

    def _on_add_custom(self) -> None:
        self._instr_ctrl.on_add_custom()


    def _add_dialog_lang_toggle(self, dlg, layout, on_change=None):
        """DE | EN toggle for the app's own dialogs (see
        sax_dialogs.add_lang_toggle). A switch goes through the toolbar
        language combo, so the whole window follows."""
        add_lang_toggle(dlg, layout, get_lang=lambda: self.lang,
                        set_lang=self._select_language, t_func=self._t,
                        on_change=on_change)

    def _select_language(self, code: str) -> None:
        idx = self._lang_combo.findData(code)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)

    def _show_welcome_dialog(self) -> None:
        """First-boot dialog. Offers opt-in for persistent JSONL log."""
        default_path = self._paths.default_history
        dlg = WelcomeDialog(self, self._t, default_path,
                            self._add_dialog_lang_toggle)
        accepted = (dlg.exec() == QDialog.DialogCode.Accepted)
        # "Decide later" leaves the question open, so the dialog returns on
        # the next launch; it is the only place history saving is offered.
        self._cfg.welcome_shown = accepted
        if accepted and dlg.keep_history():
            self._cfg.persistence_enabled = True
            # Rebuild the log on the chosen path. Existing in-memory entries
            # would be lost otherwise — but at this point the user has only
            # been in the app a few seconds at most, so that's fine.
            # The environment override wins here too, as it does at start-up.
            new_path = self._paths.history(self._cfg.effective_log_path(self._paths))
            if new_path:
                self._log = MeasurementLog(path=new_path)
                # Everything built with the previous log follows the new one,
                # or CSV export and the runs would keep using the discarded
                # in-memory log.
                self._export.set_log(self._log)
                self._runs.set_log(self._log)
                self._runs.start_run()
        self._save_settings()

    # _ask_custom_instrument lives on InstrumentController.
    # It was only invoked by _on_add_custom (also extracted), so no
    # MainWindow wrapper is retained — the controller calls itself.

    def _on_disp_changed(self, _idx):
        self._session.set_display(self._disp_combo.currentData())

    def _on_a4_changed(self, idx):
        """Toolbar concert-pitch (A4) combo changed → apply + mirror SETUP."""
        self._apply_a4(float(self._a4_combo.itemData(idx)))

    def _apply_a4(self, new_a4) -> None:
        """Apply a concert-pitch (A4) change from EITHER the toolbar or the
        SETUP combo, or the saved one at start-up. The session hands the
        value to both combos (signals blocked, so this heavy recalc runs
        EXACTLY ONCE), the engine, the drone and the pitch pipes, and opens a
        new history run. Cents are a function of frequency + A4, so the
        table's cents are invalidated — but its frequencies are kept, so the
        table is re-derived from them instead of discarding what the user
        recorded."""
        new_a4 = float(new_a4)
        # A4 change recalibrates midi assignments,
        # so the wrong-instrument counter resets too.
        self._oor_count = 0
        self._oor_banner_shown = False
        # _on_note runs on this (Qt) thread, so no reading can land in the
        # middle of the rebuild below. pause_emissions() stops the engine
        # emitting while the reference changes, and the new A4 starts a new A4
        # generation: a reading queued before the pause still arrives
        # afterwards, but carries the old generation and _on_note_event drops
        # it, and the engine discards readings it was still holding back.
        self._engine.pause_emissions()
        try:
            self._session.set_a4(new_a4)
            self._session.rederive_stats(new_a4)
        finally:
            self._engine.resume_emissions()
        # Re-seed expected blank rows on top of the remapped data so the
        # instrument's range still shows even after a clean re-derive.
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _on_record_toggle(self):
        # The run follows (RunTracker); the button follows _on_session_recording.
        self._session.set_recording(not self._session.recording)

    def _update_record_btn_style(self):
        recording = self._session.recording
        if recording:
            color = '#b7770d'   # orange = läuft
        else:
            color = '#27ae60'   # grün = pausiert, klicken zum Starten
        self._btn_record.setStyleSheet(f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """)
        self._tr.bind(self._btn_record.setText,
                      'btn_stop' if recording else 'btn_start')

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _on_reset(self):
        if QMessageBox.question(
            self, self._t('reset_title'), self._t('reset_msg'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._session.clear_stats()
            # Reset implies "I might be switching instruments" — clear the
            # cached maker/model so the next export re-prompts.
            self._export.forget_instrument_info()
            self._tuner_tab.refresh_table()

    # ── Kammerton ermitteln ───────────────────────────────────────────────────
    def _on_autotune(self):
        self._instr_ctrl.on_autotune()

    # ── Instrument-Modell-Dialog ──────────────────────────────────────────────
    # The maker/model prompt lives on ExportController;
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
        # Confirm before merging into the current session.
        # The user picked a file but might not realise the import merges
        # into the live log; an "are you sure?" beat guards against
        # mixing somebody else's CSV with their own data.
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
        except ValueError as e:
            # ValueError covers a non-raw-mode header and the size / row
            # limits; show the specific reason under the general message.
            QMessageBox.warning(self, self._t('err_title'),
                                f"{self._t('csv_import_badhdr')}\n\n{e}")
            return
        except (OSError, csv.Error) as e:
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
        self._device_poll.stop()
        self._tuner_tab.stop_refreshing()
        self._shut_down_output_users()
        # Both streams close and the audio-control thread stops: nothing may
        # keep the microphone or the speaker open after the window is gone.
        self._engine.shutdown()
        self._log.close()
        ev.accept()

    def _shut_down_output_users(self) -> None:
        """Silence and release every sound source before the streams close.
        Each step is independent: one that fails must not keep the others
        (or the engine shutdown after them) from running."""
        steps = (
            lambda: self._output_users.drop_prefix(''),
            self._setup_tab.shut_down,
            self._metro_tab.shut_down,
            self._drone_bar.shut_down,
            self._deck_tab.shut_down,
        )
        for step in steps:
            try:
                step()
            except Exception:
                pass

    def _save_session_state(self) -> None:
        """Thin delegator + persistence. The snapshot work lives on
        SessionStateController; the settings save stays here because
        the controller is intentionally agnostic about *when* to
        persist. Called from closeEvent. The per-setting saves
        scattered through the GUI remain as a belt; this is the
        suspenders."""
        self._session_state.save()
        self._save_settings()

    def _save_settings(self) -> None:
        """Write the settings now. The store never raises: a failed write
        returns False and the next save tries again, and an unreadable file
        is left alone for the whole session."""
        self._store.save(self._cfg)

    def _report_settings_problem(self) -> None:
        """Tell the player once, at start-up, that their settings file could
        not be used. Non-modal, so a locked file never blocks the window."""
        store = self._store
        if store.unreadable:
            self._info_banner.show_message(self._tr.msg(
                'settings_unreadable', path=str(store.paths.config)))
        elif store.corrupt_backup is not None:
            self._info_banner.show_message(self._tr.msg(
                'settings_damaged', path=str(store.paths.config),
                backup=str(store.corrupt_backup)))


# =============================================================================
def load_app_icon() -> QIcon | None:
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

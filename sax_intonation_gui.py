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
import datetime
import math
from functools import partial
from pathlib import Path

from PyQt6.QtCore import QLocale, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTableWidget,
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
    FILTER_MODE_DEFAULT,
    FILTER_PRESETS,
    AudioEngine,
    AudioEngineError,
    DeviceInfo,
    DeviceSelection,
    OutputState,
    format_rates,
    query_output_devices,
)
from sax_deck import DeckController
from sax_device import DeviceController
from sax_dialogs import (  # noqa: F401  (AudioPickerDialog, RangeEditorDialog,
    AudioPickerDialog,  # _promote_vendor_prefix re-exported for older importers)
    RangeEditorDialog,
    WelcomeDialog,
    _promote_vendor_prefix,
    add_lang_toggle,
)
from sax_drone import DRONE_FULL_GM, DRONE_PRESETS, DRONE_SYNTH_VOICES, DroneController
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
from sax_metronome import MetronomeController
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
from sax_pitch_pipes import PitchPipesController
from sax_session_state import SessionStateController
from sax_stats import NoteStats
from sax_table import TableController
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
APP_VERSION = '1.3.0'

# Mic-gain SETUP control: the offered dB steps + the dB→linear conversion the
# engine expects. Module-level so the combo, the startup sync, and any test all
# agree on one source.
MIC_GAIN_DB_STEPS = (-12, -6, 0, 6, 12, 18, 24)


def _db_to_linear(db: float) -> float:
    """dB gain → linear multiplier (0 dB → 1.0)."""
    return 10.0 ** (float(db) / 20.0)


# =============================================================================
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths | None = None):
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
        # Per-note statistics behind the table. Main-thread only: the engine's
        # note_event signal is delivered queued, so _on_note runs on the
        # Qt thread like every other reader and writer, and needs no lock.
        self.stats: dict[int, NoteStats] = {}
        self._recording = True   # Aufnahme läuft beim Start
        self._active_midi: int | None = None
        self._active_midi_at: datetime.datetime | None = None
        self._layout_mode: str = 'single'   # 'single' | 'matrix'
        # Built after _build_ui; selecting the start-up instrument at the end
        # of _build_ui asks for a table refresh before it exists.
        self._table_ctrl: TableController | None = None
        # Cumulative count of distinct note emissions since launch (each
        # _on_note increments). Diagnostics readout consumes this; the
        # table-level note count comes from `len(self.stats)`.
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
        # Instrument maker/model for export headers. ExportController asks once
        # and stores the answer here; Reset clears _instr_info_asked so the
        # next export asks again.
        self._last_maker = ''
        self._last_model = ''
        self._instr_info_asked = False

        # Load user config + previously-registered custom instruments before
        # building the UI so the catalog reflects them at first paint.
        # Every file the app keeps is resolved once, here, and handed to the
        # store that owns it.
        self._paths = paths if paths is not None else AppPaths.from_environment()
        self._settings = sax_config.ConfigStore(self._paths)
        sax_config.use_store(self._settings)
        sax_instruments.use_range_overrides(
            sax_instruments.RangeOverrides(self._paths.ranges))
        sax_crash.set_log_path(self._paths.crash_log)
        self._cfg = self._settings.load()
        # Apply the saved theme before any widget is built, so the global
        # stylesheet (in _build_ui) and the custom-painted widgets all read the
        # right palette from first paint. Live switches go through _apply_theme.
        sax_theme.set_active(self._cfg.theme)
        for c in sax_config.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)

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
            if self._cfg.last_instrument_key in transp_map():
                self.instrument = self._cfg.last_instrument_key
        # A saved custom instrument cannot be resolved while its file is
        # unreadable. The session runs on the default instrument, but the
        # saved choice is kept unless the player picks another one.
        keep_saved_instrument = (self._settings.customs_unreadable
                                 and self._cfg.last_instrument_key not in transp_map())
        if getattr(self._cfg, 'last_display_mode', '') in ('griff', 'klingend'):
            self.display = self._cfg.last_display_mode

        self._engine = AudioEngine()
        self._engine.set_filter_mode(
            getattr(self._cfg, 'filter_mode', FILTER_MODE_DEFAULT))
        self._engine.set_mic_gain(
            _db_to_linear(getattr(self._cfg, 'mic_gain_db', 0.0)))
        self._engine.set_prefer_wdmks(
            bool(getattr(self._cfg, 'prefer_wdmks', False)))
        # Persistence comes from config (welcome dialog), with the env var
        # SAX_INTONATION_LOG_PATH as a power-user override that always wins.
        self._log = MeasurementLog(
            path=self._paths.history(self._cfg.effective_log_path(self._paths)))
        # The first run opens after _restore_session_state below, so it is
        # stamped with the restored A4 and nickname.
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
        self._engine.signals.output_devices_changed.connect(
            self._on_output_devices_changed)
        # Sound sources the player switched on, waiting for (or parked
        # until) the output runs. See sax_output_users.
        self._output_users = OutputUsers(
            lambda: bool(self._engine.output_running), self._request_output_open)
        # Text of the "output disconnected" notice while it is shown.
        self._output_notice: str = ''

        self._build_ui()
        # Persistable session state (geometry,
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
            apply_a4=self._apply_a4,
            keep_saved_instrument=keep_saved_instrument,
        )
        # TXT / PDF / CSV / PNG-chart export flows and
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
            get_maker=lambda: self._last_maker,
            get_model=lambda: self._last_model,
            get_nickname=lambda: self._nick_edit.text(),
        )
        # Intonation-table refresh + matrix paint
        # pipeline lives on TableController.  The QTimer that drives the
        # 300 ms refresh stays wired to MainWindow._refresh_table (just
        # below) so resizeEvent's deferred refresh path also works
        # unchanged.  The controller's configure_for_mode() allocates the
        # QTableWidgetItem instances once per layout change and they are
        # reused across refresh ticks.
        self._table_ctrl = TableController(
            self,
            table=self._table,
            cfg=self._cfg,
            engine=self._engine,
            t_func=self._t,
            get_instrument_key=lambda: self.instrument,
            get_stats=lambda: self.stats,
            get_display_mode=lambda: self.display,
            get_a4=lambda: float(self._a4_combo.currentData()),
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
        self._seed_expected_notes()
        if AUDIO_OK:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4,
                                label=self._nick_edit.text().strip())
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
            t_func=self._t,
            info_banner=self._info_banner,
            audio_banner=self._audio_banner,
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
        if not self._cfg.welcome_shown and not self._settings.unreadable:
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
        self._btn_range.setObjectName('gear')   # themed via QToolButton#gear
        self._btn_range.clicked.connect(self._open_range_editor)
        il.addWidget(self._btn_range)

        self._btn_custom = QPushButton(self._t('custom_label'))
        self._btn_custom.setToolTip(self._t('custom_dlg_title'))
        self._btn_custom.setMaximumWidth(110)
        self._btn_custom.clicked.connect(self._on_add_custom)
        self._btn_custom.setObjectName('chrome')   # themed via QPushButton#chrome
        il.addWidget(self._btn_custom)

        self._nick_edit = QLineEdit()
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))
        # v0.5.7.1: 160px capped at ~12 characters, which truncated real
        # horn names like "Selmer Reference 54 Tenor #2" before the user
        # could see what they'd typed. Bump to 320 max / 280 min.
        self._nick_edit.setMinimumWidth(280)
        self._nick_edit.setMaximumWidth(320)
        self._nick_edit.editingFinished.connect(self._on_nickname_changed)
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
        self._filter_combo.setMinimumWidth(100)
        self._populate_filter_combo(self._filter_combo)
        self._filter_combo.currentIndexChanged.connect(
            self._on_filter_mode_changed)
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
        self._btn_record   = self._make_btn(
            self._t('btn_stop'), '#b7770d', self._on_record_toggle)
        self._btn_reset    = self._make_btn(self._t('btn_reset'),    '#c0392b', self._on_reset)
        self._btn_txt      = self._make_btn(self._t('btn_txt'),      '#2980b9', self._export_txt)
        self._btn_pdf      = self._make_btn(self._t('btn_pdf'),      '#8e44ad', self._export_pdf)
        self._btn_csv      = self._make_btn(self._t('btn_csv'),      '#16a085', self._export_csv)
        self._btn_chart    = self._make_btn(self._t('btn_chart'),    '#d35400', self._export_chart)
        self._btn_import   = self._make_btn(self._t('btn_import'),   '#7f8c8d', self._import_csv)

        # AUDIO IN chip, placed between Language and Import.
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
        # Non-modal info banner for hot-plug and the
        # wrong-instrument hint. Hidden by default; the GUI calls
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
        self._status_lbl.setObjectName('dimLabel')   # themed via QLabel#dimLabel
        ll3.addWidget(self._status_lbl)

        # Drone controls live on the TUNER tab,
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
        self._table_lbl.setStyleSheet('font-size:14px;font-weight:bold;padding:2px 0 6px 0;')
        self._table_lbl.setWordWrap(True)
        rl.addWidget(self._table_lbl)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(self._table_headers())
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setObjectName('matrix')   # themed via QTableWidget#matrix
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
        self._min_n_lbl.setObjectName('dimLabel')      # themed via QLabel#dimLabel
        self._min_n_spin = QSpinBox()
        self._min_n_spin.setRange(0, 999)
        self._min_n_spin.setValue(int(getattr(self._cfg, 'min_n_visible', 5)))
        self._min_n_spin.setToolTip(self._t('min_n_tip'))
        self._min_n_spin.setMinimumWidth(70)
        self._min_n_spin.setObjectName('minNspin')     # themed via QSpinBox#minNspin
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
        self._tabs.addTab(splitter, self._t('nav_tab_tuner'))
        self._tabs.addTab(self._build_metro_tab(), self._t('nav_tab_metro'))
        self._tabs.addTab(self._build_deck_tab(), self._t('nav_tab_deck'))
        self._tabs.addTab(self._build_setup_tab(), self._t('nav_tab_setup'))
        for i, key in enumerate(self._tab_keys):
            self._tabs.setTabToolTip(i, self._t(f'nav_tab_{key}_tip'))
        self._install_tab_indicators()
        # Paint the deck transport now that the tab's recording dot exists.
        self._sync_deck_transport()
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

        self.setWindowTitle(f"{self._t('window_title')} {APP_VERSION}")

        # InstrumentController owns family/instr/nickname
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

    # ── Navigation shell helpers ─────────────────────────────────────────────
    def _build_setup_tab(self) -> QWidget:
        """SETUP tab: audio-OUTPUT controls (output-device picker, the
        duplex-preference toggle, and a test-tone button that proves the
        output path — a tone plays through the mixer while the tuner keeps
        reading the mic), plus drone voice, theme, instrument, nickname,
        concert pitch, response and mic-gain settings."""
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
        self._out_device_combo.setAccessibleName(self._t('setup_output_device'))
        self._out_device_combo.currentIndexChanged.connect(
            self._on_output_device_changed)
        # Handle for the active test-tone source (None while silent).
        self._test_tone_handle = None
        # Cache of the output enumeration from the last combo population, so
        # _on_output_device_changed can resolve a chosen name's host_api
        # without re-querying the engine (N3).
        self._output_devices_cache: list = []
        self._refresh_output_device_combo()
        self._lbl_out_device = QLabel(self._t('setup_output_device'))
        og.addRow(self._lbl_out_device, self._out_device_combo)
        self._cb_prefer_duplex = QCheckBox(self._t('setup_prefer_duplex'))
        self._cb_prefer_duplex.setToolTip(self._t('setup_prefer_duplex_tip'))
        self._cb_prefer_duplex.setChecked(
            bool(getattr(self._cfg, 'output_prefer_duplex', False)))
        self._cb_prefer_duplex.toggled.connect(self._on_prefer_duplex_toggled)
        og.addRow('', self._cb_prefer_duplex)
        outer.addWidget(out_grp)

        # Drone voice — the full 128-GM picker lives here; the
        # TUNER drone bar carries the 5-preset quick row. Both drive the same
        # voice and stay in sync via DroneController.on_state_changed.
        voice_grp = QGroupBox(self._t('setup_drone_voice_group'))
        vg = QFormLayout(voice_grp)
        vg.setContentsMargins(12, 10, 12, 10)
        self._drone_voice_combo = QComboBox()
        self._drone_voice_combo.setMinimumWidth(260)
        self._drone_voice_combo.setAccessibleName(self._t('setup_drone_voice'))
        # Presets first (labelled), then the full GM catalog. Each item's data
        # is the voice id the controller resolves.
        seen_ids = set()
        # Presets, then the desktop-only synth voices (pure sine), then full GM.
        for vid, label, _program in DRONE_PRESETS + DRONE_SYNTH_VOICES + DRONE_FULL_GM:
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            self._drone_voice_combo.addItem(label, vid)
        sel_id = self._cfg.drone_voice_id
        for i in range(self._drone_voice_combo.count()):
            if self._drone_voice_combo.itemData(i) == sel_id:
                self._drone_voice_combo.setCurrentIndex(i)
                break
        self._drone_voice_combo.currentIndexChanged.connect(
            self._on_drone_voice_changed)
        self._lbl_drone_voice = QLabel(self._t('setup_drone_voice'))
        vg.addRow(self._lbl_drone_voice, self._drone_voice_combo)
        outer.addWidget(voice_grp)

        # Appearance — theme switching (dark / night / light).
        appear_grp = QGroupBox(self._t('setup_appearance_group'))
        ag = QFormLayout(appear_grp)
        ag.setContentsMargins(12, 10, 12, 10)
        self._theme_combo = QComboBox()
        self._theme_combo.setAccessibleName(self._t('setup_theme'))
        for _tname in sax_theme.THEME_ORDER:
            self._theme_combo.addItem(self._t(f'theme_{_tname}'), _tname)
        _cur_theme = sax_theme.active_name()
        for i in range(self._theme_combo.count()):
            if self._theme_combo.itemData(i) == _cur_theme:
                self._theme_combo.setCurrentIndex(i)
                break
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._lbl_theme = QLabel(self._t('setup_theme'))
        ag.addRow(self._lbl_theme, self._theme_combo)
        outer.addWidget(appear_grp)

        # Instrument — range-editor link (also reachable via the
        # toolbar gear). No inline stylesheet: it rides the global themed
        # QPushButton QSS so it re-themes on a theme switch (avoids adding to
        # the inline-style tail, task #31).
        instr_grp = QGroupBox(self._t('setup_instrument_group'))
        ig = QVBoxLayout(instr_grp)
        ig.setContentsMargins(12, 10, 12, 10)
        # Horn nickname — promote the existing config field (cfg.last_nickname,
        # used by CSV per-nickname slicing) to a first-class SETUP editor.
        # Mirrors the toolbar _nick_edit; the toolbar field stays canonical (the
        # controller + session-state read it) and both sync via _apply_nickname.
        # No inline style — rides the themed QLineEdit QSS (off the #31 tail).
        nick_row = QHBoxLayout()
        self._setup_nick_lbl = QLabel(self._t('grp_nickname'))
        nick_row.addWidget(self._setup_nick_lbl)
        self._setup_nick_edit = QLineEdit()
        self._setup_nick_edit.setObjectName('setupNickname')
        self._setup_nick_edit.setPlaceholderText(self._t('nickname_tip'))
        self._setup_nick_edit.setAccessibleName(self._t('grp_nickname'))
        self._setup_nick_edit.setText(getattr(self._cfg, 'last_nickname', '') or '')
        self._setup_nick_edit.editingFinished.connect(
            self._on_setup_nickname_changed)
        nick_row.addWidget(self._setup_nick_edit)
        ig.addLayout(nick_row)
        self._btn_setup_range = QPushButton(self._t('setup_range_edit'))
        self._btn_setup_range.setObjectName('setupRangeEdit')
        self._btn_setup_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_setup_range.clicked.connect(self._open_range_editor)
        ig.addWidget(self._btn_setup_range)
        outer.addWidget(instr_grp)

        # Concert pitch (A4) — mirrors the toolbar Kammerton combo.
        # Both combos drive the engine A4 + the (heavy) measurement recalc via
        # the shared _apply_a4, mirrored under blockSignals so the re-derive
        # runs exactly once. Items are language-independent ("NNN Hz"), so no
        # retranslate. No inline style — themed QSS (off the #31 tail).
        a4_grp = QGroupBox(self._t('grp_a4'))
        a4g = QVBoxLayout(a4_grp)
        a4g.setContentsMargins(12, 10, 12, 10)
        self._setup_a4_combo = QComboBox()
        self._setup_a4_combo.setObjectName('setupA4')
        self._setup_a4_combo.setMinimumWidth(160)
        self._setup_a4_combo.setAccessibleName(self._t('grp_a4'))
        for hz in range(430, 451):
            self._setup_a4_combo.addItem(f'{hz} Hz', hz)
        self._setup_a4_combo.setCurrentIndex(self._a4_combo.currentIndex())
        self._setup_a4_combo.currentIndexChanged.connect(
            self._on_setup_a4_changed)
        a4g.addWidget(self._setup_a4_combo)
        outer.addWidget(a4_grp)

        # Response — pitch-detection smoothing preset (Fast / Normal / Slow).
        # Mirrors the toolbar Response combo: both drive
        # cfg.filter_mode + the engine through _apply_filter_mode and stay
        # mirrored (blockSignals) so changing one updates the other without
        # double-routing. No inline stylesheet — rides the themed QComboBox
        # QSS, keeping it off the inline-style tail (#31).
        resp_grp = QGroupBox(self._t('grp_filter'))
        rg = QVBoxLayout(resp_grp)
        rg.setContentsMargins(12, 10, 12, 10)
        self._setup_filter_combo = QComboBox()
        self._setup_filter_combo.setObjectName('setupResponse')
        self._setup_filter_combo.setMinimumWidth(220)
        self._setup_filter_combo.setAccessibleName(self._t('grp_filter'))
        self._populate_filter_combo(self._setup_filter_combo)
        self._setup_filter_combo.currentIndexChanged.connect(
            self._on_setup_filter_mode_changed)
        rg.addWidget(self._setup_filter_combo)
        outer.addWidget(resp_grp)

        # Mic gain — boost a quiet mic (in dB) so it clears the detection
        # floor; default 0 dB is a no-op. Drives engine.set_mic_gain, which
        # scales ONLY the silence gate + level meter (not the signal), so the
        # cents readout is unchanged. Persisted as cfg.mic_gain_db. dB labels
        # are language-independent → no retranslate. No inline style (#31 tail).
        mic_grp = QGroupBox(self._t('setup_mic_gain'))
        mg = QVBoxLayout(mic_grp)
        mg.setContentsMargins(12, 10, 12, 10)
        self._setup_mic_gain_combo = QComboBox()
        self._setup_mic_gain_combo.setObjectName('setupMicGain')
        self._setup_mic_gain_combo.setMinimumWidth(160)
        self._setup_mic_gain_combo.setAccessibleName(self._t('setup_mic_gain'))
        self._setup_mic_gain_combo.setToolTip(self._t('setup_mic_gain_tip'))
        for _db in MIC_GAIN_DB_STEPS:
            self._setup_mic_gain_combo.addItem(
                '0 dB' if _db == 0 else f'{_db:+d} dB', _db)
        # Snap the display to the nearest offered step (the engine already uses
        # the exact persisted dB; the user can only pick a discrete step).
        _cur_db = float(getattr(self._cfg, 'mic_gain_db', 0.0))
        _best_i = min(range(self._setup_mic_gain_combo.count()),
                      key=lambda i: abs(self._setup_mic_gain_combo.itemData(i)
                                        - _cur_db))
        self._setup_mic_gain_combo.setCurrentIndex(_best_i)
        self._setup_mic_gain_combo.currentIndexChanged.connect(
            self._on_mic_gain_changed)
        mg.addWidget(self._setup_mic_gain_combo)
        # Live plain-language confirmation of what the current gain does — so the
        # control gives a VISIBLE result even though the cents readout is, by
        # design, unaffected by gain. Updated on every change + on a
        # language switch.
        self._setup_mic_gain_status = QLabel('')
        self._setup_mic_gain_status.setObjectName('setupMicGainStatus')
        self._setup_mic_gain_status.setWordWrap(True)
        self._setup_mic_gain_status.setObjectName('dimLabel')   # themed dim hint
        self._update_mic_gain_status()
        mg.addWidget(self._setup_mic_gain_status)
        outer.addWidget(mic_grp)

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
        self._test_tone_status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._test_tone_status.setVisible(False)
        tg.addWidget(self._test_tone_status)
        outer.addWidget(tone_grp)
        outer.addStretch()
        # Register every translatable SETUP widget so _retranslate can re-label
        # the whole tab on a LIVE language switch — previously the SETUP titles
        # went stale until restart. Group boxes + form-row labels.
        self._setup_i18n_groups = [
            (out_grp, 'setup_output_group'),
            (voice_grp, 'setup_drone_voice_group'),
            (appear_grp, 'setup_appearance_group'),
            (instr_grp, 'setup_instrument_group'),
            (a4_grp, 'grp_a4'),
            (resp_grp, 'grp_filter'),
            (mic_grp, 'setup_mic_gain'),
            (tone_grp, 'setup_testtone_group'),
        ]
        self._setup_i18n_labels = [
            (self._lbl_out_device, 'setup_output_device'),
            (self._lbl_drone_voice, 'setup_drone_voice'),
            (self._lbl_theme, 'setup_theme'),
            (self._setup_nick_lbl, 'grp_nickname'),
        ]
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

    def _request_output_open(self) -> None:
        """Ask for the chosen output to be opened. The answer comes back as
        an OutputStatus (_on_output_status)."""
        self._engine.request_open_output(self._current_output_selection())

    def _refresh_output_device_combo(self) -> None:
        """Fill the SETUP output combo when it is built. Later changes to the
        device list arrive from the engine's hot-plug poll
        (_on_output_devices_changed). Only enumerates; opens nothing."""
        try:
            devices = query_output_devices()
        except Exception:
            devices = []
        self._fill_output_combo(devices)

    def _fill_output_combo(self, devices) -> None:
        """System default first, then the output devices, with the saved one
        selected when it is present."""
        combo = self._out_device_combo
        was_blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._t('setup_output_default'), '')
        self._output_devices_cache = list(devices)
        saved_name = str(getattr(self._cfg, 'output_device_name', '') or '')
        select_idx = 0
        for d in devices:
            combo.addItem(f'{d.name} · {d.host_api}', d.name)
            if saved_name and d.name == saved_name:
                select_idx = combo.count() - 1
        combo.setCurrentIndex(select_idx)
        combo.blockSignals(was_blocked)

    def _on_output_devices_changed(self, devices) -> None:
        """An output was plugged in or out: offer the current list."""
        self._fill_output_combo(devices)

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
        # Re-open on the new device. Whatever is sounding carries over: the
        # mixer re-rates every source for the new stream and keeps the
        # metronome's next beat where it was due.
        self._request_output_open()

    # ── Output state ──────────────────────────────────────────────────────────
    def _on_output_status(self, status) -> None:
        """The output stream changed state (engine signal, GUI thread)."""
        if status.state == OutputState.DISCONNECTED:
            self._park_output_users()
            self._output_notice = self._t('audio_output_disconnected',
                                          name=status.device or '?')
            self._info_banner.show_message(self._output_notice)
        elif status.state == OutputState.RUNNING and self._output_notice:
            self._info_banner.hide_if_showing(self._output_notice)
            self._output_notice = ''
        self._output_users.on_status(status)

    def _park_output_users(self) -> None:
        """The output was unplugged: switch off everything that was sounding,
        so no toggle claims to play into nothing, and park it to come back on
        when the output is back."""
        users = self._output_users
        if self._test_tone_handle is not None:
            self._stop_test_tone()
            self._show_test_tone_on(False)
            users.park('tone', self._test_tone_on, self._test_tone_failed)
        if self._metro_ctrl.is_running():
            self._metro_ctrl.stop()
            users.park('metro', self._metro_ctrl.start, self._metro_revert_start)
        if self._drone_ctrl.is_enabled():
            self._drone_ctrl.set_enabled(False)
            users.park('drone', self._drone_on, self._drone_revert_toggle)
        if self._pipes_ctrl is not None:
            for midi in sorted(self._pipes_ctrl.active_midis()):
                users.park(f'pipe:{midi}', partial(self._pipe_on, midi),
                           self._pipes_failed)
            self._pipes_ctrl.release_all()
        if self._deck_state() == 'playing':
            self._deck_ctrl.stop()

    def _output_reason(self, status) -> str:
        """Why an output open failed, naming the device (and the rates it
        refused), in the player's language. '' when there is nothing to add."""
        if status is None:
            return ''
        name = status.device
        err = status.error
        if err == AudioEngineError.UNSUPPORTED_RATE and name and status.rates:
            return self._t('output_err_rate', name=name,
                           rates=format_rates(status.rates))
        if err == AudioEngineError.DEVICE_BUSY and name:
            return self._t('output_err_busy', name=name)
        if err in (AudioEngineError.NO_DEVICE,
                   AudioEngineError.DEVICE_DISCONNECTED):
            if name:
                return self._t('output_err_missing', name=name)
            return self._t('output_err_none')
        if name:
            return self._t('output_err_failed', name=name)
        return ''

    def _with_reason(self, text: str, status) -> str:
        reason = self._output_reason(status)
        return f'{text} {reason}' if reason else text

    def _on_prefer_duplex_toggled(self, checked: bool) -> None:
        # Persist the preference; it's consumed the next time the output
        # device opens. We don't re-open mid-tone — that would chop a
        # playing test tone for a setting that costs nothing to defer.
        self._cfg.output_prefer_duplex = bool(checked)
        try:
            sax_config.save_config(self._cfg)
        except Exception:
            pass

    def _on_theme_changed(self, _idx: int) -> None:
        name = self._theme_combo.currentData() or sax_theme.DEFAULT_THEME
        self._apply_theme(str(name))

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
        try:
            sax_config.save_config(self._cfg)
        except Exception:
            pass

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

    def _on_test_tone_toggled(self, on: bool) -> None:
        """Start/stop a 440 Hz test tone through the mixer. The tuner is
        deliberately NOT muted — D3 says the readout stays live while output
        sounds; this is the manual proof of that path."""
        if on:
            self._test_tone_status.setVisible(False)
            self._output_users.want('tone', self._test_tone_on,
                                    self._test_tone_failed)
        else:
            self._output_users.drop('tone')
            self._test_tone_status.setVisible(False)
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
        self._test_tone_status.setText(
            self._with_reason(self._t('setup_testtone_failed'), status))
        self._test_tone_status.setVisible(True)

    def _stop_test_tone(self) -> None:
        if self._test_tone_handle is not None:
            try:
                self._engine.stop_test_tone(self._test_tone_handle)
            except Exception:
                pass
        self._test_tone_handle = None

    def _show_test_tone_on(self, on: bool) -> None:
        """Paint the button without re-entering its toggled handler."""
        blocked = self._btn_test_tone.blockSignals(True)
        self._btn_test_tone.setChecked(on)
        self._btn_test_tone.blockSignals(blocked)
        self._btn_test_tone.setText(
            self._t('setup_testtone_stop' if on else 'setup_testtone_play'))

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
        """Public hook for the DeckController: show a pulsing red dot on the
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
        key = str(getattr(self._cfg, 'last_active_tab', 'tuner') or 'tuner')
        if key in self._tab_keys:
            self._tabs.setCurrentIndex(self._tab_keys.index(key))

    # ── METRO tab ─────────────────────────────────────────────────────────────
    # These mirror Android useMetronome (BPM_MIN/MAX, the four presets).
    # sax_metronome.py holds the authoritative constants; keep these in step.
    METRO_TIME_SIGS = ('2/4', '3/4', '4/4', '6/8')
    METRO_BPM_MIN = 30
    METRO_BPM_MAX = 300

    def _build_metro_tab(self) -> QWidget:
        """The metronome panel. Audio is driven by MetronomeController
        (sax_metronome.py). Start surfaces an inline 'not available' status
        when no output stream can open, rather than claiming to play — the
        same honesty discipline as the test-tone button. The GUI-facing
        controller API is: bpm/time_sig/volume/running,
        set_bpm/nudge_bpm/set_time_sig/set_volume, register_tap()->Optional[int],
        start/stop/toggle/is_running, on_state_changed."""
        # Local UI state, seeded from persisted config (the metronome
        # fields, read defensively).
        self._metro_bpm = self._clamp_bpm(int(getattr(self._cfg, 'last_bpm', 100)))
        self._metro_time_sig = str(getattr(self._cfg, 'last_time_sig', '4/4'))
        if self._metro_time_sig not in self.METRO_TIME_SIGS:
            self._metro_time_sig = '4/4'
        self._metro_running = False
        # The metronome controller (sax_metronome.py). on_state_changed is the
        # zero-arg callback that re-syncs this panel from controller state
        # (the controller is the single source of truth for run/bpm/sig).
        self._metro_ctrl = MetronomeController(
            self._engine.mixer, self._output_rate(),
            bpm=self._metro_bpm,
            time_sig=self._metro_time_sig,
            volume=float(getattr(self._cfg, 'click_volume', 1.0)),
            on_state_changed=self._on_metro_state_changed)

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Big BPM readout.
        self._metro_bpm_lbl = QLabel(str(self._metro_bpm))
        self._metro_bpm_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._metro_bpm_lbl.setObjectName('bpmReadout')   # themed accent number
        outer.addWidget(self._metro_bpm_lbl)
        unit = QLabel(self._t('metro_bpm'))
        unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit.setObjectName('bpmUnit')
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
            b.setObjectName('stepperLg')   # themed via QToolButton#stepperLg
        self._metro_slider = QSlider(Qt.Orientation.Horizontal)
        self._metro_slider.setAccessibleName(self._t('metro_bpm'))
        self._metro_slider.setRange(self.METRO_BPM_MIN, self.METRO_BPM_MAX)
        self._metro_slider.setValue(self._metro_bpm)
        self._metro_slider.valueChanged.connect(self._on_metro_slider)
        self._btn_metro_tap = QPushButton(self._t('metro_tap'))
        self._btn_metro_tap.setToolTip(self._t('metro_tap_tip'))
        self._btn_metro_tap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_metro_tap.clicked.connect(self._on_metro_tap)
        self._btn_metro_tap.setObjectName('chrome')   # themed via QPushButton#chrome
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
            b.setObjectName('chip')   # themed via QPushButton#chip
            self._metro_ts_btns[ts] = b
            ts_l.addWidget(b)
        outer.addWidget(ts_grp)

        # Click volume.
        vol_grp = QGroupBox(self._t('metro_volume'))
        vol_l = QHBoxLayout(vol_grp)
        vol_l.setContentsMargins(12, 8, 12, 8)
        self._metro_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._metro_vol_slider.setAccessibleName(self._t('metro_volume'))
        self._metro_vol_slider.setRange(0, 100)
        vol0 = int(round(float(getattr(self._cfg, 'click_volume', 1.0)) * 100))
        self._metro_vol_slider.setValue(max(0, min(100, vol0)))
        self._metro_vol_slider.valueChanged.connect(self._on_metro_volume)
        self._metro_vol_lbl = QLabel(f'{self._metro_vol_slider.value()}%')
        self._metro_vol_lbl.setObjectName('dimLabel')
        self._metro_vol_lbl.setMinimumWidth(44)
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
        self._metro_status.setObjectName('errorLabel')   # themed via QLabel#errorLabel
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
        self._metro_ctrl.set_time_sig(ts)

    def _on_metro_volume(self, value: int) -> None:
        self._metro_vol_lbl.setText(f'{value}%')
        vol = value / 100.0
        self._cfg.click_volume = vol
        self._metro_ctrl.set_volume(vol)

    def _on_metro_tap(self) -> None:
        self._metro_status.setVisible(False)
        # The controller updates its own bpm and fires on_state_changed, which
        # re-syncs the readout here — no manual set needed.
        self._metro_ctrl.register_tap()

    def _metro_revert_start(self, status=None) -> None:
        """Un-check the Start button (without re-firing the toggle) and show
        the unavailable status — used whenever the metronome can't actually
        sound, so the button never claims to be running."""
        blocked = self._btn_metro_start.blockSignals(True)
        self._btn_metro_start.setChecked(False)
        self._btn_metro_start.blockSignals(blocked)
        self._btn_metro_start.setText(self._t('metro_start'))
        self._metro_status.setText(
            self._with_reason(self._t('metro_unavailable'), status))
        self._metro_status.setVisible(True)

    def _on_metro_toggle(self, on: bool) -> None:
        if not on:
            self._output_users.drop('metro')
            self._metro_ctrl.stop()   # fires on_state_changed
            return
        self._metro_status.setVisible(False)
        # The metronome sounds through the output mixer: it starts once an
        # output runs (so 'running' is never shown with nothing to hear), or
        # reverts with the reason if none can open. Starting it on an output
        # that already runs leaves the stream (and anything sounding) alone.
        self._output_users.want('metro', self._metro_ctrl.start,
                                self._metro_revert_start)

    def _on_metro_state_changed(self) -> None:
        """Zero-arg slot for MetronomeController.on_state_changed (GUI thread),
        fired on every bpm/time-sig/volume/running change. The controller is
        the single source of truth: pull state from it and sync the panel so
        the run dot, Start/Stop label, BPM readout, and time-sig can't drift."""
        ctrl = self._metro_ctrl
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

    # ── Tape deck ─────────────────────────────────────────────────────────────
    # As in the Android useDeck hook: record the MIC to a single take, play it back
    # through the mixer, export it to a WAV. DeckController (sax_deck.py) owns the
    # state machine + the engine input-recording tap + WAV io; this panel is its
    # transport. Honesty discipline:
    # the pulsing-red dot + a 'recording' state appear ONLY when the controller
    # actually armed the mic input — never on a bare button press.
    #
    # The one wire beyond the drone/metro pattern: deck.pump() runs on a GUI-side
    # timer. Playback-end and cap-hit transitions are detected there (the audio
    # thread only flips a lock-free .finished flag); pump() performs the
    # transition and fires on_state_changed ON THE GUI THREAD, so buttons are
    # always relabelled on-thread.
    DECK_STATES = ('idle', 'recording', 'have_take', 'playing')

    def _build_deck_tab(self) -> QWidget:
        """The DECK tab: Record / Stop / Play / Export transport with
        state-driven enable + labels, an inline status line, and the pulsing-red
        recording dot (set_deck_recording). Wired to DeckController."""
        self._deck_played_once = False
        self._deck_last_state = None
        self._deck_last_can_record = None
        # The deck controller, with max_seconds from the clamped config field.
        self._deck_ctrl = DeckController(
            self._engine.mixer, self._output_rate(),
            engine=self._engine,           # arms the input-recording tap
            max_seconds=float(getattr(self._cfg, 'deck_max_seconds',
                                      300.0) or 300.0),
            scratch_dir=(str(getattr(self._cfg, 'deck_scratch_dir', '')
                             or '') or None),
            on_state_changed=self._on_deck_state_changed)

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel(self._t('deck_group'))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size:18px;font-weight:bold;letter-spacing:3px;')
        outer.addWidget(title)

        # Big state readout: empty / recording / take ready / playing.
        self._deck_state_lbl = QLabel('')
        self._deck_state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._deck_state_lbl.setObjectName('deckState')   # themed accent readout
        outer.addWidget(self._deck_state_lbl)

        # Transport: Record (toggle, green→red) · Stop · Play/Replay · Export.
        row = QHBoxLayout()
        self._btn_deck_record = QPushButton(self._t('deck_record'))
        self._btn_deck_record.setCheckable(True)
        self._btn_deck_record.setToolTip(self._t('deck_record_tip'))
        self._btn_deck_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_deck_record.toggled.connect(self._on_deck_record_toggle)
        self._btn_deck_record.setObjectName('transportRec')   # green/red + themed disabled
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
            b.setObjectName('transport')   # themed via QPushButton#transport
        row.addWidget(self._btn_deck_record)
        row.addWidget(self._btn_deck_stop)
        row.addWidget(self._btn_deck_play)
        row.addWidget(self._btn_deck_export)
        outer.addLayout(row)

        # Records the mic only, a single take (as in the Android app).
        hint = QLabel(self._t('deck_hint'))
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setObjectName('dimLabel')
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
        self._deck_pump_timer.start()

        return w

    def _deck_state(self) -> str:
        """Current deck state normalized into DECK_STATES. Tolerates a str or an
        enum, and hyphen OR underscore ('have-take'/'have_take'), so the GUI
        can't drift from sax_deck's exact spelling (str-enum, value 'have-take')."""
        s = self._deck_ctrl.state.value.replace('-', '_')
        return s if s in self.DECK_STATES else 'idle'

    def _deck_can_record(self) -> bool:
        """Honest pre-click probe: can a take be armed right now? Backed by the
        controller's can_record() (engine.input_running). Optimistic only when
        the controller lacks the probe — start_record()'s False-return is the
        authoritative post-click check either way."""
        try:
            return bool(self._deck_ctrl.can_record())
        except Exception:
            return False

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
        # by a bare button press (the honesty seam locked by gui_smoke).
        self.set_deck_recording(recording)

        key = {'idle': 'deck_status_empty', 'recording': 'deck_status_recording',
               'have_take': 'deck_status_have_take',
               'playing': 'deck_status_playing'}[state]
        self._deck_state_lbl.setText(self._t(key))

        self._deck_last_state = state
        self._deck_last_can_record = can_record

    def _deck_show_status(self, key: str, *, error: bool = True,
                          output_status=None) -> None:
        """Inline transport status. Red for can't/failed, green for confirmation
        (export done) — never leave a stale message of the wrong colour.
        ``output_status`` adds why the output could not open."""
        self._deck_status.setStyleSheet(
            'color:%s;font-size:12px;padding-top:4px;'
            % ('#c0392b' if error else '#2ecc71'))
        self._deck_status.setText(self._with_reason(self._t(key), output_status))
        self._deck_status.setVisible(True)

    def _on_deck_state_changed(self) -> None:
        """Zero-arg slot for DeckController.on_state_changed — always fires on
        the GUI thread (the controller marshals playback-end/cap-hit through pump()).
        Clears stale status and repaints the transport from the new state."""
        self._deck_status.setVisible(False)
        self._sync_deck_transport()

    def _deck_pump(self) -> None:
        """GUI-thread tick: let the controller detect playback-end / cap-hit
        (it fires on_state_changed when a transition happens), then re-sync if
        the state or the can_record() probe changed (e.g. the mic opened)."""
        try:
            self._deck_ctrl.pump()   # may fire on_state_changed → _sync
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
        try:
            self._deck_ctrl.stop()           # fires on_state_changed
        except Exception:
            self._sync_deck_transport()

    def _on_deck_play(self) -> None:
        self._deck_status.setVisible(False)
        # Playback sounds through the output mixer: it starts once an output
        # runs (never 'playing' with nothing to hear). Mirrors drone/metro.
        self._output_users.want('deck', self._deck_play_now,
                                self._deck_play_failed)

    def _deck_play_now(self) -> None:
        ok = False
        try:
            ok = bool(self._deck_ctrl.play())   # fires on_state_changed
        except Exception:
            ok = False
        if ok:
            self._deck_played_once = True
        else:
            self._deck_play_failed(None)

    def _deck_play_failed(self, status) -> None:
        self._deck_show_status('deck_play_unavailable', output_status=status)
        self._sync_deck_transport()

    def _on_deck_export(self) -> None:
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

    # ── Drone + pitch pipes ───────────────────────────────────────────────────
    @staticmethod
    def _fmt_semitone(n: int) -> str:
        return f'{n:+d}' if n else '0'

    def _build_drone_bar(self) -> QWidget:
        """The drone strip on the TUNER tab: on/off + 5-preset row + volume +
        semitone steppers + a pitch-pipes launcher. Wired to DroneController
        (sax_drone.py). Turning the
        drone on requires an output stream (honesty discipline — no 'on' state
        if nothing can sound). The 128-GM picker lives in SETUP and stays in
        sync via the controller's on_state_changed (single source of truth)."""
        # Seed persisted state. load_config has already coerced and clamped
        # these; a saved volume of 0.0 is a real choice and is kept.
        self._drone_voice_id = self._cfg.drone_voice_id
        self._drone_volume = self._cfg.drone_volume
        self._drone_semitones = self._cfg.drone_semitones
        # The pitch-pipes dialog, its pads and its controller are built on
        # first open.
        self._pipes_ctrl = None
        self._pipes_dlg = None
        self._pipe_btns: dict = {}
        self._drone_ctrl = DroneController(
            self._engine.mixer, self._output_rate(),
            voice_id=self._drone_voice_id,
            volume=self._drone_volume,
            semitones=self._drone_semitones,
            a4=float(self._engine.a4),
            enabled=False,
            engine=self._engine,   # enables D3 duck-attach on enable
            on_state_changed=self._on_drone_state_changed)

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
        for vid, label, _program in DRONE_PRESETS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(vid == self._drone_voice_id)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, i=vid: self._on_drone_preset(i))
            b.setObjectName('chip')   # themed via QPushButton#chip
            self._drone_preset_btns[vid] = b
            row1.addWidget(b)
        row1.addStretch()
        g.addLayout(row1)

        # Row 2: volume · semitone steppers · pitch-pipes launcher.
        row2 = QHBoxLayout()
        vlbl = QLabel(self._t('drone_volume'))
        vlbl.setObjectName('dimLabel')
        row2.addWidget(vlbl)
        self._drone_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._drone_vol_slider.setAccessibleName(self._t('drone_volume'))
        self._drone_vol_slider.setRange(0, 100)
        self._drone_vol_slider.setValue(max(0, min(100,
            int(round(self._drone_volume * 100)))))
        self._drone_vol_slider.valueChanged.connect(self._on_drone_volume)
        row2.addWidget(self._drone_vol_slider, 1)
        row2.addSpacing(10)
        slbl = QLabel(self._t('drone_semitone'))
        slbl.setObjectName('dimLabel')
        row2.addWidget(slbl)
        btn_sd = QToolButton()
        btn_sd.setText('−')
        btn_sd.setToolTip(self._t('drone_semitone_down_tip'))
        btn_sd.clicked.connect(lambda: self._drone_semitone_nudge(-1))
        self._drone_semi_lbl = QLabel(self._fmt_semitone(self._drone_semitones))
        self._drone_semi_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # drop the colour → inherits the themed QWidget text colour; keep size.
        self._drone_semi_lbl.setStyleSheet('font-size:13px;min-width:28px;')
        btn_su = QToolButton()
        btn_su.setText('+')
        btn_su.setToolTip(self._t('drone_semitone_up_tip'))
        btn_su.clicked.connect(lambda: self._drone_semitone_nudge(1))
        for b in (btn_sd, btn_su):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName('stepper')   # themed via QToolButton#stepper
        row2.addWidget(btn_sd)
        row2.addWidget(self._drone_semi_lbl)
        row2.addWidget(btn_su)
        row2.addSpacing(12)
        self._btn_pipes = QPushButton(self._t('pipes_launch'))
        self._btn_pipes.setToolTip(self._t('pipes_tip'))
        self._btn_pipes.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pipes.clicked.connect(self._open_pitch_pipes)
        self._btn_pipes.setObjectName('chrome')   # themed via QPushButton#chrome
        row2.addWidget(self._btn_pipes)
        g.addLayout(row2)

        # Inline status — shown when an audio action can't run yet.
        self._drone_status = QLabel('')
        self._drone_status.setWordWrap(True)
        self._drone_status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._drone_status.setVisible(False)
        g.addWidget(self._drone_status)
        return grp

    def _on_drone_toggle(self, on: bool) -> None:
        if not on:
            self._output_users.drop('drone')
            self._drone_ctrl.set_enabled(False)
            return
        self._drone_status.setVisible(False)
        self._output_users.want('drone', self._drone_on,
                                self._drone_revert_toggle)

    def _drone_on(self) -> None:
        """The output runs: sound the drone (built at the output's format)."""
        self._drone_ctrl.set_enabled(True)   # fires on_state_changed
        self._report_drone_fallback()

    def _report_drone_fallback(self) -> None:
        """If the SoundFont synth could not load, the drone switched to the
        pure-sine voice; say so and point the voice picker at it."""
        ctrl = self._drone_ctrl
        if not ctrl.synth_fallback:
            return
        combo = self._drone_voice_combo
        idx = combo.findData(ctrl.voice_id)
        if idx >= 0:
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(blocked)
        self._drone_status.setText(self._t('drone_sine_fallback'))
        self._drone_status.setVisible(True)

    def _drone_revert_toggle(self, status=None) -> None:
        blocked = self._btn_drone_on.blockSignals(True)
        self._btn_drone_on.setChecked(False)
        self._btn_drone_on.blockSignals(blocked)
        self._drone_status.setText(
            self._with_reason(self._t('drone_unavailable'), status))
        self._drone_status.setVisible(True)

    def _on_drone_preset(self, voice_id: str) -> None:
        self._drone_voice_id = voice_id
        self._cfg.drone_voice_id = voice_id
        self._drone_ctrl.set_voice(voice_id)   # fires on_state_changed

    def _on_drone_volume(self, value: int) -> None:
        self._drone_volume = value / 100.0
        self._cfg.drone_volume = self._drone_volume
        self._drone_ctrl.set_volume(self._drone_volume)

    def _drone_semitone_nudge(self, delta: int) -> None:
        n = max(-12, min(12, self._drone_semitones + delta))
        self._drone_semitones = n
        self._drone_semi_lbl.setText(self._fmt_semitone(n))
        self._cfg.drone_semitones = n
        self._drone_ctrl.set_semitones(n)

    def _refresh_drone_voice_widgets(self) -> None:
        """Sync the TUNER preset row + the SETUP voice combo to the current
        voice id (both drive the same drone voice)."""
        for vid, b in self._drone_preset_btns.items():
            blocked = b.blockSignals(True)
            b.setChecked(vid == self._drone_voice_id)
            b.blockSignals(blocked)
        combo = self._drone_voice_combo
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
        enabled = ctrl.is_enabled()
        blocked = self._btn_drone_on.blockSignals(True)
        self._btn_drone_on.setChecked(enabled)
        self._btn_drone_on.blockSignals(blocked)
        vid = str(ctrl.voice_id)
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
        # Built on first open; until then there is nothing to retune.
        if self._pipes_ctrl is None:
            self._pipes_ctrl = PitchPipesController(
                self._engine.mixer, self._output_rate(),
                a4=float(self._engine.a4),
                on_state_changed=self._on_pipes_state_changed)

        from PyQt6.QtWidgets import QGridLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(self._t('pipes_title'))
        # bg inherits the global QWidget/window_bg rule (themed); pads + tip +
        # status carry their own themed objectName rules.
        v = QVBoxLayout(dlg)
        tip = QLabel(self._t('pipes_tip'))
        tip.setWordWrap(True)
        tip.setObjectName('dimLabel')   # themed via QLabel#dimLabel
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
            b.setObjectName('padChip')   # themed via QPushButton#padChip
            self._pipe_btns[midi] = b
            grid.addWidget(b, i // 4, i % 4)
        v.addLayout(grid)
        self._pipes_status = QLabel('')
        self._pipes_status.setObjectName('errorLabel')   # QLabel#errorLabel
        self._pipes_status.setVisible(False)
        v.addWidget(self._pipes_status)
        # Stop any sustaining pipe when the modal closes.
        dlg.finished.connect(lambda _r: self._stop_pitch_pipes())
        self._pipes_dlg = dlg
        self._on_pipes_state_changed()         # reflect any active pad
        dlg.show()

    def _on_pipe_tapped(self, midi: int) -> None:
        key = f'pipe:{midi}'
        if self._pipes_ctrl is not None and midi in self._pipes_ctrl.active_midis():
            self._pipes_ctrl.toggle(midi)      # release: needs no output
            return
        self._pipes_status.setVisible(False)
        if self._output_users.waiting(key):
            # Tapped again before the output came: the player changed their mind.
            self._output_users.drop(key)
        else:
            self._output_users.want(key, partial(self._pipe_on, midi),
                                    self._pipes_failed)
        self._sync_pipe_btns()

    def _pipe_on(self, midi: int) -> None:
        """The output runs: sound the pad (it fires on_pipes_state_changed)."""
        if midi not in self._pipes_ctrl.active_midis():
            self._pipes_ctrl.toggle(midi)

    def _pipes_failed(self, status) -> None:
        self._pipes_status.setText(
            self._with_reason(self._t('pipes_unavailable'), status))
        self._pipes_status.setVisible(True)
        self._sync_pipe_btns()

    def _stop_pitch_pipes(self) -> None:
        self._output_users.drop_prefix('pipe:')
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
        for midi, b in self._pipe_btns.items():
            blocked = b.blockSignals(True)
            b.setChecked(midi in active)
            b.blockSignals(blocked)

    def _on_pipes_state_changed(self) -> None:
        """Zero-arg slot for PitchPipesController.on_state_changed — highlight
        the sustaining pad (if the modal is open)."""
        if self._pipe_btns:
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
        self.setWindowTitle(f"{self._t('window_title')} {APP_VERSION}")
        for i, key in enumerate(self._tab_keys):
            self._tabs.setTabText(i, self._t(f'nav_tab_{key}'))
            self._tabs.setTabToolTip(i, self._t(f'nav_tab_{key}_tip'))
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

        # Response combo (toolbar) — re-label Fast/Normal/Slow. The SETUP
        # response combo is built once in the active language alongside its
        # SETUP neighbours (drone-voice / theme), which likewise do not
        # live-retranslate; both combos carry the same data values, so
        # _apply_filter_mode keeps them in functional sync regardless.
        self._populate_filter_combo(self._filter_combo)
        self._min_n_lbl.setText(self._t('min_n_label'))
        self._min_n_spin.setToolTip(self._t('min_n_tip'))
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
        self._cb_diag.setText(self._t('show_diagnostics'))
        self._cb_diag.setToolTip(self._t('show_diagnostics_tip'))
        self._spectro_grp.setTitle(self._t('spectro_title'))
        self._data_grp.setTitle(self._t('data_panel_title'))
        self._data_panel.retranslate(self._t)
        self._audio_chip.retranslate(self._t)
        self._audio_banner.retranslate(self._t)
        self._info_banner.retranslate(self._t)

        # SETUP tab is built once, not rebuilt — re-label its titles/labels here
        # so a live language switch doesn't leave it half-translated.
        self._retranslate_setup()

        self._refresh_table()

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
        if not self._recording:
            return
        # Wrong-instrument detector. Run BEFORE the
        # OOR drop below so the counter still ticks when the user has
        # "filter to range" enabled — otherwise we'd never count notes
        # outside the range. Compute the fingered MIDI relative to the
        # currently selected instrument and bump/reset the counter.
        try:
            _t_oor = transp_map().get(self.instrument, 0)
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
                    and self._oor_count >= self._oor_threshold):
                self._oor_banner_shown = True
                instr_label = self._instr_label(self.instrument)
                self._info_banner.show_message(
                    self._t('wrong_instrument_banner_body',
                            instrument=instr_label))

        # Drop out-of-range notes when the toggle is off — keeps the table
        # bounded to the instrument's nominal range.
        if not self._cfg.allow_out_of_range:
            transp = transp_map().get(self.instrument, 0)
            lo_f, hi_f = sax_instruments.fingered_range(self.instrument)
            midi_fingered = midi_kl - transp
            if not (lo_f <= midi_fingered <= hi_f):
                return
        if midi_kl not in self.stats:
            self.stats[midi_kl] = NoteStats()
        self.stats[midi_kl].add(cents, freq)
        # Diagnostics counter: every emitted note, including ones the table
        # later discards.
        self._notes_count += 1
        # Per-measurement log. Instrument/A4 are read off the active run
        # inside the log, not from `self`, so a callback firing during a UI
        # change still attributes to the run that was active when it fired.
        midi_gr = midi_kl - transp_map().get(self.instrument, 0)
        self._log.add_measurement(midi_sounding=midi_kl,
                                   midi_fingered=midi_gr,
                                   cents=cents, freq_hz=freq)

        transp     = transp_map().get(self.instrument, 0)
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
        TableController for the actual paint; the per-cell
        paint pipeline and cell cache live on the controller."""
        if self._table_ctrl is None:      # still inside _build_ui
            return
        desired = self._desired_layout_mode()
        if desired != self._layout_mode or self._table_ctrl._current_mode is None:
            self._layout_mode = desired
            self._table_ctrl.configure_for_mode(desired)
        self._table_ctrl.refresh()

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
        QTimer.singleShot(0, self._refresh_table)

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
        applied a new A4. The autotune body already cleared self.stats;
        here we refresh the table so the new baseline
        paints. The drone and pitch pipes were retuned by _instr_set_a4."""
        a4 = float(self._engine.a4)
        self._seed_expected_notes()
        # New run at the new A4, as _apply_a4 does, so later measurements are
        # logged against the reference they were measured with.
        if AUDIO_OK and self._recording and a4 > 0:
            self._log.start_run(instrument=self.instrument, a4_hz=a4,
                                label=self._nick_edit.text().strip())
        self._refresh_table()

    def _instr_set_a4(self, hz: float) -> None:
        """Setter callback used by InstrumentController's autotune flow.
        Mirrors the v0.5.x in-line sequence: block the combo's signal so
        _on_a4_changed does NOT fire (the autotune flow does its own
        stats reset + refresh via on_a4_changed), set the index, then push
        the new A4 to the engine, drone and pitch pipes."""
        hz_int = int(hz)
        _target = hz_int - 430
        for _combo in (self._a4_combo, self._setup_a4_combo):
            if not (0 <= _target < _combo.count()):
                continue
            _combo.blockSignals(True)
            _combo.setCurrentIndex(_target)
            _combo.blockSignals(False)
        self._push_a4(float(hz_int))

    def _push_a4(self, a4: float) -> None:
        """Hand a concert-pitch value to everything that measures or sounds
        against it: the detection engine, the drone and the pitch pipes.
        Every A4 change (toolbar, SETUP, autotune, start-up restore) ends
        here, so the three can never disagree about the reference."""
        self._engine.set_a4(a4)
        self._drone_ctrl.set_a4(a4)
        if self._pipes_ctrl is not None:      # built on first open
            self._pipes_ctrl.set_a4(a4)

    def _seed_expected_notes(self) -> None:
        """Populate self.stats with empty NoteStats for the current
        instrument's expected fingered range. Existing measurements are
        preserved."""
        transp = transp_map().get(self.instrument, 0)
        lo, hi = sax_instruments.fingered_range(self.instrument)
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
        sr = int(self._engine.samplerate or 0)
        return sr if sr > 0 else int(DEFAULT_SAMPLE_RATE)

    def _output_rate(self) -> int:
        """Rate the output controllers start from. The mixer owns the output
        format and moves their sources along when it changes."""
        return int(self._engine.mixer.samplerate)

    def _device_selection_from_cfg(self) -> DeviceSelection:
        return DeviceSelection(
            name=str(getattr(self._cfg, 'audio_device_name', '') or ''),
            host_api=str(getattr(self._cfg, 'audio_device_host_api', '') or ''),
            samplerate=int(getattr(self._cfg, 'audio_device_samplerate', 0) or 0),
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

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        """Show or hide the spectrogram + diagnostics panels and persist
        the choice. Widgets remain constructed either way so flipping
        the toggle has no allocation cost on the audio thread."""
        self._cfg.show_diagnostics = bool(checked)
        sax_config.save_config(self._cfg)
        self._spectro_grp.setVisible(bool(checked))
        self._data_grp.setVisible(bool(checked))
        self._refresh_table()

    def _on_layout_pref_changed(self, _idx: int) -> None:
        """User picked Auto / List / Grid. Persist and refresh the table."""
        pref = self._layout_combo.currentData()
        if pref in ('auto', 'single', 'matrix'):
            self._cfg.layout_mode_preference = pref
            sax_config.save_config(self._cfg)
            self._refresh_table()

    def _populate_filter_combo(self, combo) -> None:
        """(Re)fill a Response combo with localized Fast / Normal / Slow,
        preserving the current selection (seeding from cfg.filter_mode on
        first build, when the combo is still empty). Signals stay blocked
        throughout, so it is safe during construction and live
        re-translation alike. Shared by the toolbar Response combo and the
        SETUP-tab response combo so the two never drift in contents."""
        cur = combo.currentData()
        if cur not in FILTER_PRESETS:
            # Seed from the engine: it ignores an unknown mode, so it holds
            # the mode actually in use.
            cur = self._engine.filter_mode
        if cur not in FILTER_PRESETS:
            cur = FILTER_MODE_DEFAULT
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._t('filter_fast'),   'fast')
        combo.addItem(self._t('filter_normal'), 'normal')
        combo.addItem(self._t('filter_slow'),   'slow')
        idx = combo.findData(cur)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.setToolTip(self._t('filter_tip'))
        combo.blockSignals(False)

    def _on_filter_mode_changed(self, _idx: int = 0) -> None:
        """Toolbar Response combo changed → apply + mirror into SETUP."""
        self._apply_filter_mode(self._filter_combo.currentData(),
                                source=self._filter_combo)

    def _on_setup_filter_mode_changed(self, _idx: int = 0) -> None:
        """SETUP Response combo changed → apply + mirror into the toolbar.
        Mirrors the toolbar Response control (Fast / Normal / Slow)."""
        self._apply_filter_mode(self._setup_filter_combo.currentData(),
                                source=self._setup_filter_combo)

    def _apply_filter_mode(self, mode, source=None) -> None:
        """Single source of truth for a Response/filter-mode change from
        EITHER the toolbar or the SETUP combo. Mirror the *other* combo
        (signals blocked so the change does not re-enter), persist
        cfg.filter_mode, and reroute the live engine. No-op on an unknown
        preset. Idempotent: re-applying the active mode changes nothing."""
        if mode not in FILTER_PRESETS:
            return
        for combo in (self._filter_combo, self._setup_filter_combo):
            if combo is source:
                continue
            idx = combo.findData(mode)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        self._cfg.filter_mode = mode
        sax_config.save_config(self._cfg)
        if AUDIO_OK:
            self._engine.set_filter_mode(mode)

    def _on_mic_gain_changed(self, _idx: int = 0) -> None:
        """SETUP mic-gain combo changed. Persist the dB and push the linear
        gain to the engine (which scales only the silence gate + level meter,
        never the signal). 0 dB = no-op."""
        db = self._setup_mic_gain_combo.currentData()
        if db is None:
            return
        db = float(db)
        self._cfg.mic_gain_db = db
        sax_config.save_config(self._cfg)
        self._engine.set_mic_gain(10.0 ** (db / 20.0))
        self._update_mic_gain_status()

    def _update_mic_gain_status(self) -> None:
        """Refresh the plain-language mic-gain effect label, so the control
        gives a visible result even though the cents readout is (by design)
        unaffected by gain."""
        lbl = self._setup_mic_gain_status
        db = int(round(float(getattr(self._cfg, 'mic_gain_db', 0.0))))
        if db == 0:
            lbl.setText(self._t('setup_mic_gain_off'))
        elif db > 0:
            lbl.setText(self._t('setup_mic_gain_boost', db=db))
        else:
            lbl.setText(self._t('setup_mic_gain_cut', db=db))

    def _retranslate_setup(self) -> None:
        """Re-label every translatable SETUP-tab widget on a language switch.
        The tab is built once (never rebuilt), so without this its titles and
        labels stay in the build-time language until restart."""
        for grp, key in self._setup_i18n_groups:
            grp.setTitle(self._t(key))
        for lbl, key in self._setup_i18n_labels:
            lbl.setText(self._t(key))
        self._cb_prefer_duplex.setText(self._t('setup_prefer_duplex'))
        self._cb_prefer_duplex.setToolTip(self._t('setup_prefer_duplex_tip'))
        self._btn_setup_range.setText(self._t('setup_range_edit'))
        self._btn_test_tone.setToolTip(self._t('setup_testtone_tip'))
        if not self._btn_test_tone.isChecked():
            self._btn_test_tone.setText(self._t('setup_testtone_play'))
        self._setup_nick_edit.setPlaceholderText(self._t('nickname_tip'))
        self._setup_mic_gain_combo.setToolTip(self._t('setup_mic_gain_tip'))
        self._update_mic_gain_status()

    def _on_min_n_changed(self, value: int) -> None:
        """User adjusted the min-N filter. Persist + redraw the table.
        The filter is purely a display gate — measurements are still
        collected; rows just hide until they accumulate enough hits."""
        self._cfg.min_n_visible = max(0, int(value))
        sax_config.save_config(self._cfg)
        self._refresh_table()

    def _on_nickname_changed(self) -> None:
        """Toolbar nickname edited → mirror into SETUP + stamp the run label."""
        self._apply_nickname(self._nick_edit.text(), source=self._nick_edit)

    def _on_setup_nickname_changed(self) -> None:
        """SETUP nickname edited → mirror into the toolbar + stamp the label.
        Mirrors the toolbar horn-nickname field."""
        self._apply_nickname(self._setup_nick_edit.text(),
                             source=self._setup_nick_edit)

    def _apply_nickname(self, text, source=None) -> None:
        """Nickname changed from EITHER the toolbar or the SETUP editor. Mirror
        the other editor (signals blocked) so both surfaces agree; the toolbar
        _nick_edit stays canonical — the InstrumentController and session-state
        read it — so it is updated before stamping. Then stamp the label onto
        the active run via the controller (a no-op when not recording)."""
        for edit in (self._nick_edit, self._setup_nick_edit):
            if edit is source:
                continue
            if edit.text() != text:
                edit.blockSignals(True)
                edit.setText(text)
                edit.blockSignals(False)
        self._instr_ctrl.on_nickname_changed()

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
        default_path = sax_config.CONFIG_DIR / sax_config.DEFAULT_LOG_FILENAME
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
            new_path = self._cfg.effective_log_path()
            if new_path:
                self._log = MeasurementLog(path=new_path)
                # The controllers were built with the previous log; point
                # them at the new one or CSV export and run metadata would
                # keep using the discarded in-memory log.
                self._export._log = self._log
                self._instr_ctrl._log = self._log
                if AUDIO_OK:
                    self._log.start_run(instrument=self.instrument,
                                        a4_hz=self._engine.a4,
                                        label=self._nick_edit.text().strip())
        sax_config.save_config(self._cfg)

    # _ask_custom_instrument lives on InstrumentController.
    # It was only invoked by _on_add_custom (also extracted), so no
    # MainWindow wrapper is retained — the controller calls itself.

    def _on_disp_changed(self, idx):
        self.display = ['griff', 'klingend'][idx]
        self._refresh_table()

    def _on_a4_changed(self, idx):
        """Toolbar concert-pitch (A4) combo changed → apply + mirror SETUP."""
        self._apply_a4(float(self._a4_combo.itemData(idx)))

    def _on_setup_a4_changed(self, idx):
        """SETUP concert-pitch (A4) combo changed → apply + mirror the toolbar.
        Mirrors the toolbar Kammerton selector."""
        self._apply_a4(float(self._setup_a4_combo.itemData(idx)))

    def _apply_a4(self, new_a4) -> None:
        """Apply a concert-pitch (A4) change from EITHER the toolbar or the
        SETUP combo. Mirror BOTH combos to new_a4 with signals blocked so this
        heavy measurement recalc runs EXACTLY ONCE and is never re-entered by
        the sibling combo. Cents are a function of frequency + A4, so the cents
        in self.stats are invalidated — but the underlying frequencies are
        immutable and still live in the log, so re-derive the table from them
        instead of discarding what the user recorded."""
        new_a4 = float(new_a4)
        hz_int = int(new_a4)
        for _combo in (self._a4_combo, self._setup_a4_combo):
            _target = hz_int - 430
            if (0 <= _target < _combo.count()
                    and _combo.currentIndex() != _target):
                _combo.blockSignals(True)
                _combo.setCurrentIndex(_target)
                _combo.blockSignals(False)
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
            self._push_a4(new_a4)
            # Re-derive from the frequencies behind the CURRENT table only.
            # The log also holds earlier sessions, other instruments, imports
            # and notes the user reset or deleted; none of those belong here.
            table_freqs = [f for st in self.stats.values() for f in st.freqs]
            remapped: dict[int, NoteStats] = {}
            for f in table_freqs:
                # Non-positive / non-finite freqs carry no tuning information
                # and would make freq_to_midi raise.
                if not math.isfinite(f) or f <= 0.0:
                    continue
                midi_round, cents = cents_dev(f, new_a4)
                if midi_round in SAX_MIDI:
                    ns = remapped.setdefault(midi_round, NoteStats())
                    ns.add(cents, f)
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
            self._btn_record.setStyleSheet(
                self._btn_record.styleSheet()
                .replace('#2ecc71', '#b7770d').replace('#27ae60', '#b7770d'))
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

        transp = transp_map().get(self.instrument, 0)
        midi_kl: int | None = None

        if self._layout_mode == 'matrix':
            # Matrix mode: map (row, col) to the sounding MIDI for the
            # clicked cell. Row = chroma index (0..11), col = octave within
            # the span the columns were painted for.
            col = self._table.columnAt(pos.x())
            if col < 0:
                return
            lo_oct, hi_oct = self._table_ctrl.painted_octave_span()
            oct_ = lo_oct + col
            if oct_ > hi_oct:
                return
            midi_visible = (oct_ + 1) * 12 + row
            if self.display == 'griff':
                midi_kl = midi_visible + transp
            else:
                midi_kl = midi_visible
            # Only offer the action if this cell actually holds played data.
            st = self.stats.get(midi_kl)
            if st is None or st.n == 0:
                return
        else:
            # Single-column mode: rows are the sorted stats after the table's
            # min-N filter (sax_table hides rows with 1..min_n-1 readings), so
            # index the same filtered list or the row maps to another note.
            min_n = max(0, int(getattr(self._cfg, 'min_n_visible', 0)))
            keys = sorted(k for k, s in self.stats.items()
                          if s.n == 0 or s.n >= min_n)
            if row >= len(keys):
                return
            midi_kl = keys[row]

        midi_gr  = midi_kl - transp
        note_str = f"{midi_note_name(midi_gr)} / {midi_note_name(midi_kl)}"

        menu = QMenu(self)   # themed via the global QMenu rule in build_app_qss
        action = menu.addAction(f"\U0001f5d1  {self._t('ctx_discard')}")
        action.setData(midi_kl)

        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen and chosen.data() == midi_kl:
            if QMessageBox.question(
                self, self._t('reset_title'),
                self._t('ctx_discard_confirm', note=note_str),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                self.stats.pop(midi_kl, None)
                self._refresh_table()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _on_reset(self):
        if QMessageBox.question(
            self, self._t('reset_title'), self._t('reset_msg'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.stats.clear()
            # Reset implies "I might be switching instruments" — clear the
            # cached maker/model so the next export re-prompts.
            self._instr_info_asked = False
            self._refresh_table()

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
        # Restore wrote the canonical toolbar widgets from cfg; mirror them into
        # the SETUP twins so both surfaces agree on launch.
        self._sync_setup_twins_from_toolbar()

    def _sync_setup_twins_from_toolbar(self) -> None:
        """Copy the canonical toolbar A4 + nickname into their SETUP twins
        (signals blocked — a programmatic sync, not a user edit). Called after
        session-state restore, which only knows about the toolbar widgets.
        Idempotent."""
        self._setup_nick_edit.blockSignals(True)
        self._setup_nick_edit.setText(self._nick_edit.text())
        self._setup_nick_edit.blockSignals(False)
        self._setup_a4_combo.blockSignals(True)
        self._setup_a4_combo.setCurrentIndex(self._a4_combo.currentIndex())
        self._setup_a4_combo.blockSignals(False)

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
        for timer in (self._device_poll, self._refresh_timer,
                      self._deck_pump_timer):
            timer.stop()
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
            self._stop_test_tone,
            self._metro_ctrl.stop,
            self._drone_ctrl.shutdown,
            self._stop_pitch_pipes,
            # The deck disarms the input tap and unregisters its playback;
            # close() is idempotent, never raises, never orphans.
            self._deck_ctrl.close,
        )
        for step in steps:
            try:
                step()
            except Exception:
                pass

    def _save_session_state(self) -> None:
        """Thin delegator + persistence. The snapshot work lives on
        SessionStateController; the save_config call stays here because
        the controller is intentionally agnostic about *when* to
        persist. Called from closeEvent. The per-setting saves
        scattered through the GUI remain as a belt; this is the
        suspenders."""
        self._session_state.save()
        sax_config.save_config(self._cfg)

    def _report_settings_problem(self) -> None:
        """Tell the player once, at start-up, that their settings file could
        not be used. Non-modal, so a locked file never blocks the window."""
        store = self._settings
        if store.unreadable:
            self._info_banner.show_message(self._t(
                'settings_unreadable', path=str(store.paths.config)))
        elif store.corrupt_backup is not None:
            self._info_banner.show_message(self._t(
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

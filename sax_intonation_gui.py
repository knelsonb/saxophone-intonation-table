#!/usr/bin/env python3
"""
Saxophon-Intonationsanalysator – GUI
=====================================
The main window, as a shell. It loads the settings, builds the session model,
the audio engine and the history log, puts the toolbar above the TUNER /
METRO / DECK / SETUP tabs, owns the banners and the tab status dots, and
wires the parts together. The parts live in their own modules: sax_toolbar,
sax_tab_tuner, sax_tab_metro, sax_tab_deck, sax_tab_setup and sax_drone_bar;
the controllers in sax_device, sax_export, sax_instrument_controller and
sax_session_state; the shared state in sax_session and sax_translator.

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

from PyQt6.QtCore import QLocale, QTimer
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabBar,
    QTabWidget,
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
from sax_controls import db_to_linear
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
from sax_instrument_controller import InstrumentController
from sax_instruments import register_custom, transp_map
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
from sax_toolbar import Toolbar
from sax_translator import Translator
from sax_widgets import (  # noqa: F401  (MatrixCellDelegate re-exported)
    AudioRecoveryBanner,
    InfoBanner,
    MatrixCellDelegate,
    StatusDot,
)

APP_NAME = 'Intonation Analyzer'
APP_VERSION = '1.4.0'

TAB_KEYS = ('tuner', 'metro', 'deck', 'setup')

# Wrong-instrument hint: this many readings in a row outside the selected
# instrument's range (about one phrase of all-wrong notes) suggest the player
# picked the wrong instrument.
WRONG_INSTRUMENT_READINGS = 12

# Hot-plug poll: cached device queries cost ~0 ms; only a re-init is
# expensive, and the engine re-inits only when the device set changed.
DEVICE_POLL_MS = 1000

# The DECK tab's recording dot pulses at about 1 Hz, as on Android.
DECK_DOT_PULSE_MS = 500
DECK_DOT_COLOURS = ('#e74c3c', '#6e2420')
METRO_DOT_COLOUR = '#2ecc71'


# =============================================================================
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths | None = None):
        super().__init__()
        # Readings since launch, for the diagnostics panel (the table counts
        # its own from the session's statistics).
        self._notes_count = 0
        # Wrong-instrument detector: consecutive readings outside the selected
        # instrument's range. A reading inside resets it, as does picking
        # another instrument or A4; the hint is shown once per selection.
        self._oor_count = 0
        self._oor_banner_shown = False

        # Every file the app keeps is resolved once, here, and handed to the
        # store that owns it. Settings and custom instruments are loaded
        # before any widget is built, so the catalog and theme are right from
        # the first paint.
        self._paths = paths if paths is not None else AppPaths.from_environment()
        self._store = sax_config.ConfigStore(self._paths)
        sax_instruments.use_range_overrides(
            sax_instruments.RangeOverrides(self._paths.ranges))
        sax_crash.set_log_path(self._paths.crash_log)
        self._cfg = cfg = self._store.load()
        sax_theme.set_active(cfg.theme)
        for c in self._store.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)

        # German-speaking locales get the original DE strings, anyone else
        # English; the saved language wins. Every translated text in the
        # window is bound to this translator and follows a switch by itself.
        lang = 'de' if QLocale.system().name().startswith('de') else 'en'
        if cfg.last_lang in ('de', 'en'):
            lang = cfg.last_lang
        self._tr = Translator(lang, self)

        # Bb tenor (the most common community-band sax) unless a saved
        # instrument is still in the catalog. A saved custom instrument cannot
        # be resolved while its file is unreadable: the session runs on the
        # default, but the saved choice is kept unless the player picks
        # another one.
        instrument = 'bb_tenor'
        if cfg.last_instrument_key in transp_map():
            instrument = cfg.last_instrument_key
        keep_saved_instrument = (self._store.customs_unreadable
                                 and cfg.last_instrument_key not in transp_map())
        display = cfg.last_display_mode if cfg.last_display_mode in ('griff', 'klingend') \
            else 'griff'
        # What is being measured: A4, instrument, display, nickname,
        # recording on/off and the per-note statistics. The combos, the
        # engine, the drone, the pitch pipes and the history log follow its
        # signals; the table and the exports read it. Readings arrive through
        # the engine's queued note signal, so it is only touched on this thread.
        self._session = SessionModel(instrument=instrument, display=display, parent=self)

        self._engine = AudioEngine()
        self._engine.set_filter_mode(cfg.filter_mode)
        self._engine.set_mic_gain(db_to_linear(cfg.mic_gain_db))
        self._engine.set_prefer_wdmks(cfg.prefer_wdmks)
        # History is kept when the welcome dialog said so; SAX_INTONATION_LOG_PATH
        # is a power-user override that always wins.
        self._log = MeasurementLog(
            path=self._paths.history(cfg.effective_log_path(self._paths)))
        # Runs follow the session. The first one opens after the session is
        # restored below, so it is stamped with the restored A4 and nickname.
        self._runs = RunTracker(self._session, self._log, enabled=AUDIO_OK)
        # Sound sources the player switched on, waiting for (or parked until)
        # the output runs. See sax_output_users.
        self._output_users = OutputUsers(
            lambda: bool(self._engine.output_running), self._request_output_open)
        # The "output disconnected" notice while it is shown.
        self._output_notice = None

        self._build_ui()
        self._session_state = SessionStateController(
            self, cfg, session=self._session, translator=self._tr,
            splitter=self._tuner_tab.splitter, apply_a4=self._apply_a4,
            keep_saved_instrument=keep_saved_instrument)
        self._export = ExportController(
            self, log=self._log, session=self._session, tr=self._tr,
            instrument_label=self._instr_ctrl.instrument_label,
            sample_rate=self._engine_sample_rate,
            add_lang_toggle=self._add_dialog_lang_toggle)
        self._session_state.restore()
        self._session.seed_expected_notes()
        self._runs.start_run()

        self._device_ctrl = DeviceController(
            self, engine=self._engine, cfg=cfg, store=self._store, tr=self._tr,
            chip=self._toolbar.audio_chip, info_banner=self._info_banner,
            audio_banner=self._audio_banner, tuner_tab=self._tuner_tab)
        signals = self._engine.signals
        if AUDIO_OK:
            signals.note_event.connect(self._on_note_event)
            signals.state_changed.connect(self._on_engine_state)
            signals.interface_appeared.connect(self._on_interface_appeared)
        # Wired even without PortAudio: an output request then fails, and the
        # failure is what turns the asking toggle back off.
        signals.output_state_changed.connect(self._on_output_status)
        signals.output_devices_changed.connect(self._setup_tab.show_output_devices)

        # Start the engine AFTER the UI exists, so a start-up PortAudio failure
        # paints a banner instead of crashing here (start never raises; it
        # sets the state and emits state_changed).
        if AUDIO_OK:
            saved = DeviceSelection(name=cfg.audio_device_name,
                                    host_api=cfg.audio_device_host_api,
                                    samplerate=cfg.audio_device_samplerate)
            # The engine's hot-plug poll and retry re-resolve the player's
            # device by name and host API against a fresh device list.
            self._engine.set_preferred_hint(saved)
            self._engine.request_start(preferred=saved,
                                       samplerate_pref=cfg.audio_samplerate_pref)
        self._device_poll = QTimer(self)
        self._device_poll.timeout.connect(self._poll_devices)
        if AUDIO_OK:
            self._device_poll.start(DEVICE_POLL_MS)
        else:
            QMessageBox.information(self, self._t('audio_error_title'),
                                    self._t('audio_error'))

        self._report_settings_problem()
        # The first-launch question about history. Skipped on every later
        # launch, and when the settings file could not be read: that player
        # is not new, their file is only locked.
        if not cfg.welcome_shown and not self._store.unreadable:
            self._show_welcome_dialog()
        self._tuner_tab.start_refreshing()

    @property
    def lang(self) -> str:
        return self._tr.lang

    def _t(self, key, **kwargs):
        """A string in the current language, for text shown once (message
        boxes, file dialogs). Text that stays on screen is bound instead."""
        return self._tr.t(key, **kwargs)

    # ── Building ─────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        tr, cfg, session = self._tr, self._cfg, self._session
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 10, 14, 10)

        toolbar = self._toolbar = Toolbar(tr=tr, cfg=cfg, engine=self._engine,
                                          session=session)
        root.addWidget(toolbar)

        # The banners sit above the tuner. The recovery banner appears only
        # while the input has FAILED; the info banner carries hot-plug news,
        # the wrong-instrument hint and settings problems, with an optional
        # action and an always-present Dismiss.
        self._audio_banner = AudioRecoveryBanner(
            tr, self._retry_audio, self._open_audio_picker)
        self._info_banner = InfoBanner(tr)

        # The drone strip sits under the tuner readout; its full voice picker
        # is in SETUP.
        self._drone_bar = DroneBar(tr=tr, cfg=cfg, engine=self._engine,
                                   session=session, output_users=self._output_users)
        self._tuner_tab = TunerTab(
            tr=tr, cfg=cfg, store=self._store, engine=self._engine, session=session,
            banners=(self._audio_banner, self._info_banner), drone_bar=self._drone_bar,
            sample_rate=self._engine_sample_rate, notes_count=lambda: self._notes_count)
        self._metro_tab = MetroTab(tr=tr, cfg=cfg, engine=self._engine,
                                   output_users=self._output_users)
        self._deck_tab = DeckTab(tr=tr, cfg=cfg, engine=self._engine,
                                 output_users=self._output_users)
        self._setup_tab = SetupTab(
            tr=tr, cfg=cfg, store=self._store, engine=self._engine,
            session=session, output_users=self._output_users,
            request_output_open=self._request_output_open,
            apply_a4=self._apply_a4,
            apply_response=self._apply_response,
            apply_theme=self._apply_theme,
            open_range_editor=self._open_range_editor,
            nickname_committed=self._runs.nickname_committed)

        # The toolbar stays above the tabs. METRO and DECK carry a status dot
        # (green while the metronome runs, pulsing red while the deck records,
        # as on Android's TabBar).
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tab_keys = list(TAB_KEYS)
        for page in (self._tuner_tab, self._metro_tab, self._deck_tab, self._setup_tab):
            self._tabs.addTab(page, '')
        for i, key in enumerate(self._tab_keys):
            tr.bind_tab(self._tabs, i, f'nav_tab_{key}', f'nav_tab_{key}_tip')
        self._install_tab_indicators()
        root.addWidget(self._tabs, 1)
        # Restore the last-active tab (allow-list coerced on load).
        if cfg.last_active_tab in self._tab_keys:
            self._tabs.setCurrentIndex(self._tab_keys.index(cfg.last_active_tab))
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # The active palette was set from the settings before any widget was
        # built; live switches go through _apply_theme.
        self.setStyleSheet(sax_theme.build_app_qss(sax_theme.active()))
        self._sync_qpalette()
        tr.bind_text(self.setWindowTitle, lambda: f"{tr.t('window_title')} {APP_VERSION}")

        # The instrument controller fills the toolbar's instrument combos and
        # follows them; it needs every widget above.
        self._instr_ctrl = InstrumentController(
            self, session=session, store=self._store, engine=self._engine, tr=tr,
            family_combo=toolbar.family_combo, instr_combo=toolbar.instr_combo,
            add_lang_toggle=self._add_dialog_lang_toggle,
            on_instrument_changed=self._on_instrument_changed,
            on_range_changed=self._on_range_changed,
            on_a4_changed=self._on_autotune_applied)
        self._instr_ctrl.refill()

        self._wire(toolbar)

    def _wire(self, toolbar: Toolbar) -> None:
        """Connect what the parts report to what the window does about it."""
        toolbar.a4_picked.connect(self._apply_a4)
        toolbar.response_picked.connect(self._apply_response)
        toolbar.layout_picked.connect(self._on_layout_picked)
        toolbar.nickname_committed.connect(self._runs.nickname_committed)
        toolbar.range_editor_requested.connect(self._open_range_editor)
        toolbar.custom_instrument_requested.connect(self._instr_ctrl.on_add_custom)
        toolbar.audio_picker_requested.connect(self._open_audio_picker)
        toolbar.autotune_requested.connect(self._instr_ctrl.on_autotune)
        toolbar.reset_requested.connect(self._on_reset)
        toolbar.import_requested.connect(self._import_csv)
        toolbar.export_requested.connect(self._export_file)
        self._setup_tab.drone_voice_picked.connect(self._drone_bar.select_voice)
        self._drone_bar.voice_changed.connect(self._setup_tab.show_drone_voice)
        self._metro_tab.running_changed.connect(self.set_metro_running)
        self._deck_tab.recording_changed.connect(self.set_deck_recording)
        # The engine measures against the session's A4 (the combos, the drone
        # and the pitch pipes follow the same signal).
        self._session.a4_changed.connect(self._engine.set_a4)
        self._session.recording_changed.connect(self._on_recording_changed)

    # ── Tab status dots ──────────────────────────────────────────────────────
    def _install_tab_indicators(self) -> None:
        bar = self._tabs.tabBar()
        self._metro_dot = StatusDot()
        self._metro_dot.set_color(METRO_DOT_COLOUR)
        bar.setTabButton(self._tab_keys.index('metro'),
                         QTabBar.ButtonPosition.RightSide, self._metro_dot)
        self._deck_dot = StatusDot()
        self._deck_dot.set_color(DECK_DOT_COLOURS[0])
        bar.setTabButton(self._tab_keys.index('deck'),
                         QTabBar.ButtonPosition.RightSide, self._deck_dot)
        # Hide AFTER setTabButton, which shows the widget: hidden before, the
        # dots would show at launch with nothing running or recording.
        self._metro_dot.setVisible(False)
        self._deck_dot.setVisible(False)
        self._deck_pulse_bright = True
        self._deck_pulse_timer = QTimer(self)
        self._deck_pulse_timer.setInterval(DECK_DOT_PULSE_MS)
        self._deck_pulse_timer.timeout.connect(self._pulse_deck_dot)

    def _pulse_deck_dot(self) -> None:
        self._deck_pulse_bright = not self._deck_pulse_bright
        self._deck_dot.set_color(DECK_DOT_COLOURS[0 if self._deck_pulse_bright else 1])

    def set_metro_running(self, running: bool) -> None:
        """Show the METRO tab's green dot while the metronome runs."""
        self._metro_dot.setVisible(bool(running))

    def set_deck_recording(self, recording: bool) -> None:
        """Show the DECK tab's pulsing red dot while a take records."""
        if recording:
            self._deck_pulse_bright = True
            self._deck_dot.set_color(DECK_DOT_COLOURS[0])
            self._deck_dot.setVisible(True)
            if not self._deck_pulse_timer.isActive():
                self._deck_pulse_timer.start()
        else:
            self._deck_pulse_timer.stop()
            self._deck_dot.setVisible(False)

    def _on_tab_changed(self, idx: int) -> None:
        # Kept live as well as on close, so a crash does not lose it.
        if 0 <= idx < len(self._tab_keys):
            self._cfg.last_active_tab = self._tab_keys[idx]

    # ── Readings ─────────────────────────────────────────────────────────────
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

    def _on_note(self, midi_kl: int, freq: float, cents: float) -> None:
        session = self._session
        if not session.recording:
            return
        # The wrong-instrument detector runs BEFORE the range filter below, so
        # it still counts when "filter to range" drops the notes it counts.
        try:
            in_range = session.in_range(midi_kl)
        except Exception:
            in_range = True
        if in_range:
            self._oor_count = 0
        else:
            self._oor_count += 1
            if (not self._oor_banner_shown
                    and self._oor_count >= WRONG_INSTRUMENT_READINGS):
                self._oor_banner_shown = True
                instrument = session.instrument
                label = self._instr_ctrl.instrument_label
                self._info_banner.show_message(
                    lambda: self._t('wrong_instrument_banner_body',
                                    instrument=label(instrument)))
        # "Filter to range" keeps the table bounded to the instrument's range.
        if not self._cfg.allow_out_of_range and not in_range:
            return
        session.add_reading(midi_kl, cents, freq)
        self._notes_count += 1
        # Instrument and A4 are latched from the log's active run, not read
        # from the session, so a reading that arrives during a change still
        # goes to the run it was measured in.
        self._log.add_measurement(midi_sounding=midi_kl,
                                  midi_fingered=midi_kl - session.transposition,
                                  cents=cents, freq_hz=freq)
        self._tuner_tab.show_reading(midi_kl, freq, cents)

    def _reset_wrong_instrument_hint(self) -> None:
        self._oor_count = 0
        self._oor_banner_shown = False

    def _engine_sample_rate(self) -> int:
        """The input's sample rate (for the cents precision), or the default
        while there is none: a stale delegate paint can race teardown."""
        sr = int(self._engine.samplerate or 0)
        return sr if sr > 0 else int(DEFAULT_SAMPLE_RATE)

    # ── Session changes ──────────────────────────────────────────────────────
    def _apply_a4(self, new_a4) -> None:
        """A concert-pitch change from the toolbar or SETUP, or the saved one
        at start-up. The session hands the value to both combos (without
        emitting, so this recalculation runs EXACTLY ONCE), the engine, the
        drone and the pitch pipes, and opens a new history run. Cents are a
        function of frequency + A4, so the table's cents are invalidated —
        its frequencies are kept, and the table is re-derived from them
        instead of discarding what the player recorded."""
        new_a4 = float(new_a4)
        self._reset_wrong_instrument_hint()
        # Readings are handled on this thread, so none can land in the middle
        # of the rebuild. pause_emissions() stops the engine emitting while
        # the reference changes, and the new A4 starts a new A4 generation: a
        # reading queued before the pause carries the old generation and
        # _on_note_event drops it.
        self._engine.pause_emissions()
        try:
            self._session.set_a4(new_a4)
            self._session.rederive_stats(new_a4)
        finally:
            self._engine.resume_emissions()
        # Re-seed the blank range rows on top of the re-derived data.
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_instrument_changed(self) -> None:
        """Another instrument was picked. The wrong-instrument hint is about
        the selected instrument, so it may fire again; the table shows the
        new instrument's range at once."""
        self._reset_wrong_instrument_hint()
        self._info_banner.hide()
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_range_changed(self) -> None:
        """The range editor saved new bounds for the same instrument: show the
        new range (the wrong-instrument hint is not reset)."""
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_autotune_applied(self) -> None:
        """Autotune applied a new A4 and cleared the table: show the range
        again. Everything that follows A4 already moved with the session."""
        self._session.seed_expected_notes()
        self._tuner_tab.refresh_table()

    def _on_recording_changed(self, on: bool) -> None:
        if on:
            self._tuner_tab.show_status('no_signal')

    def _apply_response(self, mode) -> None:
        """A Response (filter mode) pick from the toolbar or SETUP: show it in
        both (without emitting, so it is applied once), save it and route the
        live engine. Unknown modes are ignored."""
        if mode not in FILTER_PRESETS:
            return
        self._toolbar.show_response(mode)
        self._setup_tab.show_response(mode)
        self._cfg.filter_mode = mode
        self._save_settings()
        if AUDIO_OK:
            self._engine.set_filter_mode(mode)

    def _on_layout_picked(self, pref: str) -> None:
        if pref in ('auto', 'single', 'matrix'):
            self._cfg.layout_mode_preference = pref
            self._save_settings()
            self._tuner_tab.refresh_table()

    def _open_range_editor(self) -> None:
        self._instr_ctrl.open_range_editor()

    def _on_reset(self) -> None:
        if QMessageBox.question(
            self, self._t('reset_title'), self._t('reset_msg'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._session.clear_stats()
            # Reset implies "I might be switching instruments": the next
            # export asks for maker and model again.
            self._export.forget_instrument_info()
            self._tuner_tab.refresh_table()

    # ── Output ───────────────────────────────────────────────────────────────
    def _request_output_open(self) -> None:
        """Ask for the chosen output (empty name = system default, rate 0 =
        auto-negotiate) to be opened. The answer comes back as an
        OutputStatus (_on_output_status)."""
        cfg = self._cfg
        self._engine.request_open_output(DeviceSelection(
            name=cfg.output_device_name, host_api=cfg.output_device_host_api,
            samplerate=cfg.output_device_samplerate))

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
        for part in (self._setup_tab, self._metro_tab, self._drone_bar, self._deck_tab):
            part.park_output()

    # ── Audio input (DeviceController) ───────────────────────────────────────
    # Engine signals and the poll timer are connected to these window methods
    # so they are delivered on the GUI thread whichever thread emits them.
    def _open_audio_picker(self) -> None:
        self._device_ctrl.open_audio_picker()

    def _retry_audio(self) -> None:
        self._device_ctrl.retry_audio()

    def _poll_devices(self) -> None:
        self._device_ctrl.poll_devices()

    def _on_engine_state(self, state, err, msg, device) -> None:
        self._device_ctrl.on_engine_state(state, err, msg, device)

    def _on_interface_appeared(self, device: 'DeviceInfo') -> None:
        self._device_ctrl.on_interface_appeared(device)

    # ── Appearance and language ──────────────────────────────────────────────
    def _apply_theme(self, name: str) -> None:
        """Switch the live UI theme: set the active palette, re-render the
        global stylesheet, repaint the custom-painted widgets (which read
        sax_theme.active() each paint), and save the choice. Coerces any
        value; never raises into the signal."""
        pal = sax_theme.set_active(name)
        self.setStyleSheet(sax_theme.build_app_qss(pal))
        self._sync_qpalette()
        # The stylesheet re-polish repaints QSS-styled widgets; the QPainter-
        # based widgets (tuner, table, chart) need a nudge to re-read it.
        self.update()
        for widget in self.findChildren(QWidget):
            widget.update()
        self._cfg.theme = pal.name
        self._save_settings()

    def _sync_qpalette(self) -> None:
        """Set the QApplication palette from the active theme, so what the
        global QSS doesn't reach — table viewports (Base), header sections,
        scrollbars, tooltips — follows the theme too. Without this they keep
        the dark Base and read as black blocks on the light theme."""
        app = QApplication.instance()
        if app is None:
            return
        p = sax_theme.active()
        pal = QPalette()
        role = QPalette.ColorRole
        for r, colour in ((role.Window, p.window_bg), (role.Base, p.window_bg),
                          (role.AlternateBase, p.base_bg), (role.Text, p.text),
                          (role.WindowText, p.text), (role.Button, p.button_bg),
                          (role.ButtonText, p.text), (role.Highlight, p.accent),
                          (role.HighlightedText, p.text_bright),
                          (role.ToolTipBase, p.base_bg), (role.ToolTipText, p.text)):
            pal.setColor(r, QColor(colour))
        app.setPalette(pal)

    def _add_dialog_lang_toggle(self, dlg, layout, on_change=None) -> None:
        """DE | EN toggle for the app's own dialogs (sax_dialogs). A switch
        changes the whole window's language, not only the dialog's."""
        add_lang_toggle(dlg, layout, get_lang=lambda: self.lang,
                        set_lang=self._tr.set_language, t_func=self._t,
                        on_change=on_change)

    # ── Files ────────────────────────────────────────────────────────────────
    def _export_file(self, kind: str) -> None:
        {'txt': self._export.export_txt, 'pdf': self._export.export_pdf,
         'csv': self._export.export_csv, 'chart': self._export.export_chart}[kind]()

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self._t('csv_import_title'), '', self._t('csv_filter'))
        if not path:
            return
        # The import merges into the history for good; an "are you sure?"
        # beat guards against mixing somebody else's CSV into one's own data.
        confirm = QMessageBox.question(
            self, self._t('csv_import_confirm_title'),
            self._t('csv_import_confirm_body', filename=Path(path).name),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok)
        if confirm != QMessageBox.StandardButton.Ok:
            return
        try:
            runs, meas, skipped = self._log.import_raw_csv(path)
        except ValueError as e:
            # A non-raw-mode header or the size / row limits: the specific
            # reason goes under the general message.
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
            self._t('csv_import_saved', runs=runs, meas=meas, skipped=skipped))

    # ── Settings ─────────────────────────────────────────────────────────────
    def _show_welcome_dialog(self) -> None:
        """First-launch dialog: offers to keep a history of the readings."""
        dlg = WelcomeDialog(self, self._t, self._paths.default_history,
                            self._add_dialog_lang_toggle)
        accepted = (dlg.exec() == QDialog.DialogCode.Accepted)
        # "Decide later" leaves the question open, so the dialog returns on
        # the next launch; it is the only place history saving is offered.
        self._cfg.welcome_shown = accepted
        if accepted and dlg.keep_history():
            self._cfg.persistence_enabled = True
            # The environment override wins here too, as it does at start-up.
            new_path = self._paths.history(self._cfg.effective_log_path(self._paths))
            if new_path:
                # A log on the chosen path replaces the in-memory one (the
                # player has been in the app a few seconds at most). Everything
                # built with the old log follows, or CSV export and the runs
                # would keep using the discarded one.
                self._log = MeasurementLog(path=new_path)
                self._export.set_log(self._log)
                self._runs.set_log(self._log)
                self._runs.start_run()
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

    # ── Closing ──────────────────────────────────────────────────────────────
    def closeEvent(self, ev):
        # Save the session first, so the next launch lands the player back
        # where they were; a failure there must never keep the audio from
        # stopping — a bad settings file is recoverable, a dangling stream
        # is not.
        try:
            self._session_state.save()
            self._save_settings()
        except Exception:
            pass
        self._device_poll.stop()
        self._tuner_tab.stop_refreshing()
        # Silence and release every sound source before the streams close.
        # Each step is independent: one that fails must not keep the others
        # (or the engine shutdown after them) from running.
        steps = (lambda: self._output_users.drop_prefix(''),
                 self._setup_tab.shut_down, self._metro_tab.shut_down,
                 self._drone_bar.shut_down, self._deck_tab.shut_down)
        for step in steps:
            try:
                step()
            except Exception:
                pass
        # Both streams close and the audio-control thread stops: nothing may
        # keep the microphone or the speaker open after the window is gone.
        self._engine.shutdown()
        self._log.close()
        ev.accept()


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

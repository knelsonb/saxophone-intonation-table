"""The driver: the ONLY module that knows how the app is built.

Everything above this file (``app.py``, the scenario tests) talks to the app
through the harness vocabulary: instruments by key, notes by name, plain data
back. This module turns that vocabulary into the app's real user paths today:
clicking its buttons, picking from its combo boxes, answering its dialogs.

When the app is restructured (tabs split out of the main window, the audio engine
split into input / output / service modules, ``sax_*`` moved into a package) this
is the one file that changes. Each method says which internals it relies on, so
the list below is the contract the rewrite must keep or update here:

  Entry point / process
    * module ``sax_intonation_gui`` with class ``MainWindow()`` (no arguments);
      importing a fresh copy of every ``sax_*`` module per app (see ``_load``);
    * the app imports ``sounddevice`` as ``sd`` inside the audio engine; the
      fake module is installed in ``sys.modules`` before that import;
    * ``sax_audio_engine.HOST_API_OPEN_TIMEOUT_S`` (the open timeout, patched
      to FakeAudioHost.OPEN_TIMEOUT_S so a hung device costs a quarter second);
    * ``sax_audio_engine.ANALYSIS_IN_THREAD`` (patched to False: pitch
      detection runs inline in the input callback the fake host pumps, so
      every reading reaches the window before ``pump_input`` returns, in the
      order played; with the real analysis thread, readings would arrive
      whenever that thread and the Qt event loop got to them);
    * ``sax_audio_engine.CONTROL_IN_THREAD`` (patched to False: device
      commands — open, close, retry, the hot-plug poll — run inline where the
      window requests them, so a click that opens a device has opened it, and
      the window has handled the outcome, when the click returns; the real
      audio-control thread is covered by tests/contracts/test_audio_control.py);
    * data directory ``~/.intonation_analyzer`` (resolved from HOME when the
      window is built, ``sax_paths.AppPaths.from_environment``) holding
      ``config.json``, ``custom_instruments.json``, ``instrument_ranges.json``,
      ``log.jsonl`` (this month's history; earlier months beside it as
      ``log-YYYY-MM.jsonl``); ``MeasurementLog.path`` is the live history file.
  Time
    * QTimers on the window (``_device_poll``, ``_deck_pulse_timer``), the
      TUNER tab's ``_refresh_timer``, the DECK tab's ``_pump_timer`` and the
      tuner widget's are virtualised: ticked from the harness clock instead
      of real time.
  Toolbar (``_toolbar``, sax_toolbar.Toolbar)
    * ``_family_combo`` / ``_instr_combo`` (instrument), ``_disp_combo``
      (fingered / sounding), ``_a4_combo`` (430..450), ``_lang_combo``,
      ``_filter_combo``, ``_nick_edit``, ``_audio_chip`` (``_name``),
      buttons ``_btn_autotune`` ``_btn_reset`` ``_btn_txt`` ``_btn_pdf``
      ``_btn_csv`` ``_btn_chart`` ``_btn_import`` ``_btn_custom``;
  TUNER tab (``_tuner_tab``, sax_tab_tuner.TunerTab)
    * tuner: ``_tuner`` (``note`` ``cents`` ``freq`` ``active``) and
      ``_status_lbl``; table: ``_table`` (a QTableView over
      sax_table.IntonationTableModel; single layout: DisplayRole columns
      fingered/sounding/mean/std/N/bar, mean cell ForegroundRole brush = band
      colour; matrix layout: cell UserRole payload dicts), ``_caption``,
      ``layout_mode``, ``refresh_table()``, signal
      ``_table.customContextMenuRequested``; ``_min_n_spin``, ``_cb_oor``;
    * state: ``_session`` (sax_session.SessionModel: ``instrument``
      ``display`` ``a4`` ``stats``), ``lang``, ``_cfg``, ``_log``,
      ``_engine`` (``state`` ``last_error`` ``output_running`` ``samplerate``
      ``output_samplerate`` ``active_device`` ``active_output_device`` ``a4``).
  Banners
    * ``_audio_banner`` (``_msg``, ``_btn_retry``),
      ``_info_banner`` (``_msg``, ``_btn_action``); ``_engine.mixer.active_sources()``.
  METRO tab (``_metro_tab``, sax_tab_metro.MetroTab)
    * ``_slider`` ``_ts_btns`` ``_btn_start`` ``controller`` ``bpm``
      ``time_sig`` ``_status`` ``_vol_slider`` ``_btn_tap``.
  Drone / pitch pipes (``_drone_bar``, sax_drone_bar.DroneBar)
    * ``nudge_semitones()`` ``semitones`` ``voice_id`` ``controller``
      ``_btn_on`` ``_status`` ``_vol_slider`` ``_btn_pipes``
      ``pipes_controller``, ``pipes_window`` (PitchPipesWindow: ``_pads``
      C4..B4, ``status_label``); the voice is picked in SETUP's
      ``_voice_combo``; ``sax_drone`` imports ``tinysoundfont`` lazily
      (hidden to simulate a missing synth).
  DECK tab
    * ``_deck_tab`` (sax_tab_deck.DeckTab): ``_btn_record`` ``_btn_stop``
      ``_btn_play`` ``_btn_export`` ``state()`` ``_state_lbl`` ``_status``.
  SETUP tab (``_setup_tab``, sax_tab_setup.SetupTab)
    * ``_output_combo`` ``_btn_test_tone`` ``_test_tone_status``
      ``_theme_combo`` ``_mic_gain_combo`` ``_btn_range`` ``_voice_combo``.
  Dialogs (recognised in ``_classify``; filled in the ``_fill_*`` methods)
    * welcome (objectName ``welcomeDlg``), maker/model, CSV slice, custom
      instrument (property ``_lang_title_key``), ``RangeEditorDialog``
      (``_lo_spin`` ``_hi_spin`` ``_btn_save`` ``_btn_cancel`` ``_btn_restore``
      ``_error_lbl``), ``AudioPickerDialog`` (recognised, not scriptable);
      the classes live in ``sax_dialogs`` and are recognised by class name.
  Export capture
    * ``reportlab.platypus.SimpleDocTemplate.build`` (PDF story),
      ``sax_report.render_intonation_chart`` (chart arguments: the pure
      chart renderer ``sax_report.render_chart`` hands its drawing to it),
      ``MeasurementLog.import_raw_csv`` / ``.export_csv`` (results).
    * the save dialog is the static ``QFileDialog.getSaveFileName``; the app
      completes a name typed without an extension itself and asks with
      ``QMessageBox.question`` before replacing a file the dialog did not
      confirm.
  i18n
    * ``sax_i18n.STRINGS`` = ``{lang: {key: format_string}}``.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import string
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from ._patch import PatchStack
from .clock import ManualClock
from .dialogs import Dialogs
from .fake_audio import FakeAudioHost, HarnessError
from .model import (
    AudioStatus,
    AutotuneResult,
    DeckState,
    DroneState,
    ExportFailed,
    ExportModels,
    ImportResult,
    MetronomeState,
    Reading,
    Row,
    note_spellings,
)
from .tones import note_to_midi

DATA_DIR = ".intonation_analyzer"
DATA_FILES = {
    "config": "config.json",
    "customs": "custom_instruments.json",
    "ranges": "instrument_ranges.json",
    "history": "log.jsonl",
}

# Table band colours (sax_table, the mean column's ForegroundRole) -> harness band names.
_BAND_BY_COLOUR = {"#3a9e5f": "green", "#c8a020": "amber", "#c03030": "red"}

# `_lang_title_key` property of the app's own form dialogs -> harness kind.
_FORM_BY_TITLE_KEY = {
    "model_dialog_title": "maker_model",
    "csv_dialog_title": "csv_slice",
    "custom_dlg_title": "custom_instrument",
}

_FAR_MS = 2_000_000_000   # "never" for a real QTimer whose ticks the harness owns

_ACTIVE: Optional["AppDriver"] = None


def data_dir(home: Path) -> Path:
    return Path(home) / DATA_DIR


def data_file(home: Path, logical: str) -> Path:
    return data_dir(home) / DATA_FILES[logical]


def i18n_catalog() -> dict:
    """``{lang: {key: [placeholder, ...]}}`` for every UI string. Needs no app."""
    from sax_i18n import STRINGS
    fmt = string.Formatter()
    out: dict = {}
    for lang, table in STRINGS.items():
        out[lang] = {
            key: sorted({f for _lit, f, _spec, _conv in fmt.parse(text) if f})
            for key, text in table.items()
        }
    return out


def _parse_cents(text: str) -> Optional[int]:
    t = text.replace("−", "-").strip().lstrip("±")
    return int(t) if re.fullmatch(r"[+-]?\d+", t) else None


def _midi_of_cell(cell: str) -> int:
    m = re.match(r"^(.*?)(-?\d+)$", cell)
    return note_to_midi(m.group(1).split("/")[0] + m.group(2)) if m else 0


class AppDriver:
    def __init__(self, home: Path, host: FakeAudioHost, clock: ManualClock,
                 dialogs: Dialogs, lang: str = "en") -> None:
        self.home = Path(home)
        self.host = host
        self.clock = clock
        self.dialogs = dialogs
        self.lang = lang
        self.win = None
        self.errors: list[str] = []
        self.models = ExportModels()
        self._patches = PatchStack()
        self._saved_modules: dict = {}
        self._timers: dict = {}
        self._last_import: Any = None
        self._modules_swapped = False
        self.last_form_note: Optional[str] = None
        self._form_accepted = False
        self.streams_open_at_close: list = []

    # ================================================================ lifecycle
    def start(self) -> None:
        """Import a private copy of the app, build its main window offscreen."""
        global _ACTIVE
        if _ACTIVE is not None:
            raise HarnessError("another HeadlessApp is still running; close() it first")
        _ACTIVE = self
        try:
            self._start()
        except BaseException:
            self.stop()
            raise

    def _start(self) -> None:
        p = self._patches
        p.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
        for var in ("HOME", "USERPROFILE"):
            p.setenv(var, str(self.home))
        p.setenv("SAX_INTONATION_LOG_PATH", None)
        self._load_modules()

        from PyQt6.QtCore import QLocale
        from PyQt6.QtWidgets import QApplication
        self.qapp = QApplication.instance() or QApplication([])

        # Deterministic UI language on a first launch (no saved last_lang).
        p.setattr(QLocale, "system",
                  staticmethod(lambda code={"de": "de_DE"}.get(self.lang, "en_US"): QLocale(code)))
        engine = sys.modules["sax_audio_engine"]
        p.setattr(engine, "HOST_API_OPEN_TIMEOUT_S", self.host.OPEN_TIMEOUT_S)
        p.setattr(engine, "ANALYSIS_IN_THREAD", False)
        p.setattr(engine, "CONTROL_IN_THREAD", False)
        p.setattr(sys, "excepthook", self._trap_exception)

        self.dialogs.classify = self._classify
        self.dialogs.fillers = {
            "welcome": self._fill_welcome,
            "maker_model": self._fill_maker_model,
            "csv_slice": self._fill_csv_slice,
            "custom_instrument": self._fill_custom_instrument,
            "range_editor": self._fill_range_editor,
        }
        self.dialogs.install(p)
        self._install_capture_hooks()

        G = sys.modules["sax_intonation_gui"]
        self.win = G.MainWindow()          # the first-launch welcome dialog runs here
        self.win.show()
        self.qapp.processEvents()
        self._virtualize_timers()
        self.clock.subscribe(self._tick)
        # A person never touches the window in its first instant: the 100 ms deck
        # tick has enabled the Record button and the 300 ms table tick has painted
        # by then. Run those ticks (the 1 s device poll stays pending).
        self._tick(0.5)
        self.settle()
        self.check()

    def _load_modules(self) -> None:
        """Give this app its own copy of every app module.

        The app keeps module-level state (the active settings store, active
        theme, instrument registry), and some tests install their own fake
        ``sounddevice``. A fresh
        import per app makes each launch a true process start: nothing leaks in
        from another test, and nothing leaks out.
        """
        for name in [n for n in sys.modules if n.startswith("sax_") or n == "sounddevice"]:
            self._saved_modules[name] = sys.modules.pop(name)
        self._modules_swapped = True
        sys.modules["sounddevice"] = self.host.sounddevice_module()
        importlib.import_module("sax_intonation_gui")

    def _restore_modules(self) -> None:
        if not self._modules_swapped:
            return
        self._modules_swapped = False
        for name in [n for n in sys.modules if n.startswith("sax_") or n == "sounddevice"]:
            del sys.modules[name]
        sys.modules.update(self._saved_modules)
        self._saved_modules = {}

    def stop(self) -> None:
        """Close the window the way a user would, then undo every patch."""
        global _ACTIVE
        try:
            if self.win is not None:
                try:
                    self.clock.unsubscribe(self._tick)
                    self.win.close()            # closeEvent: saves session state
                    self.qapp.processEvents()
                    self.win.deleteLater()
                    from PyQt6.QtCore import QEvent
                    self.qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                finally:
                    self.win = None
            self.host.release_hangs()
            self.streams_open_at_close = [repr(s) for s in self.host.open_streams()]
        finally:
            self._timers.clear()
            self._patches.undo()
            self._restore_modules()
            if _ACTIVE is self:
                _ACTIVE = None
            # Tear the closed app down completely, now. Left to the cyclic
            # garbage collector, its Qt objects were destroyed at some random
            # point inside a later test, in the middle of that test's own Qt
            # calls, and the process crashed now and then.
            import gc

            from PyQt6.QtCore import QEvent
            for _ in range(2):
                gc.collect()
                self.qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    # ======================================================== time / event loop
    def _virtualize_timers(self) -> None:
        """Take over the app's QTimers so time only moves when the harness says.

        The timers that drive observable behaviour are re-pointed at an
        unreachable real interval and ticked from the clock instead (see
        ``_tick``). Purely cosmetic timers (spectrum plot, diagnostics panel)
        are left on real time; they change no state a test can observe.
        """
        from PyQt6.QtCore import QTimer
        w = self.win
        timers = [w._device_poll, w._deck_pulse_timer,
                  w._tuner_tab._refresh_timer, w._deck_tab._pump_timer]
        timers += w._tuner_tab._tuner.findChildren(QTimer)
        for t in timers:
            self._timers[t] = [t.interval(), 0.0]
            t.setInterval(_FAR_MS)

    def _tick(self, dt: float) -> None:
        for t, state in list(self._timers.items()):
            try:
                if not t.isActive():
                    state[1] = 0.0
                    continue
                state[1] += dt * 1000.0
                interval = max(1, state[0])
                fired = 0
                # 1e-6 ms of slack: slices of 0.05 s do not sum exactly to 1 s.
                while state[1] >= interval - 1e-6 and fired < 100:
                    state[1] -= interval
                    fired += 1
                    t.timeout.emit()
            except RuntimeError:       # the C++ timer is already gone
                self._timers.pop(t, None)

    def settle(self) -> None:
        """Let the table refresh timer fire once and drain posted Qt events."""
        if self.win is None:
            return
        self.qapp.processEvents()
        self.win._tuner_tab.refresh_table()
        self.qapp.processEvents()

    def check(self) -> None:
        """Raise whatever went wrong inside Qt since the last call."""
        self.dialogs.assert_clean()
        if self.errors:
            errs, self.errors = self.errors, []
            raise AssertionError("exception inside the app:\n" + "\n".join(errs))

    def _trap_exception(self, etype, value, tb) -> None:
        self.errors.append("".join(traceback.format_exception(etype, value, tb)))

    # ============================================================ capture hooks
    def _install_capture_hooks(self) -> None:
        p = self._patches
        models = self.models
        drv = self

        from reportlab.platypus import SimpleDocTemplate
        orig_build = SimpleDocTemplate.build

        def build(doc, flowables, *a, **k):
            models.pdf = drv._pdf_model(flowables)
            return orig_build(doc, flowables, *a, **k)
        p.setattr(SimpleDocTemplate, "build", build)

        export_mod = sys.modules["sax_report"]
        orig_chart = export_mod.render_intonation_chart

        def render_chart(**kw):
            models.chart = {
                "title": kw.get("title"), "subtitle": kw.get("subtitle"),
                "footer": kw.get("footer"), "instrument": kw.get("instrument"),
                "sample_rate": kw.get("sample_rate"),
                # Means and spreads to 0.1 ct: finer digits are numerical noise
                # that must not decide whether a golden file matches.
                "notes": [[n[0], round(float(n[1]), 1), round(float(n[2]), 1), int(n[3]),
                           round(float(n[4]), 1), int(n[5]), bool(n[6])]
                          for n in kw.get("notes", [])],
            }
            return orig_chart(**kw)
        p.setattr(export_mod, "render_intonation_chart", render_chart)

        log_cls = sys.modules["sax_intonation_log"].MeasurementLog
        orig_import = log_cls.import_raw_csv
        orig_export = log_cls.export_csv

        def import_raw_csv(log, path):
            try:
                res = orig_import(log, path)
            except BaseException as exc:
                drv._last_import = ("error", exc)
                raise
            drv._last_import = ("ok", res)
            return res

        def export_csv(log, path, **kw):
            n = orig_export(log, path, **kw)
            models.csv_rows = n
            return n
        p.setattr(log_cls, "import_raw_csv", import_raw_csv)
        p.setattr(log_cls, "export_csv", export_csv)

    @staticmethod
    def _pdf_model(flowables) -> dict:
        from reportlab.platypus import Paragraph, Spacer, Table
        out = []
        for f in flowables:
            if isinstance(f, Paragraph):
                out.append({"paragraph": f.text, "plain": f.getPlainText()})
            elif isinstance(f, Table):
                colours = {}
                for r, row in enumerate(f._cellStyles):
                    for c, cs in enumerate(row):
                        col = cs.color
                        hexval = col.hexval() if hasattr(col, "hexval") else str(col)
                        if hexval not in ("black", "0x000000", "0x0"):
                            colours[f"{r},{c}"] = hexval.replace("0x", "#")
                out.append({"table": [[str(v) for v in row] for row in f._cellvalues],
                            "text_colours": dict(sorted(colours.items()))})
            elif isinstance(f, Spacer):
                out.append({"spacer": round(float(f.height), 1)})
            else:
                out.append({"flowable": type(f).__name__})
        return {"flowables": out}

    # ================================================================== dialogs
    def _classify(self, dlg) -> Optional[str]:
        if dlg.objectName() == "welcomeDlg":
            return "welcome"
        key = dlg.property("_lang_title_key")
        if key in _FORM_BY_TITLE_KEY:
            return _FORM_BY_TITLE_KEY[key]
        name = type(dlg).__name__
        return {"RangeEditorDialog": "range_editor",
                "AudioPickerDialog": "audio_picker"}.get(name)

    @staticmethod
    def _buttons(dlg):
        from PyQt6.QtWidgets import QDialogButtonBox
        return dlg.findChildren(QDialogButtonBox)

    def _finish_form(self, dlg, answer: str) -> bool:
        """Click the Ok/accept or Cancel/reject button of the dialog's button box."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        box = self._buttons(dlg)[-1]
        wanted = (QDialogButtonBox.ButtonRole.AcceptRole if answer == "accept"
                  else QDialogButtonBox.ButtonRole.RejectRole)
        for b in box.buttons():
            if box.buttonRole(b) == wanted:
                b.click()
                break
        return int(dlg.result()) == int(QDialog.DialogCode.Accepted)

    def _fill_welcome(self, dlg, answer: str, fields: dict) -> bool:
        from PyQt6 import sip
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QCheckBox, QPushButton
        if "language" in fields:
            label = fields["language"].upper()
            # Click once and stop iterating: the click retranslates the dialog,
            # which can delete sibling widgets still in the list; touching one
            # of those afterwards dereferenced freed memory.
            target = next((b for b in dlg.findChildren(QPushButton)
                           if b.text() == label), None)
            if target is not None:
                target.click()
            self.qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            # Remembered for Welcome.language_buttons.
            self._welcome_lang_buttons = {b.text(): b.isChecked()
                                          for b in dlg.findChildren(QPushButton)
                                          if not sip.isdeleted(b)
                                          and b.text() in ("DE", "EN")}
        if "history" in fields:
            dlg.findChildren(QCheckBox)[0].setChecked(bool(fields["history"]))
        return self._finish_form(dlg, answer)

    def _fill_maker_model(self, dlg, answer: str, fields: dict) -> bool:
        from PyQt6.QtWidgets import QLineEdit
        maker, model = dlg.findChildren(QLineEdit)[:2]      # creation order: maker, model
        if "maker" in fields:
            maker.setText(fields["maker"])
        if "model" in fields:
            model.setText(fields["model"])
        return self._finish_form(dlg, answer)

    def _fill_csv_slice(self, dlg, answer: str, fields: dict) -> bool:
        from PyQt6.QtWidgets import QComboBox
        mode, run, instr, nick = dlg.findChildren(QComboBox)[:4]   # creation order
        for combo, key in ((mode, "mode"), (run, "run_id"), (instr, "instrument"),
                           (nick, "nickname")):
            if key in fields and fields[key] is not None:
                idx = combo.findData(fields[key])
                if idx < 0:
                    raise HarnessError(f"CSV slice dialog has no {key} {fields[key]!r}")
                combo.setCurrentIndex(idx)
        return self._finish_form(dlg, answer)

    def _fill_custom_instrument(self, dlg, answer: str, fields: dict) -> bool:
        from PyQt6.QtWidgets import QLineEdit, QSpinBox
        if "name" in fields:
            dlg.findChildren(QLineEdit)[0].setText(fields["name"])
        if "transpose" in fields:
            dlg.findChildren(QSpinBox)[0].setValue(int(fields["transpose"]))
        if "names" in fields:
            # Retype until the form closes with Accepted (a form that checks
            # the name itself stays open on a bad one).
            for name in fields["names"]:
                dlg.findChildren(QLineEdit)[0].setText(name)
                if self._finish_form(dlg, "accept"):
                    return True
            self._finish_form(dlg, "reject")
            return False
        return self._finish_form(dlg, answer)

    def _fill_range_editor(self, dlg, answer: str, fields: dict) -> bool:
        from PyQt6.QtWidgets import QDialog
        self.last_form_note = None
        if fields.get("restore"):
            dlg._btn_restore.click()
        for key, spin in (("lo", dlg._lo_spin), ("hi", dlg._hi_spin)):
            if key in fields:
                v = fields[key]
                spin.setValue(note_to_midi(v) if isinstance(v, str) else int(v))
        if answer == "accept":
            dlg._btn_save.click()
        else:
            dlg._btn_cancel.click()
        accepted = int(dlg.result()) == int(QDialog.DialogCode.Accepted)
        if answer == "accept" and not accepted:
            # Save is disabled while lo > hi; close the dialog so nothing hangs.
            self.last_form_note = dlg._error_lbl.text() or "range editor refused to save"
            dlg._btn_cancel.click()
        self._form_accepted = accepted
        return accepted

    # ============================================================ basic controls
    def _combo_select(self, combo, data, what: str) -> None:
        idx = combo.findData(data)
        if idx < 0:
            raise HarnessError(f"{what}: no entry {data!r}")
        combo.setCurrentIndex(idx)

    def pick_instrument(self, key: str) -> None:
        w = self.win
        family = sys.modules["sax_instruments"].family_of(key)
        if family is None:
            raise HarnessError(f"unknown instrument key {key!r}")
        self._combo_select(w._toolbar._family_combo, family, "instrument family")
        self._combo_select(w._toolbar._instr_combo, key, "instrument")

    def instrument(self) -> str:
        return self.win._session.instrument

    def set_a4(self, hz: int) -> None:
        if int(hz) != hz or not 430 <= hz <= 450:
            raise HarnessError("the A4 selector offers whole Hz from 430 to 450")
        self._combo_select(self.win._toolbar._a4_combo, int(hz), "A4")

    def a4(self) -> int:
        return int(self.win._toolbar._a4_combo.currentData())

    def engine_a4(self) -> float:
        return float(self.win._engine.a4)

    def set_display(self, mode: str) -> None:
        idx = {"fingered": 0, "sounding": 1}.get(mode)
        if idx is None:
            raise HarnessError("display mode is 'fingered' or 'sounding'")
        self.win._toolbar._disp_combo.setCurrentIndex(idx)

    def display(self) -> str:
        return {"griff": "fingered", "klingend": "sounding"}[self.win._session.display]

    def set_language(self, lang: str) -> None:
        self._combo_select(self.win._toolbar._lang_combo, lang, "language")

    def language(self) -> str:
        return self.win.lang

    def window_title(self) -> str:
        return self.win.windowTitle()

    def t(self, key: str, **kw) -> str:
        """The app's own translation of a UI string key, for the current language."""
        return self.win._t(key, **kw)

    # =================================================================== table
    def table_rows(self, played_only: bool = True) -> list[Row]:
        from PyQt6.QtCore import Qt
        self.settle()
        w = self.win
        m = w._tuner_tab._table.model()
        rows: list[Row] = []
        if w._tuner_tab.layout_mode == "single":
            griff = (w._session.display == "griff")
            for r in range(m.rowCount()):
                cells = [m.index(r, c).data() or "" for c in range(6)]
                fingered, sounding = (cells[0], cells[1]) if griff else (cells[1], cells[0])
                n = int(cells[4]) if cells[4].isdigit() else 0
                band = None
                if n:
                    brush = m.index(r, 2).data(Qt.ItemDataRole.ForegroundRole)
                    band = _BAND_BY_COLOUR.get(brush.color().name().lower())
                rows.append(Row(fingered, sounding, _parse_cents(cells[2]),
                                _parse_cents(cells[3]), n, band))
        else:
            theme = sys.modules["sax_theme"]
            for r in range(m.rowCount()):
                for c in range(m.columnCount()):
                    pay = m.index(r, c).data(Qt.ItemDataRole.UserRole)
                    if not isinstance(pay, dict) or not pay.get("n") or pay.get("mean") is None:
                        continue
                    mean = pay["mean"]
                    # The matrix cell is painted by a delegate; derive the band
                    # with the same classifier the views share.
                    band = {"in": "green", "close": "amber", "off": "red"}[
                        theme.cents_band(mean)]
                    rows.append(Row(pay["fingered_name"], pay["sounding_name"],
                                    int(round(mean)), int(round(pay["std"])),
                                    int(pay["n"]), band))
            rows.sort(key=lambda x: _midi_of_cell(x.sounding))
        return [r for r in rows if r.n > 0] if played_only else rows

    def table_caption(self) -> str:
        self.settle()
        return self.win._tuner_tab._caption.text()

    def table_reset(self, confirm: bool) -> None:
        with self.dialogs.scope():
            # Optional: a test may have scripted its own answer, which is used first.
            self.dialogs.expect("question", "Yes" if confirm else "No", optional=True)
            self.win._toolbar._btn_reset.click()

    def table_discard(self, name: str, confirm: bool) -> None:
        w = self.win
        t = w._tuner_tab._table
        if w._tuner_tab.layout_mode == "single":
            rows = self.table_rows(played_only=False)
            row = next((i for i, r in enumerate(rows)
                        if r.is_named(name) and r.n > 0), None)
            if row is None:
                raise HarnessError(f"no played row named {name!r} in the table")
            index = t.model().index(row, 0)
        else:
            index = self._matrix_cell(name)
        t.scrollTo(index)
        pos = t.visualRect(index).center()
        with self.dialogs.scope():
            self.dialogs.expect_menu(self.t("ctx_discard"))
            self.dialogs.expect("question", "Yes" if confirm else "No", optional=True)
            t.customContextMenuRequested.emit(pos)

    def _matrix_cell(self, name: str):
        """The model index of the matrix cell showing played data for
        ``name`` (fingered or sounding spelling). Uses the cell UserRole
        payload dicts the table model paints from; a cell below the minimum
        count has no mean and is not offered, as nothing is shown in it."""
        from PyQt6.QtCore import Qt
        self.settle()
        m = self.win._tuner_tab._table.model()
        for r in range(m.rowCount()):
            for c in range(m.columnCount()):
                index = m.index(r, c)
                pay = index.data(Qt.ItemDataRole.UserRole)
                if not isinstance(pay, dict) or not pay.get("n") or pay.get("mean") is None:
                    continue
                if any(name in note_spellings(pay[k])
                       for k in ("fingered_name", "sounding_name")):
                    return index
        raise HarnessError(f"no played cell named {name!r} in the matrix")

    def table_columns(self) -> list[str]:
        """Column header labels as displayed (``_table`` horizontal header):
        the six list columns, or one octave label per matrix column."""
        from PyQt6.QtCore import Qt
        self.settle()
        m = self.win._tuner_tab._table.model()
        return [m.headerData(c, Qt.Orientation.Horizontal) or ""
                for c in range(m.columnCount())]

    def stats_total_n(self) -> int:
        return sum(s.n for s in self.win._session.stats.values())

    # =================================================================== tuner
    def tuner_reading(self) -> Reading:
        self.qapp.processEvents()
        t = self.win._tuner_tab._tuner
        return Reading(note=t.note, cents=float(t.cents), freq=float(t.freq),
                       active=bool(t.active), status_text=self.win._tuner_tab._status_lbl.text())

    # ================================================================= autotune
    def autotune(self, accept: bool) -> AutotuneResult:
        before = self.a4()
        with self.dialogs.scope() as sc:
            self.dialogs.expect("warning", "Ok", optional=True)
            self.dialogs.expect("question", "Yes" if accept else "No", optional=True)
            self.win._toolbar._btn_autotune.click()
        msg = sc.last
        if msg is None:
            raise HarnessError("autotune showed no dialog")
        if msg.kind == "warning":
            return AutotuneResult("insufficient_data", msg.text, before, self.a4())
        return AutotuneResult("applied" if accept else "declined", msg.text, before, self.a4())

    # =================================================================== exports
    def _export(self, what: str, button, path: Path, maker: str, model: str,
                expect_failure: bool, extra=None, keep_existing: bool = False) -> Optional[Path]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not keep_existing:
            path.unlink()
        with self.dialogs.scope() as sc:
            self.dialogs.expect("maker_model", maker=maker, model=model, optional=True)
            if extra is not None:
                extra()
            self.dialogs.save_path(path)
            for kind in ("information", "warning", "critical"):
                self.dialogs.expect(kind, "Ok", optional=True)
            button.click()
            # A flow that stops before the save dialog (nothing to export) leaves
            # the queued path unused; drop it so it cannot answer a later dialog.
            self.dialogs._files["save"].clear()
        if path.exists():
            return path
        if expect_failure:
            return None
        raise ExportFailed(what, sc.last.text if sc.last else None)

    def export_txt(self, path, maker="", model="", expect_failure=False):
        return self._export("TXT", self.win._toolbar._btn_txt, path, maker, model, expect_failure)

    def export_pdf(self, path, maker="", model="", expect_failure=False):
        self.models.pdf = None
        return self._export("PDF", self.win._toolbar._btn_pdf, path, maker, model, expect_failure)

    def export_chart(self, path, maker="", model="", expect_failure=False):
        self.models.chart = None
        return self._export("chart", self.win._toolbar._btn_chart, path, maker, model,
                            expect_failure)

    def export_csv(self, path, mode, instrument, run_id, nickname, maker, model,
                   expect_failure, keep_existing: bool = False):
        def slice_form():
            self.dialogs.expect("csv_slice", mode=mode, instrument=instrument,
                                run_id=run_id, nickname=nickname, optional=True)
        return self._export("CSV", self.win._toolbar._btn_csv, path, maker, model,
                            expect_failure, extra=slice_form, keep_existing=keep_existing)

    # =================================================================== import
    def import_csv(self, path, confirm: bool) -> ImportResult:
        self._last_import = None
        with self.dialogs.scope() as sc:
            self.dialogs.open_path(path)
            self.dialogs.expect("question", "Ok" if confirm else "Cancel", optional=True)
            for kind in ("information", "warning", "critical"):
                self.dialogs.expect(kind, "Ok", optional=True)
            self.win._toolbar._btn_import.click()
        shown = [m for m in sc.messages if m.kind != "question"]
        text = shown[-1].text if shown else ""
        if not confirm:
            return ImportResult("cancelled")
        kind, payload = self._last_import or (None, None)
        if kind == "ok":
            runs, meas, skipped = payload
            status = ("empty" if (runs, meas, skipped) == (0, 0, 0) else
                      "imported" if meas else "nothing_new")
            return ImportResult(status, runs, meas, skipped, text)
        if kind == "error":
            status = "rejected" if isinstance(payload, ValueError) else "io_error"
            return ImportResult(status, message=text)
        return ImportResult("cancelled", message=text)

    def history_counts(self) -> tuple[int, int]:
        """(runs that hold measurements, measurements) in the history log. The
        empty run the app opens at start-up is not counted."""
        meas = self.win._log.measurements()
        return len({m.run_id for m in meas}), len(meas)

    def history_path(self) -> Optional[Path]:
        p = self.win._log.path
        return Path(p) if p else None

    # ================================================================== metronome
    def metro_start(self, bpm: Optional[int], time_sig: Optional[str]) -> None:
        w = self.win
        if bpm is not None:
            w._metro_tab._slider.setValue(int(bpm))
        if time_sig is not None:
            if time_sig not in w._metro_tab._ts_btns:
                raise HarnessError(f"time signature {time_sig!r} not offered")
            w._metro_tab._ts_btns[time_sig].click()
        if not w._metro_tab._btn_start.isChecked():
            w._metro_tab._btn_start.click()

    def metro_stop(self) -> None:
        if self.win._metro_tab._btn_start.isChecked():
            self.win._metro_tab._btn_start.click()

    def metro_state(self) -> MetronomeState:
        w = self.win
        ctrl = w._metro_tab.controller
        status = w._metro_tab._status.text() if not w._metro_tab._status.isHidden() else None
        return MetronomeState(
            running=bool(ctrl is not None and ctrl.is_running()),
            bpm=int(w._metro_tab.bpm), time_sig=str(w._metro_tab.time_sig),
            volume=w._metro_tab._vol_slider.value() / 100.0, unavailable_text=status)

    def metro_set_volume(self, percent: int) -> None:
        self.win._metro_tab._vol_slider.setValue(int(percent))

    def metro_tap(self) -> None:
        self.win._metro_tab._btn_tap.click()

    # ===================================================================== drone
    def drone_on(self, note: str, voice: Optional[str]) -> None:
        w = self.win
        if voice is not None:
            self._combo_select(w._setup_tab._voice_combo, voice, "drone voice")
        # The UI has no absolute-note picker: the drone sounds A4 shifted by up
        # to a +/-12 semitone stepper.
        want = note_to_midi(note) - 69
        if not -12 <= want <= 12:
            raise HarnessError(f"drone cannot reach {note}: the stepper covers A3..A5")
        while w._drone_bar.semitones != want:
            w._drone_bar.nudge_semitones(1 if want > w._drone_bar.semitones else -1)
        if not w._drone_bar._btn_on.isChecked():
            w._drone_bar._btn_on.click()

    def drone_off(self) -> None:
        if self.win._drone_bar._btn_on.isChecked():
            self.win._drone_bar._btn_on.click()

    def drone_synth_unavailable(self) -> None:
        """Hide the SoundFont synth package for the rest of this app's life.
        Relies on ``sax_drone`` importing ``tinysoundfont`` lazily when it first
        builds a SoundFont voice: a ``None`` entry in ``sys.modules`` makes that
        import raise ImportError, as on an install without the package."""
        saved = sys.modules.get("tinysoundfont")
        had = "tinysoundfont" in sys.modules

        def restore() -> None:
            if had:
                sys.modules["tinysoundfont"] = saved
            else:
                sys.modules.pop("tinysoundfont", None)
        self._patches.hook(restore)
        sys.modules["tinysoundfont"] = None

    def drone_state(self) -> DroneState:
        w = self.win
        ctrl = w._drone_bar.controller
        status = w._drone_bar._status.text() if not w._drone_bar._status.isHidden() else None
        return DroneState(
            enabled=bool(ctrl is not None and ctrl.is_enabled()),
            voice=str(ctrl.voice_id if ctrl is not None else w._drone_bar.voice_id),
            semitones=int(w._drone_bar.semitones),
            sine_fallback=bool(getattr(ctrl, "synth_fallback", False)),
            status_text=status)

    # ================================================================ pitch pipes
    def _pipe_midi(self, note: str) -> int:
        midi = note_to_midi(note)
        if not 60 <= midi <= 71:
            raise HarnessError(f"pitch pipes offer C4..B4, not {note}")
        return midi

    def pipes_press(self, note: str) -> None:
        bar = self.win._drone_bar
        midi = self._pipe_midi(note)
        if bar.pipes_window is None or not bar.pipes_window.isVisible():
            bar._btn_pipes.click()
        pad = bar.pipes_window._pads[midi]
        if not pad.isChecked():
            pad.click()

    def pipes_release(self, note: str) -> None:
        window = self.win._drone_bar.pipes_window
        midi = self._pipe_midi(note)
        if window is not None and window._pads[midi].isChecked():
            window._pads[midi].click()

    def pipes_active(self) -> list[int]:
        ctrl = self.win._drone_bar.pipes_controller
        return sorted(ctrl.active_midis()) if ctrl is not None else []

    def pipes_status(self) -> Optional[str]:
        window = self.win._drone_bar.pipes_window
        lbl = window.status_label if window is not None else None
        return lbl.text() if lbl is not None and not lbl.isHidden() else None

    def pipes_close(self) -> None:
        if self.win._drone_bar.pipes_window is not None:
            self.win._drone_bar.pipes_window.close()

    # ===================================================================== deck
    def deck_record(self) -> None:
        if not self.win._deck_tab._btn_record.isChecked():
            self.win._deck_tab._btn_record.click()

    def deck_stop(self) -> None:
        self.win._deck_tab._btn_stop.click()

    def deck_play(self) -> None:
        self.win._deck_tab._btn_play.click()

    def deck_export(self, path) -> Optional[Path]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.dialogs.scope():
            self.dialogs.save_path(path)
            self.win._deck_tab._btn_export.click()
        return path if path.exists() else None

    def deck_state(self) -> DeckState:
        w = self.win
        status = w._deck_tab._status.text() if not w._deck_tab._status.isHidden() else None
        return DeckState(state=w._deck_tab.state(), status_text=status,
                         label=w._deck_tab._state_lbl.text())

    def deck_can_record(self) -> bool:
        return bool(self.win._deck_tab._btn_record.isEnabled())

    # =============================================================== settings
    def set_theme(self, name: str) -> None:
        self._combo_select(self.win._setup_tab._theme_combo, name, "theme")

    def set_response(self, mode: str) -> None:
        self._combo_select(self.win._toolbar._filter_combo, mode, "response mode")

    def set_min_n(self, n: int) -> None:
        self.win._tuner_tab._min_n_spin.setValue(int(n))

    def set_filter_to_range(self, on: bool) -> None:
        self.win._tuner_tab._cb_oor.setChecked(bool(on))

    def set_nickname(self, text: str) -> None:
        self.win._toolbar._nick_edit.setText(text)
        self.win._toolbar._nick_edit.editingFinished.emit()

    def nickname(self) -> str:
        return self.win._toolbar._nick_edit.text()

    def set_mic_gain(self, db: int) -> None:
        self._combo_select(self.win._setup_tab._mic_gain_combo, db, "mic gain")

    def select_output_device(self, name: Optional[str]) -> None:
        # The SETUP list is filled when the window is built and refreshed by
        # the 1 s device poll (SETUP ``_output_combo``).
        self._combo_select(self.win._setup_tab._output_combo, name or "", "output device")

    def output_device_names(self) -> list[str]:
        # Item data of SETUP ``_output_combo``; '' is the system-default entry.
        c = self.win._setup_tab._output_combo
        return [str(c.itemData(i)) for i in range(c.count()) if c.itemData(i)]

    def test_tone(self, on: bool) -> None:
        b = self.win._setup_tab._btn_test_tone
        if b.isChecked() != bool(on):
            b.click()

    def test_tone_status(self) -> Optional[str]:
        w = self.win
        status = w._setup_tab._test_tone_status
        return status.text() if not status.isHidden() else None

    def test_tone_playing(self) -> bool:
        return self.win._setup_tab._btn_test_tone.isChecked()

    def edit_range(self, lo, hi, restore: bool, accept: bool) -> bool:
        """Open the range editor from SETUP. ``lo`` / ``hi`` are in the notation
        currently displayed (fingered or sounding), as the dialog shows them.
        Returns whether the editor accepted the values."""
        fields = {"restore": restore}
        if lo is not None:
            fields["lo"] = lo
        if hi is not None:
            fields["hi"] = hi
        self.last_form_note = None
        self._form_accepted = False
        with self.dialogs.scope():
            self.dialogs.expect("range_editor", "accept" if accept else "reject", **fields)
            self.win._setup_tab._btn_range.click()
        return self._form_accepted

    def add_custom_instrument(self, name: str, transpose: int, accept: bool = True):
        before = self.instrument()
        with self.dialogs.scope():
            self.dialogs.expect("custom_instrument", "accept" if accept else "reject",
                                name=name, transpose=transpose)
            self.dialogs.expect("warning", "Ok", optional=True)
            self.win._toolbar._btn_custom.click()
        after = self.instrument()
        return after if after != before else None

    # ================================================================== audio
    def audio_status(self) -> AudioStatus:
        w = self.win
        eng = w._engine
        dev = eng.active_device
        odev = eng.active_output_device if eng.output_running else None
        return AudioStatus(
            input_state=eng.state.value, input_error=eng.last_error.value,
            input_device=dev.name if (dev is not None and eng.input_running) else "",
            input_rate=int(eng.samplerate) if eng.input_running else 0,
            output_running=bool(eng.output_running),
            output_device=odev.name if odev else "",
            output_rate=int(eng.output_samplerate) if eng.output_running else 0,
            banner_visible=not w._audio_banner.isHidden(),
            banner_text=w._audio_banner._msg.text(),
            chip_text=w._toolbar._audio_chip._name.text(),
            status_text=w._tuner_tab._status_lbl.text())

    def audio_retry(self) -> None:
        self.win._audio_banner._btn_retry.click()

    def notice_text(self) -> Optional[str]:
        # ``_info_banner`` (InfoBanner: ``_msg``, ``_btn_action``).
        b = self.win._info_banner
        return b._msg.text() if not b.isHidden() else None

    def notice_accept(self) -> None:
        b = self.win._info_banner
        if b.isHidden() or b._btn_action.isHidden():
            raise HarnessError("no notice with an action is shown")
        b._btn_action.click()

    def output_source_count(self) -> int:
        # ``_engine.mixer.active_sources()``.
        return int(self.win._engine.mixer.active_sources())

    # ================================================================ persistence
    def config_on_disk(self) -> dict:
        p = data_file(self.home, "config")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def history_enabled(self) -> bool:
        return bool(self.win._cfg.persistence_enabled)

    # ============================================== launch / settings / language
    # Additions for the first-launch, persistence and language scenarios.

    def welcome_language_buttons(self) -> dict:
        """Checked state of the welcome dialog's DE / EN buttons after the last
        scripted welcome answer (recorded by ``_fill_welcome``)."""
        return dict(getattr(self, "_welcome_lang_buttons", {}))

    def theme(self) -> str:
        # Internal: SETUP ``_theme_combo`` item data (theme id).
        return str(self.win._setup_tab._theme_combo.currentData())

    def response(self) -> str:
        # Internal: toolbar ``_filter_combo`` item data ('fast' / 'normal' / 'slow').
        return str(self.win._toolbar._filter_combo.currentData())

    def drone_set_volume(self, percent: int) -> None:
        # Internal: drone bar ``_drone_vol_slider`` (0..100).
        self.win._drone_bar._vol_slider.setValue(int(percent))

    def drone_volume(self) -> int:
        return int(self.win._drone_bar._vol_slider.value())

    def enable_history(self) -> None:
        """Turn on history saving after the first launch, from the app's own
        controls. Looks for a checkbox carrying the welcome dialog's wording
        (``welcome_persist``); the app offers none today, so this raises."""
        from PyQt6.QtWidgets import QCheckBox, QDialog
        wording = self.t("welcome_persist")
        for cb in self.win.findChildren(QCheckBox):
            # The finished welcome dialog is still a child of the window; its
            # checkbox does nothing any more.
            node, in_dialog = cb, False
            while node is not None and node is not self.win:
                in_dialog = in_dialog or isinstance(node, QDialog)
                node = node.parentWidget()
            if cb.text() == wording and not in_dialog:
                if not cb.isChecked():
                    cb.click()
                return
        raise HarnessError("the app has no control to turn history saving on "
                           "after the first launch")

    def try_custom_instrument_names(self, names: list, transpose: int):
        """Open the custom-instrument form once and type each name in turn,
        pressing OK after each, until the form accepts one. Message boxes the
        app shows on the way are acknowledged. Returns the selected instrument
        key if it changed, else None."""
        before = self.instrument()
        with self.dialogs.scope():
            self.dialogs.expect("custom_instrument", "accept", names=list(names),
                                transpose=transpose)
            for _ in names:
                self.dialogs.expect("warning", "Ok", optional=True)
            self.win._toolbar._btn_custom.click()
        after = self.instrument()
        return after if after != before else None

    def ui_texts(self) -> list:
        """Every piece of text the user can see or reach in the window:
        ``(area, role, text)`` triples.

        ``area`` is the tab key ('tuner', 'metro', 'deck', 'setup') for widgets
        on a tab page, ``dialog:<title>`` for an open child dialog and 'window'
        for the rest. ``role`` is 'title', 'tab', 'tab_tip', 'text', 'tooltip',
        'placeholder' or 'item' (a combo-box entry). Widgets that are
        explicitly hidden are skipped; pages of a tab widget count even when
        another tab is in front, since one click shows them.
        Internals: ``_tabs`` / ``_tab_keys`` and Qt widget classes.

        The scan asks Qt (isAncestorOf / isVisibleTo) where a widget sits
        instead of walking parentWidget() in Python. Walking the chain wrapped
        Qt's private container widgets, and a stale wrapper there crashed the
        interpreter now and then. Garbage collection is paused during the scan
        so no Qt object from an earlier app instance is destroyed under it.
        """
        import gc

        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import (
            QAbstractButton,
            QComboBox,
            QDialog,
            QGroupBox,
            QLabel,
            QLineEdit,
            QTabBar,
        )
        gc.collect()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        w = self.win
        tabs = getattr(w, "_tabs", None)
        pages = []
        if tabs is not None:
            pages = [(tabs.widget(i), key) for i, key in enumerate(w._tab_keys)]
        dialogs = [d for d in w.findChildren(QDialog) if not d.isHidden()]

        def placement(widget):
            """(area, reachable) for ``widget``."""
            for dlg in dialogs:
                if dlg is widget or dlg.isAncestorOf(widget):
                    return f"dialog:{dlg.windowTitle()}", widget.isVisibleTo(dlg) or dlg is widget
            for page, key in pages:
                if page.isAncestorOf(widget):
                    return key, widget.isVisibleTo(page)
            if any(d.isAncestorOf(widget) for d in w.findChildren(QDialog)):
                return "window", False   # inside a closed dialog
            return "window", not widget.isHidden() and widget.isVisibleTo(w) or w.isHidden()

        out = [("window", "title", w.windowTitle())]
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for dlg in dialogs:
                out.append((f"dialog:{dlg.windowTitle()}", "title", dlg.windowTitle()))
            for cls in (QTabBar, QLabel, QAbstractButton, QGroupBox, QComboBox, QLineEdit):
                for widget in w.findChildren(cls):
                    area, reachable = placement(widget)
                    if not reachable:
                        continue
                    if cls is QTabBar:
                        for i in range(widget.count()):
                            out.append((area, "tab", widget.tabText(i)))
                            out.append((area, "tab_tip", widget.tabToolTip(i)))
                    elif cls in (QLabel, QAbstractButton):
                        out.append((area, "text", widget.text()))
                    elif cls is QGroupBox:
                        out.append((area, "title", widget.title()))
                    elif cls is QComboBox:
                        for i in range(widget.count()):
                            out.append((area, "item", widget.itemText(i)))
                    elif cls is QLineEdit:
                        out.append((area, "placeholder", widget.placeholderText()))
                    out.append((area, "tooltip", widget.toolTip()))
        finally:
            if gc_was_enabled:
                gc.enable()
        return [(a, r, t) for a, r, t in out if t and t.strip()]


def i18n_texts() -> dict:
    """``{lang: {key: format_string}}``: the app's UI wording. Needs no app."""
    from sax_i18n import STRINGS
    return {lang: dict(table) for lang, table in STRINGS.items()}

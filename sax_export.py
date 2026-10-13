# =============================================================================
# sax_export.py — ExportController
#
# The controller owns the four file-export flows (TXT / PDF / CSV / PNG
# chart) and their dialogs: the maker/model prompt, the CSV slice picker,
# the save dialog, the overwrite question and the messages afterwards.
# The files themselves are drawn by the pure renderers in sax_report from a
# SessionSnapshot taken once per export; the CSV slices come from
# MeasurementLog.export_csv.
#
# Design notes:
#   * Live values that change during a session (the active instrument,
#     the per-note stats dict, the UI language, the user-entered
#     maker/model/nickname) reach the controller through getter
#     callables, not via fixed references captured at construction
#     time. That way the controller reads the *current* value at
#     export time rather than freezing whatever was set on the
#     window when __init__ ran.
#   * The controller never mutates engine state and never calls
#     sax_config.save_config — the window owns persistence timing.
#   * `_instr_info_asked`, `_last_maker`, `_last_model` are read
#     and written on the parent window (self._w) so the existing
#     reset hook on MainWindow that clears the cache keeps working
#     without modification.
#   * The dialog helpers stay package-private on the controller
#     (`_ask_csv_slice`, `_ask_instrument_model`). MainWindow does
#     not call them directly; only `export_*` does.
# =============================================================================
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

import sax_report
from sax_intonation_log import MeasurementLog


def _today() -> str:
    """Date stamp for default export file names."""
    return datetime.date.today().strftime('%Y%m%d')


if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

    from sax_config import AppConfig


class ExportController:
    """Owns the TXT / PDF / CSV / PNG-chart export flows plus their
    dialogs.  Reads MeasurementLog + AppConfig + the live stats dict and
    hands a SessionSnapshot to sax_report.  Never mutates engine state."""

    def __init__(self, window: 'QMainWindow', *,
                 log: MeasurementLog, cfg: 'AppConfig', engine,
                 t_func: Callable[..., str],
                 get_instrument_key: Callable[[], str],
                 get_stats: Callable[[], dict],
                 get_lang: Callable[[], str],
                 get_maker: Callable[[], str],
                 get_model: Callable[[], str],
                 get_nickname: Callable[[], str]) -> None:
        self._w = window
        self._log = log
        self._cfg = cfg
        self._engine = engine
        self._t = t_func
        # Callables, not values — the active values change at runtime;
        # we want to read them at export time, not at controller init.
        self._get_instrument_key = get_instrument_key
        self._get_stats = get_stats
        self._get_lang = get_lang
        self._get_maker = get_maker
        self._get_model = get_model
        self._get_nickname = get_nickname

    # ── Public entry points ───────────────────────────────────────────────
    def export_txt(self) -> None:
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            self._w, self._t('txt_save_title'),
            f"intonation_{instrument}_{_today()}.txt",
            self._t('txt_filter'))
        if not path:
            return
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: sax_report.write_txt(
                snapshot, self._get_lang(), path)):
            self._saved(self._t('txt_saved', path=path))

    def export_pdf(self) -> None:
        if not sax_report.pdf_available():
            QMessageBox.critical(self._w, self._t('err_title'), self._t('reportlab_err'))
            return
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            self._w, self._t('pdf_save_title'),
            f"intonation_{instrument}_{_today()}.pdf",
            self._t('pdf_filter'))
        if not path:
            return
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: sax_report.render_pdf(
                snapshot, self._get_lang(), path)):
            self._saved(self._t('pdf_saved', path=path))

    def export_csv(self) -> None:
        w = self._w
        if not self._log.has_measurements():
            QMessageBox.information(w, self._t('export_title'),
                                    self._t('csv_no_data'))
            return

        # Optionally tag the current run with maker/model so they show up in
        # the CSV alongside the measurements. In-memory only — no rewrite of
        # the on-disk JSONL record.
        model_info = self._ask_instrument_model()
        if model_info is None:
            return
        maker, model = model_info
        if maker or model:
            self._log.set_current_run_metadata(maker=maker, model=model)

        sel = self._ask_csv_slice()
        if sel is None:
            return

        if sel['mode'] == 'instrument_avg' and not sel['instrument']:
            QMessageBox.warning(w, self._t('err_title'),
                                self._t('csv_need_instr'))
            return

        path, _ = QFileDialog.getSaveFileName(
            w, self._t('csv_save_title'),
            f"intonation_{sel['mode']}_{_today()}.csv",
            self._t('csv_filter'))
        if not path:
            return

        written = []

        def write():
            written.append(self._log.export_csv(path,
                                                mode=sel['mode'],
                                                run_id=sel['run_id'],
                                                instrument=sel['instrument'],
                                                nickname=sel['nickname']))
        if self._write(path, write):
            self._saved(self._t('csv_saved', rows=written[0], path=path))

    def export_chart(self) -> None:
        if not self._has_played_notes():
            QMessageBox.information(self._w, self._t('export_title'),
                                    self._t('chart_no_data'))
            return
        # Optional maker/model — same flow as TXT/PDF/CSV export.
        model_info = self._ask_instrument_model()
        if model_info is None:
            return
        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            self._w, self._t('chart_save_title'),
            f"intonation_chart_{instrument}_{_today()}.png",
            self._t('chart_filter'))
        if not path:
            return
        # Ensure the file ends in .png so QPixmap picks the right encoder.
        if not path.lower().endswith('.png'):
            path += '.png'
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: sax_report.render_chart(
                snapshot, self._get_lang(), path)):
            self._saved(self._t('chart_saved', path=path))

    # ── Shared steps ──────────────────────────────────────────────────────
    def _has_played_notes(self) -> bool:
        # Rows seeded for the instrument range have n == 0 until played.
        return any(st.n > 0 for st in self._get_stats().values())

    def _snapshot(self, maker: str, model: str) -> sax_report.SessionSnapshot:
        w = self._w
        instrument = self._get_instrument_key()
        return sax_report.take_snapshot(
            instrument=instrument,
            instrument_label=w._instr_label(instrument),
            a4=self._engine.a4,
            display=w.display,
            sample_rate=w._engine_sample_rate(),
            stats=self._get_stats(),
            maker=maker, model=model,
            nickname=self._get_nickname().strip())

    def _write(self, path: str, write: Callable[[], object]) -> bool:
        """Run ``write``; on failure show the error and return False."""
        try:
            write()
            return True
        except Exception as e:
            QMessageBox.critical(self._w, self._t('err_title'), str(e))
            return False

    def _saved(self, text: str) -> None:
        QMessageBox.information(self._w, self._t('export_title'), text)

    # ── Helper dialogs ────────────────────────────────────────────────────
    def _ask_instrument_model(self) -> tuple[str, str] | None:
        """Prompt for maker + model. After the first answer in a session,
        subsequent calls return the cached values without re-prompting.
        Reset clears the cache so a new instrument can be tagged."""
        w = self._w
        if w._instr_info_asked:
            return (w._last_maker, w._last_model)
        dlg = QDialog(w)
        dlg.setProperty('_lang_title_key', 'model_dialog_title')
        dlg.setWindowTitle(self._t('model_dialog_title'))
        dlg.setMinimumWidth(440)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #bbb; font-size: 13px; }
            QLineEdit {
                background: #252540; border: 1px solid #444; border-radius: 5px;
                color: #eee; padding: 6px 10px; font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #6699cc; }
            QDialogButtonBox QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 6px 18px; font-size: 13px;
            }
            QDialogButtonBox QPushButton:hover   { background: #3a6da8; }
            QDialogButtonBox QPushButton:pressed  { background: #1e3a5f; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info = QLabel(self._t('model_dialog_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        edit_maker = QLineEdit()
        edit_maker.setPlaceholderText(self._t('model_placeholder_maker'))
        edit_maker.setText(w._last_maker)

        edit_model = QLineEdit()
        edit_model.setPlaceholderText(self._t('model_placeholder_model'))
        edit_model.setText(w._last_model)

        maker_lbl = QLabel(self._t('model_label_maker'))
        model_lbl = QLabel(self._t('model_label_model'))
        form.addRow(maker_lbl, edit_maker)
        form.addRow(model_lbl, edit_model)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('model_dialog_info'))
            maker_lbl.setText(self._t('model_label_maker'))
            model_lbl.setText(self._t('model_label_model'))
            edit_maker.setPlaceholderText(self._t('model_placeholder_maker'))
            edit_model.setPlaceholderText(self._t('model_placeholder_model'))

        w._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        maker = edit_maker.text().strip()
        model = edit_model.text().strip()
        # Für nächsten Export merken — cache lives on the window so the
        # existing reset hook (MainWindow._on_reset) can clear it without
        # having to know about this controller.
        w._last_maker = maker
        w._last_model = model
        w._instr_info_asked = True
        return maker, model

    def _ask_csv_slice(self) -> dict | None:
        """Modal dialog: slice mode + (optional) run/instrument filter.

        Returns {'mode', 'run_id', 'instrument'} or None if cancelled.
        """
        w = self._w
        dlg = QDialog(w)
        dlg.setProperty('_lang_title_key', 'csv_dialog_title')
        dlg.setWindowTitle(self._t('csv_dialog_title'))
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #bbb; font-size: 13px; }
            QComboBox {
                background: #252540; border: 1px solid #444; border-radius: 5px;
                color: #eee; padding: 6px 10px; font-size: 13px; min-width: 260px;
            }
            QComboBox QAbstractItemView { background: #252540; color: #eee; }
            QDialogButtonBox QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 6px 18px; font-size: 13px;
            }
            QDialogButtonBox QPushButton:hover  { background: #3a6da8; }
            QDialogButtonBox QPushButton:pressed { background: #1e3a5f; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 14, 20, 16)

        # Closing the dialog and re-opening it is the path to picking up a
        # language switch — the captured labels here aren't re-bound. The
        # toggle at least lets the user notice they're in the wrong language.
        info = QLabel(self._t('csv_dialog_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)

        def relabel_slice_dialog():
            info.setText(self._t('csv_dialog_info'))
            # Mode combo: keep current selection, swap labels.
            cur_mode = mode_combo.currentData()
            mode_combo.blockSignals(True)
            mode_combo.clear()
            for key in MeasurementLog.SLICE_MODES:
                mode_combo.addItem(self._t(f'csv_mode_{key}'), key)
                if key == cur_mode:
                    mode_combo.setCurrentIndex(mode_combo.count() - 1)
            mode_combo.blockSignals(False)
            mode_lbl.setText(self._t('csv_mode_label'))
            run_lbl.setText(self._t('csv_run_label'))
            instr_lbl.setText(self._t('csv_instr_label'))
            nick_lbl.setText(self._t('csv_nick_label'))
            # Re-translate the "All …" sentinels at index 0 if present.
            if run_combo.itemData(0) is None:
                run_combo.setItemText(0, self._t('csv_all_runs'))
            if instr_combo.count() and instr_combo.itemData(0) is None:
                instr_combo.setItemText(0, self._t('csv_all_instruments'))
            if nick_combo.itemData(0) is None:
                nick_combo.setItemText(0, self._t('csv_all_nicks'))
            summary.setText(self._t('csv_summary',
                                     n=self._log.measurement_count(),
                                     runs=len(self._log.runs())))

        w._add_dialog_lang_toggle(dlg, layout, on_change=relabel_slice_dialog)

        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        mode_combo = QComboBox()
        for key in MeasurementLog.SLICE_MODES:
            mode_combo.addItem(self._t(f'csv_mode_{key}'), key)
        mode_lbl = QLabel(self._t('csv_mode_label'))
        form.addRow(mode_lbl, mode_combo)

        run_combo = QComboBox()
        run_combo.addItem(self._t('csv_all_runs'), None)
        for run in self._log.runs():
            label = (f"{run.started_at} · {w._instr_label(run.instrument)}"
                     f" · A={run.a4_hz:.0f} Hz")
            tail = ' '.join(x for x in (run.maker, run.model) if x)
            if tail:
                label += f" · {tail}"
            run_combo.addItem(label, run.run_id)
        run_lbl = QLabel(self._t('csv_run_label'))
        form.addRow(run_lbl, run_combo)

        instr_combo = QComboBox()
        instr_combo.addItem(self._t('csv_all_instruments'), None)
        for key in self._log.instruments():
            instr_combo.addItem(w._instr_label(key), key)
        instr_lbl = QLabel(self._t('csv_instr_label'))
        form.addRow(instr_lbl, instr_combo)

        # Nickname filter — collect from runs that have one. If nothing in the
        # log has a nickname yet, the combo will only show "All nicknames".
        nick_combo = QComboBox()
        nick_combo.addItem(self._t('csv_all_nicks'), None)
        seen_nicks = sorted({r.label for r in self._log.runs() if r.label})
        for nk in seen_nicks:
            nick_combo.addItem(nk, nk)
        nick_lbl = QLabel(self._t('csv_nick_label'))
        form.addRow(nick_lbl, nick_combo)

        layout.addLayout(form)

        n = self._log.measurement_count()
        r = len(self._log.runs())
        summary = QLabel(self._t('csv_summary', n=n, runs=r))
        summary.setStyleSheet('color: #888; font-size: 11px;')
        layout.addWidget(summary)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        def refresh_filters():
            mode = mode_combo.currentData()
            run_enabled = mode in ('raw', 'per_run_note')
            instr_required = mode == 'instrument_avg'
            instr_enabled = mode in ('raw', 'per_run_note',
                                     'per_instrument_note', 'instrument_avg',
                                     'per_nickname_note')
            nick_enabled = mode in ('raw', 'per_run_note',
                                    'per_instrument_note', 'per_nickname_note')

            run_combo.setEnabled(run_enabled)
            run_lbl.setEnabled(run_enabled)
            instr_combo.setEnabled(instr_enabled)
            instr_lbl.setEnabled(instr_enabled)
            nick_combo.setEnabled(nick_enabled)
            nick_lbl.setEnabled(nick_enabled)

            # In instrument_avg mode the user must pick one instrument; the
            # "All instruments" sentinel at index 0 is removed and reinstated
            # when leaving that mode.
            first_is_all = instr_combo.itemData(0) is None
            if instr_required and first_is_all:
                instr_combo.removeItem(0)
            elif not instr_required and not first_is_all:
                instr_combo.insertItem(0, self._t('csv_all_instruments'),
                                        None)
                instr_combo.setCurrentIndex(0)

        mode_combo.currentIndexChanged.connect(refresh_filters)
        refresh_filters()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return {
            'mode': mode_combo.currentData(),
            'run_id': run_combo.currentData() if run_combo.isEnabled() else None,
            'instrument': (instr_combo.currentData()
                           if instr_combo.isEnabled() else None),
            'nickname': (nick_combo.currentData()
                          if nick_combo.isEnabled() else None),
        }

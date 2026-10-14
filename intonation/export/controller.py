# =============================================================================
# intonation/export/controller.py — ExportController
#
# The controller owns the four file-export flows (TXT / PDF / CSV / PNG
# chart) and their dialogs: the maker/model prompt, the CSV slice picker,
# the save dialog, the overwrite question and the messages afterwards.
# The files themselves are drawn by the pure renderers in intonation.export.report from a
# SessionSnapshot taken once per export; the CSV slices come from
# MeasurementLog.export_csv.
#
# Design notes:
#   * What is exported (instrument, A4, display mode, nickname, the per-note
#     statistics) is read from the SessionModel at export time, so an
#     export always describes the session as it is on screen.
#   * The controller never mutates engine state and never
#     saves the settings — the window owns persistence timing.
#   * The maker/model answer is remembered here for the rest of the
#     session; Reset calls forget_instrument_info() so the next export
#     asks again.
#   * The dialog helpers stay package-private on the controller
#     (`_ask_csv_slice`, `_ask_instrument_model`). MainWindow does
#     not call them directly; only `export_*` does.
# =============================================================================
from __future__ import annotations

import datetime
import errno
from pathlib import Path
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

from intonation.export import report
from intonation.export.chart import ChartEncodeError
from intonation.persistence.history import MeasurementLog


def _today() -> str:
    """Date stamp for default export file names."""
    return datetime.date.today().strftime('%Y%m%d')


def _error_detail(exc: BaseException) -> str:
    """A short technical line for a failed save: the exception type and the
    errno / Windows error code, without the OS's English sentence, so the
    message stays in the user's language and still tells support what
    happened."""
    if isinstance(exc, OSError):
        parts = [errno.errorcode.get(exc.errno, '')] if exc.errno else []
        winerror = getattr(exc, 'winerror', None)
        if winerror:
            parts.append(f'WinError {winerror}')
        parts = [p for p in parts if p]
        return type(exc).__name__ + (f" ({', '.join(parts)})" if parts else '')
    return f'{type(exc).__name__}: {exc}'


if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

    from intonation.core.session import SessionModel
    from intonation.i18n.translator import Translator


class ExportController:
    """Owns the TXT / PDF / CSV / PNG-chart export flows plus their
    dialogs.  Reads the MeasurementLog and the SessionModel and hands a
    SessionSnapshot to intonation.export.report.  Never mutates engine state."""

    def __init__(self, parent: 'QWidget', *,
                 log: MeasurementLog, session: 'SessionModel',
                 tr: 'Translator',
                 instrument_label: Callable[[str], str],
                 sample_rate: Callable[[], int],
                 add_lang_toggle: Callable[..., None]) -> None:
        self._w = parent
        self._log = log
        self._session = session
        self._tr = tr
        self._t = tr.t
        # The instrument's name as the window shows it, and the input's live
        # sample rate (it sets the cents precision in the report).
        self._instr_label = instrument_label
        self._sample_rate = sample_rate
        # The window's DE | EN toggle for its own dialogs.
        self._add_lang_toggle = add_lang_toggle
        # The maker/model answer, asked once per session (see Reset).
        self._maker = ''
        self._model = ''
        self._instr_info_asked = False

    def set_log(self, log: MeasurementLog) -> None:
        """Export from another log (history was switched on mid-session)."""
        self._log = log

    def forget_instrument_info(self) -> None:
        """Ask for maker and model again on the next export."""
        self._instr_info_asked = False

    # ── Public entry points ───────────────────────────────────────────────
    def export_txt(self) -> None:
        if not self._has_played_notes():
            QMessageBox.information(self._w, self._t('export_title'),
                                    self._t('txt_no_data'))
            return
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        instrument = self._session.instrument
        path = self._ask_save_path(
            'txt_save_title', f"intonation_{instrument}_{_today()}.txt",
            'txt_filter', '.txt')
        if path is None:
            return
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: report.write_txt(
                snapshot, self._tr.lang, path)):
            self._saved(self._t('txt_saved', path=path))

    def export_pdf(self) -> None:
        if not report.pdf_available():
            QMessageBox.critical(self._w, self._t('err_title'), self._t('reportlab_err'))
            return
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        instrument = self._session.instrument
        path = self._ask_save_path(
            'pdf_save_title', f"intonation_{instrument}_{_today()}.pdf",
            'pdf_filter', '.pdf')
        if path is None:
            return
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: report.render_pdf(
                snapshot, self._tr.lang, path)):
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

        path = self._ask_save_path(
            'csv_save_title', f"intonation_{sel['mode']}_{_today()}.csv",
            'csv_filter', '.csv')
        if path is None:
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
        instrument = self._session.instrument
        path = self._ask_save_path(
            'chart_save_title', f"intonation_chart_{instrument}_{_today()}.png",
            'chart_filter', '.png')
        if path is None:
            return
        snapshot = self._snapshot(*model_info)
        if self._write(path, lambda: report.render_chart(
                snapshot, self._tr.lang, path)):
            self._saved(self._t('chart_saved', path=path))

    # ── Shared steps ──────────────────────────────────────────────────────
    def _has_played_notes(self) -> bool:
        return self._session.has_played_notes()

    def _snapshot(self, maker: str, model: str) -> report.SessionSnapshot:
        session = self._session
        instrument = session.instrument
        return report.take_snapshot(
            instrument=instrument,
            instrument_label=self._instr_label(instrument),
            a4=session.a4,
            display=session.display,
            sample_rate=self._sample_rate(),
            stats=session.stats,
            maker=maker, model=model,
            nickname=session.label)

    def _ask_save_path(self, title_key: str, default_name: str,
                       filter_key: str, ext: str) -> str | None:
        """Ask for a file name; None when the user cancels.

        A name typed without the extension gets it here. The save dialog
        only confirmed replacing the name it was given, so when that name
        changed and the completed one already exists, ask again before
        anything overwrites it.
        """
        path, _ = QFileDialog.getSaveFileName(
            self._w, self._t(title_key), default_name, self._t(filter_key))
        if not path:
            return None
        if path.lower().endswith(ext):
            return path
        path += ext
        target = Path(path)
        if target.exists():
            answer = QMessageBox.question(
                self._w, self._t('export_overwrite_title'),
                self._t('export_overwrite_body',
                        name=target.name, folder=str(target.parent)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return None
        return path

    def _write(self, path: str, write: Callable[[], object]) -> bool:
        """Run ``write``; on failure explain it in the user's language,
        naming the file, and return False."""
        name = Path(path).name
        try:
            write()
            return True
        except PermissionError as e:
            # Most often the old file is open in Excel or a PDF viewer
            # (Windows refuses to replace it) or the folder is read-only.
            text = self._t('export_err_locked', name=name)
            detail = e
        except OSError as e:
            text = self._t('export_err_os', name=name)
            detail = e
        except ChartEncodeError as e:
            text = self._t('chart_err_encode', name=name, fmt=e.fmt)
            detail = e
        except Exception as e:
            text = self._t('export_err_other', name=name)
            detail = e
        QMessageBox.critical(
            self._w, self._t('err_title'),
            text + '\n\n' + self._t('export_err_detail', detail=_error_detail(detail)))
        return False

    def _saved(self, text: str) -> None:
        QMessageBox.information(self._w, self._t('export_title'), text)

    # ── Helper dialogs ────────────────────────────────────────────────────
    def _ask_instrument_model(self) -> tuple[str, str] | None:
        """Prompt for maker + model. After the first answer in a session,
        subsequent calls return the cached values without re-prompting.
        Reset clears the cache so a new instrument can be tagged."""
        w = self._w
        if self._instr_info_asked:
            return (self._maker, self._model)
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
        edit_maker.setText(self._maker)

        edit_model = QLineEdit()
        edit_model.setPlaceholderText(self._t('model_placeholder_model'))
        edit_model.setText(self._model)

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

        self._add_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        maker = edit_maker.text().strip()
        model = edit_model.text().strip()
        # Für nächsten Export merken.
        self._maker = maker
        self._model = model
        self._instr_info_asked = True
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

        self._add_lang_toggle(dlg, layout, on_change=relabel_slice_dialog)

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
            label = (f"{run.started_at} · {self._instr_label(run.instrument)}"
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
            instr_combo.addItem(self._instr_label(key), key)
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

# =============================================================================
# sax_export.py — ExportController
#
# Extracted from sax_intonation_gui.py in Phase 5 of the refactor. The
# controller owns the four file-export flows (TXT / PDF / CSV / PNG
# chart) and the two helper dialogs (CSV slice picker,
# instrument-model prompt) that used to be inlined on MainWindow.
#
# Design notes:
#   * Live values that change during a session (the active instrument,
#     the per-note stats dict, the UI language, the user-entered
#     maker/model/nickname) reach the controller through getter
#     callables, not via fixed references captured at construction
#     time. That way the controller reads the *current* value at
#     export time rather than freezing whatever was set on the
#     window when __init__ ran.
#   * The controller is pure-presentation: it reads MeasurementLog +
#     AppConfig + the live stats dict, prompts the user via
#     QFileDialog, and writes a file on disk. It never mutates
#     engine state and never calls sax_config.save_config — the
#     window owns persistence timing.
#   * `_instr_info_asked`, `_last_maker`, `_last_model` are read
#     and written on the parent window (self._w) so the existing
#     reset hook on MainWindow that clears the cache keeps working
#     without modification.
#   * The dialog helpers stay package-private on the controller
#     (`_ask_csv_slice`, `_ask_instrument_model`). MainWindow does
#     not call them directly; only `export_*` does.
#   * No behavioural change vs. the inlined implementation — file
#     formats, dialog text, slice modes and chart parameters are
#     preserved verbatim.
# =============================================================================
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QComboBox,
    QDialogButtonBox, QFileDialog, QMessageBox,
)

import sax_instruments
from sax_intonation_log import MeasurementLog
from sax_intonation_chart import render_intonation_chart
from sax_i18n import STRINGS

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow
    from sax_config import AppConfig


class ExportController:
    """Owns the TXT / PDF / CSV / PNG-chart export flows plus their
    helper dialogs.  Pure-presentation: reads MeasurementLog +
    AppConfig + the live stats dict, produces a file on disk via
    QFileDialog.  Never mutates engine state."""

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
        # Local imports keep the module decoupled from the GUI file's
        # internal helpers while still using the same implementations.
        from sax_intonation_gui import (
            TRANSP_MAP, CHROMA, format_cents, midi_note_name, _today,
        )
        w = self._w
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        maker, model = model_info

        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            w, self._t('txt_save_title'),
            f"intonation_{instrument}_{_today()}.txt",
            self._t('txt_filter'))
        if not path:
            return
        transp = TRANSP_MAP.get(instrument, 0)
        instr_key = f'instr_long_{instrument}'
        lines = [
            self._t('txt_header'),
            '=' * 54,
            self._t('txt_instr', name=self._t(instr_key)),
        ]
        if maker:
            lines.append(self._t('txt_maker', maker=maker))
        if model:
            lines.append(self._t('txt_model', model=model))
        lines += [
            (self._t('txt_transp', note=CHROMA[transp % 12])
             if transp else self._t('txt_no_transp')),
            self._t('txt_a4', hz=self._engine.a4),
            self._t('txt_date', dt=datetime.datetime.now().strftime('%d.%m.%Y %H:%M')),
            '',
            self._t('txt_col_header',
                    fingered=self._t('pdf_col_finger'),
                    sounding=self._t('pdf_col_sound'),
                    mean=self._t('txt_col_mean'),
                    std=self._t('txt_col_std'),
                    n='N'),
            '-' * 62,
        ]
        with w._lock:
            items = sorted(self._get_stats().items())
        sr_now = w._engine_sample_rate()
        a4 = self._engine.a4
        for midi_kl, st in items:
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            mean_str = format_cents(st.mean, note_freq, sr_now)
            std_str = format_cents(st.std, note_freq, sr_now).lstrip('+-')
            lines.append(
                f"{midi_note_name(midi_kl - transp):<12} {midi_note_name(midi_kl):<12}"
                f" {mean_str:>7}   {std_str:>6}  {st.n:>5}  {self._make_bar(st.mean, 24)}")
        lines += ['', self._t('txt_total', total=sum(s.n for _,s in items), notes=len(items))]
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            QMessageBox.information(w, self._t('export_title'), self._t('txt_saved', path=path))
        except Exception as e:
            QMessageBox.critical(w, self._t('err_title'), str(e))

    def export_pdf(self) -> None:
        from sax_intonation_gui import (
            TRANSP_MAP, CHROMA, format_cents, midi_note_name, _today,
        )
        w = self._w
        try:
            from reportlab.lib.pagesizes import A4 as RL_A4
            from reportlab.lib import colors
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            QMessageBox.critical(w, self._t('err_title'), self._t('reportlab_err'))
            return

        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        maker, model = model_info

        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            w, self._t('pdf_save_title'),
            f"intonation_{instrument}_{_today()}.pdf",
            self._t('pdf_filter'))
        if not path:
            return

        transp = TRANSP_MAP.get(instrument, 0)
        doc    = SimpleDocTemplate(path, pagesize=RL_A4,
                                   leftMargin=20*mm, rightMargin=20*mm,
                                   topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story  = []

        ts_title = ParagraphStyle('T', parent=styles['Title'],
                                  fontSize=18, textColor=colors.HexColor('#1a237e'), spaceAfter=4)
        ts_sub   = ParagraphStyle('S', parent=styles['Normal'],
                                  fontSize=11, textColor=colors.HexColor('#555'), spaceAfter=2)

        story.append(Paragraph(self._t('pdf_title'), ts_title))
        story.append(Paragraph(self._t(f'instr_long_{instrument}'), ts_sub))
        if maker:
            story.append(Paragraph(self._t('pdf_maker', maker=maker), ts_sub))
        if model:
            story.append(Paragraph(self._t('pdf_model', model=model), ts_sub))
        story.append(Paragraph(
            (self._t('pdf_transp', note=CHROMA[transp % 12], n=transp)
             if transp else self._t('pdf_no_transp')), ts_sub))
        story.append(Paragraph(self._t('pdf_a4', hz=self._engine.a4), ts_sub))
        story.append(Paragraph(
            self._t('pdf_created', dt=datetime.datetime.now().strftime('%d.%m.%Y %H:%M')), ts_sub))
        story.append(Spacer(1, 10*mm))

        with w._lock:
            items = sorted(self._get_stats().items())

        if not items:
            story.append(Paragraph(self._t('pdf_no_data'), styles['Normal']))
        else:
            total_n = sum(s.n for _, s in items)
            story.append(Paragraph(
                self._t('pdf_summary', notes=len(items), total=total_n),
                ParagraphStyle('I', parent=styles['Normal'],
                               fontSize=10, textColor=colors.HexColor('#666'), spaceAfter=6)))

            data = [[self._t('pdf_col_finger'), self._t('pdf_col_sound'),
                     self._t('pdf_col_mean'), self._t('pdf_col_std'),
                     self._t('pdf_col_n'), self._t('pdf_col_tend')]]
            sr_now = w._engine_sample_rate()
            a4 = self._engine.a4
            for midi_kl, st in items:
                note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
                mean_str = format_cents(st.mean, note_freq, sr_now)
                std_str = format_cents(st.std, note_freq, sr_now).lstrip('+-')
                data.append([
                    midi_note_name(midi_kl - transp), midi_note_name(midi_kl),
                    mean_str,
                    f"±{std_str}" if st.n > 1 else '–',
                    str(st.n),
                    self._make_bar_ascii(st.mean),
                ])

            tbl = Table(data, colWidths=[30*mm, 30*mm, 28*mm, 22*mm, 16*mm, 44*mm], repeatRows=1)
            ts = TableStyle([
                ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#1a237e')),
                ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
                ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',     (0,0), (-1,-1), 10),
                ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
                ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),
                 [colors.HexColor('#f5f5ff'), colors.HexColor('#eeeeff')]),
                ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor('#aaaacc')),
                ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
                ('TOPPADDING',   (0,0), (-1,-1), 4),
                ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ])
            for r, (_, st) in enumerate(items, 1):
                c = (colors.HexColor('#1b5e20') if abs(st.mean) <= 5 else
                     colors.HexColor('#e65100') if abs(st.mean) <= 12 else
                     colors.HexColor('#b71c1c'))
                ts.add('TEXTCOLOR', (2,r), (2,r), c)
                ts.add('FONTNAME',  (2,r), (2,r), 'Helvetica-Bold')
            tbl.setStyle(ts)
            story.append(tbl)

        try:
            doc.build(story)
            QMessageBox.information(w, self._t('export_title'), self._t('pdf_saved', path=path))
        except Exception as e:
            QMessageBox.critical(w, self._t('err_title'), str(e))

    def export_csv(self) -> None:
        from sax_intonation_gui import _today
        w = self._w
        if not self._log.measurements():
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

        try:
            n = self._log.export_csv(path,
                                     mode=sel['mode'],
                                     run_id=sel['run_id'],
                                     instrument=sel['instrument'],
                                     nickname=sel['nickname'])
            QMessageBox.information(w, self._t('export_title'),
                                    self._t('csv_saved', rows=n, path=path))
        except Exception as e:
            QMessageBox.critical(w, self._t('err_title'), str(e))

    def export_chart(self) -> None:
        from sax_intonation_gui import (
            TRANSP_MAP, midi_note_name, _today,
        )
        w = self._w
        with w._lock:
            items = sorted(self._get_stats().items())

        if not items:
            QMessageBox.information(w, self._t('export_title'),
                                    self._t('chart_no_data'))
            return

        # Optional maker/model — same flow as TXT/PDF/CSV export.
        model_info = self._ask_instrument_model()
        if model_info is None:
            return
        maker, model = model_info

        instrument = self._get_instrument_key()
        path, _ = QFileDialog.getSaveFileName(
            w, self._t('chart_save_title'),
            f"intonation_chart_{instrument}_{_today()}.png",
            self._t('chart_filter'))
        if not path:
            return
        # Ensure the file ends in .png so QPixmap picks the right encoder.
        if not path.lower().endswith('.png'):
            path += '.png'

        transp = TRANSP_MAP.get(instrument, 0)
        disp_griff = (w.display == 'griff')
        sr_now = w._engine_sample_rate()
        a4 = self._engine.a4
        # v0.6 Phase-4 (Item 7): pass fingered MIDI per note so the chart
        # renderer can colour out-of-range bars distinctly.
        lo_f, hi_f = sax_instruments.fingered_range(instrument)
        notes = []
        for midi_kl, st in items:
            midi_gr = midi_kl - transp
            display_name = (midi_note_name(midi_gr) if disp_griff
                            else midi_note_name(midi_kl))
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            in_range = (lo_f <= midi_gr <= hi_f)
            notes.append((display_name, st.mean, st.std, st.n, note_freq,
                          midi_gr, in_range))

        instr_long = self._t(f'instr_long_{instrument}')
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        if maker or model:
            subtitle = self._t('chart_subtitle_id',
                                instr=instr_long,
                                maker=maker, model=model,
                                a4=self._engine.a4, dt=dt)
        else:
            subtitle = self._t('chart_subtitle',
                                instr=instr_long,
                                a4=self._engine.a4, dt=dt)
        total = sum(s.n for _, s in items)
        footer = self._t('chart_footer', notes=len(items), total=total)

        try:
            render_intonation_chart(
                notes=notes,
                title=self._t('chart_title'),
                subtitle=subtitle,
                footer=footer,
                output_path=path,
                sample_rate=sr_now,
                instrument=instrument,
            )
            QMessageBox.information(w, self._t('export_title'),
                                    self._t('chart_saved', path=path))
        except Exception as e:
            QMessageBox.critical(w, self._t('err_title'), str(e))

    # ── Helper dialogs ────────────────────────────────────────────────────
    def _ask_instrument_model(self) -> tuple[str, str] | None:
        """Prompt for maker + model. After the first answer in a session,
        subsequent calls return the cached values without re-prompting.
        Reset clears the cache so a new instrument can be tagged."""
        w = self._w
        if getattr(w, '_instr_info_asked', False):
            return (getattr(w, '_last_maker', ''),
                    getattr(w, '_last_model', ''))
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
        if hasattr(w, '_last_maker'):
            edit_maker.setText(w._last_maker)

        edit_model = QLineEdit()
        edit_model.setPlaceholderText(self._t('model_placeholder_model'))
        if hasattr(w, '_last_model'):
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
                                     n=len(self._log.measurements()),
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

        n = len(self._log.measurements())
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

    # ── Export-only formatting helpers ────────────────────────────────────
    @staticmethod
    def _make_bar(cents, w=20):
        half = w // 2
        fill = int(min(1.0, abs(cents) / 40.0) * half)
        if cents > 1:
            return ' '*half + '│' + '█'*fill + '░'*(half-fill)
        elif cents < -1:
            return '░'*(half-fill) + '█'*fill + '│' + ' '*half
        return ' '*half + '│' + ' '*half

    @staticmethod
    def _make_bar_ascii(cents, w=16):
        half = w // 2
        fill = int(min(1.0, abs(cents) / 40.0) * half)
        if cents > 1:  return ' '*half + '|' + '#'*fill + '.'*(half-fill)
        if cents < -1: return '.'*(half-fill) + '#'*fill + '|' + ' '*half
        return ' '*half + '|' + ' '*half

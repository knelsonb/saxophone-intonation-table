# =============================================================================
# sax_tab_tuner.py — the TUNER tab
#
# Left: the window's banners (placed here, owned by the window), the tuner
# readout and its status line, the drone bar, and the optional spectrum
# analyzer + diagnostics panels. Right: the intonation table with its
# caption, the minimum-count filter and the "filter to instrument range"
# switch.
#
# The table shows the SessionModel's statistics in one of two layouts:
#   'single' — one row per note (played notes and the instrument's range);
#   'matrix' — piano-roll: 12 chromatic rows × one column per octave, covering
#              the instrument's range (cells outside it are greyed out).
# The layout follows the player's choice (cfg.layout_mode_preference) or,
# on 'auto', the width the table has. It repaints every 300 ms, on a
# language or display switch, and when the window asks (refresh_table).
#
# The window feeds readings in through show_reading(); everything else this
# tab shows comes from the session and the settings it is given.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sax_audio_engine import AUDIO_OK
from sax_pitch import format_cents, midi_note_name
from sax_table import MATRIX_COL_WIDTH, MATRIX_HEADER_W, TableController
from sax_widgets import (
    CentBarDelegate,
    DataPanelWidget,
    SpectrumAnalyzerWidget,
    TunerWidget,
)

if TYPE_CHECKING:
    from sax_audio_engine import AudioEngine
    from sax_config import AppConfig, ConfigStore
    from sax_session import SessionModel
    from sax_translator import Translator

# The table repaints this often from the session's statistics.
REFRESH_INTERVAL_MS = 300
# Hysteresis: enter matrix at the calculated threshold, drop back to single
# only when noticeably narrower. Prevents thrash near the edge.
MATRIX_HYSTERESIS = 48    # px


class TunerTab(QWidget):
    """The TUNER tab; see the module comment."""

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 store: 'ConfigStore', engine: 'AudioEngine',
                 session: 'SessionModel', banners: Sequence[QWidget],
                 drone_bar: QWidget, sample_rate: Callable[[], int],
                 notes_count: Callable[[], int],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._cfg = cfg
        self._store = store
        self._session = session
        self._sample_rate = sample_rate
        self._layout_mode = 'single'
        # The splitter's widths are saved and restored with the session.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(6)
        self.splitter.addWidget(self._build_left(engine, banners, drone_bar, notes_count))
        self.splitter.addWidget(self._build_right())
        self.splitter.setSizes([420, 620])
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.splitter)

        self._table_ctrl = TableController(
            table=self._table, caption=self._caption,
            bar_delegate=self._bar_delegate, cfg=cfg, session=session,
            tr=tr, sample_rate=sample_rate)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self.refresh_table)
        # The captions are drawn per refresh, in the language of the moment.
        tr.language_changed.connect(lambda _lang: self.refresh_table())
        session.display_changed.connect(lambda _mode: self.refresh_table())

    @property
    def layout_mode(self) -> str:
        """'single' or 'matrix': the layout the table shows now."""
        return self._layout_mode

    @property
    def table_controller(self) -> TableController:
        return self._table_ctrl

    # ── Building ─────────────────────────────────────────────────────────────
    def _build_left(self, engine, banners, drone_bar, notes_count) -> QWidget:
        tr = self._tr
        left = QWidget()
        box = QVBoxLayout(left)
        box.setContentsMargins(0, 0, 6, 0)
        for banner in banners:
            box.addWidget(banner)

        self._tuner = TunerWidget()
        # Expanding horizontally but only Preferred vertically: the tuner
        # takes its full minimum height (260px) at the top, and the spectrum
        # analyzer claims the space left over when the panels are visible.
        self._tuner.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
        box.addWidget(self._tuner)
        self._status_lbl = QLabel()
        tr.bind(self._status_lbl.setText, 'no_signal')
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setObjectName('dimLabel')   # themed via QLabel#dimLabel
        box.addWidget(self._status_lbl)

        # The drone strip sits directly under the readout — quick access while
        # tuning against it.
        box.addWidget(drone_bar)

        # "Show spectrum analyzer & diagnostics", next to the panels it shows.
        footer = QHBoxLayout()
        footer.setContentsMargins(4, 0, 4, 0)
        self._cb_diag = QCheckBox()
        tr.bind(self._cb_diag.setText, 'show_diagnostics')
        self._cb_diag.setChecked(self._cfg.show_diagnostics)
        tr.bind(self._cb_diag.setToolTip, 'show_diagnostics_tip')
        self._cb_diag.toggled.connect(self._on_diagnostics_toggled)
        footer.addWidget(self._cb_diag)
        footer.addStretch()
        box.addLayout(footer)

        # Always constructed, so the toggle is just show/hide: no
        # reconstruction cost when flipped repeatedly. They poll only while
        # visible.
        live_engine = engine if AUDIO_OK else None
        self._spectro_grp = QGroupBox()
        tr.bind(self._spectro_grp.setTitle, 'spectro_title')
        spectro_box = QVBoxLayout(self._spectro_grp)
        spectro_box.setContentsMargins(6, 6, 6, 6)
        self._spectro = SpectrumAnalyzerWidget(live_engine)
        spectro_box.addWidget(self._spectro)
        box.addWidget(self._spectro_grp, 1)

        self._data_grp = QGroupBox()
        tr.bind(self._data_grp.setTitle, 'data_panel_title')
        data_box = QVBoxLayout(self._data_grp)
        data_box.setContentsMargins(6, 6, 6, 6)
        self._data_panel = DataPanelWidget(live_engine, tr, notes_count,
                                           lambda: self._cfg)
        data_box.addWidget(self._data_panel)
        box.addWidget(self._data_grp)

        self._spectro_grp.setVisible(self._cfg.show_diagnostics)
        self._data_grp.setVisible(self._cfg.show_diagnostics)
        return left

    def _build_right(self) -> QWidget:
        tr = self._tr
        right = QWidget()
        box = QVBoxLayout(right)
        box.setContentsMargins(6, 0, 0, 0)
        # Written by the table controller on every refresh (and so in the
        # current language), not bound here.
        self._caption = QLabel(tr.t('table_empty_hint'))
        self._caption.setStyleSheet('font-size:14px;font-weight:bold;padding:2px 0 6px 0;')
        self._caption.setWordWrap(True)
        box.addWidget(self._caption)

        table = self._table = QTableView()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setObjectName('matrix')   # themed via QTableView#matrix
        box.addWidget(table)
        self._bar_delegate = CentBarDelegate(table, sample_rate_getter=self._sample_rate)
        table.setItemDelegateForColumn(5, self._bar_delegate)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_context_menu)

        # Min-N filter: hide notes with fewer measurements than the threshold,
        # so single accidental blips don't pollute the view. Default 5 =
        # "actually held", the autotune requirement too.
        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 0)
        min_n_lbl = QLabel()
        tr.bind(min_n_lbl.setText, 'min_n_label')
        min_n_lbl.setObjectName('dimLabel')      # themed via QLabel#dimLabel
        self._min_n_spin = QSpinBox()
        self._min_n_spin.setRange(0, 999)
        self._min_n_spin.setValue(self._cfg.min_n_visible)
        tr.bind(self._min_n_spin.setToolTip, 'min_n_tip')
        self._min_n_spin.setMinimumWidth(70)
        self._min_n_spin.setObjectName('minNspin')     # themed via QSpinBox#minNspin
        self._min_n_spin.valueChanged.connect(self._on_min_n_changed)
        row.addWidget(min_n_lbl)
        row.addWidget(self._min_n_spin)
        # "Filter to instrument range" sits beside the min-N filter: both
        # gate what the table shows. Checked = filter ON (stored inverted as
        # cfg.allow_out_of_range).
        self._cb_oor = QCheckBox()
        tr.bind(self._cb_oor.setText, 'allow_oor')
        self._cb_oor.setChecked(not self._cfg.allow_out_of_range)
        tr.bind(self._cb_oor.setToolTip, 'allow_oor_tip')
        self._cb_oor.toggled.connect(self._on_oor_toggled)
        row.addSpacing(16)
        row.addWidget(self._cb_oor)
        row.addStretch()
        box.addLayout(row)
        return right

    # ── Readout ──────────────────────────────────────────────────────────────
    def show_reading(self, midi_sounding: int, freq: float, cents: float) -> None:
        """A note was measured: show it on the tuner and the status line."""
        session = self._session
        fingered = midi_note_name(midi_sounding - session.transposition)
        sounding = midi_note_name(midi_sounding)
        sr = self._sample_rate()
        self._tuner.sample_rate = sr
        self._tuner.set_note(fingered if session.display == 'griff' else sounding,
                             freq, cents)
        self._tr.bind(self._status_lbl.setText,
                      'status_fmt', fingered=fingered, sounding=sounding,
                      freq=freq, cents_str=format_cents(cents, freq, sr),
                      a4=session.a4)

    def show_status(self, key: str, **kwargs) -> None:
        """Put a message on the status line under the tuner."""
        self._tr.bind(self._status_lbl.setText, key, **kwargs)

    def set_sample_rate(self, sr: int) -> None:
        """The input runs at ``sr``: the tuner's cents precision follows it."""
        self._tuner.sample_rate = sr

    # ── Table ────────────────────────────────────────────────────────────────
    def start_refreshing(self) -> None:
        self._refresh_timer.start()

    def stop_refreshing(self) -> None:
        self._refresh_timer.stop()

    def refresh_table(self) -> None:
        """Choose the layout for the width the table has now (the choice
        depends on the live viewport), then repaint."""
        desired = self._desired_layout_mode()
        if desired != self._layout_mode or self._table_ctrl.mode is None:
            self._layout_mode = desired
            self._table_ctrl.configure_for_mode(desired)
        self._table_ctrl.refresh()

    def _desired_layout_mode(self) -> str:
        """The player's choice first; on 'auto', matrix as long as the
        viewport fits at least two octave columns, else the single list for
        readability. Matrix never truncates a playable note — wider
        instruments scroll horizontally instead of dropping back to single."""
        pref = self._cfg.layout_mode_preference
        if pref in ('single', 'matrix'):
            return pref
        width = self._table.viewport().width()
        if width <= 0:
            width = self._table.width()
        if width <= 0:
            return 'single'
        floor = MATRIX_HEADER_W + 2 * MATRIX_COL_WIDTH
        if self._layout_mode == 'matrix':
            return 'matrix' if width >= floor - MATRIX_HYSTERESIS else 'single'
        return 'matrix' if width >= floor else 'single'

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # Defer the layout decision one event-loop tick so the splitter has
        # passed the new size on to the table's viewport before it is read;
        # otherwise a quick shrink can leave the table in matrix mode.
        QTimer.singleShot(0, self.refresh_table)

    def _on_context_menu(self, pos) -> None:
        """Right-click a played note: offer to discard its readings."""
        table = self._table
        index = table.indexAt(pos)
        if not index.isValid():
            return
        session = self._session
        transp = session.transposition
        # The note the clicked row (or matrix cell, through the octave span
        # on screen) shows.
        midi_kl = self._table_ctrl.model.midi_at(index.row(), index.column())
        if midi_kl is None:
            return
        if self._layout_mode == 'matrix':
            # Only offer the action if this cell actually holds played data.
            st = session.stats.get(midi_kl)
            if st is None or st.n == 0:
                return

        t = self._tr.t
        note = f"{midi_note_name(midi_kl - transp)} / {midi_note_name(midi_kl)}"
        menu = QMenu(self)   # themed via the global QMenu rule in build_app_qss
        action = menu.addAction(f"\U0001f5d1  {t('ctx_discard')}")
        action.setData(midi_kl)
        chosen = menu.exec(table.viewport().mapToGlobal(pos))
        if chosen and chosen.data() == midi_kl:
            if QMessageBox.question(
                self, t('reset_title'), t('ctx_discard_confirm', note=note),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                session.discard(midi_kl)
                self.refresh_table()

    # ── Settings ─────────────────────────────────────────────────────────────
    def _on_diagnostics_toggled(self, checked: bool) -> None:
        self._cfg.show_diagnostics = bool(checked)
        self._store.save(self._cfg)
        self._spectro_grp.setVisible(bool(checked))
        self._data_grp.setVisible(bool(checked))
        self.refresh_table()

    def _on_min_n_changed(self, value: int) -> None:
        # A display gate only: readings are still collected; rows just hide
        # until they have enough.
        self._cfg.min_n_visible = max(0, int(value))
        self._store.save(self._cfg)
        self.refresh_table()

    def _on_oor_toggled(self, checked: bool) -> None:
        # Refresh now, so the matrix shrinks back to the nominal range as soon
        # as the filter goes on.
        self._cfg.allow_out_of_range = not checked
        self._store.save(self._cfg)
        self.refresh_table()

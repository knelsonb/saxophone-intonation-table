# =============================================================================
# sax_table.py — TableController
#
# Extracted from sax_intonation_gui.py in Phase 5 of the refactor. The
# controller owns the intonation table's two view modes (single-column
# list and octave-by-pitch-class matrix), the per-cell paint pipeline,
# and the 300 ms QTimer-driven refresh cycle.
#
# Design notes:
#   * Live values that change during a session (the active instrument,
#     the per-note stats dict, the display mode, the A4 reading) reach
#     the controller through getter callables, not via fixed references
#     captured at construction time. That way every refresh tick reads
#     the *current* values rather than ones frozen at __init__.
#   * v0.6 (Phase 5) lands Legolas W8: cell QTableWidgetItem objects are
#     allocated ONCE per layout change in configure_for_mode().  The
#     refresh() body methods read self._cells[(r, c)] and mutate the
#     already-existing item via setText / setData / setBackground /
#     setForeground.  Previously every 300 ms tick allocated up to 84
#     fresh QTableWidgetItem objects (12 pitch classes × 7 octaves in
#     matrix mode), at ~16,800 Qt allocations per minute under a held
#     practice session.  Now: zero allocations on the refresh hot path.
#   * The QTimer ownership stays on MainWindow.  The window's
#     _refresh_table delegates to TableController.refresh() after
#     deciding the layout mode — keeping the timer wiring local to
#     MainWindow preserves Qt signal lifetimes and the resizeEvent
#     debounce path.
#   * No behavioural change vs. the inlined implementation — colors,
#     formatting, NaN guards, header labels, min-N gating, OOR
#     handling, and the active-cell highlight are preserved verbatim.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QStyledItemDelegate,
)

import sax_instruments

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow, QTableWidget
    from sax_config import AppConfig


class TableController:
    """Owns the intonation table's two view modes (single-column list and
    octave-by-pitch-class matrix), the per-cell paint pipeline, and the
    QTimer-driven 300ms refresh cycle.

    v0.6 (Phase 5): cell items are allocated ONCE per layout change in
    configure_for_mode() — subsequent refresh ticks only mutate the
    already-existing QTableWidgetItem instances via setText/setData/
    setBackground.  Previously every 300ms tick allocated up to 84 fresh
    QTableWidgetItem objects (12 rows × 7 octaves in matrix mode), at
    ~16,800 Qt object allocations per minute under a held practice
    session.  Legolas W8.
    """

    def __init__(self, window: 'QMainWindow', *,
                 table: 'QTableWidget',
                 cfg: 'AppConfig',
                 engine,
                 t_func: Callable[..., str],
                 get_instrument_key: Callable[[], str],
                 get_stats: Callable[[], dict],
                 get_display_mode: Callable[[], str],
                 get_a4: Callable[[], float]) -> None:
        self._w = window
        self._table = table
        self._cfg = cfg
        self._engine = engine
        self._t = t_func
        self._get_instrument_key = get_instrument_key
        self._get_stats = get_stats
        self._get_display_mode = get_display_mode
        self._get_a4 = get_a4
        # Cell cache: (row, col) -> QTableWidgetItem.  Populated in
        # configure_for_mode(); read and mutated by refresh().
        self._cells: dict[tuple[int, int], QTableWidgetItem] = {}
        # Current view mode tag so refresh() knows which body to run.
        # None until the first configure_for_mode() call — refresh()
        # bails silently in that window so a stray pre-configure tick
        # never crashes.
        self._current_mode: str | None = None
        # Matrix mode also caches the resolved octave range that
        # configure_for_mode() set up.  refresh_matrix() reads it back
        # rather than recomputing — and a range change between configure
        # and refresh would mean the cell cache is stale anyway, which
        # the caller (MainWindow._refresh_table) handles by reconfiguring
        # whenever the layout flips.
        self._matrix_lo_oct: int = 0
        self._matrix_hi_oct: int = -1

    # -------------------------------------------------------------------------
    # Public surface
    # -------------------------------------------------------------------------
    def configure_for_mode(self, mode: str) -> None:
        """Swap the table between 'single' and 'matrix' layouts.

        Clears the cell cache, calls setRowCount / setColumnCount, then
        allocates one QTableWidgetItem per (row, col) cell and parks it
        in self._cells.  Subsequent refresh() calls mutate those cached
        items in place — no further QTableWidgetItem allocations on the
        300 ms hot path (Legolas W8).
        """
        w = self._w
        # Lazily build the delegates the way the original code did so
        # the matrix view's custom paint and the single view's bar
        # delegate both work without per-call construction.
        if not hasattr(w, '_default_delegate'):
            w._default_delegate = QStyledItemDelegate(self._table)
        # _matrix_delegate is built in MainWindow's _build_ui paths
        # historically; it may not exist yet if configure_for_mode is
        # called before that.  Build defensively.
        if not hasattr(w, '_matrix_delegate'):
            # MatrixCellDelegate lives in sax_intonation_gui; import
            # lazily to avoid a circular import at module load.
            from sax_intonation_gui import MatrixCellDelegate
            w._matrix_delegate = MatrixCellDelegate(
                self._table, sample_rate_getter=w._engine_sample_rate)

        # Drop stale items before reconfiguring.  Without this, a
        # single→matrix switch would leave the old single-mode items
        # parented to the table after setRowCount/setColumnCount churn.
        self._cells.clear()
        self._table.clear()

        if mode == 'matrix':
            lo_oct, hi_oct = self._matrix_octave_range()
            self._matrix_lo_oct = lo_oct
            self._matrix_hi_oct = hi_oct
            n_oct = hi_oct - lo_oct + 1
            self._table.setColumnCount(n_oct)
            self._table.setRowCount(12)
            self._table.verticalHeader().setVisible(True)
            self._table.verticalHeader().setDefaultSectionSize(
                w._MATRIX_ROW_HEIGHT)
            # Fixed column widths — playable notes always render at full
            # cell size. If the columns don't all fit, Qt's horizontal
            # scrollbar takes over instead of the cells getting squished.
            hh = self._table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            hh.setDefaultSectionSize(w._MATRIX_COL_WIDTH)
            for c in range(n_oct):
                self._table.setColumnWidth(c, w._MATRIX_COL_WIDTH)
            self._table.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._table.setItemDelegateForColumn(5, w._default_delegate)
            self._table.setItemDelegate(w._matrix_delegate)
            # Allocate cell items ONCE — refresh ticks reuse them.
            for r in range(12):
                for c in range(n_oct):
                    item = QTableWidgetItem('')
                    self._table.setItem(r, c, item)
                    self._cells[(r, c)] = item
        else:
            self._table.setColumnCount(6)
            self._table.verticalHeader().setVisible(False)
            self._table.verticalHeader().setDefaultSectionSize(28)
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch)
            self._table.setHorizontalHeaderLabels(self._table_headers())
            # Restore the default per-table delegate, then the bar
            # delegate for the tendency column.
            self._table.setItemDelegate(w._default_delegate)
            self._table.setItemDelegateForColumn(5, w._bar_delegate)
            # Row count for single-mode is determined per-refresh by the
            # filtered stats list — _refresh_single() calls setRowCount
            # and grows the cell cache to match.  Cache starts empty.

        self._current_mode = mode

    def refresh(self) -> None:
        """Re-paint all cells from the current stats dict.  Bails
        silently if configure_for_mode() has not been called yet — a
        stray pre-configure tick must never crash."""
        if self._current_mode is None:
            return
        if self._current_mode == 'matrix':
            self._refresh_matrix()
        else:
            self._refresh_single()

    # -------------------------------------------------------------------------
    # Matrix range helpers — public on the controller so MainWindow's
    # context-menu code (which maps a click position to a sounding MIDI)
    # can ask the controller for the current low octave without re-
    # implementing the half-step-beyond logic.
    # -------------------------------------------------------------------------
    def _matrix_octave_range(self) -> tuple[int, int]:
        """(lo_octave, hi_octave) inclusive to display for the current
        instrument.  Spans the instrument's nominal fingered range AND
        any actually-played notes outside it — overtones, altissimo, and
        accidentals get their own cells so nothing gets truncated.

        Half-step-beyond rule: if the low note is exactly a C (the start
        of its octave) we pad one column below so the B a half-step
        lower is visible; if the high note is exactly a B (the end of
        its octave) we pad one column above so the C a half-step higher
        is visible.  Extra context octaves beyond that are configurable
        via cfg.matrix_extra_octaves."""
        # Imported lazily to keep this module's import-time graph
        # decoupled from sax_intonation_gui's heavyweight imports.
        from sax_intonation_gui import TRANSP_MAP
        instrument = self._get_instrument_key()
        display = self._get_display_mode()
        transp = TRANSP_MAP.get(instrument, 0)
        lo_f, hi_f = sax_instruments.fingered_range(instrument)
        if display == 'griff':
            lo_midi, hi_midi = lo_f, hi_f
        else:
            lo_midi, hi_midi = lo_f + transp, hi_f + transp
        # Played-note expansion (only when OOR is allowed).
        with self._w._lock:
            played = [m for m, st in self._get_stats().items() if st.n > 0]
        if played and self._cfg.allow_out_of_range:
            if display == 'griff':
                played = [m - transp for m in played]
            lo_midi = min(lo_midi, min(played))
            hi_midi = max(hi_midi, max(played))
        lo_oct = lo_midi // 12 - 1
        hi_oct = hi_midi // 12 - 1
        # Half-step-beyond rule.
        if lo_midi % 12 == 0:      # low note is C → show B in the column below
            lo_oct -= 1
        if hi_midi % 12 == 11:     # high note is B → show C in the column above
            hi_oct += 1
        # Configurable extra context on each side.
        extra = max(0, int(getattr(self._cfg, 'matrix_extra_octaves', 0)))
        lo_oct -= extra
        hi_oct += extra
        # Clamp to non-negative octaves (MIDI octave -1 not useful for
        # any real instrument in this app).
        lo_oct = max(0, lo_oct)
        return (lo_oct, hi_oct)

    def _matrix_octave_count(self) -> int:
        lo, hi = self._matrix_octave_range()
        return hi - lo + 1

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _table_headers(self) -> list[str]:
        return [self._t('col_fingered'), self._t('col_sounding'),
                self._t('col_mean'), self._t('col_std'),
                self._t('col_n'), self._t('col_tendency')]

    def _active_midi_now(self):
        """Currently-played MIDI if the highlight is still fresh, else
        None.  Reads the window's _active_midi / _active_midi_at pair
        directly — they're updated from the audio thread's _on_note
        path and that wiring stays on MainWindow."""
        import datetime
        w = self._w
        if (w._active_midi_at is None
                or (datetime.datetime.now() - w._active_midi_at)
                .total_seconds() > 1.5):
            return None
        return w._active_midi

    def _ensure_single_cells(self, n_rows: int) -> None:
        """Grow / shrink the single-mode cell cache to n_rows × 6.
        Items are allocated once and reused; rows removed by the user's
        min-N filter drop their cached items so we don't leak Qt
        objects across long sessions."""
        # Grow if needed.
        for r in range(n_rows):
            for c in range(6):
                if (r, c) not in self._cells:
                    item = QTableWidgetItem('')
                    self._table.setItem(r, c, item)
                    self._cells[(r, c)] = item
        # Drop any cached cells past the new row count.  setRowCount
        # has already removed them from the table widget; we just need
        # to evict them from our dict so a future grow re-allocates.
        stale = [k for k in self._cells if k[0] >= n_rows]
        for k in stale:
            del self._cells[k]

    # -------------------------------------------------------------------------
    # Single-column body
    # -------------------------------------------------------------------------
    def _refresh_single(self) -> None:
        from sax_intonation_gui import (
            TRANSP_MAP, midi_note_name, format_cents,
        )
        w = self._w
        instrument = self._get_instrument_key()
        display = self._get_display_mode()
        transp = TRANSP_MAP.get(instrument, 0)
        disp_griff = (display == 'griff')

        if disp_griff:
            hdrs = [self._t('col_fingered'), self._t('col_sounding')]
        else:
            hdrs = [self._t('col_sounding'), self._t('col_fingered')]
        self._table.setHorizontalHeaderLabels(
            hdrs + [self._t('col_mean'), self._t('col_std'),
                    self._t('col_n'), self._t('col_tendency')])

        with w._lock:
            raw_items = sorted(self._get_stats().items())

        # Min-N filter: rows with 1..min_n-1 measurements are below
        # threshold and hidden as noise. N=0 seeded blanks still show so
        # the instrument range stays visible as a guide.
        min_n = max(0, int(getattr(self._cfg, 'min_n_visible', 0)))
        items = [(m, s) for (m, s) in raw_items
                 if s.n == 0 or s.n >= min_n]

        self._table.setRowCount(len(items))
        # Ensure the cell cache matches the new row count BEFORE we
        # start mutating cells (Legolas W8 — no per-row allocations on
        # the hot path).
        self._ensure_single_cells(len(items))

        played_n = 0
        active_midi = self._active_midi_now()
        sr_now = w._engine_sample_rate()
        a4 = self._engine.a4
        for row, (midi_kl, st) in enumerate(items):
            midi_gr = midi_kl - transp
            kl_name = midi_note_name(midi_kl)
            gr_name = midi_note_name(midi_gr)
            n1, n2  = (gr_name, kl_name) if disp_griff else (kl_name, gr_name)
            mean    = st.mean
            has_data = st.n > 0
            if has_data:
                played_n += 1

            col = (QColor('#3a9e5f') if abs(mean) <= 5 else
                   QColor('#c8a020') if abs(mean) <= 12 else QColor('#c03030'))
            dim_col = QColor('#555')

            # Nominal frequency for this MIDI: drives the precision floor.
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            mean_str = format_cents(mean, note_freq, sr_now) if has_data else '–'
            if st.n > 1:
                std_str = '±' + format_cents(st.std, note_freq, sr_now).lstrip('+-')
            else:
                std_str = '–'

            is_active = (active_midi == midi_kl)
            values = [
                n1, n2,
                mean_str,
                std_str,
                str(st.n) if has_data else '–',
                '',
            ]
            for c, val in enumerate(values):
                item = self._cells[(row, c)]
                item.setText(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Foreground colour rules mirror the original verbatim.
                if c == 2:
                    item.setForeground(col if has_data else dim_col)
                elif not has_data:
                    item.setForeground(dim_col)
                else:
                    # Restore default foreground when a previously-empty
                    # row gets data on a later tick (otherwise the dim
                    # gray would stick from an earlier paint).
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)
                if c == 5 and has_data:
                    item.setData(Qt.ItemDataRole.UserRole,
                                 {'cents': mean, 'freq': note_freq})
                elif c == 5:
                    # Clear stale bar data so the delegate paints empty.
                    item.setData(Qt.ItemDataRole.UserRole, None)
                if is_active:
                    item.setBackground(QColor('#2c5a8a'))
                else:
                    # Clear any prior active-tint so a row that lost the
                    # highlight returns to default background colors.
                    item.setData(Qt.ItemDataRole.BackgroundRole, None)

        total = sum(s.n for _, s in items)
        if not items or played_n == 0:
            label = self._t('table_empty_hint')
        else:
            label = self._t('table_summary', notes=played_n, total=total)
        w._table_lbl.setText(label)

    # -------------------------------------------------------------------------
    # Matrix body
    # -------------------------------------------------------------------------
    def _refresh_matrix(self) -> None:
        from sax_intonation_gui import (
            TRANSP_MAP, midi_note_name, CHROMA,
        )
        w = self._w
        instrument = self._get_instrument_key()
        display = self._get_display_mode()
        transp = TRANSP_MAP.get(instrument, 0)
        disp_griff = (display == 'griff')
        lo_f, hi_f = sax_instruments.fingered_range(instrument)
        # Use the range cached by configure_for_mode so the cell layout
        # and the per-cell math line up.  If the range has drifted
        # (e.g. a played-note expansion), the next layout-mode tick on
        # MainWindow's side will reconfigure and the cache will refresh.
        lo_oct = self._matrix_lo_oct
        hi_oct = self._matrix_hi_oct
        if hi_oct < lo_oct:
            # Defensive: never configured for matrix; bail.
            return
        octaves = list(range(lo_oct, hi_oct + 1))

        # Fingered (griff) mode uses RELATIVE octave labels (-1, 0, +1)
        # centered on the instrument's middle octave — saxophone players
        # read "low Bb" as Bb3 in SPN, which feels mis-octaved against an
        # absolute scale. Concert (klingend) mode keeps absolute octave
        # numbers because sounding pitch IS absolute.
        if disp_griff:
            mid_oct = (lo_oct + hi_oct) // 2
            header_strings = [
                self._t('matrix_oct_rel_label', n=(o - mid_oct))
                for o in octaves
            ]
        else:
            header_strings = [
                self._t('matrix_oct_label', n=o) for o in octaves
            ]
        self._table.setHorizontalHeaderLabels(header_strings)
        # Row labels: pitch class only (octave lives in the column header).
        self._table.setVerticalHeaderLabels(
            [c.split('/')[0] for c in CHROMA])

        with w._lock:
            stats_by_midi = dict(self._get_stats())
        active_midi = self._active_midi_now()
        a4 = self._engine.a4

        played_n = 0
        in_range_cells = 0
        for r in range(12):
            for c, oct_ in enumerate(octaves):
                midi_visible = (oct_ + 1) * 12 + r
                if disp_griff:
                    midi_fingered = midi_visible
                    midi_sounding = midi_visible + transp
                else:
                    midi_sounding = midi_visible
                    midi_fingered = midi_visible - transp
                in_range = lo_f <= midi_fingered <= hi_f
                if in_range:
                    in_range_cells += 1

                st = stats_by_midi.get(midi_sounding)
                # Apply min-N gate so single-blip cells don't render with
                # arbitrary cents readings. Same rule as single-column:
                # a cell with 1..min_n-1 hits is treated as if it has no
                # data yet.
                min_n = max(0, int(
                    getattr(self._cfg, 'min_n_visible', 0)))
                has_data = (st is not None and st.n > 0
                            and st.n >= min_n)
                if has_data:
                    played_n += 1
                # MatrixCellDelegate reads this dict and paints all six
                # data fields the single-column view shows per row:
                # fingered name, sounding name, mean, std, N, bar.
                # Nominal frequency at A4 drives the precision floor.
                note_freq = a4 * (
                    2.0 ** ((midi_sounding - 69) / 12.0))
                payload = {
                    'mean':          st.mean if has_data else None,
                    'std':           st.std if has_data else None,
                    'n':             st.n if st is not None else 0,
                    'in_range':      in_range,
                    'active':        (active_midi == midi_sounding),
                    'fingered_name': midi_note_name(midi_fingered),
                    'sounding_name': midi_note_name(midi_sounding),
                    'freq':          note_freq,
                }
                # Reuse the cached item — no allocation on the hot path.
                item = self._cells.get((r, c))
                if item is None:
                    # Defensive: cache miss (would happen only if the
                    # octave range drifted out from under us).  Allocate
                    # once and park it; future ticks will reuse.
                    item = QTableWidgetItem('')
                    self._table.setItem(r, c, item)
                    self._cells[(r, c)] = item
                item.setData(Qt.ItemDataRole.UserRole, payload)

        if in_range_cells == 0:
            label = self._t('table_empty_hint')
        else:
            label = self._t('table_matrix_title',
                             played=played_n, total=in_range_cells)
        w._table_lbl.setText(label)

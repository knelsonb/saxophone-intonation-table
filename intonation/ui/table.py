# =============================================================================
# intonation/ui/table.py — the intonation table: a model over the session's statistics
#
# IntonationTableModel presents the SessionModel's per-note statistics in one
# of two layouts:
#   'single' — one row per note (played notes and the instrument's range,
#              minus notes still below the minimum count), six columns:
#              fingered, sounding, mean, spread, N and a cents bar;
#   'matrix' — piano-roll: 12 chromatic rows × one column per octave, each
#              cell a payload dict a delegate paints (MatrixCellDelegate).
# It is rebuilt from the session on every refresh (the 300 ms tick of the
# TUNER tab), so the table always shows the session as it is now; rows and
# cells change in place, so the view keeps its scroll position.
#
# TableController sets the view up for a layout (delegates, header sizing,
# row height) and writes the caption above the table.
#
# The roles the view and the harness read:
#   DisplayRole        single-layout cell text;
#   ForegroundRole     the mean column's band colour (dim when unplayed);
#   BackgroundRole     the row of the note played last;
#   UserRole           single layout, bar column: {'cents', 'freq'};
#                      matrix layout: the cell payload (see MatrixCellDelegate).
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QHeaderView, QStyledItemDelegate

from intonation.core.pitch import CHROMA, format_cents, midi_note_name
from intonation.ui.theme import cents_band
from intonation.ui.widgets import MatrixCellDelegate

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QLabel, QTableView

    from intonation.core.session import SessionModel
    from intonation.i18n.translator import Translator
    from intonation.persistence.config import AppConfig

# Text colour of the mean column for each cents_band() result.
_BAND_COLOR = {'in': '#3a9e5f', 'close': '#c8a020', 'off': '#c03030'}
_DIM_COLOR = '#555'
_ACTIVE_ROW_COLOR = '#2c5a8a'

# Matrix geometry. The TUNER tab's width-driven layout choice uses the same
# numbers, so a matrix it picks always fits two octave columns.
MATRIX_COL_WIDTH = 112    # min usable width per octave column (px)
MATRIX_HEADER_W = 64      # row-header width (note names)
MATRIX_ROW_HEIGHT = 60    # tall enough for: names + mean+std + bar + N
SINGLE_ROW_HEIGHT = 28

SINGLE_COLUMNS = 6
MEAN_COLUMN = 2
BAR_COLUMN = 5

_CENTER = int(Qt.AlignmentFlag.AlignCenter)
_BRUSHES: dict[str, QBrush] = {}


def _brush(colour: str) -> QBrush:
    """One shared brush per colour: data() is asked for it on every paint."""
    brush = _BRUSHES.get(colour)
    if brush is None:
        brush = _BRUSHES[colour] = QBrush(QColor(colour))
    return brush


def _note_freq(a4: float, midi: int) -> float:
    """Nominal frequency of ``midi`` at ``a4``: sets the cents precision."""
    return a4 * (2.0 ** ((midi - 69) / 12.0))


class _Row:
    """One single-layout row, as painted. Rows compare equal when they look
    the same, so a refresh repaints only what changed."""
    __slots__ = ('midi', 'texts', 'played', 'colour', 'bar', 'active')

    def __init__(self, midi, texts, played, colour, bar, active):
        self.midi = midi
        self.texts = texts
        self.played = played
        self.colour = colour
        self.bar = bar
        self.active = active

    def __eq__(self, other) -> bool:
        return (isinstance(other, _Row)
                and (self.midi, self.texts, self.colour, self.bar, self.active)
                == (other.midi, other.texts, other.colour, other.bar, other.active))


def _changed_span(old: list, new: list) -> Optional[tuple[int, int]]:
    """(first, last) index where ``new`` differs from ``old`` (compared up
    to the shorter length), or None when they agree."""
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    return (changed[0], changed[-1]) if changed else None


class IntonationTableModel(QAbstractTableModel):
    """The session's statistics as a table; see the module comment."""

    def __init__(self, *, cfg: 'AppConfig', session: 'SessionModel',
                 tr: 'Translator', sample_rate: Callable[[], int],
                 parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._session = session
        self._t = tr.t
        self._sample_rate = sample_rate
        self._layout: Optional[str] = None
        self._rows: list[_Row] = []
        self._headers: list[str] = []
        self._row_headers = [c.split('/')[0] for c in CHROMA]
        # Matrix: the octave span the columns are laid out for, and one
        # payload per (row, column).
        self._lo_oct = 0
        self._hi_oct = -1
        self._cells: list[list[dict]] = []
        # The line above the table, written with every fill.
        self.caption = ''

    # ── Layout ───────────────────────────────────────────────────────────────
    @property
    def layout(self) -> Optional[str]:
        """'single', 'matrix', or None before the first set_layout()."""
        return self._layout

    def set_layout(self, layout: str) -> None:
        """Switch layout (or lay the matrix out again for the live octave
        span) and fill it from the session."""
        self.beginResetModel()
        self._layout = layout
        self._rows = []
        self._cells = []
        if layout == 'matrix':
            self._lo_oct, self._hi_oct = self.octave_range()
            self._fill_matrix()
        else:
            self._rows = self._single_rows()
        self.endResetModel()

    def painted_octave_span(self) -> tuple[int, int]:
        """(lo_octave, hi_octave) the matrix columns are laid out for. Column
        ``c`` shows octave ``lo_octave + c``. Code that maps a clicked column
        back to a note must use this span, which is the one on screen,
        rather than the live range."""
        return (self._lo_oct, self._hi_oct)

    def octave_range(self) -> tuple[int, int]:
        """(lo_octave, hi_octave) inclusive the matrix should show now: the
        instrument's fingered range and, when out-of-range notes are allowed,
        every played note outside it (overtones, altissimo, accidentals), so
        nothing is truncated.

        Half-step-beyond rule: a range starting on a C gets the octave below
        (to show the B a half step lower); one ending on a B gets the octave
        above. cfg.matrix_extra_octaves adds context on both sides."""
        session = self._session
        transp = session.transposition
        lo_f, hi_f = session.fingered_range()
        griff = session.display == 'griff'
        lo_midi, hi_midi = (lo_f, hi_f) if griff else (lo_f + transp, hi_f + transp)
        played = [m for m, st in session.stats.items() if st.n > 0]
        if played and self._cfg.allow_out_of_range:
            if griff:
                played = [m - transp for m in played]
            lo_midi = min(lo_midi, min(played))
            hi_midi = max(hi_midi, max(played))
        lo_oct = lo_midi // 12 - 1
        hi_oct = hi_midi // 12 - 1
        if lo_midi % 12 == 0:      # low note is C → show B in the column below
            lo_oct -= 1
        if hi_midi % 12 == 11:     # high note is B → show C in the column above
            hi_oct += 1
        extra = self._cfg.matrix_extra_octaves
        # MIDI octave -1 is no use for any real instrument.
        return (max(0, lo_oct - extra), hi_oct + extra)

    # ── Refresh ──────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Repaint from the session: only the rows or matrix rows that look
        different are reported as changed. The matrix is laid out again first
        when its live octave span moved (display toggle, instrument or range
        change, a note played outside the span)."""
        if self._layout is None:
            return
        headers = list(self._headers)
        if self._layout == 'matrix':
            if self.octave_range() != self.painted_octave_span():
                self.set_layout('matrix')
                return
            old_cells = self._cells
            self._fill_matrix()
            span = _changed_span(old_cells, self._cells)
            if span is not None:
                self.dataChanged.emit(self.index(span[0], 0),
                                      self.index(span[1], self.columnCount() - 1))
        else:
            old = self._rows
            rows = self._single_rows()
            if len(rows) > len(old):
                self.beginInsertRows(QModelIndex(), len(old), len(rows) - 1)
                self._rows = rows
                self.endInsertRows()
            elif len(rows) < len(old):
                self.beginRemoveRows(QModelIndex(), len(rows), len(old) - 1)
                self._rows = rows
                self.endRemoveRows()
            else:
                self._rows = rows
            span = _changed_span(old, rows)
            if span is not None:
                self.dataChanged.emit(self.index(span[0], 0),
                                      self.index(span[1], SINGLE_COLUMNS - 1))
        if self._headers != headers and self.columnCount():
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0,
                                        self.columnCount() - 1)

    def _single_rows(self) -> list[_Row]:
        t = self._t
        session = self._session
        transp = session.transposition
        griff = session.display == 'griff'
        names = [t('col_fingered'), t('col_sounding')]
        if not griff:
            names.reverse()
        self._headers = names + [t('col_mean'), t('col_std'), t('col_n'),
                                 t('col_tendency')]
        # Rows with 1..min_n-1 readings are below the threshold and hidden as
        # noise; unplayed (seeded) rows stay so the range shows as a guide.
        min_n = self._cfg.min_n_visible
        items = [(m, s) for m, s in sorted(session.stats.items())
                 if s.n == 0 or s.n >= min_n]
        active = session.active_midi()
        sr = self._sample_rate()
        a4 = session.a4
        rows = []
        played_n = total = 0
        for midi_kl, st in items:
            kl_name = midi_note_name(midi_kl)
            gr_name = midi_note_name(midi_kl - transp)
            played = st.n > 0
            freq = _note_freq(a4, midi_kl)
            total += st.n
            if played:
                played_n += 1
                mean_str = format_cents(st.mean, freq, sr)
                colour = _BAND_COLOR[cents_band(st.mean)]
            else:
                mean_str = '–'
                colour = _DIM_COLOR
            std_str = ('±' + format_cents(st.std, freq, sr).lstrip('+-')
                       if st.n > 1 else '–')
            first, second = (gr_name, kl_name) if griff else (kl_name, gr_name)
            rows.append(_Row(
                midi_kl,
                (first, second, mean_str, std_str, str(st.n) if played else '–', ''),
                played, colour,
                {'cents': st.mean, 'freq': freq} if played else None,
                active == midi_kl))
        if not items or played_n == 0:
            self.caption = t('table_empty_hint')
        else:
            self.caption = t('table_summary', notes=played_n, total=total)
        return rows

    def _fill_matrix(self) -> None:
        t = self._t
        session = self._session
        transp = session.transposition
        griff = session.display == 'griff'
        lo_f, hi_f = session.fingered_range()
        octaves = range(self._lo_oct, self._hi_oct + 1)
        # Fingered display labels octaves relative to the instrument's middle
        # octave (-1, 0, +1): saxophone players read "low Bb" as Bb3 in SPN,
        # which feels mis-octaved against an absolute scale. Sounding display
        # keeps absolute octave numbers because sounding pitch IS absolute.
        if griff:
            mid = (self._lo_oct + self._hi_oct) // 2
            self._headers = [t('matrix_oct_rel_label', n=o - mid) for o in octaves]
        else:
            self._headers = [t('matrix_oct_label', n=o) for o in octaves]
        stats = session.stats
        active = session.active_midi()
        a4 = session.a4
        # A cell with 1..min_n-1 readings is shown as not yet played, as the
        # single layout hides such a row.
        min_n = self._cfg.min_n_visible
        played_n = in_range_cells = 0
        cells = []
        for r in range(12):
            row = []
            for o in octaves:
                midi_visible = (o + 1) * 12 + r
                if griff:
                    midi_fingered, midi_sounding = midi_visible, midi_visible + transp
                else:
                    midi_fingered, midi_sounding = midi_visible - transp, midi_visible
                in_range = lo_f <= midi_fingered <= hi_f
                in_range_cells += in_range
                st = stats.get(midi_sounding)
                has_data = st is not None and st.n > 0 and st.n >= min_n
                played_n += has_data
                row.append({
                    'mean':          st.mean if has_data else None,
                    'std':           st.std if has_data else None,
                    'n':             st.n if st is not None else 0,
                    'in_range':      in_range,
                    'active':        active == midi_sounding,
                    'fingered_name': midi_note_name(midi_fingered),
                    'sounding_name': midi_note_name(midi_sounding),
                    'freq':          _note_freq(a4, midi_sounding),
                })
            cells.append(row)
        self._cells = cells
        if in_range_cells == 0:
            self.caption = t('table_empty_hint')
        else:
            self.caption = t('table_matrix_title', played=played_n, total=in_range_cells)

    # ── Reading back ─────────────────────────────────────────────────────────
    def midi_at(self, row: int, col: int) -> Optional[int]:
        """The sounding note shown at (row, col), or None. A matrix column
        maps through the octave span on screen."""
        if self._layout == 'matrix':
            octave = self._lo_oct + col
            if not (0 <= row < 12 and col >= 0 and octave <= self._hi_oct):
                return None
            midi_visible = (octave + 1) * 12 + row
            if self._session.display == 'griff':
                return midi_visible + self._session.transposition
            return midi_visible
        if 0 <= row < len(self._rows):
            return self._rows[row].midi
        return None

    # ── QAbstractTableModel ──────────────────────────────────────────────────
    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return 12 if self._layout == 'matrix' else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid() or self._layout is None:
            return 0
        if self._layout == 'matrix':
            return max(0, self._hi_oct - self._lo_oct + 1)
        return SINGLE_COLUMNS

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._headers[section] if 0 <= section < len(self._headers) else None
        if self._layout == 'matrix' and 0 <= section < 12:
            return self._row_headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if self._layout == 'matrix':
            if (role == Qt.ItemDataRole.UserRole and r < len(self._cells)
                    and c < len(self._cells[r])):
                return self._cells[r][c]
            return None
        if r >= len(self._rows):
            return None
        row = self._rows[r]
        if role == Qt.ItemDataRole.DisplayRole:
            return row.texts[c]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return _CENTER
        if role == Qt.ItemDataRole.ForegroundRole:
            # The mean carries the band colour; an unplayed row is dim.
            if c == MEAN_COLUMN or not row.played:
                return _brush(row.colour)
            return None
        if role == Qt.ItemDataRole.BackgroundRole:
            return _brush(_ACTIVE_ROW_COLOR) if row.active else None
        if role == Qt.ItemDataRole.UserRole and c == BAR_COLUMN:
            return row.bar
        return None


class TableController:
    """Sets the table view up for a layout (delegates, header sizing, row
    height) over an IntonationTableModel, and writes the caption."""

    def __init__(self, *,
                 table: 'QTableView',
                 caption: 'QLabel',
                 bar_delegate: QStyledItemDelegate,
                 cfg: 'AppConfig',
                 session: 'SessionModel',
                 tr: 'Translator',
                 sample_rate: Callable[[], int]) -> None:
        self._table = table
        self._caption = caption
        self._bar_delegate = bar_delegate
        self.model = IntonationTableModel(cfg=cfg, session=session, tr=tr,
                                          sample_rate=sample_rate, parent=table)
        table.setModel(self.model)
        # Delegates swapped in by configure_for_mode: the plain one for the
        # single layout's text columns, the painted cell for the matrix.
        self._default_delegate = QStyledItemDelegate(table)
        self._matrix_delegate = MatrixCellDelegate(table, sample_rate_getter=sample_rate)
        # The octave span the matrix columns were last sized for.
        self._sized_span = (0, -1)

    @property
    def mode(self) -> Optional[str]:
        """The layout shown: 'single', 'matrix', or None before the first
        configure_for_mode()."""
        return self.model.layout

    def configure_for_mode(self, mode: str) -> None:
        """Lay the table out for 'single' or 'matrix' and fill it."""
        self.model.set_layout(mode)
        self._configure_view(mode)
        self._caption.setText(self.model.caption)

    def refresh(self) -> None:
        """Repaint from the session (laying the matrix out again when its
        span moved). Before the first configure_for_mode() it does nothing."""
        if self.model.layout is None:
            return
        self.model.refresh()
        if self.model.layout == 'matrix' and self.painted_octave_span() != self._sized_span:
            self._configure_view('matrix')
        self._caption.setText(self.model.caption)

    def painted_octave_span(self) -> tuple[int, int]:
        return self.model.painted_octave_span()

    def _matrix_octave_range(self) -> tuple[int, int]:
        return self.model.octave_range()

    def _configure_view(self, mode: str) -> None:
        table = self._table
        header = table.horizontalHeader()
        if mode == 'matrix':
            table.verticalHeader().setVisible(True)
            table.verticalHeader().setDefaultSectionSize(MATRIX_ROW_HEIGHT)
            # Fixed column widths: playable notes always render at full cell
            # size; when the columns don't all fit, the horizontal scrollbar
            # takes over instead of squeezing the cells.
            header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            header.setDefaultSectionSize(MATRIX_COL_WIDTH)
            for c in range(self.model.columnCount()):
                table.setColumnWidth(c, MATRIX_COL_WIDTH)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            table.setItemDelegateForColumn(BAR_COLUMN, self._default_delegate)
            table.setItemDelegate(self._matrix_delegate)
            self._sized_span = self.painted_octave_span()
        else:
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(SINGLE_ROW_HEIGHT)
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setItemDelegate(self._default_delegate)
            table.setItemDelegateForColumn(BAR_COLUMN, self._bar_delegate)

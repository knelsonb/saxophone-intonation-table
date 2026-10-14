"""Matrix cell painting: what each kind of cell looks like.

The table model hands the matrix delegate one payload dict per cell (see
intonation.ui.table). A cell outside the keyed range that nobody played stays
blank; one that WAS played (low A on an alto, altissimo) shows its data like a
guide cell, with a dashed border instead of a solid one, so a measured note
never looks like an empty cell.
"""
from __future__ import annotations

import pytest
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QApplication, QStyleOptionViewItem

from intonation.ui import theme
from intonation.ui.widgets import MatrixCellDelegate

W, H = 120, 60


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def payload(in_range: bool, mean, n: int) -> dict:
    return {"mean": mean, "std": 2.0 if mean is not None else None, "n": n,
            "in_range": in_range, "active": False, "fingered_name": "A3",
            "sounding_name": "C3", "freq": 130.8}


def paint(qapp, data: dict) -> QImage:
    model = QStandardItemModel(1, 1)
    item = QStandardItem()
    item.setData(data, Qt.ItemDataRole.UserRole)
    model.setItem(0, 0, item)
    img = QImage(W, H, QImage.Format.Format_RGB32)
    img.fill(QColor("#ff00ff"))
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, W, H)
    painter = QPainter(img)
    MatrixCellDelegate().paint(painter, option, model.index(0, 0))
    painter.end()
    return img


def pixels(img: QImage, y: int) -> list[int]:
    return [img.pixel(x, y) & 0xFFFFFF for x in range(W)]


def ink(img: QImage, background: str) -> int:
    """How many pixels differ from the cell background."""
    bg = QColor(background).rgb() & 0xFFFFFF
    return sum(p != bg for y in range(H) for p in pixels(img, y))


def gaps_along_top(img: QImage, background: str) -> int:
    """Background pixels on the top border line, between the corners."""
    bg = QColor(background).rgb() & 0xFFFFFF
    return sum(p == bg for p in pixels(img, 0)[4:-4])


def test_an_unplayed_cell_outside_the_keyed_range_is_blank(qapp):
    img = paint(qapp, payload(in_range=False, mean=None, n=0))
    assert ink(img, theme.active().window_bg) == 0


def test_a_played_cell_outside_the_keyed_range_shows_its_data(qapp):
    played = paint(qapp, payload(in_range=False, mean=-7.0, n=12))
    guide = paint(qapp, payload(in_range=True, mean=-7.0, n=12))
    bg = theme.active().window_bg
    # About as much ink as the same data in a guide cell, nothing like blank.
    assert ink(played, bg) > 0.5 * ink(guide, theme.active().base_bg) > 100


def test_a_played_cell_outside_the_keyed_range_has_a_dashed_border(qapp):
    played = paint(qapp, payload(in_range=False, mean=-7.0, n=12))
    guide = paint(qapp, payload(in_range=True, mean=-7.0, n=12))
    assert gaps_along_top(guide, theme.active().base_bg) == 0
    gaps = gaps_along_top(played, theme.active().window_bg)
    assert 10 < gaps < (W - 8) - 10                # both dashes and gaps

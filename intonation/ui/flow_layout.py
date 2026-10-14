"""
FlowLayout — Qt layout that reflows children onto the next row when the
parent gets narrower than the children's combined width.

Ported from Qt's official `examples/widgets/layouts/flowlayout` (BSD-3
licensed, redistributable). The original is C++; this is a Python rewrite
that follows the same `heightForWidth`/`minimumSize` contract.
"""

from __future__ import annotations

from PyQt6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QSizePolicy, QStyle


class FlowLayout(QLayout):

    def __init__(self, parent=None, margin: int = 0,
                 hspacing: int = -1, vspacing: int = -1):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(QMargins(margin, margin, margin, margin))
        self._hspace = hspacing
        self._vspace = vspacing
        self._items: list = []

    def addItem(self, item):
        self._items.append(item)

    def horizontalSpacing(self) -> int:
        if self._hspace >= 0:
            return self._hspace
        return self._smart_spacing(
            QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self) -> int:
        if self._vspace >= 0:
            return self._vspace
        return self._smart_spacing(
            QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        s = QSize()
        for item in self._items:
            s = s.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0
        for item in self._items:
            wid = item.widget()
            space_x = self.horizontalSpacing()
            if space_x == -1 and wid is not None:
                space_x = wid.style().layoutSpacing(
                    QSizePolicy.ControlType.PushButton,
                    QSizePolicy.ControlType.PushButton,
                    Qt.Orientation.Horizontal)
            space_y = self.verticalSpacing()
            if space_y == -1 and wid is not None:
                space_y = wid.style().layoutSpacing(
                    QSizePolicy.ControlType.PushButton,
                    QSizePolicy.ControlType.PushButton,
                    Qt.Orientation.Vertical)
            next_x = x + item.sizeHint().width() + space_x
            if (next_x - space_x > effective.right() and line_height > 0):
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y() + m.bottom()

    def _smart_spacing(self, pm: QStyle.PixelMetric) -> int:
        parent = self.parent()
        if parent is None:
            return -1
        if parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        return parent.spacing()

"""Generate the Intonation Analyzer application icon.

Renders a single source PNG at 512×512 with QPainter, then derives a
multi-size .ico for Windows. PIL is used only for the ICO step and is
optional at build time — without it the script still produces icon.png.

Output:
    assets/icon.png    (512×512 RGBA)
    assets/icon.ico    (Windows multi-size: 16/32/48/64/128/256)

The icon shows a green-yellow-red intonation needle on a dark disk —
visually distinctive at favicon size and immediately readable as "tuner".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter,
                          QPen, QPixmap, QPolygonF, QRadialGradient)
from PyQt6.QtWidgets import QApplication


SIZE = 512
OUT_DIR = Path(__file__).resolve().parent.parent / 'assets'


def render_icon() -> QPixmap:
    pix = QPixmap(SIZE, SIZE)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Outer dark disk with a subtle radial gradient.
    grad = QRadialGradient(SIZE / 2, SIZE / 2 - 40, SIZE / 1.6)
    grad.setColorAt(0.0, QColor('#2a2a3e'))
    grad.setColorAt(1.0, QColor('#0e0e18'))
    p.setBrush(QBrush(grad))
    p.setPen(QPen(QColor('#3a3a55'), 8))
    margin = 16
    p.drawEllipse(QRectF(margin, margin, SIZE - 2 * margin, SIZE - 2 * margin))

    # Cents scale (green ±5, yellow ±15, red beyond), drawn as pie wedges.
    cx, cy = SIZE / 2, SIZE / 2 + 30
    r_outer = SIZE / 2 - 60
    r_inner = r_outer - 36
    arc_rect = QRectF(cx - r_outer, cy - r_outer, 2 * r_outer, 2 * r_outer)
    p.setPen(Qt.PenStyle.NoPen)

    def pie(deg_lo, deg_hi, color):
        p.setBrush(QColor(color))
        # drawPie expects start angle and span angle in 1/16ths of a degree.
        start = int((90 - deg_hi) * 16)
        span = int((deg_hi - deg_lo) * 16)
        p.drawPie(arc_rect, start, span)

    pie(-120, -30, '#c03030')
    pie(-30, -8, '#c8a020')
    pie(-8, 8, '#3a9e5f')
    pie(8, 30, '#c8a020')
    pie(30, 120, '#c03030')

    # Mask out the centre to leave just the colored arc ring.
    p.setBrush(QColor('#13131e'))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(cx, cy), r_inner, r_inner)

    # Needle pointing slightly sharp (about +6 ct → ~13° from straight up).
    needle_angle_deg = 12
    rad = math.radians(90 + needle_angle_deg)
    tip = QPointF(cx + (r_outer - 6) * math.cos(rad),
                  cy - (r_outer - 6) * math.sin(rad))
    base_w = 14
    perp = math.radians(needle_angle_deg)
    bx1 = cx + base_w * math.cos(perp + math.pi / 2)
    by1 = cy - base_w * math.sin(perp + math.pi / 2)
    bx2 = cx + base_w * math.cos(perp - math.pi / 2)
    by2 = cy - base_w * math.sin(perp - math.pi / 2)
    p.setBrush(QColor('#f0f0f8'))
    p.setPen(QPen(QColor('#c0c0d0'), 2))
    p.drawPolygon(QPolygonF([tip, QPointF(bx1, by1), QPointF(bx2, by2)]))

    # Pivot cap.
    p.setBrush(QColor('#1e2030'))
    p.setPen(QPen(QColor('#888'), 3))
    p.drawEllipse(QPointF(cx, cy), 16, 16)

    # Title arc at the top.
    p.setPen(QColor('#d8d8e8'))
    f = QFont()
    f.setFamilies(['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans'])
    f.setBold(True)
    f.setPixelSize(54)
    p.setFont(f)
    p.drawText(QRectF(0, 30, SIZE, 70),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               'IA')

    p.end()
    return pix


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    pix = render_icon()
    png_path = OUT_DIR / 'icon.png'
    if not pix.save(str(png_path), 'PNG'):
        print(f'failed to write {png_path}', file=sys.stderr)
        return 1
    print(f'wrote {png_path}')

    # Build ICO via Pillow if available (optional dependency for build time).
    ico_path = OUT_DIR / 'icon.ico'
    try:
        from PIL import Image
    except ImportError:
        print('Pillow not installed; skipping .ico generation')
        print('  pip install Pillow   to enable it')
        return 0
    img = Image.open(png_path)
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format='ICO', sizes=sizes)
    print(f'wrote {ico_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

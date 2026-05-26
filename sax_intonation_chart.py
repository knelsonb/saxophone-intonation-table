"""
PNG chart rendering for the intonation table.

Uses QPainter to draw a shareable bar chart of mean cents deviation per note.
No new dependencies — PyQt6 is already in. The result is written via
QPixmap.save(), which is wired for PNG/JPEG/BMP by the bundled image plugins.

The chart is intentionally simple: one bar per note, sorted by sounding MIDI,
centered on a zero line, colored green/yellow/red by magnitude. Anyone the
file is shared with should be able to read it without a legend.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QPixmap, QPolygonF,
)


def _sans(point_size: int, bold: bool = False) -> QFont:
    """Return a sans-serif font that resolves on Windows, macOS, and Linux.

    QFont('Sans') / QFont('Monospace') are Linux aliases and do not match a
    real family on Windows — Qt then falls back to a font with no glyphs and
    everything renders as boxes. Setting the family-list explicitly with the
    family-name fallback chain plus a style hint avoids that.
    """
    f = QFont()
    f.setFamilies(["Segoe UI", "Arial", "Helvetica", "DejaVu Sans",
                   "Liberation Sans"])
    f.setStyleHint(QFont.StyleHint.SansSerif)
    f.setPointSize(point_size)
    if bold:
        f.setBold(True)
    return f


def _mono(point_size: int) -> QFont:
    f = QFont()
    f.setFamilies(["Consolas", "Menlo", "DejaVu Sans Mono",
                   "Liberation Mono", "Courier New"])
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(point_size)
    return f


# Chart dimensions in pixels.
_WIDTH = 1100
_HEIGHT = 640

# Layout offsets inside the chart.
_MARGIN_L = 70
_MARGIN_R = 30
_MARGIN_T = 110
_MARGIN_B = 90

# Y axis: ±MAX_CENTS shown. Bars beyond clamp to the edge with a marker.
_MAX_CENTS = 50.0


def render_intonation_chart(
    *,
    notes: list[tuple],
    title: str,
    subtitle: str,
    footer: str,
    output_path: str,
    sample_rate: int = 44100,
    instrument: str | None = None,
) -> None:
    """Render a bar chart and save to `output_path`.

    `notes` is a list of tuples, one per note in display order:
    * legacy 4-tuple: (label, mean_cents, std_cents, n)
    * v0.5.6 5-tuple: (label, mean_cents, std_cents, n, freq_hz)
    * v0.6 7-tuple:   (label, mean_cents, std_cents, n, freq_hz,
                       midi_fingered, in_range_bool)

    The 5-tuple feeds the frequency-adaptive cent precision used when
    drawing the numeric value above each bar. ``sample_rate`` is the
    engine's negotiated rate at chart time and gates that precision.

    ``instrument`` (v0.6 Phase-4 Item 7) is the instrument-catalog key for
    looking up the fingered range when the 7-tuple form isn't used. Any
    note whose fingered MIDI lands outside ``fingered_range(instrument)``
    is rendered in a desaturated grey with a small ⚠ marker so the user
    can see at a glance which notes are out of the instrument's nominal
    band (overtones, altissimo, accidentals).

    The image format is inferred from the file extension by QPixmap.save —
    .png works without any extra plugins on the standard PyQt6 install.
    """
    pix = QPixmap(_WIDTH, _HEIGHT)
    pix.fill(QColor(20, 20, 28))

    # Resolve the fingered range once (chart loop doesn't need to re-import
    # sax_instruments for every bar). If lookup fails or no instrument key
    # was supplied, treat every note as in-range — chart still renders.
    in_range_fn = _make_in_range_fn(instrument)

    p = QPainter(pix)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_header(p, title, subtitle)
        _draw_axes(p)
        if notes:
            _draw_bars(p, notes, sample_rate=sample_rate,
                       in_range_fn=in_range_fn)
        else:
            _draw_no_data(p)
        _draw_footer(p, footer)
    finally:
        p.end()

    # Returns False if PyQt6's image plugin chain refuses the format — for
    # example a stripped-down conda build with no PNG plugin. Surface that
    # instead of silently writing a 0-byte file.
    if not pix.save(output_path):
        raise RuntimeError(
            f"Failed to save chart image to {output_path!r}. The image "
            "format may not be supported by this PyQt6 install.")


def _make_in_range_fn(instrument: str | None):
    """Return a callable ``(midi_fingered) -> bool``. Looks up the
    instrument's fingered range via ``sax_instruments.fingered_range`` so
    out-of-range notes can be styled distinctly. Falls back to a
    "everything is in range" function if the instrument is unknown or the
    lookup raises — chart never crashes on a stale instrument key."""
    if not instrument:
        return lambda _midi: True
    try:
        from sax_instruments import fingered_range
        lo, hi = fingered_range(instrument)
    except Exception:
        return lambda _midi: True
    lo_i = int(lo)
    hi_i = int(hi)
    return lambda midi: (midi is not None
                         and lo_i <= int(midi) <= hi_i)


def _draw_header(p: QPainter, title: str, subtitle: str) -> None:
    p.setPen(QColor(240, 240, 250))
    p.setFont(_sans(20, bold=True))
    p.drawText(QRectF(_MARGIN_L, 20, _WIDTH - _MARGIN_L - _MARGIN_R, 36),
               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               title)
    p.setPen(QColor(170, 170, 190))
    p.setFont(_sans(11))
    p.drawText(QRectF(_MARGIN_L, 56, _WIDTH - _MARGIN_L - _MARGIN_R, 22),
               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               subtitle)


def _plot_area() -> QRectF:
    return QRectF(_MARGIN_L, _MARGIN_T,
                  _WIDTH - _MARGIN_L - _MARGIN_R,
                  _HEIGHT - _MARGIN_T - _MARGIN_B)


def _draw_axes(p: QPainter) -> None:
    area = _plot_area()
    # Background card.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(28, 28, 38))
    p.drawRoundedRect(area.adjusted(-6, -6, 6, 6), 8, 8)

    # Gridlines at ±50, ±25, 0.
    p.setFont(_mono(9))
    for ct in (-50, -25, 0, 25, 50):
        y = _cent_to_y(ct, area)
        if ct == 0:
            p.setPen(QPen(QColor(140, 140, 170), 1.6))
        elif abs(ct) == 25:
            p.setPen(QPen(QColor(70, 70, 90), 1, Qt.PenStyle.DashLine))
        else:
            p.setPen(QPen(QColor(60, 60, 78), 1))
        p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        p.setPen(QColor(170, 170, 190))
        sign = '+' if ct > 0 else ''
        p.drawText(QRectF(0, y - 10, _MARGIN_L - 8, 20),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{sign}{ct} ct")

    # ±5 ct "in tune" band.
    y_hi = _cent_to_y(5, area)
    y_lo = _cent_to_y(-5, area)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(40, 110, 60, 70))
    p.drawRect(QRectF(area.left(), y_hi,
                      area.width(), y_lo - y_hi))


def _cent_to_y(cents: float, area: QRectF) -> float:
    norm = max(-1.0, min(1.0, cents / _MAX_CENTS))
    # Positive cents -> upward (smaller y).
    mid = area.top() + area.height() / 2
    return mid - norm * (area.height() / 2)


def _format_cents_label(value_cents: float, freq_hz: float,
                        sample_rate: int) -> str:
    """Local mirror of sax_intonation_gui.format_cents to keep the chart
    module dependency-free (no GUI import). See that helper for the
    derivation behind the tier thresholds.

    v0.5.7.8: guard against non-finite inputs (mirrors the v0.5.7.2 guard
    added to sax_intonation_gui.format_cents). Chart export can be reached
    with corrupted freq_hz from CSV import, and int(round(NaN)) raises
    ValueError. Keep the placeholder glyph identical to format_cents."""
    if (not math.isfinite(value_cents) or not math.isfinite(freq_hz)
            or freq_hz <= 0):
        return '–'
    sr = float(sample_rate) if sample_rate else 44100.0
    if sr <= 0 or freq_hz <= 0:
        floor_ct = 0.3
    else:
        floor_ct = 173.0 * float(freq_hz) / sr
    if floor_ct <= 0.3:
        v = float(value_cents)
        return f"{'+' if v >= 0 else '-'}{abs(v):.1f}"
    if floor_ct <= 0.7:
        v = round(float(value_cents) * 2.0) / 2.0
        return f"{'+' if v >= 0 else '-'}{abs(v):.1f}"
    v = round(float(value_cents))
    return f"{'+' if v >= 0 else '-'}{abs(int(v))}"


def _draw_bars(p: QPainter,
               notes: list[tuple],
               sample_rate: int = 44100,
               in_range_fn=None) -> None:
    if in_range_fn is None:
        in_range_fn = lambda _midi: True
    area = _plot_area()
    n = len(notes)
    slot = area.width() / n
    bar_w = max(8.0, min(40.0, slot * 0.55))
    zero_y = _cent_to_y(0, area)

    p.setFont(_mono(9))
    for i, row in enumerate(notes):
        # Accept legacy 4-tuple and v0.5.6 5-tuple with note freq, plus
        # v0.6 7-tuple with midi_fingered + explicit in_range flag.
        midi_fingered = None
        explicit_in_range = None
        if len(row) >= 7:
            (label, mean, std, count, freq,
             midi_fingered, explicit_in_range) = (row[0], row[1], row[2],
                                                   row[3], row[4], row[5], row[6])
        elif len(row) >= 5:
            label, mean, std, count, freq = row[0], row[1], row[2], row[3], row[4]
        else:
            label, mean, std, count = row[0], row[1], row[2], row[3]
            freq = 0.0
        if explicit_in_range is None:
            note_in_range = in_range_fn(midi_fingered)
        else:
            note_in_range = bool(explicit_in_range)
        cx = area.left() + slot * (i + 0.5)
        # v0.5.7.9: skip bar/whisker geometry when mean is non-finite.
        # _cent_to_y(NaN) yields NaN; passing NaN to drawRoundedRect is
        # undefined Qt behavior. Still render the note label so the gap
        # is explicable, plus a "(no data)" subtitle under it.
        if not math.isfinite(mean):
            p.setPen(QColor(210, 210, 220))
            if bar_w < 22:
                p.save()
                p.translate(cx, area.bottom() + 8)
                p.rotate(-45)
                p.drawText(QRectF(-60, -6, 60, 14),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           label)
                p.restore()
            else:
                p.drawText(QRectF(cx - slot / 2, area.bottom() + 6,
                                  slot, 16),
                           Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                           label)
            p.setPen(QColor(140, 140, 160))
            p.setFont(_mono(8))
            p.drawText(QRectF(cx - slot / 2, area.bottom() + 26,
                              slot, 14),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       "(no data)")
            p.setFont(_mono(9))
            continue
        top_y = _cent_to_y(mean, area)
        # Color by magnitude — same thresholds as the in-app tuner widget.
        # v0.6 Phase-4 (Item 7): notes outside the instrument's fingered
        # range render in a desaturated grey instead of the green-yellow-
        # red gradient, so an altissimo overtone doesn't visually scream
        # "out of tune" when it's really just out of range.
        m = abs(mean)
        if not note_in_range:
            col = QColor(140, 140, 150)
        elif m <= 5:
            col = QColor(60, 220, 100)
        elif m <= 15:
            col = QColor(255, 200, 40)
        else:
            col = QColor(240, 80, 80)

        rect = QRectF(cx - bar_w / 2,
                      min(zero_y, top_y),
                      bar_w,
                      abs(top_y - zero_y))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(rect, 3, 3)

        # Out-of-range marker: a small ⚠ glyph above the bar (or below if
        # the bar tip is already high). Visually distinct from the cent
        # label so it doesn't get mistaken for a value.
        if not note_in_range:
            warn_y = area.top() - 2 if mean >= 0 else area.bottom() - 10
            p.setPen(QColor(220, 180, 80))
            p.setFont(_sans(11, bold=True))
            p.drawText(QRectF(cx - slot / 2, warn_y, slot, 14),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       '⚠')
            p.setFont(_mono(9))

        # Clamp indicator if the value went past ±MAX_CENTS.
        if abs(mean) > _MAX_CENTS:
            arrow_y = area.top() if mean > 0 else area.bottom() - 6
            p.setBrush(col)
            poly = QPolygonF([QPointF(cx, arrow_y),
                              QPointF(cx - 6, arrow_y + (6 if mean > 0 else -6)),
                              QPointF(cx + 6, arrow_y + (6 if mean > 0 else -6))])
            p.drawPolygon(poly)

        # Std-deviation whisker at the top of the bar.
        if count > 1 and std > 0:
            top_w = _cent_to_y(mean + std, area)
            bot_w = _cent_to_y(mean - std, area)
            p.setPen(QPen(QColor(230, 230, 240, 200), 1.5))
            p.drawLine(QPointF(cx, top_w), QPointF(cx, bot_w))
            p.drawLine(QPointF(cx - 4, top_w), QPointF(cx + 4, top_w))
            p.drawLine(QPointF(cx - 4, bot_w), QPointF(cx + 4, bot_w))

        # Numeric cent label above (or below for negative) the bar tip,
        # snapped to the precision the measurement supports at the
        # negotiated sample rate.
        if bar_w >= 14:
            cent_txt = _format_cents_label(mean, freq, sample_rate)
            p.setPen(QColor(220, 220, 235))
            p.setFont(_mono(8))
            if mean >= 0:
                label_y = max(area.top(), top_y - 14)
            else:
                label_y = min(area.bottom() - 12, top_y + 2)
            p.drawText(QRectF(cx - slot / 2, label_y, slot, 12),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       cent_txt)
            p.setFont(_mono(9))

        # Note label below the chart.
        p.setPen(QColor(210, 210, 220))
        # Rotate labels if they would overlap.
        if bar_w < 22:
            p.save()
            p.translate(cx, area.bottom() + 8)
            p.rotate(-45)
            p.drawText(QRectF(-60, -6, 60, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       label)
            p.restore()
        else:
            p.drawText(QRectF(cx - slot / 2, area.bottom() + 6,
                              slot, 16),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       label)

        # N count below the label, smaller and dimmer.
        p.setPen(QColor(140, 140, 160))
        p.setFont(_mono(8))
        p.drawText(QRectF(cx - slot / 2, area.bottom() + 26,
                          slot, 14),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   f"n={count}")
        p.setFont(_mono(9))


def _draw_no_data(p: QPainter) -> None:
    area = _plot_area()
    p.setPen(QColor(150, 150, 170))
    p.setFont(_sans(14))
    p.drawText(area, Qt.AlignmentFlag.AlignCenter,
               "No measurements to chart yet")


def _draw_footer(p: QPainter, footer: str) -> None:
    p.setPen(QColor(140, 140, 160))
    p.setFont(_sans(10))
    p.drawText(QRectF(_MARGIN_L, _HEIGHT - 26,
                      _WIDTH - _MARGIN_L - _MARGIN_R, 18),
               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               footer)

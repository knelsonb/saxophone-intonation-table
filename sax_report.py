"""Report renderers: TXT, PDF and PNG chart from one frozen session snapshot.

ExportController (sax_export) owns the dialogs: it asks for maker/model and a
file name, builds a SessionSnapshot once from the window, and hands it to one
of the renderers here. The renderers never open a dialog or read the window,
so the same snapshot always gives the same file, and a renderer can be called
from a test or a script without a MainWindow.

Every file is written through sax_atomic.atomic_open: a failed export leaves
any previous file with that name untouched. Errors propagate to the caller,
which explains them to the user.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from xml.sax.saxutils import escape as _xml_escape

import sax_instruments
from sax_atomic import atomic_open
from sax_i18n import STRINGS
from sax_intonation_chart import render_intonation_chart
from sax_pitch import CHROMA, format_cents, midi_note_name
from sax_theme import cents_band

# PDF mean-column text colour for each cents_band() result.
_PDF_BAND_COLOR = {'in': '#1b5e20', 'close': '#e65100', 'off': '#b71c1c'}


@dataclass(frozen=True)
class NoteRow:
    """One played note of the table, in both spellings."""
    midi_sounding: int
    midi_fingered: int
    sounding: str
    fingered: str
    mean: float
    std: float
    n: int
    freq_hz: float          # nominal frequency of the sounding note at A4
    band: str               # cents_band(mean): 'in' / 'close' / 'off'
    in_range: bool          # fingered note inside the instrument's range


@dataclass(frozen=True)
class SessionSnapshot:
    """Everything an export needs, taken from the window at one moment."""
    instrument: str
    instrument_label: str   # long display name, in the window's language
    transposition: int      # semitones, sounding minus fingered
    a4: float
    display: str            # 'griff' (fingered) or 'klingend' (sounding)
    sample_rate: int        # gates the cent precision of printed values
    rows: tuple[NoteRow, ...]
    maker: str
    model: str
    nickname: str
    created: datetime.datetime

    @property
    def total(self) -> int:
        return sum(r.n for r in self.rows)


def take_snapshot(*, instrument: str, instrument_label: str, a4: float,
                  display: str, sample_rate: int, stats: Mapping,
                  maker: str = '', model: str = '', nickname: str = '',
                  created: datetime.datetime | None = None) -> SessionSnapshot:
    """Freeze the live table into a SessionSnapshot.

    ``stats`` maps sounding MIDI to an object with ``mean``, ``std`` and
    ``n``. Rows seeded for the instrument range have n == 0 until played;
    exports list measured notes only.
    """
    transp = sax_instruments.transp_map().get(instrument, 0)
    lo_f, hi_f = sax_instruments.fingered_range(instrument)
    rows = []
    for midi_kl, st in sorted((k, st) for k, st in stats.items() if st.n > 0):
        midi_gr = midi_kl - transp
        rows.append(NoteRow(
            midi_sounding=midi_kl,
            midi_fingered=midi_gr,
            sounding=midi_note_name(midi_kl),
            fingered=midi_note_name(midi_gr),
            mean=st.mean,
            std=st.std,
            n=st.n,
            freq_hz=a4 * (2.0 ** ((midi_kl - 69) / 12.0)),
            band=cents_band(st.mean),
            in_range=(lo_f <= midi_gr <= hi_f),
        ))
    return SessionSnapshot(
        instrument=instrument, instrument_label=instrument_label,
        transposition=transp, a4=a4, display=display,
        sample_rate=sample_rate, rows=tuple(rows),
        maker=maker, model=model, nickname=nickname,
        created=created or datetime.datetime.now())


def translator(lang: str) -> Callable[..., str]:
    """``_t(key, **kwargs)`` for ``lang``, with the window's fallbacks."""
    table = STRINGS.get(lang, STRINGS['de'])

    def _t(key: str, **kwargs) -> str:
        s = table.get(key, STRINGS['de'].get(key, f'[{key}]'))
        return s.format(**kwargs) if kwargs else s
    return _t


def format_stamp(dt: datetime.datetime, lang: str) -> str:
    """Date and time as a reader of ``lang`` writes it: 13.10.2026 16:44 in
    German, 2026-10-13 16:44 (ISO) in English."""
    return dt.strftime('%d.%m.%Y %H:%M' if lang == 'de' else '%Y-%m-%d %H:%M')


# ── TXT ────────────────────────────────────────────────────────────────────
def render_txt(snapshot: SessionSnapshot, lang: str) -> str:
    """The plain-text report, without a trailing newline."""
    _t = translator(lang)
    s = snapshot
    lines = [
        _t('txt_header'),
        '=' * 54,
        _t('txt_instr', name=s.instrument_label),
    ]
    if s.maker:
        lines.append(_t('txt_maker', maker=s.maker))
    if s.model:
        lines.append(_t('txt_model', model=s.model))
    lines += [
        _txt_transposition(s.transposition, _t),
        _t('txt_a4', hz=s.a4),
        _t('txt_date', dt=format_stamp(s.created, lang)),
        '',
        _t('txt_col_header',
           fingered=_t('pdf_col_finger'),
           sounding=_t('pdf_col_sound'),
           mean=_t('txt_col_mean'),
           std=_t('txt_col_std'),
           n='N'),
        '-' * 62,
    ]
    for r in s.rows:
        mean_str = format_cents(r.mean, r.freq_hz, s.sample_rate)
        std_str = format_cents(r.std, r.freq_hz, s.sample_rate).lstrip('+-')
        lines.append(
            f"{r.fingered:<12} {r.sounding:<12}"
            f" {mean_str:>7}   {std_str:>6}  {r.n:>5}  {_make_bar(r.mean, 24)}")
    lines += ['', _t('txt_total', total=s.total, notes=len(s.rows))]
    return '\n'.join(lines)


def write_txt(snapshot: SessionSnapshot, lang: str, path: Path | str) -> None:
    with atomic_open(path) as f:
        f.write(render_txt(snapshot, lang))


def _txt_transposition(transp: int, _t: Callable[..., str]) -> str:
    """The header line naming the transposition.

    For whole octaves "fingered C sounds as C" would read like no
    transposition at all, so those say the octave instead.
    """
    if not transp:
        return _t('txt_no_transp')
    if transp % 12:
        return _t('txt_transp', note=CHROMA[transp % 12])
    octaves = abs(transp) // 12
    if octaves == 1:
        return _t('txt_transp_8va' if transp > 0 else 'txt_transp_8vb')
    return _t('txt_transp_octaves_up' if transp > 0 else 'txt_transp_octaves_down',
              n=octaves)


def _make_bar(cents, w=20):
    half = w // 2
    fill = int(min(1.0, abs(cents) / 40.0) * half)
    if cents > 1:
        return ' '*half + '│' + '█'*fill + '░'*(half-fill)
    elif cents < -1:
        return '░'*(half-fill) + '█'*fill + '│' + ' '*half
    return ' '*half + '│' + ' '*half


# ── PDF ────────────────────────────────────────────────────────────────────
def pdf_available() -> bool:
    """reportlab is optional; the PDF button explains when it is missing."""
    try:
        import reportlab.platypus  # noqa: F401
    except ImportError:
        return False
    return True


def render_pdf(snapshot: SessionSnapshot, lang: str, path: Path | str) -> None:
    """Write the PDF report. Raises ImportError without reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4 as RL_A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    _t = translator(lang)
    s = snapshot
    transp = s.transposition
    styles = getSampleStyleSheet()
    story = []

    ts_title = ParagraphStyle('T', parent=styles['Title'],
                              fontSize=18, textColor=colors.HexColor('#1a237e'), spaceAfter=4)
    ts_sub   = ParagraphStyle('S', parent=styles['Normal'],
                              fontSize=11, textColor=colors.HexColor('#555'), spaceAfter=2)

    story.append(Paragraph(_t('pdf_title'), ts_title))
    story.append(Paragraph(_xml_escape(s.instrument_label), ts_sub))
    if s.maker:
        story.append(Paragraph(_t('pdf_maker', maker=_xml_escape(s.maker)), ts_sub))
    if s.model:
        story.append(Paragraph(_t('pdf_model', model=_xml_escape(s.model)), ts_sub))
    story.append(Paragraph(
        (_t('pdf_transp', note=CHROMA[transp % 12], n=transp)
         if transp else _t('pdf_no_transp')), ts_sub))
    story.append(Paragraph(_t('pdf_a4', hz=s.a4), ts_sub))
    # The PDF keeps the German-style date in both languages; its golden
    # file pins that, unlike the TXT report and the chart.
    story.append(Paragraph(
        _t('pdf_created', dt=s.created.strftime('%d.%m.%Y %H:%M')), ts_sub))
    story.append(Spacer(1, 10*mm))

    if not s.rows:
        story.append(Paragraph(_t('pdf_no_data'), styles['Normal']))
    else:
        story.append(Paragraph(
            _t('pdf_summary', notes=len(s.rows), total=s.total),
            ParagraphStyle('I', parent=styles['Normal'],
                           fontSize=10, textColor=colors.HexColor('#666'), spaceAfter=6)))

        data = [[_t('pdf_col_finger'), _t('pdf_col_sound'),
                 _t('pdf_col_mean'), _t('pdf_col_std'),
                 _t('pdf_col_n'), _t('pdf_col_tend')]]
        for r in s.rows:
            mean_str = format_cents(r.mean, r.freq_hz, s.sample_rate)
            std_str = format_cents(r.std, r.freq_hz, s.sample_rate).lstrip('+-')
            data.append([
                r.fingered, r.sounding,
                mean_str,
                f"±{std_str}" if r.n > 1 else '–',
                str(r.n),
                _make_bar_ascii(r.mean),
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
        for i, r in enumerate(s.rows, 1):
            c = colors.HexColor(_PDF_BAND_COLOR[r.band])
            ts.add('TEXTCOLOR', (2,i), (2,i), c)
            ts.add('FONTNAME',  (2,i), (2,i), 'Helvetica-Bold')
        tbl.setStyle(ts)
        story.append(tbl)

    with atomic_open(path, binary=True) as f:
        doc = SimpleDocTemplate(f, pagesize=RL_A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        doc.build(story)


def _make_bar_ascii(cents, w=16):
    half = w // 2
    fill = int(min(1.0, abs(cents) / 40.0) * half)
    if cents > 1:
        return ' '*half + '|' + '#'*fill + '.'*(half-fill)
    if cents < -1:
        return '.'*(half-fill) + '#'*fill + '|' + ' '*half
    return ' '*half + '|' + ' '*half


# ── Chart ──────────────────────────────────────────────────────────────────
def render_chart(snapshot: SessionSnapshot, lang: str, path: Path | str) -> None:
    """Write the PNG bar chart. Raises sax_intonation_chart.ChartEncodeError
    when Qt cannot encode the image, OSError when the file cannot be written."""
    _t = translator(lang)
    s = snapshot
    disp_griff = (s.display == 'griff')
    # Fingered MIDI and the range flag let the chart grey out notes outside
    # the instrument's nominal range.
    notes = [((r.fingered if disp_griff else r.sounding),
              r.mean, r.std, r.n, r.freq_hz, r.midi_fingered, r.in_range)
             for r in s.rows]
    dt = format_stamp(s.created, lang)
    if s.maker or s.model:
        subtitle = _t('chart_subtitle_id', instr=s.instrument_label,
                      maker=s.maker, model=s.model, a4=s.a4, dt=dt)
    else:
        subtitle = _t('chart_subtitle', instr=s.instrument_label, a4=s.a4, dt=dt)
    render_intonation_chart(
        notes=notes,
        title=_t('chart_title'),
        subtitle=subtitle,
        footer=_t('chart_footer', notes=len(s.rows), total=s.total),
        output_path=str(path),
        sample_rate=s.sample_rate,
        instrument=s.instrument,
    )

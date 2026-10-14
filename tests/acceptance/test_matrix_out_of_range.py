"""Played notes outside the keyed range, in the matrix layout.

The matrix's guide cells are the notes the instrument's keys give. A note I play
outside them (low A on an alto, altissimo) gets a cell with its cents, drawn
with a dashed border (tests/contracts/test_matrix_cells.py), and the caption
counts it apart from the guide cells, so it never claims more cells played than
there are.
"""
from __future__ import annotations

from tests.harness import Tone

MATRIX = {"layout_mode_preference": "matrix"}
ALTO_GUIDE_CELLS = 33                            # fingered Bb3..F#6


def on(instrument: str, **config) -> dict:
    return {"last_instrument_key": instrument, **config}


def test_notes_outside_the_keys_get_cells_and_are_counted_apart(make_app):
    app = make_app(config=on("eb_alto", **MATRIX))
    app.launch()
    for note in ("D#4", "C6", "C3"):             # fingered C5, A6, A3
        app.play(Tone(note=note, timbre="sax", seconds=1.0))

    rows = {r.fingered: r for r in app.table.rows()}
    assert sorted(rows) == ["A3", "A6", "C5"]
    assert all(r.mean is not None and r.n >= 5 for r in rows.values())
    assert app.table.caption == app.text(
        "table_matrix_title_outside", played=1, total=ALTO_GUIDE_CELLS, outside=2)


def test_with_only_guide_notes_played_the_caption_is_unchanged(make_app):
    app = make_app(config=on("eb_alto", **MATRIX))
    app.launch()
    app.play(Tone(note="D#4", timbre="sax", seconds=1.0))
    assert app.table.caption == app.text(
        "table_matrix_title", played=1, total=ALTO_GUIDE_CELLS)


def test_altissimo_kept_by_the_filter_has_a_column_in_the_matrix(make_app):
    """With filter-to-range on, fingered D7 is kept (the alto reaches it), so
    the matrix grows a column for it rather than hiding it."""
    app = make_app(config=on("eb_alto", allow_out_of_range=False, **MATRIX))
    app.launch()
    for note in ("D#4", "F6"):                   # fingered C5, D7
        app.play(Tone(note=note, timbre="sax", seconds=1.0))

    assert [r.fingered for r in app.table.rows()] == ["C5", "D7"]
    assert app.table.caption == app.text(
        "table_matrix_title_outside", played=1, total=ALTO_GUIDE_CELLS, outside=1)


def test_the_caption_reads_in_german(make_app):
    app = make_app(config=on("eb_alto", last_lang="de", **MATRIX))
    app.launch()
    for note in ("D#4", "C3"):                   # fingered C5, A3
        app.play(Tone(note=note, timbre="sax", seconds=1.0))
    assert app.table.caption == app.text(
        "table_matrix_title_outside", played=1, total=ALTO_GUIDE_CELLS, outside=1)
    assert "außerhalb des Rasters" in app.table.caption


def test_a_brief_squeak_adds_no_empty_column(make_app):
    """A stray with fewer readings than a cell needs to show would only add
    a blank column; once it has enough, its column appears."""
    app = make_app(config=on("eb_alto", **MATRIX))
    app.launch()
    before = app.table.columns
    app.play(Tone(note="C2", seconds=0.2))       # a few readings only
    assert 0 < app.history_counts[1] < 5         # measured, below min n
    assert app.table.columns == before

    app.play(Tone(note="C2", seconds=1.0))
    assert len(app.table.columns) > len(before)

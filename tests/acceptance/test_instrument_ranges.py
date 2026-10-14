"""An instrument's two ranges, as a player meets them.

The table's blank guide rows show the notes the instrument's keys give (the
keyed range). Whether a played note counts as out of range, for the
wrong-instrument hint, the filter-to-range switch and the greyed bars in the
chart, is judged on the wider range skilled players reach (the extended range):
altissimo, high string positions, high brass.
"""
from __future__ import annotations

import pytest

from tests.harness import Tone


def on(instrument: str, **config) -> dict:
    return {"last_instrument_key": instrument, **config}


FILTER_ON = {"allow_out_of_range": False}


def test_the_alto_guide_rows_run_from_low_bb_to_high_f_sharp(make_app):
    """The keys of an alto give fingered Bb3 to F#6: those are the blank rows."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[0].sounding) == ("A#/Bb3", "C#/Db3")
    assert (listed[-1].fingered, listed[-1].sounding) == ("F#/Gb6", "A5")
    assert len(listed) == 33


def test_the_baritone_guide_rows_include_its_low_a(make_app):
    app = make_app(config=on("eb_bari"))
    app.launch()
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[-1].fingered) == ("A3", "F#/Gb6")


@pytest.mark.parametrize("instrument, note", [
    ("eb_alto", "F6"),        # fingered altissimo D7
    ("eb_alto", "G#6"),       # fingered altissimo F7
    ("cello", "B4"),
    ("violin", "F#6"),
    ("trp_bb", "A#5"),        # fingered high C
    ("horn_f", "F5"),         # fingered C6
])
def test_a_note_a_skilled_player_reaches_is_kept_and_not_flagged(make_app, instrument, note):
    """Held well past the hint's threshold with filter-to-range on: the note is
    kept and no wrong-instrument hint appears."""
    app = make_app(config=on(instrument, **FILTER_ON))
    app.launch()
    app.play(Tone(note=note, seconds=2.0))
    assert app.table.row(note, by="sounding").n >= 10
    assert app.audio.notice is None


def test_notes_beyond_the_alto_are_dropped_by_the_filter_and_flagged(make_app):
    """Fingered F#7, G7 and G#7 are past the alto's extended top F7. The hint
    needs three different such notes (see test_wrong_instrument_hint.py)."""
    app = make_app(config=on("eb_alto", **FILTER_ON))
    app.launch()
    for note in ("A6", "A#6", "B6"):            # fingered F#7, G7, G#7
        app.play(Tone(note=note, seconds=1.0))
    assert app.table.rows() == []
    assert app.audio.notice is not None


def test_the_chart_greys_only_notes_beyond_the_extended_range(make_app):
    """Low A has no key on an alto (greyed); altissimo F7 is reachable (not greyed)."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in ("C3", "A4", "G#6"):            # fingered A3, F#5, F7
        app.play(Tone(note=note, timbre="sax", seconds=1.0))
    app.export.chart()
    flags = {row[0]: row[6] for row in app.export.chart_model["notes"]}
    assert flags == {"A3": False, "F#/Gb5": True, "F7": True}


def test_an_edited_range_moves_the_guide_rows_but_keeps_altissimo_in_range(make_app):
    """Narrowing the alto to C4..C6 in the range editor changes the blank rows
    (seen after a restart: narrowing in place leaves the old blank rows, a
    separate known issue); altissimo is still a note the alto reaches, so the
    filter keeps it."""
    app = make_app(config=on("eb_alto", **FILTER_ON))
    app.launch()
    assert app.settings.edit_range("C4", "C6")
    app.restart()
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[-1].fingered) == ("C4", "C6")

    app.play(Tone(note="G#6", seconds=2.0))     # fingered F7
    assert app.table.row("G#6", by="sounding").n >= 10
    assert app.audio.notice is None


def test_the_generic_c_instrument_counts_every_note_as_in_range(make_app):
    """'C Instrument' has no instrument behind it, so a flute or violin player
    using it is never told a note is out of range: high notes are kept by the
    filter and raise no hint."""
    app = make_app(config=on("c", **FILTER_ON))
    app.launch()
    for note in ("A6", "B6", "C#7"):
        app.play(Tone(note=note, seconds=1.0))
    assert len(app.table.rows()) == 3
    assert app.audio.notice is None


def test_a_custom_instrument_without_a_range_counts_every_note_as_in_range(make_app):
    app = make_app(config=on("eb_alto", **FILTER_ON))
    app.launch()
    key = app.settings.add_custom_instrument("Test Horn", transpose=0)
    assert key and app.instrument == key
    for note in ("C2", "A6", "C#7"):
        app.play(Tone(note=note, seconds=1.0))
    assert len(app.table.rows()) == 3
    assert app.audio.notice is None


def test_narrowing_the_range_drops_the_blank_rows_at_once(make_app):
    """No restart needed: the rows run C4..C6 right after the edit, and a note
    I played below the new range keeps its row."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    app.play(Tone(note="C#3", timbre="sax", seconds=1.0))    # fingered Bb3
    assert app.settings.edit_range("C4", "C6")

    listed = [r.fingered for r in app.table.rows(played_only=False)]
    assert listed[0] == "A#/Bb3"                 # played, so kept
    assert listed[1:2] == ["C4"] and listed[-1] == "C6"
    assert len(listed) == 1 + 25

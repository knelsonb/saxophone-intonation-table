"""Playing and reading the table: what a player sees while practising.

Scales on the alto with the 5/12 cent colour rule, the lowest and highest notes
of several instruments, switching instruments and the fingered/sounding toggle,
and cleaning the table up again (reset, right-click discard) in both the list
and the matrix layout.
"""
from __future__ import annotations

import re

import pytest

from tests.harness import Tone

MATRIX = {"layout_mode_preference": "matrix"}


def on(instrument: str, **config) -> dict:
    """Seed config: start the app with ``instrument`` already selected, so the
    table shows only that instrument's range."""
    return {"last_instrument_key": instrument, **config}


def plain(cell: str) -> str:
    """'F#/Gb5' -> 'F#5': a table cell as a name ``Tone`` accepts."""
    m = re.match(r"^(.*?)(-?\d+)$", cell)
    return m.group(1).split("/")[0] + m.group(2)


def sounding(app) -> list[str]:
    return [r.sounding for r in app.table.rows()]


def octaves(app) -> list[int]:
    """The octave numbers heading the matrix columns ('Oct 3' -> 3)."""
    return [int(re.search(r"[+-]?\d+$", c).group()) for c in app.table.columns]


def skeleton(app) -> list[str]:
    """Every note the table lists, played or not."""
    return [r.sounding for r in app.table.rows(played_only=False)]


# ------------------------------------------------------------------ scales
# An Eb alto sounds a major sixth below what is fingered.
ALTO_C_MAJOR = [  # (fingered, sounding as the table writes it)
    ("C4", "D#/Eb3"), ("D4", "F3"), ("E4", "G3"), ("F4", "G#/Ab3"),
    ("G4", "A#/Bb3"), ("A4", "C4"), ("B4", "D4"), ("C5", "D#/Eb4"),
]


@pytest.mark.parametrize("cents, band", [(0, "green"), (+10, "amber"), (-20, "red")],
                         ids=["in-tune", "ten-sharp", "twenty-flat"])
def test_an_alto_scale_lists_every_note_with_its_offset_and_colour(make_app, cents, band):
    """Fingered C major on the alto, played evenly off: every note gets its row
    under both names with the offset I played and the matching colour."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    for _fingered, cell in ALTO_C_MAJOR:
        app.play(Tone(note=plain(cell), cents=cents, timbre="sax", seconds=0.75))

    rows = app.table.rows()
    assert [(r.fingered, r.sounding) for r in rows] == ALTO_C_MAJOR
    for r in rows:
        assert (r.mean, r.band) == (cents, band), r
        assert r.n >= 10, r
    assert app.table.caption == app.text(
        "table_summary", notes=8, total=app.table.total_measurements)


@pytest.mark.parametrize("played, shown, band", [
    (+5.3, +5, "green"), (-5.3, -5, "green"), (+5.6, +6, "amber"),
    (+12.3, +12, "amber"), (-12.3, -12, "amber"), (+12.6, +13, "red"),
])
def test_the_colour_matches_the_whole_cents_number_shown(make_app, played, shown, band):
    """Green up to 5 cents, amber up to 12, judged on the number I read in the
    table: '+5' is never amber and '+12' is never red."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    app.play(Tone(note="A4", cents=played, timbre="sax", seconds=1.0))

    row = app.table.row("A4", by="sounding")
    assert (row.mean, row.band) == (shown, band)


# ------------------------------------------------------------------- ranges
@pytest.mark.parametrize("instrument, low, high", [
    ("bb_bass", ("A3", "G1"), ("C7", "A#/Bb4")),
    ("tuba_f", ("D1", "D1"), ("E4", "E4")),
    ("flute_piccolo", ("D4", "D5"), ("A6", "A7")),
    ("bb_soprano", ("A3", "G3"), ("C7", "A#/Bb6")),
    ("bb_tenor", ("A3", "G2"), ("C7", "A#/Bb5")),
    ("eb_bari", ("A3", "C2"), ("C7", "D#/Eb5")),
])
def test_the_lowest_and_highest_notes_read_as_the_right_note_in_tune(make_app, instrument,
                                                                     low, high):
    """The table's range runs from the instrument's lowest to its highest note,
    and both ends, played in tune, land in the right row at about 0 cents."""
    app = make_app(config=on(instrument))
    app.launch()
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[0].sounding) == low
    assert (listed[-1].fingered, listed[-1].sounding) == high

    for fingered, cell in (low, high):
        app.table.reset()
        app.play(Tone(note=plain(cell), timbre="sax", seconds=1.0))
        rows = app.table.rows()
        assert [(r.fingered, r.sounding) for r in rows] == [(fingered, cell)]
        assert abs(rows[0].mean) <= 2 and rows[0].n >= 10, rows[0]
        assert app.tuner.reading().note == fingered


@pytest.mark.parametrize("instrument, note", [
    ("tuba_f", "D1"), ("tuba_bbb", "E1"), ("tbn_bass", "G#/Ab1"), ("euph_bc", "E2"),
])
def test_a_low_brass_pedal_with_a_weak_fundamental_reads_the_right_note(make_app,
                                                                        instrument, note):
    """Low brass pedal tones carry little fundamental; the app must still name
    the note itself, not the octave above."""
    app = make_app(config=on(instrument))
    app.launch()

    app.play(Tone(note=plain(note), timbre="weak_fundamental", seconds=1.0))

    rows = app.table.rows()
    assert [r.sounding for r in rows] == [note]
    assert abs(rows[0].mean) <= 2 and rows[0].n >= 10, rows[0]


@pytest.mark.parametrize("a4", [
    440,
    pytest.param(431, marks=pytest.mark.xfail(strict=True, reason=(
        "Sauron-11: 27 Hz detection floor clips piano A0 at A4=431 (26.94 Hz), "
        "reads +5 ct"))),
    pytest.param(430, marks=pytest.mark.xfail(strict=True, reason=(
        "Sauron-11: 27 Hz detection floor clips piano A0 at A4=430 (26.88 Hz), "
        "reads +9 ct"))),
])
def test_the_lowest_piano_a_reads_in_tune_at_any_concert_pitch(make_app, a4):
    """A piano tuned to a lower concert pitch still has an A0 the app can read."""
    app = make_app(config=on("piano"))
    app.launch()
    app.set_a4(a4)

    app.play(Tone(note="A0", a4=a4, timbre="sine", seconds=1.5))

    row = app.table.row("A0", by="sounding")
    assert abs(row.mean) <= 1, row


# ------------------------------------------------- instrument and display
@pytest.mark.parametrize("instrument, fingered", [
    ("eb_alto", "F#/Gb5"), ("bb_tenor", "B5"), ("bb_soprano", "B4"),
    ("clar_bb", "B4"), ("horn_f", "E5"), ("flute_c", "A4"),
])
def test_concert_a_reads_as_each_instruments_own_fingered_note(make_app, instrument,
                                                               fingered):
    """The same 440 Hz is a different written note on each horn; the tuner and
    the table give me the one I finger."""
    app = make_app(config=on(instrument))
    app.launch()

    app.play(Tone(note="A4", timbre="sax", seconds=1.0))

    row = app.table.row("A4", by="sounding")
    assert (row.fingered, row.mean) == (fingered, 0)
    assert app.tuner.reading().note == fingered
    status = app.tuner.reading().status_text
    assert fingered in status and "A4" in status


def test_the_display_toggle_swaps_which_name_comes_first(make_app):
    """Fingered view leads with the written note; sounding view leads with the
    concert note, in the tuner and in the table headers."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    fingered_col, sounding_col = app.text("col_fingered"), app.text("col_sounding")

    app.play(Tone(note="A4", seconds=1.0))
    assert app.display == "fingered"
    assert app.tuner.reading().note == "F#/Gb5"
    assert app.table.columns[:2] == [fingered_col, sounding_col]

    app.set_display("sounding")
    assert app.table.columns[:2] == [sounding_col, fingered_col]
    app.play(Tone(note="A4", seconds=1.0))
    assert app.tuner.reading().note == "A4"
    row = app.table.row("A4")
    assert (row.fingered, row.sounding, row.mean) == ("F#/Gb5", "A4", 0)


@pytest.mark.xfail(strict=True, reason=(
    "Frodo-3: switching instrument relabels the previous instrument's readings "
    "with the new transposition"))
def test_switching_instrument_does_not_pass_off_the_old_readings_as_the_new_ones(make_app):
    """I played A4 on the alto, then picked the tenor: the tenor table must not
    show my alto notes as tenor B5. Either the app clears the table (asking
    first) or it keeps the alto readings with the alto."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    app.play(Tone(note="A4", seconds=1.0))
    assert app.table.row("A4", by="sounding").fingered == "F#/Gb5"

    with app.dialogs.scope():
        app.dialogs.expect("question", "Yes", optional=True)    # "clear the table?"
        app.pick_instrument("bb_tenor")

    assert app.table.rows() == []


@pytest.mark.xfail(strict=True, reason=(
    "Frodo-3: maker/model from the previous instrument is reused for the next "
    "instrument's export"))
def test_a_new_instrument_asks_for_its_own_maker_and_model(make_app):
    """The alto is a Selmer; the tenor I switch to is a Yamaha. Its export must
    ask again instead of labelling the tenor a Selmer."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    app.play(Tone(note="A4", seconds=1.0))
    app.export.txt(maker="Selmer", model="Mark VI")
    asked = app.dialogs.forms_shown.count("maker_model")
    assert asked == 1

    with app.dialogs.scope():
        app.dialogs.expect("question", "Yes", optional=True)
        app.pick_instrument("bb_tenor")
    app.play(Tone(note="C4", seconds=1.0))
    app.export.txt(maker="Yamaha", model="YTS-62")

    assert app.dialogs.forms_shown.count("maker_model") == 2


# ------------------------------------------------------------ reset, min-N
def test_reset_clears_every_reading_and_starts_counting_afresh(make_app):
    """After Reset the table is empty and the next note starts from one take."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in ("C4", "E4", "G4"):
        app.play(Tone(note=note, seconds=1.0))
    first_n = app.table.row("C4", by="sounding").n
    assert app.table.total_measurements > 0

    app.table.reset()

    assert app.table.rows() == []
    assert app.table.total_measurements == 0
    assert app.table.caption == app.text("table_empty_hint")
    app.play(Tone(note="C4", seconds=1.0))
    assert sounding(app) == ["C4"]
    assert abs(app.table.row("C4", by="sounding").n - first_n) <= 2


@pytest.mark.xfail(strict=True, reason=(
    "Frodo-13: Reset also removes the instrument's range rows"))
def test_reset_keeps_the_instruments_range_on_screen(make_app):
    """Reset throws away readings, not the guide rows that show my range."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    before = skeleton(app)
    app.play(Tone(note="C4", seconds=1.0))

    app.table.reset()

    assert skeleton(app) == before
    assert app.table.rows() == []


@pytest.mark.xfail(strict=True, reason=(
    "Frodo-13: a note with fewer readings than the minimum vanishes, guide row "
    "included"))
def test_a_note_below_the_minimum_count_stays_listed(make_app):
    """A short note under the minimum count is shown dimmed; its row does not
    disappear from the range until enough readings arrive."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    app.settings.min_n(10)
    assert "D4" in skeleton(app)

    app.play(Tone(note="D4", timbre="sax", seconds=0.3))     # about 8 readings

    assert app.table.total_measurements > 0
    assert "D4" in skeleton(app)


@pytest.mark.xfail(strict=True, reason=(
    "Frodo-13: browsing instruments leaves their blank range rows behind"))
def test_looking_at_another_instrument_leaves_no_blank_rows_behind(make_app):
    """Peeking at the bari and coming back to the alto shows the alto's range
    only, not a mix of both."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    before = skeleton(app)

    app.pick_instrument("eb_bari")
    app.pick_instrument("eb_alto")

    assert skeleton(app) == before


# ------------------------------------------------------------------ discard
def _three_notes_and_a_short_one(app) -> None:
    """C4, E4 and G4 held; D4 too short to pass a minimum count of 10."""
    app.settings.min_n(10)
    for note, seconds in (("C4", 1.0), ("D4", 0.3), ("E4", 1.0), ("G4", 1.0)):
        app.play(Tone(note=note, timbre="sax", seconds=seconds))
    assert sounding(app) == ["C4", "E4", "G4"]


@pytest.mark.parametrize("layout", ["list", "matrix"])
def test_discarding_a_note_removes_that_note_even_with_hidden_short_notes(make_app, layout):
    """Right-click E4 > discard removes E4, even though a short D4 below the
    minimum count sits unseen just before it."""
    app = make_app(config=on("eb_alto", **(MATRIX if layout == "matrix" else {})))
    app.launch()
    _three_notes_and_a_short_one(app)

    app.table.discard("E4")

    assert sounding(app) == ["C4", "G4"]


def test_matrix_keeps_a_low_tenor_note_and_its_menu_after_switching_to_sounding(make_app):
    """On the tenor, low A (sounding G2) stays in the matrix when I switch to
    sounding names, the columns relabel to concert octaves, and right-click on
    it discards G2 itself."""
    app = make_app(config=on("bb_tenor", **MATRIX))
    app.launch()
    app.play(Tone(note="G2", timbre="sax", seconds=1.0))
    app.play(Tone(note="C4", timbre="sax", seconds=1.0))
    assert sounding(app) == ["G2", "C4"]
    assert all(re.search(r"[+-]\d+$", c) for c in app.table.columns)   # relative octaves

    app.set_display("sounding")

    assert sounding(app) == ["G2", "C4"]
    assert 2 in octaves(app)
    app.table.discard("G2")
    assert sounding(app) == ["C4"]


def test_matrix_columns_follow_an_instrument_change(make_app):
    """Switching from alto to bari in sounding view adds the bari's lower
    octave, and its low C2 shows up and can be discarded."""
    app = make_app(config=on("eb_alto", last_display_mode="klingend", **MATRIX))
    app.launch()
    alto_octaves = octaves(app)
    assert 3 in alto_octaves                       # alto low A sounds C3

    app.pick_instrument("eb_bari")

    bari_octaves = octaves(app)
    assert 2 in bari_octaves                       # bari low A sounds C2
    assert min(bari_octaves) < min(alto_octaves)
    app.play(Tone(note="C2", timbre="sax", seconds=1.0))
    app.play(Tone(note="C4", timbre="sax", seconds=1.0))
    assert [(r.fingered, r.sounding) for r in app.table.rows()] == [("A3", "C2"),
                                                                    ("A5", "C4")]
    app.table.discard("C2")
    assert sounding(app) == ["C4"]

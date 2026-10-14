"""The "is this the right instrument?" hint, as a player meets it.

The hint appears once three different notes, each actually measured, fall
outside the selected instrument's extended range. A long held note, repeated
attacks on one note, or an altissimo run must never raise it; playing a clearly
different instrument does. The hint offers the range editor, for a player whose
instrument simply reaches further than the catalogue says.
"""
from __future__ import annotations

from tests.harness import Tone


def on(instrument: str, **config) -> dict:
    return {"last_instrument_key": instrument, **config}


FILTER_ON = {"allow_out_of_range": False}
# Sounding notes far below an alto (fingered A2, B2, C#3, D3).
BASS_LINE = ("C2", "D2", "E2", "F2")


def test_a_long_held_cello_note_above_the_keyed_range_raises_no_hint(make_app):
    """Cello B4 is above the guide rows (which stop at G4) but well inside what
    a cellist reaches; five seconds of it is ordinary playing."""
    app = make_app(config=on("cello"))
    app.launch()
    app.play(Tone(note="B4", seconds=5.0))
    assert app.table.row("B4", by="sounding").n >= 50
    assert app.audio.notice is None


def test_one_note_beyond_the_range_held_or_repeated_raises_no_hint(make_app):
    """Fingered F#7 is past the alto's extended top, but one note is one note:
    held for six seconds and then tongued again four times, still no hint."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    app.play(Tone(note="A6", seconds=6.0))      # fingered F#7
    for _ in range(4):
        app.play(Tone(note="A6", seconds=0.8))
    assert app.table.row("A6", by="sounding").n >= 60
    assert app.audio.notice is None


def test_an_alto_altissimo_run_raises_no_hint(make_app):
    """Fingered D7, D#7, E7 and F7 on the alto: four different notes above the
    keys, all of them altissimo a skilled player reaches."""
    app = make_app(config=on("eb_alto", **FILTER_ON))
    app.launch()
    for note in ("F6", "F#6", "G6", "G#6"):     # fingered D7, D#7, E7, F7
        app.play(Tone(note=note, timbre="sax", seconds=1.5))
    assert len(app.table.rows()) == 4
    assert app.audio.notice is None


def test_a_clearly_wrong_instrument_raises_the_hint_on_the_third_note(make_app):
    """A bass line played with the alto selected: two stray notes could be
    anything, the third different one brings the hint, with the range editor
    on offer."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:2]:
        app.play(Tone(note=note, seconds=1.0))
    assert app.audio.notice is None

    app.play(Tone(note=BASS_LINE[2], seconds=1.0))

    assert app.audio.notice == app.text("wrong_instrument_banner_body",
                                        instrument=app.text("instr_long_eb_alto"))
    assert app.audio.notice_action == app.text("wrong_instrument_edit_range")


def test_the_hint_is_shown_once_per_instrument_and_again_after_a_change(make_app):
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=1.0))
    assert app.audio.notice is not None
    app.audio.dismiss_notice()
    app.play(Tone(note=BASS_LINE[3], seconds=1.0))
    assert app.audio.notice is None             # already asked for this pick

    app.pick_instrument("bb_tenor")
    app.pick_instrument("eb_alto")
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=1.0))
    assert app.audio.notice is not None


def test_edit_range_on_the_hint_opens_the_editor_for_the_selected_instrument(make_app):
    """From the hint I lower the alto's range to fingered A2; the bass notes are
    then the alto's and filter-to-range keeps them."""
    app = make_app(config=on("eb_alto", **FILTER_ON))
    app.launch()
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=1.0))
    assert app.table.rows() == []

    app.dialogs.expect("range_editor", "accept", match="Alto", lo="A2")
    app.audio.accept_notice()

    assert app.dialogs.forms_shown[-1] == "range_editor"
    assert app.dialogs.pending() == []
    assert app.audio.notice is None
    app.play(Tone(note="C2", seconds=1.0))      # fingered A2
    assert app.table.row("C2", by="sounding").n >= 5


# Sounding notes an alto reaches (fingered D5, E5, F#5, G#5, A#5, C6).
ALTO_SCALE = ("F4", "G4", "A4", "B4", "C#5", "D#5")


def test_strays_close_together_still_count_across_in_range_playing(make_app):
    """Two bass notes, ten seconds on the alto, a third bass note: all three
    came within one stretch of playing, so the hint appears."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:2]:
        app.play(Tone(note=note, seconds=1.0))
    app.play(Tone(note="A4", timbre="sax", seconds=10.0))
    assert app.audio.notice is None

    app.play(Tone(note=BASS_LINE[2], seconds=1.0))
    assert app.audio.notice is not None


def test_squeaks_spread_over_a_long_practice_raise_no_hint(make_app):
    """A squeak above the alto's altissimo every half minute of scales is a
    player on the right horn: each squeak is forgotten before the next."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for squeak in ("A6", "A#6", "B6"):          # fingered F#7, G7, G#7
        for note in ALTO_SCALE:
            app.play(Tone(note=note, timbre="sax", seconds=5.0))
        app.play(Tone(note=squeak, seconds=0.6))
        assert app.table.row(squeak, by="sounding").n >= 3
    assert app.audio.notice is None


def test_a_blip_too_short_to_be_confirmed_does_not_count(make_app):
    """In the default response mode the detector confirms a pitch after
    roughly 0.2 s, so three different 0.1 s blips leave nothing to count. Held
    for 0.4 s, the same three notes are measured and do count."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=0.1))
    assert app.table.rows() == []
    assert app.audio.notice is None

    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=0.4))
    assert app.audio.notice is not None


def test_in_the_fast_response_mode_short_notes_count(make_app):
    """The fast mode confirms a pitch after roughly 0.1 s, and a confirmed
    note counts: the player chose speed over caution."""
    app = make_app(config=on("eb_alto", filter_mode="fast"))
    app.launch()
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=0.15))
    assert app.audio.notice is not None


def test_changing_a4_keeps_the_strays_heard_so_far(make_app):
    """A new concert pitch changes neither the instrument nor its range."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:2]:
        app.play(Tone(note=note, seconds=1.0))
    app.set_a4(442)
    app.play(Tone(note=BASS_LINE[2], seconds=1.0, a4=442.0))
    assert app.audio.notice is not None


def test_reset_forgets_the_strays_heard_so_far(make_app):
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:2]:
        app.play(Tone(note=note, seconds=1.0))
    app.table.reset()
    app.play(Tone(note=BASS_LINE[2], seconds=1.0))
    assert app.audio.notice is None


def test_widening_the_range_from_setup_forgets_the_strays_it_now_covers(make_app):
    """Two bass notes, then I lower the alto to fingered A2 in SETUP (before
    any hint): those two are now in range. Two lower notes after that are
    only the second and third strays heard, not the third and fourth."""
    app = make_app(config=on("eb_alto"))
    app.launch()
    for note in BASS_LINE[:2]:                  # C2, D2: fingered A2, B2
        app.play(Tone(note=note, seconds=1.0))
    assert app.settings.edit_range("A2", "F#6")

    for note in ("A1", "B1"):                   # fingered F#2, G#2
        app.play(Tone(note=note, seconds=1.0))
    assert app.audio.notice is None


def test_the_hint_and_its_action_read_in_german(make_app):
    """The action carries the same label as the SETUP button that opens the
    same editor."""
    app = make_app(config=on("eb_alto", last_lang="de"))
    app.launch()
    for note in BASS_LINE[:3]:
        app.play(Tone(note=note, seconds=1.0))
    assert app.audio.notice == app.text("wrong_instrument_banner_body",
                                        instrument=app.text("instr_long_eb_alto"))
    assert "Tonumfang" in app.audio.notice
    assert app.audio.notice_action == app.text("setup_range_edit") == "Tonumfang bearbeiten …"

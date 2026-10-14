"""The range editor from the keyboard: type a note name, press Enter.

Enter saves what was typed. It must never press Restore Default, which used to
snap the field back and close nothing, so the edit was silently lost. A name
that does not parse keeps the dialog open with the error shown, and nothing is
saved.
"""
from __future__ import annotations

import json


def on(instrument: str, **config) -> dict:
    return {"last_instrument_key": instrument, **config}


def saved_ranges(app) -> dict:
    path = app.file("ranges")
    return json.loads(path.read_text(encoding="utf-8"))["ranges"] if path.exists() else {}


def test_typing_a_top_note_and_pressing_enter_saves_it(make_app):
    """The alto's keys stop at F#6; I type F6 and press Enter."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert app.settings.type_range(hi="F6")

    assert app.settings.range_error is None
    assert saved_ranges(app) == {"eb_alto": [58, 89]}     # Bb3..F6 fingered
    app.restart()
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[-1].fingered) == ("A#/Bb3", "F6")


def test_typing_a_bottom_note_and_pressing_enter_saves_it(make_app):
    """German players may write H for B natural; H3 is understood."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert app.settings.type_range(lo="H3")

    assert saved_ranges(app) == {"eb_alto": [59, 90]}     # B3..F#6 fingered


def test_enter_on_a_name_that_does_not_parse_saves_nothing(make_app):
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert not app.settings.type_range(hi="X9")

    assert app.settings.range_error == app.text("range_note_name_invalid")
    assert saved_ranges(app) == {}


def test_a_typo_in_the_bottom_field_saves_nothing(make_app):
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert not app.settings.type_range(lo="X9")

    assert app.settings.range_error == app.text("range_note_name_invalid")
    assert saved_ranges(app) == {}


def test_a_typo_in_one_field_blocks_a_good_edit_in_the_other(make_app):
    """A valid new top with a bottom that does not parse: saving the top alone
    would keep a bottom I did not ask for, so nothing is saved."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert not app.settings.type_range(lo="X9", hi="F6")

    assert app.settings.range_error == app.text("range_note_name_invalid")
    assert saved_ranges(app) == {}


def test_clicking_save_with_the_focus_still_in_a_field_saves_what_was_typed(make_app):
    """On macOS a mouse click on Save leaves the focus in the note field, so
    the field never reports that editing finished. The typed names are still
    what gets saved."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert app.settings.type_range(lo="C4", hi="F6", click_save=True)

    assert saved_ranges(app) == {"eb_alto": [60, 89]}     # C4..F6 fingered


def test_restore_default_after_a_typo_lets_enter_save_the_default(make_app):
    """I mistype the top, Enter refuses it, I press Restore Default: the field
    shows the default again and Enter saves it."""
    app = make_app(config=on("eb_alto"))
    app.launch()

    assert app.settings.type_range(hi="X9", then_restore=True)

    assert app.settings.range_error is None
    assert saved_ranges(app) == {}                        # the default is no override
    listed = app.table.rows(played_only=False)
    assert (listed[0].fingered, listed[-1].fingered) == ("A#/Bb3", "F#/Gb6")

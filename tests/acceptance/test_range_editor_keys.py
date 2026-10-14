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

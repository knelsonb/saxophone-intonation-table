"""First launch, saved settings, language and custom instruments.

A player opens the app for the first time, answers (or postpones) the history
question, sets it up the way they like and expects to find it that way the next
day: after an upgrade from an older release, with a damaged or locked settings
file, in German or English, with their own instrument in the list.

Starting the app as a script (``python sax_intonation_gui.py``) is outside what
the harness can drive: it imports the app as a module and builds the window
itself, so the ``__main__`` launch path needs a subprocess test of its own.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager

import pytest

from tests.harness import DEFAULT_INPUT, Tone, fixture_path, i18n_texts

TEXTS = i18n_texts()
VERSION = re.compile(r"\d+\.\d+\.\d+\S*$")
RAW_KEY = re.compile(r"^\[[a-z0-9_]+\]$")
UNFILLED = re.compile(r"\{[a-z_][a-z0-9_]*(:[^}]*)?\}")


def only_in(lang: str, other: str) -> set[str]:
    """Plain UI strings (no placeholders) that exist only in ``lang``."""
    theirs = set(TEXTS[other].values())
    return {text.strip() for key, text in TEXTS[lang].items()
            if "{" not in text and len(text.strip()) > 2
            and text != TEXTS[other].get(key) and text not in theirs}


def stale_texts(app, old: str, new: str, area: str | None = None) -> list:
    old_only = only_in(old, new)
    return [(a, role, text) for a, role, text in app.ui_texts()
            if text.strip() in old_only and (area is None or a.startswith(area))]


def user_was_told_about(app, needle: str) -> bool:
    """A message box or any visible text mentions ``needle``."""
    shown = [m.text + " " + m.title for m in app.dialogs.messages]
    shown += [text for _area, _role, text in app.ui_texts()]
    return any(needle in s for s in shown)


@contextmanager
def unreadable(path):
    """The file exists but cannot be opened (another program holds it, or its
    permissions were changed)."""
    path.chmod(0)
    try:
        yield
    finally:
        path.chmod(0o644)


# ======================================================================= welcome
def test_deciding_later_keeps_asking_and_keeps_no_history(fresh_app):
    """'Decide later' means the question comes back and nothing is written to
    a history file in the meantime."""
    fresh_app.welcome.decide_later()
    fresh_app.launch()
    fresh_app.pick_instrument("eb_alto")
    fresh_app.play(Tone(note="A4", seconds=1.0))
    assert fresh_app.table.row("A4", by="sounding").n >= 5

    fresh_app.welcome.decide_later()
    fresh_app.restart()

    assert fresh_app.welcome.shown
    assert not fresh_app.settings.history_enabled
    assert not fresh_app.file("history").exists()
    assert fresh_app.table.rows() == []


def test_opting_in_saves_every_reading_and_exports_read_the_saved_history(fresh_app):
    """After 'save history' the readings land in the log file, and a CSV export
    in the next session still contains them."""
    fresh_app.welcome.opt_in()
    fresh_app.launch()
    fresh_app.pick_instrument("eb_alto")
    fresh_app.play(Tone(note="A4", seconds=1.0))
    fresh_app.play(Tone(note="C5", cents=-10, seconds=1.0))
    runs, played = fresh_app.history_counts
    assert runs == 1 and played >= 10
    assert fresh_app.file("history").exists()

    fresh_app.restart()

    assert not fresh_app.welcome.shown
    assert fresh_app.table.rows() == []                 # a new session starts empty
    assert fresh_app.history_counts == (1, played)
    exported = fresh_app.export.csv("raw")
    assert fresh_app.export.csv_rows == played
    assert "eb_alto" in exported.read_text(encoding="utf-8")


@pytest.mark.xfail(
    strict=True,
    reason="Frodo-22: history can only be turned on later by editing config.json")
def test_history_can_be_turned_on_later_without_editing_files(fresh_app):
    """A player who said 'no history' on day one can change their mind from
    the app itself."""
    fresh_app.welcome.continue_without_history()
    fresh_app.launch()
    assert not fresh_app.settings.history_enabled

    fresh_app.settings.enable_history()

    assert fresh_app.settings.history_enabled
    fresh_app.restart()
    assert fresh_app.settings.history_enabled


def test_the_welcome_language_buttons_always_keep_one_language_selected(fresh_app):
    """Clicking the language that is already active leaves it selected."""
    fresh_app.welcome.decide_later(language="en")       # already English
    fresh_app.launch()
    assert fresh_app.welcome.language_buttons == {"DE": False, "EN": True}

    fresh_app.welcome.decide_later(language="de")
    fresh_app.restart()
    assert fresh_app.welcome.language_buttons == {"DE": True, "EN": False}
    assert fresh_app.ui_language == "de"


@pytest.mark.xfail(
    strict=True,
    reason="New-1-1: last_lang defaults to 'en', so the system-locale default is always overridden")
def test_first_launch_on_a_german_system_starts_in_german(make_app):
    """With no saved settings, the app speaks the system language."""
    app = make_app(lang="de", first_launch=True)
    app.welcome.decide_later()
    app.launch()

    assert app.ui_language == "de"
    assert app.window_title.startswith(TEXTS["de"]["window_title"])


# ============================================================ saved settings
def test_every_setting_a_player_touches_survives_a_restart(app):
    """Theme, language, instrument, concert pitch, display, response, tempo and
    the drone set-up are all where the player left them; a silent drone stays
    silent."""
    app.settings.theme("light")
    app.settings.language("de")
    app.pick_instrument("bb_tenor")
    app.set_a4(442)
    app.set_display("sounding")
    app.settings.response("slow")
    app.metronome.start(bpm=132, time_sig="3/4")
    app.metronome.set_volume(40)
    app.metronome.stop()
    app.drone.on("A3", voice="sine")
    app.drone.off()
    app.drone.set_volume(0)

    app.restart()

    assert app.settings.theme_name == "light"
    assert app.ui_language == "de"
    assert (app.instrument, app.a4, app.engine_a4) == ("bb_tenor", 442, 442.0)
    assert app.display == "sounding"
    assert app.settings.response_mode == "slow"
    state = app.metronome.state
    assert (state.bpm, state.time_sig, state.volume) == (132, "3/4", pytest.approx(0.4))
    drone = app.drone.state
    assert (drone.voice, drone.semitones, app.drone.volume) == ("sine", -12, 0)


def test_an_unknown_response_mode_in_the_settings_file_falls_back_to_normal(make_app):
    """A hand-edited or future value shows up as Normal, and is saved back as
    Normal."""
    app = make_app(config={"filter_mode": "turbo"})
    app.launch()
    assert app.settings.response_mode == "normal"
    app.close()
    assert app.read_config()["filter_mode"] == "normal"


@pytest.mark.parametrize("fixture, kept", [
    ("config_v0.6.json", dict(instrument="bb_tenor", a4=443, bpm=100, theme="dark")),
    ("config_v1.0.json", dict(instrument="eb_alto", a4=442, bpm=None, theme=None)),
    ("config_v1.1.json", dict(instrument="eb_bari", a4=440, bpm=100, theme="light")),
], ids=["v0.6", "v1.0", "v1.1"])
def test_an_upgraded_settings_file_keeps_old_choices_and_takes_new_ones(make_app, fixture,
                                                                         kept):
    """Settings saved by an older release open as they were; new choices made
    after the upgrade stick, and the old ones are still there afterwards."""
    old = json.loads(fixture_path(fixture).read_text(encoding="utf-8"))
    app = make_app(config=fixture_path(fixture),
                   customs=fixture_path("custom_instruments.json"))
    app.launch()
    assert (app.instrument, app.a4) == (kept["instrument"], kept["a4"])
    assert app.metronome.state.bpm == (kept["bpm"] or old.get("last_bpm", 100))
    assert app.settings.theme_name == (kept["theme"] or old.get("theme", "dark"))
    assert app.drone.state.voice == "organ"             # field newer than every fixture

    app.settings.theme("night")
    app.metronome.start(bpm=88)
    app.metronome.stop()
    app.drone.set_volume(25)
    app.restart()

    assert (app.instrument, app.a4, app.engine_a4) == (kept["instrument"], kept["a4"],
                                                       float(kept["a4"]))
    assert app.settings.theme_name == "night"
    assert app.metronome.state.bpm == 88
    assert app.drone.volume == 25


def test_a_corrupt_settings_file_does_not_stop_the_app(make_app):
    """Garbage in config.json: the app opens on defaults, works, and leaves a
    valid settings file behind."""
    app = make_app(config=b'{"last_a4_hz": 442, "theme": "li')
    app.welcome.decide_later()                          # nothing is known any more
    app.launch()

    assert app.a4 == 440
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    assert app.table.row("A4", by="sounding").mean == 0
    app.close()
    assert isinstance(app.read_config(), dict)


def test_a_corrupt_settings_file_is_reported_and_kept_for_repair(make_app):
    """The player is told their settings could not be read, and the damaged
    file is kept so a hand edit can be repaired instead of lost."""
    damaged = b'{"last_a4_hz": 442, "persistence_enabled": true,, }'
    app = make_app(config=damaged)
    app.dialogs.auto_acknowledge("information", "warning", "critical")
    app.dialogs.expect("welcome", "reject", optional=True)
    app.launch()
    told = user_was_told_about(app, "config.json")
    app.close()

    assert told
    kept = [p for p in app.file("config").parent.iterdir()
            if p.is_file() and p.read_bytes() == damaged]
    assert kept


def test_a_locked_settings_file_is_left_exactly_as_it_was(make_app):
    """If config.json cannot be opened, the session runs on defaults and the
    file is not overwritten; once it is readable again the old settings return."""
    app = make_app(config={"last_a4_hz": 442, "theme": "light"})
    path = app.file("config")
    before = path.read_bytes()
    app.dialogs.expect("welcome", "reject", optional=True)
    with unreadable(path):
        app.launch()
        assert app.a4 == 440
        app.set_a4(445)
        app.settings.theme("night")
        app.close()
    assert path.read_bytes() == before

    app.launch()
    assert (app.a4, app.settings.theme_name) == (442, "light")


def test_a_locked_settings_file_is_reported_instead_of_greeting_a_new_user(make_app):
    """A player whose settings file is locked is told so, and is not asked the
    first-launch question again as if they were new."""
    app = make_app(config={"last_a4_hz": 442})
    app.dialogs.auto_acknowledge("information", "warning", "critical")
    app.dialogs.expect("welcome", "reject", optional=True)
    with unreadable(app.file("config")):
        app.launch()
        greeted = app.welcome.shown
        told = user_was_told_about(app, "config.json")
        app.close()

    assert not greeted
    assert told


# ================================================================== language
def test_switching_language_relabels_the_title_and_tabs_and_keeps_the_version(app):
    """The window title (with its version number), the tab names and the tab
    tooltips follow a mid-session language switch, both ways."""
    version = VERSION.search(app.window_title).group(0)
    for lang in ("de", "en"):
        app.settings.language(lang)
        words = TEXTS[lang]
        assert app.window_title == f"{words['window_title']} {version}"
        texts = app.ui_texts()
        tabs = [t for _a, role, t in texts if role == "tab"]
        tips = [t for _a, role, t in texts if role == "tab_tip"]
        keys = ("tuner", "metro", "deck", "setup")
        assert tabs == [words[f"nav_tab_{k}"] for k in keys]
        assert tips == [words[f"nav_tab_{k}_tip"] for k in keys]


@pytest.mark.parametrize("start, switch_to", [("en", "de"), ("de", "en")],
                         ids=["en-to-de", "de-to-en"])
def test_after_a_language_switch_no_text_is_left_in_the_old_language(make_app, start,
                                                                     switch_to):
    """Every label, button, combo entry and tooltip on every tab follows the
    switch; nothing still speaks the previous language."""
    app = make_app(lang=start)
    app.launch()
    app.settings.language(switch_to)
    assert stale_texts(app, start, switch_to) == []


@pytest.mark.parametrize("lang", ["en", "de"])
def test_no_text_shows_a_raw_key_or_an_unfilled_placeholder(make_app, lang):
    """No '[some_key]' fallbacks and no '{name}' holes anywhere, in either
    language, also after switching back and forth."""
    app = make_app(lang=lang)
    app.launch()
    other = "en" if lang == "de" else "de"
    for current in (lang, other, lang):
        app.settings.language(current)
        leaks = [(a, r, t) for a, r, t in app.ui_texts()
                 if RAW_KEY.match(t.strip()) or UNFILLED.search(t)]
        assert leaks == [], f"in {current}"


def test_questions_asked_after_a_language_switch_use_the_new_language(app):
    """A confirmation that pops up after switching to German is in German."""
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    app.settings.language("de")

    app.table.reset()

    question = app.dialogs.last_message()
    assert question.text == TEXTS["de"]["reset_msg"]
    assert app.table.rows() == []


def test_an_open_pitch_pipes_window_follows_a_language_switch(app):
    """The pitch pipes window that is open while the language changes is
    relabelled too."""
    app.pipes.press("A4")
    app.settings.language("de")

    titles = [t for a, role, t in app.ui_texts()
              if a.startswith("dialog:") and role == "title"]
    assert TEXTS["de"]["pipes_title"] in titles
    assert stale_texts(app, "en", "de", area="dialog:") == []


def test_the_audio_problem_banner_follows_a_language_switch(app):
    """A banner that is up when the language changes is re-worded."""
    app.host.unplug(DEFAULT_INPUT)
    app.advance(1.0)
    english = app.audio.status().banner_text
    assert app.audio.status().banner_visible

    app.settings.language("de")

    german = app.audio.status().banner_text
    assert german != english
    assert DEFAULT_INPUT in german


@pytest.mark.xfail(
    strict=True,
    reason="Frodo-14: German 'Aufnahme' names both the measurement switch and the deck tab")
def test_german_uses_different_words_for_measuring_and_for_the_recorder_tab(make_app):
    """'Aufnahme' should not name both the measurement switch on the tuner and
    the tape-recorder tab."""
    app = make_app(lang="de")
    app.launch()
    texts = app.ui_texts()
    deck_tab = next(t for a, role, t in texts
                    if role == "tab" and t == TEXTS["de"]["nav_tab_deck"])
    tuner_words = [t for a, role, t in texts if a in ("tuner", "window") and role == "text"]
    assert not [t for t in tuner_words if deck_tab.lower() in t.lower()]


@pytest.mark.xfail(
    strict=True,
    reason="Frodo-15: the German UI writes B natural as B, not H")
def test_german_note_names_call_b_natural_h(make_app):
    """In the German UI the note B natural is written H, as German players
    read it (and as the range editor already accepts)."""
    app = make_app(lang="de")
    app.launch()
    app.pick_instrument("eb_alto")
    app.set_display("sounding")
    app.play(Tone(note="B4", seconds=1.0))
    assert app.table.rows()[0].sounding == "H4"


def test_german_toolbar_buttons_are_in_german(make_app):
    """The export and reset buttons do not stay English in the German UI."""
    app = make_app(lang="de")
    app.launch()
    buttons = [t for _a, role, t in app.ui_texts() if role == "text"]
    english = [t for t in buttons
               if re.search(r"\bExport (TXT|PDF|CSV)\b|\bReset\b", t)]
    assert english == []


# ======================================================== custom instruments
def test_a_custom_instrument_keeps_its_name_and_transposition_after_restart(app):
    """An instrument the player added reads like the real one it stands for
    (here: an Eb horn, 9 semitones), today and after a restart."""
    key = app.settings.add_custom_instrument("Mezzo Test", transpose=-9)
    assert key == "custom_mezzo_test"
    app.play(Tone(note="A4", seconds=1.0))
    assert app.table.row("A4", by="sounding").fingered == "F#/Gb5"

    app.restart()

    assert app.instrument == key
    app.play(Tone(note="A4", seconds=1.0))
    assert app.table.row("A4", by="sounding").fingered == "F#/Gb5"
    saved = json.loads(app.file("customs").read_text(encoding="utf-8"))
    assert [(c["key"], c["transp"]) for c in saved] == [(key, -9)]


def test_a_custom_instrument_names_its_exports(app):
    """The report says the player's own name for the instrument and the file
    name suggested for it does too."""
    key = app.settings.add_custom_instrument("Mezzo Test", transpose=-9)
    app.play(Tone(note="A4", seconds=1.0))

    report = app.export.txt(maker="Keilwerth", model="SX90")

    suggested = app.dialogs.file_dialogs[-1][2]
    assert key in suggested
    assert "Mezzo Test" in report.read_text(encoding="utf-8")
    app.export.pdf()
    plain = " ".join(f.get("plain", "") for f in app.export.pdf_model["flowables"])
    assert "Mezzo Test" in plain
    assert key in app.export.csv("raw").read_text(encoding="utf-8")


@pytest.mark.parametrize("name, reason_key", [
    ("", "custom_err_name"),
    ("   ", "custom_err_name"),
    ("!!! ---", "custom_err_chars"),
])
def test_a_custom_instrument_name_needs_a_letter_or_a_digit(app, name, reason_key):
    """A blank or symbols-only name is refused with a reason, and nothing is
    added or saved."""
    before = app.instrument

    assert app.settings.add_custom_instrument(name, transpose=-9) is None

    assert app.dialogs.last_message().text == app.text(reason_key)
    assert app.instrument == before
    assert not app.file("customs").exists()


def test_names_differing_only_in_case_are_separate_and_a_repeat_updates(app):
    """'Mezzo' and 'mezzo' are two instruments; adding 'Mezzo' again changes its
    transposition instead of making a third."""
    assert app.settings.add_custom_instrument("Mezzo", transpose=-9) == "custom_mezzo"
    assert app.settings.add_custom_instrument("mezzo", transpose=-2) == "custom_mezzo_2"
    assert app.settings.add_custom_instrument("Mezzo", transpose=-14) == "custom_mezzo"

    saved = json.loads(app.file("customs").read_text(encoding="utf-8"))
    assert sorted((c["key"], c["transp"]) for c in saved) == [
        ("custom_mezzo", -14), ("custom_mezzo_2", -2)]


@pytest.mark.xfail(
    strict=True,
    reason="Frodo-17: the custom-instrument form checks the name after closing, losing the input")
def test_a_rejected_custom_name_can_be_fixed_in_the_same_form(app):
    """After a refused name the form stays open with what was typed, so the
    player corrects it instead of starting over."""
    key = app.settings.try_custom_instrument_names(["!!!", "Mezzo"], transpose=-9)
    assert key == "custom_mezzo"


def test_an_unreadable_customs_file_is_left_untouched(make_app):
    """If custom_instruments.json cannot be opened, adding an instrument works
    for the session but the file is not rewritten; the saved instruments are
    all back once it is readable."""
    app = make_app(customs=fixture_path("custom_instruments.json"))
    path = app.file("customs")
    before = path.read_bytes()
    with unreadable(path):
        app.launch()
        assert app.settings.add_custom_instrument("Session Only", transpose=-2) \
            == "custom_session_only"
        app.close()
    assert path.read_bytes() == before

    app.launch()
    app.pick_instrument("custom_mezzo_soprano")
    app.pick_instrument("custom_old_style")
    assert app.instrument == "custom_old_style"


def test_an_unreadable_customs_file_does_not_lose_the_chosen_instrument(make_app):
    """A player on their own instrument whose customs file is briefly locked
    finds that instrument selected again once the file is readable."""
    app = make_app(config={"last_instrument_key": "custom_mezzo_soprano"},
                   customs=fixture_path("custom_instruments.json"))
    with unreadable(app.file("customs")):
        app.launch()
        app.close()

    app.launch()
    assert app.instrument == "custom_mezzo_soprano"

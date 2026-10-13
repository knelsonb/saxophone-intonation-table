"""End-to-end proof that the headless harness works.

Each test is a short user story told only through the harness API: launch the
app, play a note into the (fake) microphone, read the table, listen to the
(fake) speaker. None of them imports the app's modules.
"""
from __future__ import annotations

import math

import pytest

from tests.acceptance._scenarios import play_known_table
from tests.harness import DEFAULT_INPUT, Tone, assert_golden


def cents_between(hz: float, reference: float) -> float:
    return 1200.0 * math.log2(hz / reference)


# --------------------------------------------------------------------- tuning
def test_a_sine_at_440_hz_reads_in_tune_on_an_alto(app):
    # An Eb alto sounds a concert A4 when the player fingers F#5.
    assert app.audio.status().input_state == "running"
    assert app.table.rows() == []
    app.pick_instrument("eb_alto")

    app.play(Tone(note="A4", timbre="sine", seconds=1.5))

    row = app.table.row("A4", by="sounding")
    assert row.fingered == "F#/Gb5"
    assert row.mean == 0
    assert row.n >= 5
    assert row.band == "green"
    assert app.table.row("F#5", by="fingered") == row     # either spelling finds it
    assert app.tuner.reading().cents_shown == 0


def test_a_note_ten_cents_sharp_is_amber(app):
    app.pick_instrument("eb_alto")

    app.play(Tone(note="A4", cents=+10, seconds=1.5))

    row = app.table.row("A4", by="sounding")
    assert row.mean == 10
    assert row.band == "amber"      # green up to 5 cents, amber up to 12, red beyond


def test_txt_export_of_a_known_table_matches_the_golden_file(app):
    play_known_table(app)

    path = app.export.txt(maker="Selmer", model="Mark VI")

    assert_golden(path, "expected/txt_known_table_en.txt")


# --------------------------------------------------------------------- output
def test_metronome_at_120_bpm_clicks_every_half_second(app):
    state = app.metronome.start(bpm=120, time_sig="4/4")
    assert state.running and state.bpm == 120

    heard = app.capture_output(seconds=3.0)

    onsets = heard.onsets()
    block = app.host.output_stream.blocksize / app.host.output_stream.samplerate
    assert len(onsets) >= 6
    for gap in heard.onset_intervals():
        assert gap == pytest.approx(0.5, abs=block)     # within one output block
    assert not heard.is_silent()


def test_a_sine_drone_on_a3_sounds_220_hz(app):
    state = app.drone.on("A3", voice="sine")
    assert state.enabled and not state.sine_fallback

    heard = app.capture_output(seconds=1.0)

    assert abs(cents_between(heard.pitch(near=220.0), 220.0)) < 3.0
    app.drone.off()
    app.advance(0.3)                       # let the release tail fade
    assert app.capture_output(seconds=0.5).is_silent()


# ------------------------------------------------------------------- failures
def test_unplugging_the_microphone_is_reported_and_replugging_recovers(app):
    app.pick_instrument("eb_alto")
    app.host.unplug(DEFAULT_INPUT)

    app.advance(1.0)                      # the app polls for devices once a second

    status = app.audio.status()
    assert status.input_state == "failed"
    assert status.input_error == "device_disconnected"
    assert status.banner_visible and DEFAULT_INPUT in status.banner_text
    assert status.chip_text == "No device"
    assert app.play(Tone(note="A4", seconds=1.0)) == 0      # nothing reaches the app
    assert app.table.rows() == []

    app.host.plug(DEFAULT_INPUT)
    app.advance(1.0)

    assert app.audio.status().input_state == "running"
    assert not app.audio.status().banner_visible
    app.play(Tone(note="A4", seconds=1.5))
    assert app.table.row("A4", by="sounding").n >= 5


# ---------------------------------------------------------------- persistence
def test_restart_keeps_the_concert_pitch_and_language(app):
    app.set_a4(442)
    app.settings.language("de")
    assert app.ui_language == "de"

    app.restart()

    assert app.a4 == 442
    assert app.engine_a4 == 442.0
    assert app.ui_language == "de"
    assert app.window_title.startswith("Intonations-Analysator")
    assert app.read_config()["last_a4_hz"] == 442


def test_first_launch_asks_about_history_until_the_user_answers(fresh_app):
    fresh_app.welcome.decide_later()
    fresh_app.launch()
    assert fresh_app.welcome.shown
    assert not fresh_app.settings.history_enabled

    fresh_app.welcome.decide_later()          # "decide later" => asked again next time
    fresh_app.restart()
    assert fresh_app.welcome.shown

    fresh_app.welcome.opt_in()
    fresh_app.restart()
    assert fresh_app.welcome.shown
    assert fresh_app.settings.history_enabled
    assert fresh_app.settings.history_path == fresh_app.file("history")

    fresh_app.restart()                       # answered: no dialog this time
    assert not fresh_app.welcome.shown


# ---------------------------------------------------------------- harness rules
def test_a_dialog_nobody_scripted_fails_the_test_instead_of_hanging(fresh_app):
    # A first launch with no scripted welcome answer must not block: the harness
    # records the surprise and raises it from launch().
    with pytest.raises(AssertionError, match="unexpected welcome dialog"):
        fresh_app.launch()

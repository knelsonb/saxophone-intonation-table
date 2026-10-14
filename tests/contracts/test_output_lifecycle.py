"""Output lifecycle details the device scenarios do not reach.

The scenarios in tests/acceptance cover the main paths (unplug, replug,
switch, refusal). These pin the rest of the contract: an output switch that
fails switches off what was sounding and says why, every kind of source is
shown off while the output is gone and comes back with it, and nothing keeps
sounding into a closed stream.
"""
from __future__ import annotations

from tests.harness import DEFAULT_OUTPUT

HEADPHONES = "USB Headphones"


def test_a_switch_to_an_output_that_refuses_stops_the_metronome_and_says_why(make_app, host):
    host.add_output(HEADPHONES, rates=(44100,), default_rate=44100)
    app = make_app()
    app.launch()
    assert app.metronome.start(bpm=120).running

    host.set_rates(HEADPHONES, ())
    host.remove_device(DEFAULT_OUTPUT)          # nothing to fall back to
    app.settings.output_device(HEADPHONES)

    state = app.metronome.state
    assert not state.running
    assert HEADPHONES in (state.unavailable_text or "")
    assert app.capture_output(1.0).is_silent()


def test_every_source_shows_off_while_unplugged_and_comes_back(app):
    app.metronome.start(bpm=120)
    app.drone.on("A4", voice="sine")
    app.pipes.press("C4")
    app.settings.test_tone(True)

    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(1.0)

    assert not app.metronome.state.running
    assert not app.drone.state.enabled
    assert app.pipes.active == []
    assert not app.settings.test_tone_playing
    assert DEFAULT_OUTPUT in (app.audio.notice or "")

    app.host.plug(DEFAULT_OUTPUT)
    app.advance(1.0)

    assert app.metronome.state.running
    assert app.drone.state.enabled
    assert app.pipes.active == [60]
    assert app.settings.test_tone_playing
    assert app.audio.notice is None
    assert not app.capture_output(1.0).is_silent()


def test_switching_a_source_off_while_unplugged_keeps_it_off_after_replug(app):
    app.drone.on("A4", voice="sine")
    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(1.0)

    app.drone.off()                             # already shown off: no change
    app.metronome.start(bpm=100)                # cannot start: no output
    assert not app.metronome.state.running

    app.host.plug(DEFAULT_OUTPUT)
    app.advance(1.0)

    assert app.drone.state.enabled, "the parked drone comes back"
    assert not app.metronome.state.running, "a failed start is not retried later"

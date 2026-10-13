"""Practice tools: metronome, drone, pitch pipes, test tone and tape deck.

Each test is something a player does while practising: start the click, switch
to headphones mid-exercise, tune against a drone, record a phrase and listen
back. Everything is judged by what comes out of the (fake) speaker or what the
app says on screen.
"""
from __future__ import annotations

import array
import math
import wave

import pytest

from tests.harness import DEFAULT_OUTPUT, OutputCapture, Tone

HEADPHONES = "USB Headphones"        # a second speaker that runs at 44.1 kHz
MONITORS = "Studio Monitors"         # a second speaker that runs at 96 kHz


def cents_between(hz: float, reference: float) -> float:
    return 1200.0 * math.log2(hz / reference)


def in_tune(heard: OutputCapture, hz: float, tolerance: float = 3.0) -> bool:
    found = heard.pitch(near=hz, search_cents=300.0)
    return found > 0 and abs(cents_between(found, hz)) < tolerance


def block_seconds(app) -> float:
    stream = app.host.output_stream
    return stream.blocksize / stream.samplerate


def downbeats(heard: OutputCapture) -> list[bool]:
    """For each click heard, whether it is the accented one. The accent is the
    click that is clearly higher (or, failing that, clearly louder)."""
    clicks = [heard.slice(t, t + 0.03) for t in heard.onsets()]
    for values in ([c.pitch() for c in clicks], [c.peak() for c in clicks]):
        if values and max(values) > 1.1 * min(values):
            middle = (max(values) + min(values)) / 2.0
            return [v > middle for v in values]
    return [False] * len(clicks)


def has_gaps(heard: OutputCapture, window_s: float = 0.005) -> bool:
    """True when a steady tone drops out anywhere, even for a few milliseconds."""
    n = max(1, int(window_s * heard.sr))
    peaks = [heard.slice(i / heard.sr, (i + n) / heard.sr).peak()
             for i in range(0, len(heard) - n, n)]
    return min(peaks) < 0.25 * max(peaks)


def read_wav(path) -> OutputCapture:
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth()) == (1, 2)
        pcm = array.array("h", w.readframes(w.getnframes()))
        return OutputCapture([v / 32768.0 for v in pcm], w.getframerate())


@pytest.fixture()
def two_speakers(make_app, host):
    """The usual 48 kHz speakers plus headphones at 44.1 kHz, both present
    from the start so SETUP lists them."""
    host.add_output(HEADPHONES, rates=(44100,), default_rate=44100)
    app = make_app()
    app.launch()
    return app


# ------------------------------------------------------------------- metronome
def test_metronome_in_3_4_accents_the_first_beat_of_every_bar(app):
    """120 bpm in 3/4: a click every half second, every third one accented.
    Stop means stop."""
    state = app.metronome.start(bpm=120, time_sig="3/4")
    assert state.running and (state.bpm, state.time_sig) == (120, "3/4")

    heard = app.capture_output(3.2)

    for gap in heard.onset_intervals():
        assert gap == pytest.approx(0.5, abs=block_seconds(app))
    assert downbeats(heard) == [True, False, False] * 2 + [True]

    assert not app.metronome.stop().running
    app.advance(0.2)                                  # the last click rings out
    assert app.capture_output(1.0).is_silent()


def test_stopping_and_starting_again_quickly_gives_one_steady_beat(app):
    """Stop and restart within a beat: one chain of clicks, no doubled or
    stray click from the old run, and the new run starts on a downbeat."""
    app.metronome.start(bpm=120, time_sig="4/4")
    app.advance(0.7)                                  # a beat is already queued
    app.metronome.stop()
    app.metronome.start()

    heard = app.capture_output(2.9)

    assert len(heard.onsets()) == 6
    for gap in heard.onset_intervals():
        assert gap == pytest.approx(0.5, abs=block_seconds(app))
    assert downbeats(heard) == [True, False, False, False, True, False]


def test_a_tempo_change_while_running_applies_from_the_next_beat(app):
    """Slowing from 120 to 60 bpm mid-run: the click due next still comes, then
    the clicks are a second apart."""
    app.metronome.start(bpm=120, time_sig="4/4")
    app.advance(1.2)

    assert app.metronome.start(bpm=60).bpm == 60
    heard = app.capture_output(3.5)

    onsets = heard.onsets()
    assert onsets[0] < 0.5 + block_seconds(app)
    assert len(onsets) == 4
    for gap in heard.onset_intervals():
        assert gap == pytest.approx(1.0, abs=block_seconds(app))


@pytest.mark.xfail(strict=True, reason="Sauron-13: a time-signature change mid-run "
                                       "puts the downbeat on an arbitrary beat")
def test_changing_the_time_signature_mid_run_puts_the_downbeat_where_a_bar_starts(app):
    """Switching 4/4 to 3/4 right after a downbeat: the accent comes either on
    the next click (new bar now) or after the current 4/4 bar finishes."""
    app.metronome.start(bpm=120, time_sig="4/4")
    app.advance(2.2)                                  # beat 5 of the run just sounded

    app.metronome.start(time_sig="3/4")
    heard = app.capture_output(3.0)

    assert len(heard.onsets()) == 6
    new_bar_now = [True, False, False, True, False, False]
    after_this_bar = [False, False, False, True, False, False]
    assert downbeats(heard) in (new_bar_now, after_this_bar)


def test_switching_to_headphones_keeps_the_bar_count(two_speakers):
    """Mid-bar switch to the headphones: the clicks carry on at the same tempo
    and the accent still lands on beat one of the bar."""
    app = two_speakers
    app.metronome.start(bpm=120, time_sig="4/4")
    before = app.capture_output(1.2)                  # beats 1, 2, 3 of the first bar
    assert downbeats(before) == [True, False, False]

    app.settings.output_device(HEADPHONES)
    after = app.capture_output(2.2)

    assert app.audio.status().output_device == HEADPHONES
    assert app.metronome.state.running
    assert downbeats(after) == [False, True, False, False, False]   # beat 4, then bar 2
    for gap in after.onset_intervals():
        assert gap == pytest.approx(0.5, abs=block_seconds(app))


@pytest.mark.xfail(strict=True, reason="New-practice-2: after an output switch the next "
                                       "click comes at once instead of on the beat")
def test_switching_to_headphones_does_not_shift_the_beat(two_speakers):
    """The click after the switch comes one beat after the last one, so a
    player keeping time does not get pulled off the beat."""
    app = two_speakers
    app.metronome.start(bpm=120, time_sig="4/4")
    before = app.capture_output(1.2)
    app.settings.output_device(HEADPHONES)
    after = app.capture_output(1.0)

    gap = (before.duration - before.onsets()[-1]) + after.onsets()[0]
    assert gap == pytest.approx(0.5, abs=0.1)


@pytest.mark.xfail(strict=True, reason="Frodo-4, Sauron-7: a replugged speaker is never "
                                       "reopened; the metronome shows running but is silent")
def test_metronome_is_heard_again_after_the_speaker_is_replugged(app):
    """Speaker cable pulled and pushed back in mid-practice: the metronome
    clicks again, or it shows that it stopped. Never 'running' and silent."""
    app.metronome.start(bpm=120, time_sig="4/4")
    app.advance(1.0)
    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(1.0)                                  # the app polls devices once a second
    app.host.plug(DEFAULT_OUTPUT)
    app.advance(2.0)

    state = app.metronome.state
    heard = app.capture_output(2.0)
    if state.running:
        assert len(heard.onsets()) >= 3
    else:
        assert state.unavailable_text


@pytest.mark.xfail(strict=True, reason="Uruk2-10: closing the window leaves the output "
                                       "stream open")
def test_closing_the_window_releases_the_speaker(app):
    """Close the app with the metronome and drone going: nothing keeps the
    sound device open afterwards."""
    app.metronome.start(bpm=100)
    app.drone.on("A4", voice="sine")
    app.advance(0.5)

    assert app.close() == []


# ----------------------------------------------------------------------- drone
def test_drone_without_the_synth_falls_back_to_sine_says_so_and_follows_a4(app):
    """On an install without the SoundFont synth the drone still sounds, as a
    sine, tells the player why, and moves to 442 Hz when A4 is set to 442."""
    app.drone.synth_unavailable()

    state = app.drone.on("A4")                        # the saved voice is a SoundFont one

    assert state.enabled and state.sine_fallback and state.voice == "sine"
    assert state.status_text and "sine" in state.status_text.lower()
    assert in_tune(app.capture_output(1.0), 440.0)

    app.set_a4(442)

    assert app.drone.state.enabled
    assert in_tune(app.capture_output(1.0), 442.0)


def test_the_tuner_keeps_reading_the_player_while_the_drone_sounds(app):
    """With an A drone going, a played E is still measured and shown live, and
    the drone keeps sounding."""
    app.pick_instrument("eb_alto")
    app.drone.on("A4", voice="sine")

    app.play(Tone(note="E4", timbre="sax", seconds=1.5), release=0)

    reading = app.tuner.reading()
    assert reading.active and reading.cents_shown == 0
    assert "E4" in reading.status_text
    assert in_tune(app.capture_output(0.5), 440.0)
    app.silence(0.4)
    row = app.table.row("E4", by="sounding")
    assert row.mean == 0 and row.n >= 5


def test_playing_in_unison_with_the_drone_is_measured(app):
    """Tuning a long A against an A drone is the whole point of a drone: the
    played A must show up in the tuner and the table."""
    app.pick_instrument("eb_alto")
    app.drone.on("A4", voice="sine")

    app.play(Tone(note="A4", timbre="sax", cents=-6, seconds=1.5))

    row = app.table.row("A4", by="sounding")
    assert row.mean == -6 and row.n >= 5


def test_the_drone_follows_a_speaker_switch_in_tune_without_dropouts(make_app, host):
    """Switch from the 48 kHz speakers to 96 kHz monitors with the drone on:
    it keeps sounding at 440 Hz with no gaps."""
    host.add_output(MONITORS, rates=(96000,), default_rate=96000)
    app = make_app()
    app.launch()
    app.drone.on("A4", voice="sine")
    assert in_tune(app.capture_output(0.5), 440.0)

    app.settings.output_device(MONITORS)
    heard = app.capture_output(1.0)

    assert heard.sr == 96000 and app.drone.state.enabled
    assert in_tune(heard, 440.0)
    assert not has_gaps(heard.slice(0.2))            # after the 0.2 s switch-over fade


@pytest.mark.xfail(strict=True, reason="Frodo-4, Uruk3-2: a replugged speaker is never "
                                       "reopened; after recovery the drone would drop part of "
                                       "every larger block")
def test_the_drone_survives_the_speaker_coming_back_at_a_new_rate(app):
    """Speaker unplugged, then back at 96 kHz (its settings changed): the drone
    sounds again, in tune and without dropouts, or says it stopped."""
    app.drone.on("A4", voice="sine")
    app.advance(0.5)
    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(1.0)
    app.host.set_rates(DEFAULT_OUTPUT, (96000,))
    app.host.plug(DEFAULT_OUTPUT)
    app.advance(2.0)

    state = app.drone.state
    heard = app.capture_output(1.0)
    if state.enabled:
        assert in_tune(heard, 440.0) and not has_gaps(heard)
    else:
        assert state.status_text


# ----------------------------------------------------------------- pitch pipes
@pytest.mark.parametrize("rate", [44100, 48000, 96000])
def test_pitch_pipe_a_is_440_hz_on_every_output_rate(make_app, host, rate):
    """The A pipe is 440 Hz whatever rate the speaker runs at."""
    host.set_rates(DEFAULT_OUTPUT, (rate,))
    app = make_app()
    app.launch()

    app.pipes.press("A4")
    heard = app.capture_output(0.5)

    assert heard.sr == rate
    assert in_tune(heard, 440.0)
    assert app.audio.status().output_rate == rate


def test_a_speaker_switch_never_leaves_a_pitch_pipe_out_of_tune(two_speakers):
    """Switching to the headphones while a pipe sounds: whatever still sounds is
    in tune, and the pad shows whether it is still on."""
    app = two_speakers
    app.pipes.press("A4")
    app.capture_output(0.3)

    app.settings.output_device(HEADPHONES)
    heard = app.capture_output(0.5)

    if 69 in app.pipes.active:
        assert in_tune(heard, 440.0)
    else:
        assert heard.is_silent()


@pytest.mark.xfail(strict=True, reason="New-practice-3: a speaker switch silences a "
                                       "held pitch pipe (the drone keeps sounding)")
def test_a_held_pitch_pipe_keeps_sounding_after_a_speaker_switch(two_speakers):
    """The pipe the player is tuning to keeps sounding on the new speaker, the
    same way the drone does."""
    app = two_speakers
    app.pipes.press("A4")
    app.capture_output(0.3)

    app.settings.output_device(HEADPHONES)
    heard = app.capture_output(0.5)

    assert app.pipes.active == [69]
    assert in_tune(heard.slice(0.1), 440.0)


@pytest.mark.xfail(strict=True, reason="Sauron-5: a held pitch pipe keeps the old A4 "
                                       "after a concert-pitch change")
def test_a_held_pitch_pipe_follows_a_concert_pitch_change(app):
    """Changing A4 to 442 while the A pipe sounds retunes it to 442 Hz."""
    app.pipes.press("A4")
    app.capture_output(0.3)

    app.set_a4(442)

    assert app.pipes.active == [69]
    assert in_tune(app.capture_output(0.5).slice(0.1), 442.0)


# ------------------------------------------------------------------- test tone
def test_the_test_tone_leaves_no_stale_tone_after_an_output_rate_change(two_speakers):
    """Test tone on, then switch from 48 kHz speakers to 44.1 kHz headphones:
    nothing comes out at the wrong pitch."""
    app = two_speakers
    app.settings.test_tone(True)
    assert in_tune(app.capture_output(0.5), 440.0)

    app.settings.output_device(HEADPHONES)
    heard = app.capture_output(1.0)

    assert heard.sr == 44100
    assert heard.is_silent() or in_tune(heard, 440.0)


@pytest.mark.xfail(strict=True, reason="New-practice-1: after an output switch the test "
                                       "tone falls silent while its button still says Stop")
def test_the_test_tone_button_tells_the_truth_after_an_output_switch(two_speakers):
    """After the switch the test tone either still plays at 440 Hz or its
    button shows it is off."""
    app = two_speakers
    app.settings.test_tone(True)
    app.capture_output(0.5)

    app.settings.output_device(HEADPHONES)
    heard = app.capture_output(1.0)

    if app.settings.test_tone_playing:
        assert in_tune(heard, 440.0)
    else:
        assert heard.is_silent()


# ------------------------------------------------------------------ tape deck
def test_the_deck_plays_back_and_exports_the_note_i_recorded(app, tmp_path):
    """Record a Bb, listen back through the 48 kHz speakers, export it: the
    playback and the WAV file are both the Bb that was played."""
    bb = Tone(note="Bb4", timbre="sax", seconds=1.5)
    assert app.deck.record(1.5, bb).state == "have_take"

    assert app.deck.play().state == "playing"
    assert in_tune(app.capture_output(0.5), bb.hz, tolerance=5.0)
    assert app.deck.play(wait=True).state == "have_take"       # plays to the end

    take = read_wav(app.deck.export(tmp_path / "phrase.wav"))
    assert take.sr == 44100
    assert take.duration == pytest.approx(1.5, abs=0.1)
    assert in_tune(take, bb.hz)


@pytest.mark.xfail(strict=True, reason="Frodo-19: hitting the recording length limit "
                                       "stops the take without a word")
def test_a_recording_that_hits_the_length_limit_says_so(make_app, tmp_path):
    """With a 1 s limit (300 s by default), a 2 s phrase stops at 1 s and the
    deck tells the player the take was cut at the limit."""
    app = make_app(config={"deck_max_seconds": 1})
    app.launch()

    state = app.deck.record(2.0, Tone(note="A4", seconds=2.0))

    assert state.state == "have_take"
    take = read_wav(app.deck.export(tmp_path / "cut.wav"))   # its message replaces the status
    assert take.duration == pytest.approx(1.0, abs=0.1)
    assert state.status_text


@pytest.mark.xfail(strict=True, reason="Frodo-19: the export dialog suggests no file name "
                                       "and 'Take exported.' does not say where")
def test_exporting_a_take_suggests_a_wav_name_and_says_where_it_went(app, tmp_path):
    """The save dialog offers a .wav name to accept, and the confirmation names
    the file that was written."""
    app.deck.record(1.0, Tone(note="A4", seconds=1.0))

    path = app.deck.export(tmp_path / "long-tone.wav")

    assert path is not None and path.exists()
    _kind, _caption, suggested, _filter = app.dialogs.file_dialogs[-1]
    assert suggested.lower().endswith(".wav")
    assert "long-tone.wav" in (app.deck.state.status_text or "")


@pytest.mark.xfail(strict=True, reason="Frodo-19: an unexported take is discarded on close "
                                       "without a word")
def test_an_unexported_take_is_still_there_after_a_restart(app):
    """Record, close the app without exporting, open it again: the take is
    still there to play or export."""
    app.deck.record(1.0, Tone(note="A4", seconds=1.0))
    assert app.deck.state.state == "have_take"

    app.restart()

    assert app.deck.state.state == "have_take"


@pytest.mark.xfail(strict=True, reason="Sauron-6: deck playback is not re-rated on an "
                                       "output switch and plays about 147 cents flat")
def test_deck_playback_stays_in_tune_when_switching_to_headphones(two_speakers):
    """Listening back to an A, switch to the 44.1 kHz headphones mid-take: the
    A still sounds at 440 Hz."""
    app = two_speakers
    app.deck.record(2.0, Tone(note="A4", seconds=2.0))
    app.deck.play()
    assert in_tune(app.capture_output(0.3), 440.0, tolerance=5.0)

    app.settings.output_device(HEADPHONES)
    heard = app.capture_output(0.5)

    assert app.deck.state.state == "playing"
    assert in_tune(heard, 440.0, tolerance=5.0)

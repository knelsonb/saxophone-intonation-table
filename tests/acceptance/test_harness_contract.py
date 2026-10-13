"""The harness's own contract: each sub-API and each fake-device failure mode,
exercised once through real user paths so scenario authors can trust them.

These tests describe app behaviour only as far as the harness needs it (a busy
microphone is reported as busy, an exported take is a WAV). Detailed scenario
coverage belongs in separate acceptance files.
"""
from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from tests.harness import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    Dialogs,
    FakeAudioHost,
    HarnessError,
    ManualClock,
    Tone,
)


def cents_between(hz: float, reference: float) -> float:
    return 1200.0 * math.log2(hz / reference)


# ---------------------------------------------------------------------- tones
@pytest.mark.parametrize("tone", [
    Tone(note="A4", timbre="sine"),
    Tone(note="A4", timbre="sax"),
    Tone(note="A4", timbre="weak_fundamental"),
    Tone(note="A4", timbre="sax", vibrato=(5.5, 8.0)),
    Tone(freq=442.0, timbre="sine"),
], ids=["sine", "sax", "weak-fundamental", "sax-vibrato", "explicit-freq"])
def test_each_timbre_is_heard_as_the_note_that_was_played(app, tone):
    app.pick_instrument("eb_alto")
    app.play(Tone(**{**tone.__dict__, "seconds": 1.5}))
    row = app.table.row("A4", by="sounding")
    expected = round(cents_between(tone.hz, 440.0))
    assert abs(row.mean - expected) <= 3
    assert row.n >= 5


def test_a_tone_describes_its_own_frequency():
    assert Tone(note="A4").hz == pytest.approx(440.0)
    assert Tone(note="A3").hz == pytest.approx(220.0)
    assert Tone(note="A4", a4=442.0).hz == pytest.approx(442.0)
    assert Tone(note="A4", cents=1200).hz == pytest.approx(880.0)
    with pytest.raises(ValueError):
        Tone()
    with pytest.raises(ValueError):
        Tone(note="A4", freq=440.0)


# ---------------------------------------------------- microphone failure modes
def test_a_busy_microphone_is_reported_and_retry_recovers(make_app, host):
    host.set_fail(DEFAULT_INPUT, "busy")
    app = make_app()
    app.launch()
    status = app.audio.status()
    assert (status.input_state, status.input_error) == ("failed", "device_busy")
    assert status.banner_visible and status.chip_text == "No device"

    host.set_fail(DEFAULT_INPUT, None)
    assert app.audio.retry().input_state == "running"


def test_a_microphone_that_hangs_on_open_times_out_and_retry_recovers(make_app, host):
    host.set_fail(DEFAULT_INPUT, "hang")
    app = make_app()
    app.launch()                                  # costs one open timeout, no real hang
    status = app.audio.status()
    assert (status.input_state, status.input_error) == ("failed", "hostapi_failure")
    assert "timed out" in status.banner_text

    host.set_fail(DEFAULT_INPUT, None)
    assert app.audio.retry().input_state == "running"
    app.close()
    assert host.open_streams() == []              # the late stream was disposed of


@pytest.mark.parametrize("fail, rates", [
    (None, ()),                                   # accepts no rate at all
    ("open_rate", (44100, 48000)),                # probes pass, the real open refuses
], ids=["no-usable-rate", "lying-driver"])
def test_a_microphone_that_refuses_every_rate_is_reported(make_app, host, fail, rates):
    host.remove_device(DEFAULT_INPUT)
    host.add_input("Odd Mic", rates=rates, fail=fail, default_rate=44100)
    app = make_app()
    app.launch()
    status = app.audio.status()
    assert (status.input_state, status.input_error) == ("failed", "unsupported_rate")
    assert status.banner_visible


# ---------------------------------------------------------- output failure modes
def test_the_test_tone_says_so_when_no_speaker_works(make_app, host):
    host.set_fail(DEFAULT_OUTPUT, "busy")
    app = make_app()
    app.launch()
    app.settings.test_tone(True)
    assert not app.settings.test_tone_playing
    assert app.settings.test_tone_error
    assert app.audio.status().output_running is False


def test_the_test_tone_sounds_at_440_hz_through_the_output_rate(app):
    app.settings.test_tone(True)
    assert app.settings.test_tone_playing
    heard = app.capture_output(seconds=0.5)
    assert heard.sr == 48000
    assert abs(cents_between(heard.pitch(near=440.0), 440.0)) < 3.0
    app.settings.test_tone(False)
    app.advance(0.3)
    assert app.capture_output(seconds=0.5).is_silent()


# ------------------------------------------------------------------ table flows
def _play_three_notes(app, cents=0):
    app.pick_instrument("eb_alto")
    for note in ("C4", "E4", "G4"):
        app.play(Tone(note=note, cents=cents, seconds=1.0))


def test_discard_and_reset_ask_first_and_honour_the_answer(app):
    _play_three_notes(app)
    assert [r.sounding for r in app.table.rows()] == ["C4", "E4", "G4"]

    app.table.discard("E4", confirm=False)
    assert [r.sounding for r in app.table.rows()] == ["C4", "E4", "G4"]
    app.table.discard("E4")
    assert [r.sounding for r in app.table.rows()] == ["C4", "G4"]

    app.table.reset(confirm=False)
    assert len(app.table.rows()) == 2
    app.table.reset()
    assert app.table.rows() == []

    with pytest.raises(HarnessError):
        app.table.discard("E4")                            # nothing left to discard


def test_autotune_needs_three_notes_then_proposes_a_new_concert_pitch(app):
    app.pick_instrument("eb_alto")
    app.play(Tone(note="C4", seconds=1.0))
    result = app.autotune.run()
    assert result.status == "insufficient_data" and result.a4_after == 440

    _play_three_notes(app, cents=+20)               # everything 20 cents sharp
    declined = app.autotune.run(accept=False)
    assert (declined.status, declined.a4_after) == ("declined", 440)
    assert len(app.table.rows()) == 3

    applied = app.autotune.run(accept=True)
    assert applied.status == "applied"
    assert (applied.a4_before, applied.a4_after) == (440, 445)
    assert app.a4 == 445 and app.engine_a4 == 445.0
    assert app.table.rows() == []                    # measurements restart at the new pitch


def test_changing_a4_re_reads_the_same_playing_against_the_new_pitch(app):
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    assert app.table.row("A4", by="sounding").mean == 0
    app.set_a4(442)
    # 440 Hz played against A=442 is 8 cents flat.
    assert app.table.row("A4", by="sounding").mean == -8


def test_the_display_can_switch_between_fingered_and_sounding_names(app):
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    app.set_display("sounding")
    assert app.display == "sounding"
    row = app.table.row("A4")
    assert (row.fingered, row.sounding) == ("F#/Gb5", "A4")


# --------------------------------------------------------- settings that persist
def test_a_custom_instrument_and_an_edited_range_survive_a_restart(app):
    key = app.settings.add_custom_instrument("Mezzo Test", transpose=5)
    assert key == "custom_mezzo_test" and app.instrument == key

    assert app.settings.edit_range("C3", "C5")
    assert app.settings.range_error is None
    assert not app.settings.edit_range("C6", "C5")           # low above high is refused
    assert app.settings.range_error

    app.restart()
    assert app.instrument == "custom_mezzo_test"
    listed = app.table.rows(played_only=False)               # the instrument's range
    assert (listed[0].fingered, listed[-1].fingered, len(listed)) == ("C3", "C5", 25)


# ------------------------------------------------------- pitch pipes and the deck
def test_pitch_pipes_sound_the_pressed_note_until_released(app):
    app.pipes.press("A4")
    assert app.pipes.active == [69]
    assert abs(cents_between(app.capture_output(0.5).pitch(near=440.0), 440.0)) < 3.0
    app.pipes.press("C4")                                     # pads can be chorded
    assert app.pipes.active == [60, 69]
    app.pipes.release("A4")
    app.pipes.release("C4")
    app.advance(0.3)
    assert app.capture_output(0.5).is_silent()
    with pytest.raises(HarnessError):
        app.pipes.press("A5")                                 # only C4..B4 exist


def test_the_deck_records_the_microphone_and_exports_a_wav(app, tmp_path):
    assert app.deck.state.state == "idle"
    assert app.deck.record(1.0, Tone(note="A4", seconds=1.0)).state == "have_take"

    path = app.deck.export(tmp_path / "take.wav")
    with wave.open(str(path)) as w:
        assert w.getframerate() == 44100
        assert abs(w.getnframes() - 44100) < 2048        # one microphone block of slack
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.abs(data).max() > 5000

    app.deck.play()
    assert app.deck.state.state == "playing"
    heard = app.capture_output(0.5)
    assert abs(cents_between(heard.pitch(near=440.0), 440.0)) < 5.0
    assert app.deck.play(wait=True).state == "have_take"       # runs to the end


# ---------------------------------------------------------------- harness itself
def test_closing_an_app_leaves_the_process_as_it_found_it(make_app):
    import os
    import sys

    from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox

    watched = ("sounddevice", "sax_config", "sax_intonation_gui", "sax_audio_engine")
    modules = {name: sys.modules.get(name) for name in watched}
    qt = [(c, n, c.__dict__[n]) for c, n in ((QMessageBox, "question"), (QDialog, "exec"),
                                            (QFileDialog, "getSaveFileName"))]
    environment = {k: os.environ.get(k) for k in ("HOME", "SAX_INTONATION_LOG_PATH")}
    hook = sys.excepthook

    app = make_app()
    app.launch()
    assert sys.modules["sax_config"] is not modules["sax_config"]       # a private copy
    assert os.environ["HOME"] == str(app.home)
    app.close()

    assert {name: sys.modules.get(name) for name in watched} == modules
    assert {k: os.environ.get(k) for k in environment} == environment
    assert [(c, n, c.__dict__[n]) for c, n, _ in qt] == qt
    assert sys.excepthook is hook


def test_a_scripted_answer_nobody_used_is_reported():
    dialogs = Dialogs()
    dialogs.expect("question", "Yes")
    dialogs.save_path("/nowhere.txt")
    with pytest.raises(AssertionError, match="never used"):
        dialogs.assert_none_pending()


def test_the_fake_host_delivers_whole_blocks_and_carries_the_rest():
    clock = ManualClock()
    host = FakeAudioHost(clock)
    host.add_input("Mic", rates=(44100,))
    sd = host.sounddevice_module()
    seen = []
    stream = sd.InputStream(samplerate=44100, blocksize=100, channels=1, dtype="float32",
                            callback=lambda indata, frames, t, s: seen.append(frames),
                            device=None)
    assert host.pump_input(np.zeros(250)) == 0       # not started: audio is lost
    stream.start()
    assert host.pump_input(np.zeros(250)) == 200
    assert host.pump_input(np.zeros(50)) == 100      # 50 carried + 50 = one more block
    assert seen == [100, 100, 100]

    host.unplug("Mic")
    assert sd.query_devices() == []
    assert host.pump_input(np.zeros(500)) == 0 and host.dropped_input_frames > 0
    with pytest.raises(sd.PortAudioError):
        stream.close()                                # a dead stream complains, as PortAudio does


def test_the_clock_pulls_output_in_whole_blocks_without_losing_time():
    clock = ManualClock()
    host = FakeAudioHost(clock)
    host.add_output("Speaker", rates=(48000,))
    sd = host.sounddevice_module()
    stream = sd.OutputStream(samplerate=48000, blocksize=2400, channels=1, dtype="float32",
                             callback=lambda out, frames, t, s: out.fill(0.25), device=None)
    stream.start()
    clock.advance(1.0)
    cap = host.output_capture()
    assert len(cap) == 48000 and cap.sr == 48000
    assert cap.peak() == pytest.approx(0.25)
    clock.advance(0.03)                              # 1440 frames: less than a block yet
    assert len(host.output_capture()) == 48000
    clock.advance(0.03)                              # 2880 in total: one block, 480 carried
    assert len(host.output_capture()) == 48000 + 2400

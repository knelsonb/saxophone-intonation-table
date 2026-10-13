"""Audio devices, hot-plug and long sessions, from the player's chair.

A player plugs and unplugs headphones and interfaces, finds a device busy
because a video call still holds it, and leaves the app open through a whole
practice session. These scenarios check that the app says what happened, says
what to do about it, and gets back to listening and sounding on its own where
it can.
"""
from __future__ import annotations

import gc
import json
import math
import time
import tracemalloc

import pytest

from tests.harness import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    FakeAudioHost,
    Tone,
)

INTERFACE = "Scarlett 2i2"

# Length of the simulated practice session. Every simulated second costs about
# 5 ms of test time, mostly the app's own 300 ms table repaint, so 30 minutes
# runs in roughly 11 s.
SESSION_MINUTES = 30


def cents_between(hz: float, reference: float) -> float:
    return 1200.0 * math.log2(hz / reference)


def heard_440(app, seconds: float = 0.5) -> bool:
    heard = app.capture_output(seconds)
    if heard.is_silent():
        return False
    return abs(cents_between(heard.pitch(near=440.0), 440.0)) < 3.0


def is_listening(app, note: str = "A4") -> bool:
    """Play a note into the microphone and see whether the table took it."""
    before = app.table.total_measurements
    app.play(Tone(note=note, seconds=1.5))
    return app.table.total_measurements >= before + 5


def visible_audio_text(app) -> str:
    """Everything the window currently says about audio, in one string."""
    status = app.audio.status()
    parts = [status.status_text, status.chip_text, app.audio.notice or ""]
    if status.banner_visible:
        parts.append(status.banner_text)
    return "\n".join(parts)


# =================================================================== output
@pytest.mark.xfail(strict=True, reason="Frodo-4: an unplugged output is never noticed; "
                                       "the test tone keeps claiming to play")
def test_unplugging_the_speaker_stops_the_test_tone_and_says_so(app):
    """Pull the headphones while the test tone plays: the app notices and the
    button stops claiming to play."""
    app.settings.test_tone(True)
    assert heard_440(app)

    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(2.0)

    assert app.audio.status().output_running is False
    assert not app.settings.test_tone_playing


@pytest.mark.xfail(strict=True, reason="Frodo-4: no output polling; a replugged speaker "
                                       "stays silent until restart")
def test_the_test_tone_sounds_again_after_the_speaker_is_plugged_back_in(app):
    """Unplug and replug the speaker, press the test tone again: it is heard."""
    app.settings.test_tone(True)
    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(2.0)
    app.host.plug(DEFAULT_OUTPUT)
    app.advance(2.0)

    app.settings.test_tone(False)
    app.settings.test_tone(True)

    assert app.settings.test_tone_playing
    assert heard_440(app)


@pytest.mark.xfail(strict=True, reason="Frodo-4: the metronome stays silent after the "
                                       "speaker is replugged")
def test_the_metronome_keeps_clicking_after_the_speaker_is_plugged_back_in(app):
    """A running metronome survives a headphone unplug and replug."""
    app.metronome.start(bpm=120, time_sig="4/4")
    assert len(app.capture_output(2.0).onsets()) >= 3

    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(2.0)
    app.host.plug(DEFAULT_OUTPUT)
    app.advance(2.0)

    assert app.metronome.state.running
    assert len(app.capture_output(2.0).onsets()) >= 3


@pytest.mark.xfail(strict=True, reason="Frodo-4: the SETUP output list is filled once; "
                                       "new outputs never appear until restart")
def test_headphones_plugged_in_after_launch_appear_in_setup(app):
    """Headphones plugged in mid-session show up in the output list."""
    assert DEFAULT_OUTPUT in app.settings.output_devices

    app.host.add_output("USB Headphones", rates=(48000,))
    app.advance(2.0)

    assert "USB Headphones" in app.settings.output_devices


def test_picking_the_speaker_again_after_a_replug_brings_the_sound_back(app):
    """Today's way out after a replug: choose the speaker in SETUP, and the
    test tone works again without a restart."""
    app.settings.test_tone(True)
    app.host.unplug(DEFAULT_OUTPUT)
    app.advance(2.0)
    app.host.plug(DEFAULT_OUTPUT)
    app.advance(2.0)

    app.settings.output_device(DEFAULT_OUTPUT)
    app.settings.test_tone(False)
    app.settings.test_tone(True)

    assert app.audio.status().output_device == DEFAULT_OUTPUT
    assert heard_440(app)


def test_a_speaker_that_refuses_every_rate_says_what_to_do_and_recovers(app):
    """When the only speaker refuses every rate, the test tone and metronome
    tell the player to pick an output; once the OS setting is fixed, pressing
    them again works."""
    app.host.set_rates(DEFAULT_OUTPUT, ())

    app.settings.test_tone(True)
    assert not app.settings.test_tone_playing
    assert "choose an output device" in (app.settings.test_tone_error or "")
    metro = app.metronome.start(bpm=100)
    assert not metro.running
    assert "pick one under setup" in (metro.unavailable_text or "").lower()

    app.host.set_rates(DEFAULT_OUTPUT, (48000,))
    app.settings.test_tone(True)
    assert app.settings.test_tone_error is None
    assert heard_440(app)
    app.settings.test_tone(False)
    assert app.metronome.start(bpm=100).running
    assert len(app.capture_output(2.0).onsets()) >= 3


@pytest.mark.xfail(strict=True, reason="New-6-2: the output failure text names neither "
                                       "the device nor the refused rate")
def test_a_refused_speaker_is_named_in_the_message(app):
    """The test-tone error says WHICH output failed, so the player knows what
    to fix."""
    app.host.set_rates(DEFAULT_OUTPUT, ())

    app.settings.test_tone(True)

    assert DEFAULT_OUTPUT in (app.settings.test_tone_error or "")


# ==================================================================== input
def test_an_unplugged_mic_is_named_and_listening_resumes_on_replug(app):
    """Pull the mic: the banner names it and says to plug it back in. Plug it
    back in: the banner goes and the table records again, no click needed."""
    app.pick_instrument("eb_alto")
    assert is_listening(app)

    app.host.unplug(DEFAULT_INPUT)
    app.advance(1.0)

    status = app.audio.status()
    assert status.banner_visible
    assert f'"{DEFAULT_INPUT}"' in status.banner_text
    assert "plug it back in" in status.banner_text
    assert not is_listening(app)

    app.host.plug(DEFAULT_INPUT)
    app.advance(1.0)

    status = app.audio.status()
    assert (status.input_state, status.input_device) == ("running", DEFAULT_INPUT)
    assert not status.banner_visible
    assert is_listening(app, "C5")


@pytest.mark.xfail(strict=True, reason="New-6-3: Retry with the interface still unplugged "
                                       "says 'No audio input found' and drops its name")
def test_retry_with_the_interface_still_unplugged_keeps_naming_it(make_app, host):
    """Clicking Retry before plugging the interface back in must not claim
    there is no input at all while the laptop mic is right there."""
    host.add_input(INTERFACE, rates=(48000,))
    app = make_app(config={"audio_device_name": INTERFACE})
    app.launch()
    assert app.audio.status().input_device == INTERFACE
    host.unplug(INTERFACE)
    app.advance(1.0)

    status = app.audio.retry()

    assert status.banner_visible
    assert INTERFACE in status.banner_text


def test_retry_while_unplugged_then_replugging_returns_to_listening(make_app, host):
    """Retry too early, then plug the interface in: the app picks it up by
    itself."""
    host.add_input(INTERFACE, rates=(48000,))
    app = make_app(config={"audio_device_name": INTERFACE})
    app.launch()
    app.pick_instrument("eb_alto")
    host.unplug(INTERFACE)
    app.advance(1.0)
    assert app.audio.retry().input_state == "failed"

    host.plug(INTERFACE)
    app.advance(1.0)

    assert app.audio.status().input_device == INTERFACE
    assert is_listening(app)


def test_a_busy_mic_says_what_to_do_and_retry_returns_to_listening(make_app, host):
    """Another app holds the mic: the banner says to close it and click Retry.
    Doing exactly that gets the tuner listening."""
    host.set_fail(DEFAULT_INPUT, "busy")
    app = make_app()
    app.launch()
    app.pick_instrument("eb_alto")

    status = app.audio.status()
    assert (status.input_state, status.input_error) == ("failed", "device_busy")
    assert status.banner_visible
    assert "Close that app and click Retry" in status.banner_text
    assert not is_listening(app)

    host.set_fail(DEFAULT_INPUT, None)
    assert app.audio.retry().input_state == "running"
    assert not app.audio.status().banner_visible
    assert is_listening(app)


@pytest.mark.xfail(strict=True, reason='New-6-1: the busy banner reads \'"?" is busy\'; '
                                       "the device that failed to open is unnamed")
def test_the_busy_banner_names_the_busy_mic(make_app, host):
    """'"?" is busy' tells the player nothing; the banner names the device."""
    host.set_fail(DEFAULT_INPUT, "busy")
    app = make_app()
    app.launch()

    assert f'"{DEFAULT_INPUT}" is busy' in app.audio.status().banner_text


def test_a_mic_that_vanishes_while_opening_reconnects_when_plugged_back(make_app, host):
    """The mic disappears just as the app opens it (PortAudio 'invalid
    device'). It is reported as missing, not busy, and replugging it is enough."""
    host.set_fail(DEFAULT_INPUT, "vanish_on_open")
    app = make_app()
    app.launch()
    app.pick_instrument("eb_alto")

    status = app.audio.status()
    assert (status.input_state, status.input_error) == ("failed", "no_device")
    assert "busy" not in status.banner_text

    app.advance(1.0)
    host.plug(DEFAULT_INPUT)
    app.advance(1.0)

    assert app.audio.status().input_state == "running"
    assert is_listening(app)


@pytest.mark.xfail(strict=True, reason="Uruk1-3: a device that vanishes during the rate "
                                       "probe is reported as an unsupported rate")
def test_a_mic_that_vanishes_during_the_rate_check_is_reported_missing(make_app, host):
    """A mic pulled while the app checks its sample rates is reported as gone
    (not as a rate problem), and replugging it reconnects."""
    host.set_fail(DEFAULT_INPUT, "vanish_on_probe")
    app = make_app()
    app.launch()

    status = app.audio.status()
    assert status.input_error in ("no_device", "device_disconnected")

    app.advance(1.0)
    host.plug(DEFAULT_INPUT)
    app.advance(1.0)
    assert app.audio.status().input_state == "running"


# ------------------------------------------------- saved interface vs fallback
def _launch_without_the_saved_interface(make_app, host):
    host.add_input(INTERFACE, rates=(48000,))
    host.unplug(INTERFACE)
    app = make_app(config={"audio_device_name": INTERFACE})
    app.launch()
    assert app.audio.status().input_device == DEFAULT_INPUT     # the fallback mic
    return app


@pytest.mark.xfail(strict=True, reason="Frodo-5: falling back to another mic shows no "
                                       "notice naming the missing interface")
def test_starting_without_the_saved_interface_says_so(make_app, host):
    """Launching with the interface unplugged tells the player the app is on
    another mic and which one it was looking for."""
    app = _launch_without_the_saved_interface(make_app, host)

    shown = visible_audio_text(app)
    assert INTERFACE in shown and DEFAULT_INPUT in shown


@pytest.mark.xfail(strict=True, reason="Frodo-5: the fallback mic overwrites the saved "
                                       "interface, so it is never used again")
def test_the_saved_interface_is_used_again_after_a_session_on_the_fallback(make_app, host):
    """One session on the laptop mic must not make the app forget the
    interface: with it plugged in again, the next launch uses it."""
    app = _launch_without_the_saved_interface(make_app, host)
    app.close()

    host.plug(INTERFACE)
    app.launch()

    assert app.audio.status().input_device == INTERFACE


def test_plugging_the_interface_back_in_gets_it_used(make_app, host):
    """Plug the interface in mid-session: the app either switches to it or
    offers to, and accepting the offer switches."""
    app = _launch_without_the_saved_interface(make_app, host)

    host.plug(INTERFACE)
    app.advance(1.0)

    if app.audio.status().input_device != INTERFACE:
        assert INTERFACE in (app.audio.notice or "")
        app.audio.accept_notice()
    status = app.audio.status()
    assert (status.input_state, status.input_device) == ("running", INTERFACE)
    app.pick_instrument("eb_alto")
    assert is_listening(app)


# ------------------------------------------------------------ rates and hangs
@pytest.mark.parametrize("rates, expected", [
    ((44100, 48000, 88200, 96000, 192000), 48000),
    ((44100, 88200, 96000, 192000), 44100),
    ((96000, 192000), 96000),
], ids=["48k-preferred", "44k-when-no-48k", "high-rate-only-device"])
def test_auto_sample_rate_prefers_48k_then_44k(make_app, host, rates, expected):
    """An interface that offers 192 kHz is still opened at 48 or 44.1 kHz,
    where detection is cheap; higher rates only when nothing else works."""
    host.remove_device(DEFAULT_INPUT)
    host.add_input("Rate Mic", rates=rates, default_rate=max(rates))
    app = make_app()
    app.launch()
    app.pick_instrument("eb_alto")

    assert app.audio.status().input_rate == expected
    assert is_listening(app)


def test_the_speaker_also_opens_at_48k_when_it_offers_more(make_app, host):
    """The output side follows the same rate preference."""
    host.remove_device(DEFAULT_OUTPUT)
    host.add_output("Rate Speaker", rates=(44100, 48000, 96000, 192000),
                    default_rate=192000)
    app = make_app()
    app.launch()
    app.settings.test_tone(True)

    assert app.audio.status().output_rate == 48000
    assert heard_440(app)


def _timed_retry(app) -> float:
    t0 = time.perf_counter()
    app.audio.retry()
    return time.perf_counter() - t0


def test_a_hung_mic_freezes_the_window_no_longer_than_the_open_timeout(app):
    """A driver that stops answering costs one open timeout, then the banner
    says it timed out and Retry is still there."""
    app.host.set_fail(DEFAULT_INPUT, "hang")

    frozen = _timed_retry(app)

    status = app.audio.status()
    assert status.input_state == "failed" and "timed out" in status.banner_text
    assert frozen < FakeAudioHost.OPEN_TIMEOUT_S + 1.0


@pytest.mark.xfail(strict=True, reason="Legolas-10: the open timeout is paid again for "
                                       "every candidate rate on the UI thread")
def test_a_hung_mic_with_many_rates_freezes_no_longer_than_one_timeout(make_app, host):
    """A hung interface that lists three rates must not freeze the window three
    times as long."""
    host.remove_device(DEFAULT_INPUT)
    host.add_input("Hung Mic", rates=(44100, 48000, 96000))
    app = make_app()
    app.launch()
    host.set_fail("Hung Mic", "hang")

    frozen = _timed_retry(app)

    assert app.audio.status().input_state == "failed"
    assert frozen < 2 * FakeAudioHost.OPEN_TIMEOUT_S


# ============================================================ long sessions
def _history_lines(app) -> list[dict]:
    path = app.file("history")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _retained_by_app_while(action) -> int:
    """Bytes of Python memory allocated during ``action`` and still alive
    after it. The harness's own recording of the speaker is left out: it keeps
    minutes of output on purpose."""
    gc.collect()
    tracemalloc.start()
    try:
        action()
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    snapshot = snapshot.filter_traces([
        tracemalloc.Filter(False, "*/tests/harness/*"),
        tracemalloc.Filter(False, tracemalloc.__file__),
    ])
    return sum(stat.size for stat in snapshot.statistics("filename"))


def test_a_thirty_minute_practice_session_leaves_nothing_behind(make_app, host):
    """Half an hour of scales, metronome, drone, pads, test tone and one mic
    replug: sound sources come and go without piling up, nothing keeps sounding
    afterwards, idle time costs no memory and the history file grows only by
    what was played."""
    app = make_app(config={"persistence_enabled": True})
    app.launch()
    app.pick_instrument("eb_alto")
    clock = host.clock
    peak_sources = 0
    left_after_block = []

    def idle(seconds: float) -> None:
        nonlocal peak_sources
        app.advance(max(0.0, seconds))
        peak_sources = max(peak_sources, app.audio.output_sources)

    session_end = 60.0 * SESSION_MINUTES
    for block in range(SESSION_MINUTES // 5 - 1):   # five-minute practice blocks
        block_end = 300.0 * (block + 1)
        for note, cents in (("C4", -6), ("E4", 9), ("G4", 0)):
            app.play(Tone(note=note, cents=cents, seconds=1.5, timbre="sax"))
        app.metronome.start(bpm=60 + 10 * block, time_sig="4/4")
        idle(100.0)
        app.drone.on("A3", voice="sine")
        app.play(Tone(note="E4", seconds=1.5, timbre="sax"))
        idle(50.0)
        app.drone.off()
        app.metronome.stop()
        if block % 2 == 0:
            app.settings.test_tone(True)
            idle(1.0)
            app.settings.test_tone(False)
        else:
            app.pipes.press("A4")
            idle(2.0)
            app.pipes.release("A4")
            app.pipes.close()
        if block == 2:
            host.unplug(DEFAULT_INPUT)
            idle(2.0)
            host.plug(DEFAULT_INPUT)
        idle(block_end - 1.0 - clock.now)
        assert app.capture_output(1.0).is_silent(), f"still sounding after block {block + 1}"
        assert not app.tuner.reading().active
        # A switched-off drone may stay parked in the mix for its next use;
        # anything beyond that would be a source that never went away.
        left_after_block.append(app.audio.output_sources)

    assert 1 <= peak_sources <= 3
    assert max(left_after_block) <= 1
    assert len(set(left_after_block)) == 1, f"sources piled up: {left_after_block}"
    assert app.audio.status().input_state == "running"

    # The last five minutes: the metronome runs while the player reads the
    # table. That holds on to no Python memory and writes no history.
    size_before = app.file("history").stat().st_size
    app.metronome.start(bpm=90)
    idle(30.0)
    retained = _retained_by_app_while(lambda: idle(30.0))
    idle(session_end - 1.0 - clock.now)
    app.metronome.stop()
    idle(session_end - clock.now)
    assert clock.now == pytest.approx(session_end, abs=0.1)
    assert retained < 256 * 1024, f"{retained} bytes retained over 30 idle seconds"
    assert app.file("history").stat().st_size == size_before

    # The history file holds exactly what was measured: one line per run that
    # recorded something, one per measurement, nothing for idle time.
    lines = _history_lines(app)
    runs, measurements = app.history_counts
    assert measurements == app.table.total_measurements
    assert measurements >= 4 * 5 * (SESSION_MINUTES // 5 - 1)   # every held note counted
    assert sum(1 for x in lines if x.get("kind") == "measurement") == measurements
    assert sum(1 for x in lines if x.get("kind") == "run") == runs
    assert size_before < 400 * (runs + measurements)


def _old_history(months: int = 3, per_month: int = 40) -> str:
    lines = []
    for m in range(1, months + 1):
        run_id = f"oldrun{m:06d}"
        lines.append(json.dumps({
            "kind": "run", "run_id": run_id, "started_at": f"2025-{m:02d}-10T19:00:00",
            "instrument": "eb_alto", "a4_hz": 440.0, "maker": "", "model": "", "label": ""}))
        for i in range(per_month):
            lines.append(json.dumps({
                "kind": "measurement", "run_id": run_id,
                "timestamp": f"2025-{m:02d}-10T19:{i // 60:02d}:{i % 60:02d}",
                "instrument": "eb_alto", "a4_hz": 440.0, "midi_sounding": 69,
                "midi_fingered": 78, "cents": 2.0, "freq_hz": 440.5}))
    return "\n".join(lines) + "\n"


def test_last_years_history_is_moved_out_of_the_live_log(make_app):
    """The expected bound: the live history file holds the current month,
    and older months are kept beside it rather than re-read on every start."""
    old = _old_history()
    app = make_app(config={"persistence_enabled": True}, history=old)
    app.launch()
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.5))

    live = app.file("history").read_text(encoding="utf-8")
    assert '"2025-' not in live
    kept = "".join(p.read_text(encoding="utf-8", errors="replace")
                   for p in app.file("history").parent.iterdir()
                   if p.is_file() and p != app.file("history"))
    assert kept.count('"kind": "measurement"') >= 3 * 40

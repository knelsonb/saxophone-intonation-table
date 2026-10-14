"""The metronome keeps its beat across an output reopen.

Opening another output device (or reopening the same one after a replug)
restarts the mixer's clock, possibly at another sample rate
(``Mixer.set_format``). The pending beat moves with it, keeping its distance in
time, and the metronome carries its grid on from where that beat actually
sounds: the next click comes one beat after the last one heard, the beat
index (and so the accent pattern) continues, and later beats are counted at
the new rate. Nothing outside the mixer has to resynchronise the metronome.
"""
from __future__ import annotations

import numpy as np

from intonation.audio.mixer import Mixer
from intonation.audio.sources.metronome import MetronomeController

SR = 48000
MB = 1024


def _record(metro):
    """Wrap the click's trigger: beat index -> (onset sample, accent). The
    trigger runs inside the mixer's render, while its clock still reads the
    block's first sample."""
    seen = {}
    click = metro.click_source
    orig = click.trigger

    def trigger(accent, offset=0):
        seen[metro._beat] = (metro._mixer.clock + offset, accent)
        orig(accent, offset)
    click.trigger = trigger
    return seen


def _run_to_beat(mixer, metro, target_beat, block=MB):
    """Render blocks until the metronome has fired ``target_beat``."""
    out = np.zeros(block, dtype=np.float32)
    guard = 0
    while metro._beat < target_beat:
        mixer.render(out, block)
        guard += 1
        assert guard < 100000, "metronome failed to advance"


def test_the_beat_chain_survives_a_clock_reset():
    """reset_clock (a reopen at the same rate) keeps the next beat scheduled
    and the beat index continuing."""
    mixer = Mixer(max_block=MB, samplerate=SR)
    metro = MetronomeController(mixer, SR, bpm=200, time_sig="4/4")
    beats = _record(metro)
    metro.start()
    _run_to_beat(mixer, metro, 5)
    due_in = metro._onset(6) - mixer.clock

    mixer.reset_clock()

    assert mixer.pending_events() == 1, "the next beat must stay scheduled"
    _run_to_beat(mixer, metro, 6)
    assert beats[6][0] == due_in, "the beat comes when it was due, not at once"


def test_a_rate_change_keeps_the_beat_in_time_and_the_bar_count():
    """48 kHz -> 44.1 kHz mid-bar: the next click comes one beat after the last
    one (in seconds), the accent pattern continues, and later beats are an
    interval apart at the new rate."""
    mixer = Mixer(max_block=MB, samplerate=SR)
    metro = MetronomeController(mixer, SR, bpm=120, time_sig="4/4")
    beats = _record(metro)
    metro.start()
    _run_to_beat(mixer, metro, 5)                 # beat 6 is not a downbeat
    since_last_s = (mixer.clock - beats[5][0]) / SR

    mixer.set_format(44100, 1000)
    _run_to_beat(mixer, metro, 8, block=1000)

    gap_s = since_last_s + beats[6][0] / 44100
    assert abs(gap_s - 0.5) < 1.0 / 44100, f"beat gap across the switch {gap_s:.5f} s"
    assert beats[7][0] - beats[6][0] == 22050, "beats after the switch count at 44.1 kHz"
    assert beats[8][0] - beats[7][0] == 22050
    assert [beats[i][1] for i in (5, 6, 7, 8)] == [False, False, False, True], (
        "bar 3 starts on beat 8, as it would have without the switch")


def test_a_stopped_metronome_starts_at_the_new_rate():
    """A format change while stopped schedules nothing; the next start counts
    at the mixer's current rate."""
    mixer = Mixer(max_block=MB, samplerate=SR)
    metro = MetronomeController(mixer, SR, bpm=120)
    beats = _record(metro)
    mixer.set_format(44100, MB)
    assert mixer.pending_events() == 0 and not metro.is_running()

    metro.start()
    _run_to_beat(mixer, metro, 1)
    assert beats[1][0] - beats[0][0] == 22050


def test_two_resets_never_double_schedule_a_beat():
    """Each reset moves the one pending beat; none is added."""
    mixer = Mixer(max_block=MB, samplerate=SR)
    metro = MetronomeController(mixer, SR, bpm=200)
    metro.start()
    mixer.reset_clock()
    mixer.set_format(SR, MB)
    assert mixer.pending_events() == 1

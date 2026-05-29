"""METRO-RESYNC-B (v1.1 hardening ledger / #12): the Option-B controller
resync() keeps the metronome PHASE-CORRECT across a GUI output-device switch —
it resumes on the NEXT beat index (accent pattern preserved), not restarted at
beat 0 like the old stop()/start() bracket (Option A).

A device switch resets the mixer clock (open_output_device -> mixer.reset_clock,
which clears the schedule). resync() re-anchors the beat chain onto the reset
clock continuing the global beat index, so is_accent() stays consistent: no
spurious downbeat mid-bar, and the click is never request_stopped (no pop).
"""
from __future__ import annotations

import numpy as np

from sax_mixer import Mixer
from sax_metronome import MetronomeController

SR = 48000
MB = 1024


def _run_to_beat(mixer, metro, target_beat, out):
    """Render blocks until the metronome has fired at least ``target_beat``."""
    guard = 0
    while metro._beat < target_beat:
        out[:] = 0.0
        mixer.render(out, MB)
        guard += 1
        assert guard < 100000, "metronome failed to advance"


def test_resync_continues_beat_index_across_device_switch():
    """resync() resumes on the NEXT beat index, not beat 0 (the stop/start reset)."""
    mixer = Mixer(max_block=MB)
    metro = MetronomeController(mixer, SR, bpm=200, time_sig="4/4")
    metro.start()
    out = np.zeros(MB, dtype=np.float32)
    _run_to_beat(mixer, metro, 5, out)         # advance several beats into the run
    fired_before = metro._beat
    assert fired_before >= 5

    # --- simulate a GUI-orchestrated output-device switch ---
    mixer.reset_clock()                        # open_output_device() does this
    assert mixer.pending_events() == 0, "the reopen clears the pending schedule"
    metro.resync()                             # Option B: re-anchor, keep the index
    assert mixer.pending_events() >= 1, "resync must reschedule the next beat"
    assert metro._anchor_beat == fired_before + 1, "anchored at the continued index"

    target = fired_before + 1
    _run_to_beat(mixer, metro, target, out)
    assert metro._beat == target, (
        f"beat index must CONTINUE across the switch: expected {target}, got "
        f"{metro._beat} (stop()/start() would have reset it to 0)")


def test_resync_preserves_accent_pattern_not_a_downbeat_reset():
    """The resumed beat carries the accent the CONTINUOUS grid dictates — proving
    it's a real resync, not a reset (which would resume on an accented beat 0)."""
    mixer = Mixer(max_block=MB)
    metro = MetronomeController(mixer, SR, bpm=200, time_sig="4/4")

    accents = {}                               # beat_index -> accent flag at fire
    click = metro.click_source
    _orig = click.trigger
    click.trigger = lambda ac, off=0, *a: (accents.__setitem__(metro._beat, ac),
                                           _orig(ac, off, *a))[1]
    metro.start()
    out = np.zeros(MB, dtype=np.float32)
    _run_to_beat(mixer, metro, 5, out)         # next beat is 6 (6 % 4 == 2 -> not a downbeat)
    fired_before = metro._beat

    mixer.reset_clock()
    metro.resync()
    target = fired_before + 1
    _run_to_beat(mixer, metro, target, out)

    assert not metro.is_accent(target), (
        f"test setup: resumed beat {target} should be a non-downbeat")
    assert accents.get(target) is False, (
        f"resumed beat {target} fired as accent={accents.get(target)} — a reset "
        f"to beat 0 would fire an ACCENT; resync must preserve the mid-bar phase")


def test_resync_is_noop_when_stopped():
    """resync() on a stopped metronome must not schedule or raise."""
    mixer = Mixer(max_block=MB)
    metro = MetronomeController(mixer, SR, bpm=120)
    metro.resync(48000)                        # not running
    assert mixer.pending_events() == 0
    assert not metro.is_running()


def test_resync_updates_samplerate():
    """resync(new_sr) adopts the new output rate (re-synthesised click)."""
    mixer = Mixer(max_block=MB)
    metro = MetronomeController(mixer, SR, bpm=120)
    metro.start()
    metro.resync(44100)
    assert metro._samplerate == 44100

"""STAB-THREAD-BOUNDARY (v1.1 hardening ledger): no audio/render-thread path may
fire a controller's on_state_changed (the GUI relabels widgets from it) — only
GUI-thread control methods may. Audit + regression locks.

Audit of every on_state_changed (`_emit`) call site:
  * DeckController: the delicate case — a playback source exhausts and a recording
    hits its cap on the AUDIO thread, but each only flips a lock-free flag;
    DeckController.pump() (GUI tick) observes it and does the transition + _emit.
    LOCKED by test_deck.py::test_playback_end_transitions_to_have_take_only_via_pump
    ("render (audio thread) must NOT perform the transition") +
    test_cap_hit_auto_disarm_finalizes_to_have_take_via_pump.
  * MetronomeController: the beat event (sax_metronome.py:413, "fires on the render
    thread") triggers the click + reschedules — never _emit. _emit only at
    start/stop/set_bpm/set_time_sig/set_volume (GUI). LOCKED HERE.
  * PitchPipesController: _emit only at toggle/release/clear (GUI); source.render +
    the mixer's auto-reap (render thread) never touch the controller callback.
    LOCKED HERE.
  * DroneController: _emit only at the set_* GUI methods (sax_drone.py:511+);
    render() + set_duck_target (the D3 render-thread duck path) never _emit —
    GUI-thread-only by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from sax_mixer import Mixer
from sax_metronome import MetronomeController

PP = pytest.importorskip("sax_pitch_pipes", reason="pitch pipes module")

SR = 48000
MB = 1024


def test_metronome_beat_event_does_not_fire_on_state_changed():
    """The per-beat event runs on the RENDER thread (sax_metronome.py:413). It
    must trigger the click + reschedule but NEVER fire on_state_changed — only
    GUI-thread start/stop/set_* may notify the GUI."""
    mixer = Mixer(max_block=MB)
    fired = [0]
    metro = MetronomeController(
        mixer, SR, bpm=240,
        on_state_changed=lambda *a: fired.__setitem__(0, fired[0] + 1))
    metro.start()
    baseline = fired[0]            # start() legitimately notifies (GUI thread)

    click = metro.click_source     # count beats so 'no _emit' isn't vacuous
    beats = [0]
    _orig = click.trigger
    click.trigger = lambda *a, **k: (beats.__setitem__(0, beats[0] + 1),
                                     _orig(*a, **k))[1]
    out = np.zeros(MB, dtype=np.float32)
    for _ in range(500):           # render thread fires ~40 beats @240bpm
        out[:] = 0.0
        mixer.render(out, MB)

    assert beats[0] >= 5, "beats must actually fire on the render thread (else vacuous)"
    assert fired[0] == baseline, (
        f"on_state_changed fired {fired[0] - baseline}x from the render-thread "
        f"beat event — the GUI callback must only fire from GUI-thread methods")


def test_pitchpipe_render_thread_fade_does_not_fire_on_state_changed():
    """A released pad fades + is reaped on the RENDER thread (source.render +
    the mixer's auto-reap). The controller's on_state_changed must only fire from
    the GUI-thread toggle/release path, never from render."""
    mixer = Mixer(max_block=MB)
    fired = [0]
    pipes = PP.PitchPipesController(
        mixer, SR,
        on_state_changed=lambda *a: fired.__setitem__(0, fired[0] + 1))
    pipes.toggle(69)               # sound A4   (GUI -> notifies)
    pipes.toggle(69)               # release it (GUI -> notifies)
    baseline = fired[0]

    out = np.zeros(MB, dtype=np.float32)
    for _ in range(200):           # render: pad fades to finished, mixer reaps it
        out[:] = 0.0
        mixer.render(out, MB)

    assert mixer.active_sources() == 0, (
        "the released pad must have been reaped during render (else vacuous)")
    assert fired[0] == baseline, (
        "on_state_changed fired from the render-thread fade/reap — it must only "
        "fire from the GUI-thread toggle path")

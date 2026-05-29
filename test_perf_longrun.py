"""PERF-LONGRUN (v1.1 hardening ledger): a long session stays bounded — memory
does not grow with runtime, and the mixer's scheduler + source list stay
O(active), not O(history).

The risk this guards: the metronome fires + reschedules a beat on a one-beat
rolling horizon (~2000 beats in 10 sim-minutes here). If render() failed to reap
a fired event (sax_mixer.py:296 pops it), or a source were re-registered instead
of reused, the scheduler queue / source list — and memory — would grow with
elapsed time. We render 10 minutes of sim-time (cheap: numpy blocks in a loop,
no wall-clock wait, no audio device) and assert the queue depth, source count,
dropped-event canary, and retained memory all stay flat. The beat counter proves
the fire+reschedule loop actually ran, so the bounded assertions aren't vacuous.
"""
from __future__ import annotations

import tracemalloc

import numpy as np

from sax_mixer import Mixer, TestToneSource
from sax_metronome import MetronomeController

SR = 48000
MB = 1024


def test_longrun_mixer_metronome_is_bounded():
    mixer = Mixer(max_block=MB)
    mixer.register(TestToneSource(440.0, SR, MB, gain=0.1))   # a steady source too
    metro = MetronomeController(mixer, SR, bpm=200)           # ~2000 beats / 10 min
    metro.start()

    # Count beats actually fired by wrapping the click's trigger — without this
    # the bounded-queue check could pass simply because nothing ever fired.
    click = metro.click_source
    beats = [0]
    _orig_trigger = click.trigger

    def _counting_trigger(*a, **k):
        beats[0] += 1
        return _orig_trigger(*a, **k)

    click.trigger = _counting_trigger

    out = np.zeros(MB, dtype=np.float32)
    for _ in range(50):                       # warm up before snapshotting memory
        out[:] = 0.0
        mixer.render(out, MB)

    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    n_blocks = (SR * 600) // MB               # 10 minutes of sim-time
    max_pending = max_sources = 0
    for _ in range(n_blocks):
        out[:] = 0.0
        mixer.render(out, MB)
        max_pending = max(max_pending, mixer.pending_events())
        max_sources = max(max_sources, mixer.active_sources())
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = cur - base

    assert 1950 <= beats[0] <= 2050, (
        f"expected ~2000 beats over 10 min @200bpm, got {beats[0]} — the "
        f"fire+reschedule loop must actually run for the bounds below to mean "
        f"anything")
    assert max_pending <= 4, (
        f"scheduler queue reached {max_pending} entries over ~2000 beats — fired "
        f"events must be reaped (the rolling horizon is O(active), not O(history))")
    assert max_sources <= 3, (
        f"source list reached {max_sources} — the click must be reused/reaped, "
        f"not re-registered per beat")
    assert mixer.dropped_events == 0, (
        f"{mixer.dropped_events} dropped events — the scheduler fell behind / a "
        f"beat landed in the past (drift)")
    assert retained < 64_000, (
        f"render retained {retained} B over 10 sim-minutes — memory that grows "
        f"with runtime (a per-block / per-beat leak)")

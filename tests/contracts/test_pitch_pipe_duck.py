"""D3 contract for pitch pads: a held pad is a duck consumer like the drone.

Mic bleed of a sounding pad is ducked, and a player holding the pad's own note
is told apart from that bleed (it does not follow the duck) and measured. The
drone has the same contract in tests/acceptance/test_practice_tools.py.
"""
from __future__ import annotations

import numpy as np

import sax_pitch_pipes as pp
from sax_mixer import Mixer
from tests.harness import Tone


class _Engine:
    """The duck surface of the engine, recording what is attached."""

    def __init__(self) -> None:
        self.consumers: list = []

    def attach_duck_consumer(self, c) -> None:
        if c not in self.consumers:
            self.consumers.append(c)

    def detach_duck_consumer(self, c=None) -> None:
        self.consumers = [] if c is None else [x for x in self.consumers if x is not c]


def test_a_sounding_pad_is_a_duck_consumer_until_it_is_gone():
    eng = _Engine()
    ctrl = pp.PitchPipesController(Mixer(max_block=1024), 48000, engine=eng)
    ctrl.toggle(69)
    ctrl.toggle(64)
    assert [s.midi for s in eng.consumers] == [69, 64]
    ctrl.toggle(69)                    # released: still audible, still ducked
    assert len(eng.consumers) == 2
    ctrl.silence()
    assert eng.consumers == []


def test_a_ducked_pad_is_quieter():
    src = pp.PitchPipeSource(69, 48000, 1024)
    block = np.zeros(1024, dtype=np.float32)
    peaks = []
    for target in (1.0, 0.3):
        src.set_duck_target(target)
        for _ in range(30):            # past the attack and the duck glide
            block.fill(0.0)
            src.render(block, 1024, 0)
        peaks.append(float(np.abs(block).max()))
    assert peaks[1] < 0.35 * peaks[0]


def test_playing_in_unison_with_a_held_pitch_pipe_is_measured(app):
    """Tuning a long A against the A pipe: the played A shows up in the table,
    as it does against an A drone."""
    app.pick_instrument("eb_alto")
    app.pipes.press("A4")

    app.play(Tone(note="A4", timbre="sax", cents=-6, seconds=1.5))

    row = app.table.row("A4", by="sounding")
    assert row.mean == -6 and row.n >= 5
    assert app.pipes.active == [69]

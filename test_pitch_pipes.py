"""Acceptance net for ``sax_pitch_pipes`` (parity Sprint 3).

Pitch pipes are pure-numpy sine pads over the chromatic reference octave
(C4..B4 = MIDI 60..71), tuned at the current A4. No TSF, no Qt — pure math +
the Mixer, so the whole net runs on the always-runnable suite.

Pre-written against Sauron's firm S3 contract (msg 3430); skip-guarded so it
auto-activates the moment sax_pitch_pipes lands.
"""
from __future__ import annotations

import numpy as np
import pytest

from sax_mixer import Mixer

pp = pytest.importorskip(
    "sax_pitch_pipes",
    reason="sax_pitch_pipes not landed yet (Sauron's S3 lane) — auto-activates on land")


def _expected_freq(midi, a4):
    return a4 * (2.0 ** ((midi - 69) / 12.0))


# ---------------------------------------------------------------------------
# 1. Note table + frequency math.
# ---------------------------------------------------------------------------
def test_pitch_pipe_notes_are_c4_to_b4():
    assert list(pp.PITCH_PIPE_NOTES) == list(range(60, 72)), (
        "default pitch pipes span the chromatic reference octave C4..B4 "
        "(MIDI 60..71)")


@pytest.mark.parametrize("midi,a4,expected", [
    pytest.param(69, 440.0, 440.0, id="A4_at_440"),
    pytest.param(60, 440.0, 261.6255653, id="C4_at_440"),
    pytest.param(71, 440.0, 493.8833013, id="B4_at_440"),
    pytest.param(69, 442.0, 442.0, id="A4_at_442"),
    pytest.param(69, 430.0, 430.0, id="A4_at_430"),
    pytest.param(60, 442.0, 442.0 * 2 ** (-9 / 12), id="C4_at_442_nondefault"),
])
def test_pitch_pipe_freq(midi, a4, expected):
    assert pp.pitch_pipe_freq(midi, a4) == pytest.approx(expected, rel=1e-6)


def test_pitch_pipe_freq_octave_doubles():
    for a4 in (430.0, 440.0, 442.0):
        assert pp.pitch_pipe_freq(72, a4) == pytest.approx(
            2.0 * pp.pitch_pipe_freq(60, a4), rel=1e-9)


def test_pitch_pipe_freq_nondefault_a4_scales_whole_octave():
    """At a non-440 A4 every note shifts by the same ratio (the table is
    relative to the reference, not hardcoded Hz)."""
    ratio = 442.0 / 440.0
    for midi in pp.PITCH_PIPE_NOTES:
        assert pp.pitch_pipe_freq(midi, 442.0) == pytest.approx(
            pp.pitch_pipe_freq(midi, 440.0) * ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. PitchPipesController — toggle / active set / release / a4.
# ---------------------------------------------------------------------------
def _ctrl(a4=440.0):
    return pp.PitchPipesController(Mixer(max_block=4096), 48000, a4=a4)


def test_controller_starts_with_no_active_pipes():
    c = _ctrl()
    assert list(c.active_midis()) == [] or c.active_midis() == frozenset()


def test_toggle_adds_then_removes_a_pipe():
    c = _ctrl()
    c.toggle(64)
    assert 64 in c.active_midis(), "toggle should sound the pad"
    c.toggle(64)
    assert 64 not in c.active_midis(), "toggle again should release it"


def test_multiple_pipes_sound_together():
    c = _ctrl()
    for m in (60, 64, 67):          # a C-major triad
        c.toggle(m)
    active = set(c.active_midis())
    assert {60, 64, 67} <= active


def test_release_all_clears_every_pipe():
    c = _ctrl()
    for m in (60, 64, 67):
        c.toggle(m)
    c.release_all()
    assert set(c.active_midis()) == set(), "release_all must silence every pad"


def test_active_pipes_are_pitched_sources_on_the_mixer():
    """A sounding pitch pipe sustains a pitch, so it must report active_midi
    (unlike the metronome click) — feeding get_sounding_output_midis() / the
    D3 vote-exclude. Verify via the mixer the controller drives."""
    m = Mixer(max_block=4096)
    c = pp.PitchPipesController(m, 48000, a4=440.0)
    c.toggle(67)
    assert 67 in m.sounding_midis(), (
        "a sounding pitch pipe must appear in the mixer's sounding MIDIs "
        "(pitched source, active_midi set)")


def test_set_a4_retunes_active_pipes():
    c = _ctrl(a4=440.0)
    c.set_a4(442.0)
    # The note set is unchanged; only the tuning reference moved.
    c.toggle(69)
    assert 69 in c.active_midis()


def test_pipes_produce_audio_through_mixer():
    m = Mixer(max_block=4096)
    c = pp.PitchPipesController(m, 48000, a4=440.0)
    c.toggle(69)
    out = np.zeros(4096, dtype=np.float32)
    m.render(out, 4096)
    assert float(np.max(np.abs(out))) > 1e-3, "a sounding pipe must produce audio"

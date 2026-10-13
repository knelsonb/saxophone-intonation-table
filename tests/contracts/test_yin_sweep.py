"""YIN sweep contract: every MIDI note the table can show is read correctly.

The catalog's sounding ranges span MIDI 21..108 (A0..C8), but the older accuracy
nets only cover 82..2093 Hz. This sweep drives the engine's REAL detection path
(silence gate -> YIN -> confirm/lock -> emit) through ``sim_harness.detect_tone``
for every note, at both common sample rates, with three timbres:

  sine              one partial
  sax_like          five harmonics, fundamental strongest
  weak_fundamental  2nd harmonic at twice the fundamental amplitude, the signal
                    that tempts a detector to answer an octave high

Contract per cell: the dominant confirmed MIDI equals the target, the cents error
is within 1 ct up to MIDI 84 and 3 ct above, and the engine's own cents readout
agrees with the independently computed error.

Cells that miss the contract today are listed in ``KNOWN_TOP_OCTAVE`` with their
measured numbers and marked ``xfail(strict=True)``: fixing the detector turns them
into XPASS failures, which is the prompt to delete the entry. A companion test
bounds the drift of the same region so it cannot get worse unnoticed.

Cost: about 700 detections at roughly 15 ms each, so about 12 s in total.
"""
from __future__ import annotations

import pytest

import sim_harness as sim

HOPS = 10          # DEFAULT_N-sample hops fed per cell; about 4 confirmed notes
MIDI_LOW = 21      # A0
MIDI_HIGH = 108    # C8
TIMBRES = ('sine', 'sax', 'weak_fundamental')
RATES = (44100, 48000)

_ENGINE_CENTS_AGREEMENT = 0.05   # ct between the engine readout and our math
_TOP_OCTAVE_CEILING_CT = 12.0    # regression ceiling; measured worst 10.17 ct
_OCTAVE_START = 97


def _hz(midi: int, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((midi - 69) / 12.0)


def _tolerance_ct(midi: int) -> float:
    return 1.0 if midi <= 84 else 3.0


# (timbre, sample rate) -> {midi: measured behaviour}. Measured on the engine as
# of v1.2.1 with HOPS hops at A4=440. Above about 2.1 kHz the period is under
# 21 samples at 44.1 kHz, so the integer-lag YIN loses sub-sample accuracy.
KNOWN_TOP_OCTAVE: dict[tuple[str, int], dict[int, str]] = {
    ('sine', 44100): {
        102: '+3.35 ct', 103: '+4.33 ct', 104: '+4.44 ct',
        106: '+4.39 ct', 107: '+6.95 ct'},
    ('sax', 44100): {
        103: '+4.80 ct', 104: '+7.49 ct', 107: '+10.17 ct',
        108: 'no note confirmed'},
    ('weak_fundamental', 44100): {
        103: '+4.31 ct', 104: '+6.00 ct', 105: 'no note confirmed',
        107: '+8.33 ct', 108: 'reads MIDI 96, an octave low (-1198 ct)'},
    ('sine', 48000): {
        102: '+3.12 ct', 103: '+3.09 ct', 106: '+4.16 ct', 107: '+5.87 ct'},
    ('sax', 48000): {
        100: '+3.48 ct', 101: '+3.92 ct', 102: '+4.58 ct', 103: '+5.15 ct',
        104: '+3.41 ct', 107: '+8.18 ct', 108: '+5.09 ct'},
    ('weak_fundamental', 48000): {
        101: '+3.36 ct', 102: '+3.85 ct', 103: '+4.16 ct', 107: '+6.82 ct',
        108: 'no note confirmed'},
}

# Cells where the wrong note (or no note) comes out; inexact cells start with a
# signed cents figure.
_IDENTITY_FAILURES = {
    (t, sr, m)
    for (t, sr), cells in KNOWN_TOP_OCTAVE.items()
    for m, what in cells.items()
    if not what.startswith(('+', '-'))
}


def _sweep_params():
    for timbre in TIMBRES:
        for sr in RATES:
            for midi in range(MIDI_LOW, MIDI_HIGH + 1):
                known = KNOWN_TOP_OCTAVE.get((timbre, sr), {}).get(midi)
                marks = []
                if known:
                    marks.append(pytest.mark.xfail(
                        strict=True, raises=AssertionError,
                        reason=f'measured {known}'))
                yield pytest.param(timbre, sr, midi, marks=marks,
                                   id=f'{timbre}-{sr}-m{midi}')


def _detect(midi: int, timbre: str, sr: int, a4: float = 440.0):
    return sim.detect_tone(_hz(midi, a4), kind=timbre, a4=a4, sr=sr, hops=HOPS)


def _assert_contract(midi: int, timbre: str, sr: int, a4: float = 440.0) -> None:
    res = _detect(midi, timbre, sr, a4)
    assert res.emitted, f'MIDI {midi}: no note confirmed'
    assert res.dominant_midi == midi, (
        f'MIDI {midi} read as {res.dominant_midi} (A4={a4:g}, {sr} Hz, {timbre})')
    err = res.cents_error()
    assert abs(err) <= _tolerance_ct(midi), (
        f'MIDI {midi}: {err:+.2f} ct exceeds +-{_tolerance_ct(midi):g} ct '
        f'(A4={a4:g}, {sr} Hz, {timbre})')
    assert res.median_cents == pytest.approx(err, abs=_ENGINE_CENTS_AGREEMENT), (
        'engine cents readout disagrees with the frequency it reports')


@pytest.mark.parametrize('timbre,sr,midi', list(_sweep_params()))
def test_every_note_reads_correctly(timbre: str, sr: int, midi: int) -> None:
    _assert_contract(midi, timbre, sr)


def _ceiling_params():
    for timbre in TIMBRES:
        for sr in RATES:
            for midi in range(_OCTAVE_START, MIDI_HIGH + 1):
                if (timbre, sr, midi) in _IDENTITY_FAILURES:
                    continue
                yield pytest.param(timbre, sr, midi, id=f'{timbre}-{sr}-m{midi}')


@pytest.mark.parametrize('timbre,sr,midi', list(_ceiling_params()))
def test_top_octave_error_stays_bounded(timbre: str, sr: int, midi: int) -> None:
    """The cells that miss the 3 ct contract still return the right note, and
    within a fixed ceiling, so the known inaccuracy cannot silently grow."""
    res = _detect(midi, timbre, sr)
    assert res.dominant_midi == midi
    assert abs(res.cents_error()) <= _TOP_OCTAVE_CEILING_CT


def test_known_failure_table_matches_the_sweep_range() -> None:
    """Typos in KNOWN_TOP_OCTAVE would create xfails that never run."""
    for (timbre, sr), cells in KNOWN_TOP_OCTAVE.items():
        assert timbre in TIMBRES
        assert sr in RATES
        assert all(_OCTAVE_START <= m <= MIDI_HIGH for m in cells)


# ---------------------------------------------------------------------------
# A4 boundaries. The reference is user-settable 430..450 Hz; the contract is
# the same, with the target frequency computed from that A4.
# ---------------------------------------------------------------------------
_A4_NOTES = (21, 33, 45, 69, 84, 93, 100, 108)

# (a4, sr, midi) -> measured behaviour.
KNOWN_A4_EDGES: dict[tuple[float, int, int], str] = {
    (430.0, 44100, 100): '+3.60 ct',
    (450.0, 44100, 100): '+4.02 ct',
    (450.0, 44100, 108): '+13.99 ct',
    (450.0, 48000, 108): '+10.92 ct',
}


def _a4_params():
    for a4 in (430.0, 450.0):
        for sr in RATES:
            for midi in _A4_NOTES:
                known = KNOWN_A4_EDGES.get((a4, sr, midi))
                marks = []
                if known:
                    marks.append(pytest.mark.xfail(
                        strict=True, raises=AssertionError,
                        reason=f'measured {known}'))
                yield pytest.param(a4, sr, midi, marks=marks,
                                   id=f'a4_{a4:g}-{sr}-m{midi}')


@pytest.mark.parametrize('a4,sr,midi', list(_a4_params()))
def test_a4_range_edges(a4: float, sr: int, midi: int) -> None:
    _assert_contract(midi, 'sax', sr, a4)

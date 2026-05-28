"""Acceptance net for the D3 mic-coordination POLICY (``sax_coordination``).

The policy is the pure decision logic behind "keep the tuner readout LIVE
while the drone sounds, suppress the leakage" (D3): vote-exclude the output's
own sounding MIDIs from the incumbent-pitch vote, and duck the output on
*suspicion* of mic bleed. It is a pure function of per-detection-frame inputs
plus an internal counter — no audio, no Qt, no sounddevice, no numpy — so the
whole contract is unit-testable deterministically here.

This is the "E" test in Treebeard's net, written against the CANONICAL
interface the Orchestrator locked (Decision dataclass + OutputCoordinator).
The consumer wiring (engine input callback → coordinator.update() → drone
set_gain) is correctly deferred to Sprint 3; the policy is built + locked now.

Calibration of the assertions:
  * The CRISP invariants are asserted EXACTLY — vote-exclude equals the
    sounding set, and the suspicion off-by-one fires at exactly
    ``suspicion_frames`` CONSECUTIVE matches (2 must NOT fire when the
    threshold is 3; 3 must).
  * The duck RAMP is asserted by robust INVARIANTS — bounds [duck_depth, 1.0],
    convergence toward duck_depth under sustained suspicion, and release
    toward 1.0 after leakage clears — NOT by an exact per-frame step count,
    which depends on the implementation's frame-cadence math. This locks the
    behaviour without over-coupling to one ramp formula.

TDD note: skips cleanly until ``sax_coordination`` lands (Sauron's lane), then
auto-activates. It must show PASSED — not skipped — for Sprint 1 done.
"""
from __future__ import annotations

import dataclasses

import pytest

sax_coordination = pytest.importorskip(
    'sax_coordination',
    reason="sax_coordination not landed yet (Sauron's D3 policy lane) — "
           "E test auto-activates when the module appears")

Decision = sax_coordination.Decision
OutputCoordinator = sax_coordination.OutputCoordinator


# ---------------------------------------------------------------------------
# Helpers — drive the coordinator with a frame sequence.
# ---------------------------------------------------------------------------
def _run(coord, frames):
    """Feed a list of (detected_midi, sounding_set) tuples; return the list of
    Decision results, one per frame."""
    return [coord.update(d, s) for (d, s) in frames]


# ---------------------------------------------------------------------------
# 1. Vote-exclude — excluded_midis is exactly the sounding output set.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sounding", [
    pytest.param(frozenset(), id="empty"),
    pytest.param(frozenset({60}), id="single"),
    pytest.param(frozenset({60, 67, 72}), id="multiple"),
])
def test_excluded_midis_equals_sounding_set(sounding):
    coord = OutputCoordinator()
    d = coord.update(detected_midi=None, sounding_output_midis=sounding)
    assert d.excluded_midis == sounding, (
        f"excluded_midis must equal the sounding output set so mic-bleed of "
        f"those notes can't win the incumbent vote; got {d.excluded_midis}")


def test_excluded_midis_independent_of_detected_note():
    """The exclusion set is the OUTPUT's sounding notes regardless of what the
    mic currently detects."""
    coord = OutputCoordinator()
    d = coord.update(detected_midi=64, sounding_output_midis=frozenset({60}))
    assert d.excluded_midis == frozenset({60})


# ---------------------------------------------------------------------------
# 2. Suspicion off-by-one — the headline lock (2 vs 3).
# ---------------------------------------------------------------------------
def test_suspicion_fires_at_exactly_threshold_consecutive_frames():
    """Default suspicion_frames=3: two consecutive matching frames must NOT
    fire suspicion; the third MUST. This locks the 2-vs-3 off-by-one."""
    coord = OutputCoordinator(suspicion_frames=3)
    s = frozenset({60})
    d1 = coord.update(60, s)
    d2 = coord.update(60, s)
    d3 = coord.update(60, s)
    assert d1.suspicious is False, "frame 1 of a streak must not be suspicious"
    assert d2.suspicious is False, (
        "frame 2 must NOT fire suspicion when the threshold is 3 (off-by-one)")
    assert d3.suspicious is True, "frame 3 (== threshold) MUST fire suspicion"


def test_suspicion_streak_resets_on_non_matching_frame():
    """A frame whose detected note is NOT in the sounding set breaks the
    consecutive streak; the counter restarts from zero."""
    coord = OutputCoordinator(suspicion_frames=3)
    s = frozenset({60})
    coord.update(60, s)        # 1
    coord.update(60, s)        # 2
    d_break = coord.update(64, s)   # break — 64 not sounding
    assert d_break.suspicious is False
    d1 = coord.update(60, s)   # streak restarts: 1
    d2 = coord.update(60, s)   # 2
    assert d1.suspicious is False
    assert d2.suspicious is False, (
        "after a break, two matches must not re-fire (streak restarted)")
    d3 = coord.update(60, s)   # 3 → fires again
    assert d3.suspicious is True


def test_detected_none_never_matches():
    """detected_midi=None (no confident pitch) is never 'in' the sounding set,
    so it can never accumulate suspicion even with output sounding."""
    coord = OutputCoordinator(suspicion_frames=3)
    s = frozenset({60})
    results = _run(coord, [(None, s)] * 10)
    assert all(not d.suspicious for d in results), (
        "None detection must never trigger suspicion")


@pytest.mark.parametrize("threshold", [1, 2, 5])
def test_custom_suspicion_threshold(threshold):
    coord = OutputCoordinator(suspicion_frames=threshold)
    s = frozenset({72})
    results = _run(coord, [(72, s)] * threshold)
    # The first (threshold-1) frames are not suspicious; the threshold-th is.
    assert all(not d.suspicious for d in results[:threshold - 1]), (
        f"frames before frame {threshold} must not be suspicious")
    assert results[threshold - 1].suspicious is True, (
        f"frame {threshold} must fire suspicion for threshold={threshold}")


# ---------------------------------------------------------------------------
# 3. Duck level — robust invariants (bounds, convergence, release).
# ---------------------------------------------------------------------------
def test_duck_level_open_before_any_suspicion():
    coord = OutputCoordinator(duck_depth=0.30)
    d = coord.update(64, frozenset({60}))   # detected != sounding, no suspicion
    assert d.duck_level == pytest.approx(1.0), (
        f"duck_level must be 1.0 (fully open) with no suspicion, got {d.duck_level}")


def test_duck_level_stays_within_bounds():
    """Across a long mixed run, duck_level must never leave [duck_depth, 1.0]."""
    depth = 0.30
    coord = OutputCoordinator(duck_depth=depth)
    s = frozenset({60})
    frames = ([(60, s)] * 50) + ([(64, s)] * 50) + ([(60, s)] * 50)
    for d in _run(coord, frames):
        assert depth - 1e-6 <= d.duck_level <= 1.0 + 1e-6, (
            f"duck_level {d.duck_level} left the [{depth}, 1.0] band")


def test_duck_converges_to_depth_under_sustained_suspicion():
    """Sustained mic-bleed (detected == sounding for many frames) must ramp
    the duck down to ~duck_depth. Frame count is large so convergence holds
    for any reasonable per-frame ramp step."""
    depth = 0.30
    coord = OutputCoordinator(duck_depth=depth)
    s = frozenset({60})
    results = _run(coord, [(60, s)] * 4000)
    assert results[-1].suspicious is True
    assert results[-1].duck_level == pytest.approx(depth, abs=0.02), (
        f"sustained suspicion must duck to ~{depth}, got {results[-1].duck_level}")


def test_duck_decreases_once_suspicious():
    """Ducking must actually reduce the level below fully-open under sustained
    suspicion (direction check, independent of step size)."""
    coord = OutputCoordinator(duck_depth=0.30)
    s = frozenset({60})
    results = _run(coord, [(60, s)] * 200)
    assert results[-1].duck_level < 1.0, (
        "sustained suspicion must reduce duck_level below 1.0")


def test_duck_releases_toward_open_after_leakage_clears():
    """After ducking, once leakage clears (detected no longer in the sounding
    set), the level ramps back toward 1.0 (past any post-duck confirm window).
    """
    depth = 0.30
    coord = OutputCoordinator(duck_depth=depth)
    s = frozenset({60})
    # First duck hard under sustained suspicion …
    ducked = _run(coord, [(60, s)] * 4000)
    assert ducked[-1].duck_level == pytest.approx(depth, abs=0.05)
    # … then clear the leakage for a long stretch and expect release to open.
    released = _run(coord, [(None, s)] * 4000)
    assert released[-1].duck_level == pytest.approx(1.0, abs=0.02), (
        f"duck must release toward 1.0 after leakage clears, "
        f"got {released[-1].duck_level}")


# ---------------------------------------------------------------------------
# 3b. Duck ramp RATE-INVARIANCE — closes the gap that let a real bug through.
#
# The duck ramp must take the same WALL-CLOCK time (~duck_ms) regardless of
# sample rate, because the engine holds the detection-hop DURATION constant
# (~46.44 ms) and scales hop SAMPLES with the rate. A correct policy therefore
# ramps over the SAME number of detection frames at every sample rate.
#
# This is the assertion the rate-AGNOSTIC bounds above could NOT make (by
# design — to leave the ramp formula free), which is exactly why a
# rate-dependent duck_ms defect (hop_ms derived from samples/sr, correct only
# at 44.1k → ramp 2/4/8 frames at 44.1k/96k/192k) slipped past them. This lock
# fails on that bug and passes once the cadence is rate-invariant — WITHOUT
# pinning the literal frame count (no formula coupling, ramp-shape freedom
# preserved).
# ---------------------------------------------------------------------------
def _frames_to_floor(coord, depth, *, midi=60, limit=100000):
    """Drive sustained suspicion (detected == sounding) and return the number
    of update() frames until duck_level first settles at the floor (depth)."""
    s = frozenset({midi})
    for i in range(1, limit + 1):
        d = coord.update(midi, s)
        if d.duck_level <= depth + 1e-9:
            return i
    raise AssertionError(f"duck never reached floor {depth} within {limit} frames")


@pytest.mark.parametrize("duck_ms", [80.0, 160.0, 40.0])
def test_duck_ramp_frame_count_is_rate_invariant(duck_ms):
    """For a given duck_ms, the frames-to-floor count must be IDENTICAL across
    sample rates — the detection cadence is rate-invariant, so the ramp is
    measured in a fixed number of ~46.44 ms frames no matter the rate. Parametrized
    over several duck_ms so it proves rate-independence generally, not a
    coincidence at the 80 ms default."""
    depth = 0.30
    counts = {
        sr: _frames_to_floor(
            OutputCoordinator(duck_depth=depth, duck_ms=duck_ms, samplerate=sr),
            depth)
        for sr in (44100, 96000, 192000)
    }
    assert len(set(counts.values())) == 1, (
        f"duck ramp frame-count must be identical across sample rates for "
        f"duck_ms={duck_ms} (cadence is rate-invariant); got {counts}. A "
        f"rate-dependent count means duck_ms is derived from samples/sr "
        f"(correct only at 44.1k) instead of the constant ~46.44 ms hop.")


# ---------------------------------------------------------------------------
# 4. Purity / independence / Decision shape.
# ---------------------------------------------------------------------------
def test_instances_are_independent():
    a = OutputCoordinator(suspicion_frames=3)
    b = OutputCoordinator(suspicion_frames=3)
    s = frozenset({60})
    a.update(60, s); a.update(60, s); a.update(60, s)   # a → suspicious
    # b is untouched: its first frame must not inherit a's streak.
    assert b.update(60, s).suspicious is False, (
        "coordinators must not share counter state")


def test_decision_is_frozen_dataclass_with_expected_fields():
    coord = OutputCoordinator()
    d = coord.update(60, frozenset({60}))
    assert dataclasses.is_dataclass(d)
    # Frozen: attribute assignment must raise.
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.duck_level = 0.5   # type: ignore[misc]
    # Fields present with the contracted types.
    assert isinstance(d.excluded_midis, frozenset)
    assert isinstance(d.duck_level, float)
    assert isinstance(d.suspicious, bool)

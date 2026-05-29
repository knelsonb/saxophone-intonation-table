"""Acceptance net for ``sax_metronome`` — the Sprint-2 metronome.

Covers the Orchestrator's Sprint-2 acceptance items against the four pure
test-seams Sauron built in (msg request 3342):
  * beat-index -> sample-offset math, incl. the integer-round-ONCE drift proof
  * accent-on-downbeat per time signature
  * tap-tempo: average of the last 4 inter-tap intervals, reset on >2s gap
  * click volume scaling (render-observable via the ClickSource MixerSource)
plus the MetronomeController state machine + on_state_changed + the mixer
scheduling integration.

Pure + numpy only (no Qt, no sounddevice, no audio hardware): the metronome
imports the real Mixer and drives it deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from sax_mixer import Mixer
import sax_metronome as M
from sax_metronome import (
    BPM_MIN, BPM_MAX, beat_onset_sample, bpm_from_tap_times, beats_per_bar,
    clamp_bpm, clamp_volume, ClickSource, MetronomeController,
)


# ===========================================================================
# 1. beat_onset_sample — the drift-zero core (seam 1).
# ===========================================================================
@pytest.mark.parametrize("idx,bpm,sr,expected", [
    pytest.param(0, 120, 48000, 0, id="beat0_is_zero"),
    pytest.param(1, 120, 48000, 24000, id="120bpm@48k"),
    pytest.param(1, 120, 44100, 22050, id="120bpm@44k1"),
    pytest.param(4, 120, 48000, 96000, id="four_beats"),
    pytest.param(2, 60, 44100, 88200, id="60bpm_two_beats"),
])
def test_beat_onset_sample_known_values(idx, bpm, sr, expected):
    assert beat_onset_sample(idx, bpm, sr) == expected


def test_beat_onset_sample_returns_int():
    assert isinstance(beat_onset_sample(3, 130, 44100), int)


def test_beat_onset_round_once_is_bounded_and_does_not_drift():
    """Round-ONCE keeps every beat within 0.5 sample of its true onset, for
    all beats, forever. A naive accumulator of rounded intervals drifts
    without bound — this test contrasts the two to lock the drift-killer.
    Uses 130 bpm @ 44.1k, whose interval (20353.84... samples) is NON-integer
    so the contrast is visible."""
    sr, bpm = 44100, 130
    interval = sr * 60.0 / bpm
    max_err_once = 0.0
    acc = 0
    step = round(interval)          # the naive "next += rounded interval"
    max_err_acc = 0.0
    for idx in range(20000):
        true = idx * interval
        max_err_once = max(max_err_once, abs(beat_onset_sample(idx, bpm, sr) - true))
        max_err_acc = max(max_err_acc, abs(acc - true))
        acc += step
    assert max_err_once <= 0.5, (
        f"round-once onset error must stay <=0.5 sample; got {max_err_once}")
    assert max_err_acc > 1.0, (
        "control: a rounded-interval accumulator must visibly drift "
        f"(proves the test can see drift) — got {max_err_acc}")


def test_beat_onset_5min_at_120bpm_under_1ms():
    """Acceptance: <1ms drift over 5 min @120bpm. At 44.1k, 1ms = 44.1 samples;
    round-once keeps it <=0.5 sample for every beat (here 120bpm*5min=600 beats)."""
    sr, bpm = 44100, 120
    interval = sr * 60.0 / bpm
    for idx in range(601):
        assert abs(beat_onset_sample(idx, bpm, sr) - idx * interval) <= 0.5


# ===========================================================================
# 2. clamp_bpm / clamp_volume.
# ===========================================================================
@pytest.mark.parametrize("raw,expected", [
    (BPM_MIN, BPM_MIN), (BPM_MAX, BPM_MAX),
    (BPM_MIN - 1, BPM_MIN), (0, BPM_MIN), (-100, BPM_MIN),
    (BPM_MAX + 1, BPM_MAX), (10000, BPM_MAX),
    (120.4, 120), (120.6, 121),
])
def test_clamp_bpm(raw, expected):
    assert clamp_bpm(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    (0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (-1.0, 0.0), (2.0, 1.0),
    (float("nan"), 0.0),
])
def test_clamp_volume(raw, expected):
    assert clamp_volume(raw) == pytest.approx(expected)


# ===========================================================================
# 3. Tap-tempo (seam 2): avg of last 4 intervals, reset on >2s gap.
# ===========================================================================
def test_bpm_from_tap_times_needs_two_taps():
    assert bpm_from_tap_times([]) is None
    assert bpm_from_tap_times([1.0]) is None


def test_bpm_from_tap_times_two_taps():
    # 0.5s interval -> 60/0.5 = 120 bpm.
    assert bpm_from_tap_times([0.0, 0.5]) == 120


def test_bpm_from_tap_times_averages_last_four_intervals():
    # Five even 0.5s taps (4 intervals) -> 120 bpm.
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert bpm_from_tap_times(times) == 120


def test_bpm_from_tap_times_only_last_four_intervals_count():
    # A long-ago fast pair then four 0.5s intervals; only the last 4 average.
    # 6 taps -> 5 intervals; the window keeps the last 4 (all 0.5) -> 120.
    times = [0.0, 0.1, 0.6, 1.1, 1.6, 2.1]
    assert bpm_from_tap_times(times) == 120


def test_bpm_from_tap_times_resets_after_long_gap():
    # Gap of 4s (>2s) starts a fresh run; only [5.0, 5.5] count -> 120.
    times = [0.0, 0.5, 1.0, 5.0, 5.5]
    assert bpm_from_tap_times(times) == 120


def test_bpm_from_tap_times_single_tap_after_gap_is_none():
    times = [0.0, 0.5, 5.0]   # last run is just [5.0] -> no interval
    assert bpm_from_tap_times(times) is None


@pytest.mark.parametrize("interval,expected", [
    pytest.param(0.1, BPM_MAX, id="too_fast_clamps_to_max"),   # 600 -> 300
    pytest.param(2.0, 30, id="slow_exactly_30"),               # 60/2 = 30
])
def test_bpm_from_tap_times_clamps(interval, expected):
    assert bpm_from_tap_times([0.0, interval]) == expected


# ===========================================================================
# 4. Accent-on-downbeat per time signature (seam 3).
# ===========================================================================
@pytest.mark.parametrize("time_sig,n", [
    ("2/4", 2), ("3/4", 3), ("4/4", 4), ("6/8", 6),
])
def test_beats_per_bar(time_sig, n):
    assert beats_per_bar(time_sig) == n


def test_beats_per_bar_garbage_defaults_to_four():
    assert beats_per_bar("nonsense") == 4


@pytest.mark.parametrize("time_sig,n", [
    ("2/4", 2), ("3/4", 3), ("4/4", 4), ("6/8", 6),
])
def test_is_accent_only_on_downbeat(time_sig, n):
    ctrl = MetronomeController(Mixer(max_block=4096), 48000, time_sig=time_sig)
    assert ctrl.beats_per_bar == n
    for beat in range(n * 3):   # three full bars
        expected = (beat % n == 0)
        assert ctrl.is_accent(beat) is expected, (
            f"{time_sig}: beat {beat} accent={ctrl.is_accent(beat)}, "
            f"expected {expected}")


# ===========================================================================
# 5. ClickSource (seam 4): volume scaling, accent vs beat, placement, protocol.
# ===========================================================================
def _click_peak(volume, *, accent=True, frames=8192, sr=48000):
    cs = ClickSource(sr, volume=volume)
    cs.trigger(accent=accent, offset=0)
    out = np.zeros(frames, dtype=np.float32)
    cs.render(out, frames, 0)
    return float(np.max(np.abs(out)))


def test_click_volume_scales_linearly():
    full = _click_peak(1.0)
    half = _click_peak(0.5)
    assert full > 0.0
    assert half == pytest.approx(0.5 * full, rel=0.02), (
        f"click peak must scale ~linearly with volume: full={full} half={half}")


def test_click_volume_zero_is_silent():
    assert _click_peak(0.0) == 0.0


def test_accent_click_louder_than_beat_click():
    assert _click_peak(1.0, accent=True) > _click_peak(1.0, accent=False)


def test_click_active_midi_is_none():
    assert ClickSource(48000).active_midi is None


def test_click_render_noop_when_not_triggered():
    cs = ClickSource(48000)
    out = np.full(4096, 3.0, dtype=np.float32)
    cs.render(out, 4096, 0)
    assert np.all(out == 3.0), "render must be a no-op (no alloc) between clicks"


def test_click_placed_at_trigger_offset():
    cs = ClickSource(48000, volume=1.0)
    cs.trigger(accent=True, offset=500)
    out = np.zeros(8192, dtype=np.float32)
    cs.render(out, 8192, 0)
    assert np.all(out[:500] == 0.0), "nothing before the trigger offset"
    assert np.any(np.abs(out[500:]) > 1e-3), "click energy starts at the offset"


def test_click_spanning_block_boundary_resumes():
    """A click longer than one block must resume mid-envelope on the next
    render (carry the _pos cursor), not restart or drop."""
    sr = 48000
    cs = ClickSource(sr, volume=1.0)
    cs.trigger(accent=True, offset=0)
    # Render in two short blocks shorter than the ~38ms envelope (~1824 samples).
    a = np.zeros(256, dtype=np.float32); cs.render(a, 256, 0)
    b = np.zeros(256, dtype=np.float32); cs.render(b, 256, 256)
    assert np.any(np.abs(a) > 1e-3), "first block has click energy"
    assert np.any(np.abs(b) > 1e-3), "click continues into the second block"


def test_click_source_integrates_with_real_mixer():
    m = Mixer(max_block=8192)
    cs = ClickSource(48000, volume=1.0)
    m.register(cs)
    cs.trigger(accent=True, offset=0)
    out = np.zeros(8192, dtype=np.float32)
    m.render(out, 8192)
    assert np.any(np.abs(out) > 1e-3), "click must sound through the mixer"
    assert m.sounding_midis() == frozenset(), "clicks are unpitched"


def test_click_rejects_bad_samplerate():
    with pytest.raises(ValueError):
        ClickSource(0)


# ===========================================================================
# 6. MetronomeController — state machine + on_state_changed + scheduling.
# ===========================================================================
def _ctrl(**kw):
    return MetronomeController(Mixer(max_block=4096), 48000, **kw)


def test_controller_defaults():
    c = _ctrl()
    assert c.bpm == 100
    assert c.time_sig == "4/4"
    assert c.volume == 1.0
    assert c.running is False and c.is_running() is False


def test_controller_clamps_constructor_args():
    c = _ctrl(bpm=9999, time_sig="7/8", volume=5.0)
    assert c.bpm == BPM_MAX
    assert c.time_sig == "4/4"   # invalid sig -> default
    assert c.volume == 1.0


def test_set_bpm_clamps_and_nudge():
    c = _ctrl()
    c.set_bpm(9999); assert c.bpm == BPM_MAX
    c.set_bpm(0); assert c.bpm == BPM_MIN
    c.set_bpm(120); c.nudge_bpm(5); assert c.bpm == 125
    c.nudge_bpm(-10000); assert c.bpm == BPM_MIN


def test_set_volume_propagates_to_click_source():
    c = _ctrl()
    c.set_volume(0.4)
    assert c.volume == pytest.approx(0.4)
    assert c.click_source.volume == pytest.approx(0.4)


def test_register_tap_injectable_clock_sets_bpm():
    c = _ctrl()
    assert c.register_tap(now=0.0) is None       # first tap of the run
    bpm = c.register_tap(now=0.5)                # 0.5s -> 120
    assert bpm == 120
    assert c.bpm == 120                          # tap also sets the controller


def test_on_state_changed_fires_on_changes_only():
    fired = []
    c = MetronomeController(Mixer(max_block=4096), 48000,
                            on_state_changed=lambda: fired.append(1))
    n0 = len(fired)
    c.set_bpm(150); assert len(fired) == n0 + 1          # changed -> fires
    c.set_bpm(150); assert len(fired) == n0 + 1          # no-op -> no fire
    c.set_time_sig("3/4"); assert len(fired) == n0 + 2
    c.set_volume(0.3); assert len(fired) == n0 + 3
    c.start(); assert len(fired) == n0 + 4
    c.stop(); assert len(fired) == n0 + 5


def test_start_registers_click_and_schedules_first_beat():
    m = Mixer(max_block=4096)
    c = MetronomeController(m, 48000, bpm=120)
    c.start()
    assert c.is_running() is True
    assert m.active_sources() == 1, "start must register the click source"
    assert m.pending_events() >= 1, "start must schedule the first beat"


def test_stop_reaps_click_after_drain():
    m = Mixer(max_block=4096)
    c = MetronomeController(m, 48000)
    c.start()
    c.stop()
    assert c.is_running() is False
    # stop() no longer hard-unregisters mid-click (that truncates the burst ->
    # pop). It requests stop; the Mixer reaps the click once any in-flight burst
    # has drained. With nothing sounding, the next render reaps it.
    out = np.zeros(4096, dtype=np.float32)
    m.render(out, 4096)
    assert m.active_sources() == 0, "the click must be reaped after stop drains"


def test_stop_does_not_truncate_inflight_click():
    """stop() while a click is sounding lets the burst finish (no truncation
    pop), then the Mixer reaps the source once it has decayed to zero."""
    m = Mixer(max_block=64)            # small blocks -> the burst spans many
    c = MetronomeController(m, 48000)
    c.start()
    c._click.trigger(accent=True, offset=0)   # fire a click now
    out = np.zeros(64, dtype=np.float32)
    m.render(out, 64)
    assert c._click._buf is not None, "a click should be mid-burst"
    c.stop()
    assert c._click._buf is not None, "stop must NOT truncate the in-flight click"
    tail = 0.0
    for _ in range(60):                # drain well past the 38 ms burst
        out[:] = 0.0
        m.render(out, 64)
        tail = float(np.max(np.abs(out)))
    assert c._click._buf is None, "the burst must finish on its own"
    assert m.active_sources() == 0, "then the Mixer reaps the click"
    assert tail < 1e-3, "the click tail must decay to ~0, not hard-cut"


def test_toggle_flips_running():
    c = _ctrl()
    c.toggle(); assert c.is_running() is True
    c.toggle(); assert c.is_running() is False


def test_start_is_idempotent():
    m = Mixer(max_block=4096)
    c = MetronomeController(m, 48000)
    c.start(); c.start()
    assert m.active_sources() == 1, "double start must not double-register"


# ===========================================================================
# 7. Device-switch chain — UNIT lock for the Option-A regression.
#
# An output-device reopen calls mixer.reset_clock(0), which clears the
# scheduled-event horizon. Because the metronome's rolling horizon only
# reschedules from a FIRED event, a reopen mid-play silently kills the beat
# chain (Gandalf/Sauron's catch). The fix is a GUI-layer stop()/start()
# bracket (Frodo's lane, locked by gui_smoke). These UNIT tests complement
# that: they run on the ALWAYS-RUNNABLE suite (no PyQt6 — gui_smoke skips
# there), locking the underlying controller/mixer mechanism the GUI fix
# rides on, so the regression is covered on every test run, not just CI.
# ===========================================================================
def test_reset_clock_silently_kills_running_metronome_chain():
    """Locks the BUG mechanism: after a reset_clock (the reopen signal), a
    running metronome's beat chain stays dead — no event fires to reschedule
    the next beat. This documents exactly why the device-switch reopen must be
    bracketed with stop()/start() at the orchestrating (GUI) layer."""
    block = 2048
    m = Mixer(max_block=block)
    c = MetronomeController(m, 48000, bpm=120)
    c.start()
    out = np.zeros(block, dtype=np.float32)
    m.render(out, block)                      # beat 0 fires, beat 1 scheduled
    assert m.pending_events() >= 1, "a next beat should be scheduled after the first fires"

    m.reset_clock(0)                          # mimic the output-device reopen
    assert m.pending_events() == 0, "reset_clock clears the scheduled horizon"
    for _ in range(5):
        m.render(out, block)
    assert m.pending_events() == 0, (
        "bug locked: the chain does NOT self-heal after reset_clock — it stays "
        "dead, which is why the GUI must bracket the reopen with stop()/start()")
    assert c.is_running() is True, (
        "the silent-failure signature: the controller still reports running "
        "while no beats are scheduled")


def test_stop_start_reestablishes_chain_after_reset():
    """Locks the Option-A FIX mechanism: bracketing the reopen with
    stop()/start() re-anchors and reschedules, reviving the beat chain."""
    block = 2048
    m = Mixer(max_block=block)
    c = MetronomeController(m, 48000, bpm=120)
    c.start()
    out = np.zeros(block, dtype=np.float32)
    m.render(out, block)
    m.reset_clock(0)
    assert m.pending_events() == 0           # chain dead

    c.stop()
    c.start()                                # the Option-A bracket
    assert m.pending_events() >= 1, "stop()+start() must reschedule the beat chain"
    out2 = np.zeros(block, dtype=np.float32)
    m.render(out2, block)
    assert np.any(np.abs(out2) > 1e-3), "clicks resume after the stop/start bracket"
    assert m.pending_events() >= 1, "the revived chain keeps rescheduling"
    assert m.dropped_events == 0, "revival must not drop events"


def test_running_metronome_produces_clicks_through_mixer():
    """Integration: a running metronome at 120bpm@48k, driven block-by-block
    through the real mixer, must actually sound clicks and never drop events."""
    sr, bpm, block = 48000, 120, 2048
    m = Mixer(max_block=block)
    c = MetronomeController(m, sr, bpm=bpm)
    c.start()
    out = np.zeros(block, dtype=np.float32)
    energy = 0.0
    # ~2 seconds of audio = ~4 beats at 120bpm.
    for _ in range(int(sr * 2 / block) + 1):
        m.render(out, block)
        energy += float(np.sum(np.abs(out)))
    assert energy > 0.0, "a running metronome must produce audible clicks"
    assert m.dropped_events == 0, "no scheduled beat may be dropped (drift-zero)"

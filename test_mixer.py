"""Acceptance net for ``sax_mixer`` — the pull-based output mixer.

This is the executable form of "Gandalf's contract" (D2, drift budget
zero) plus the no-allocation bar (plan §2). The mixer imports only numpy —
no sounddevice, no Qt — so every assertion here runs with no audio
hardware and no mock, exactly as the module docstring promises.

Test surface (Treebeard's A–G net, mapped to the contract):
  * clock: starts at 0, advances by exactly ``frames`` per render.
  * render: additive sum of sources, clamp to [-1, 1], silence when empty,
    over-long ``frames`` clamps to ``max_block``, 2-D output broadcast.
  * scheduling drift gate (the four locked edge cases):
      - normal: event fires at offset ``abs_sample - t0``
      - boundary: event at ``t0 + frames`` waits for the NEXT block
        (half-open interval; kills double-fire-on-boundary)
      - two-in-one-block: both fire, ascending sample order
      - past: dropped, ``dropped_events`` increments, never fires
      - blocksize-invariant: an event at sample N fires at N regardless
        of how the stream chunks its callbacks
  * source registry: register/unregister, idempotent register, snapshot.
  * coordination: ``sounding_midis`` collects pitched sources, drops None.
  * TestToneSource: sine energy, gain clamp, additive, zero-allocation.

Conventions match the existing suite: ``pytest.mark.parametrize`` with
named ``pytest.param`` ids for tables; plain ``def test_*`` for stateful
cases; every assert carries an expected-vs-got message.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from sax_mixer import Mixer, TestToneSource, GainGlide


# ---------------------------------------------------------------------------
# Test doubles — minimal MixerSource implementations.
# ---------------------------------------------------------------------------
class ConstSource:
    """Adds a constant value to every sample. Pre-allocates its buffer so it
    honours the no-alloc contract; used for additive / clamp / registry
    tests where a known, exact contribution is wanted."""

    def __init__(self, value: float, max_block: int, midi=None):
        self._buf = np.full(max_block, float(value), dtype=np.float32)
        self._midi = midi

    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        out[:frames] += self._buf[:frames]

    @property
    def active_midi(self):
        return self._midi


# ---------------------------------------------------------------------------
# 1. Clock
# ---------------------------------------------------------------------------
def test_clock_starts_at_zero():
    m = Mixer(max_block=128)
    assert m.clock == 0, f"fresh mixer clock should be 0, got {m.clock}"


def test_clock_advances_by_frames():
    m = Mixer(max_block=128)
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 64)
    assert m.clock == 64, f"clock should be 64 after one 64-frame render, got {m.clock}"
    m.render(out, 32)
    assert m.clock == 96, f"clock should accumulate to 96, got {m.clock}"


def test_render_zero_or_negative_frames_is_noop():
    m = Mixer(max_block=128)
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 0)
    m.render(out, -5)
    assert m.clock == 0, f"non-positive frames must not advance clock, got {m.clock}"


def test_reset_clock_sets_value_clears_events_keeps_dropped():
    m = Mixer(max_block=128)
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 100)                       # clock -> 100
    assert m.schedule(50, lambda off: None) is False  # past -> dropped
    assert m.dropped_events == 1
    m.schedule(150, lambda off: None)        # pending
    assert m.pending_events() == 1
    m.reset_clock(0)
    assert m.clock == 0, "reset_clock must set the clock"
    assert m.pending_events() == 0, "reset_clock must clear pending events"
    assert m.dropped_events == 1, (
        "dropped_events is a lifetime diagnostic — reset_clock must NOT zero it")


# ---------------------------------------------------------------------------
# 2. Render — additive sum + clamp
# ---------------------------------------------------------------------------
def test_render_empty_mix_is_silence():
    m = Mixer(max_block=64)
    out = np.full(64, 9.0, dtype=np.float32)
    m.render(out, 64)
    assert np.all(out == 0.0), "no sources registered must yield pure silence"


def test_render_sums_sources_additively():
    m = Mixer(max_block=64)
    m.register(ConstSource(0.3, 64))
    m.register(ConstSource(0.4, 64))
    out = np.zeros(64, dtype=np.float32)
    m.render(out, 64)
    assert np.allclose(out, 0.7, atol=1e-6), (
        f"two sources 0.3 + 0.4 should sum to 0.7, got {out[0]}")


@pytest.mark.parametrize("values,expected", [
    pytest.param((0.8, 0.8), 1.0, id="positive_overflow_clamps_to_+1"),
    pytest.param((-0.8, -0.8), -1.0, id="negative_overflow_clamps_to_-1"),
    pytest.param((0.5, 0.25), 0.75, id="within_rails_unclamped"),
    pytest.param((5.0, 5.0), 1.0, id="far_positive_clamps"),
])
def test_render_clamps_to_rails(values, expected):
    m = Mixer(max_block=32)
    for v in values:
        m.register(ConstSource(v, 32))
    out = np.zeros(32, dtype=np.float32)
    m.render(out, 32)
    assert np.allclose(out, expected, atol=1e-6), (
        f"sum{values} should clamp/resolve to {expected}, got {out[0]}")


def test_render_overlong_frames_clamps_to_max_block():
    m = Mixer(max_block=16)
    m.register(ConstSource(0.5, 16))
    out = np.zeros(64, dtype=np.float32)
    m.render(out, 64)                       # asks for 64, max_block is 16
    assert m.clock == 16, (
        f"render must clamp frames to max_block (16), clock got {m.clock}")
    assert np.allclose(out[:16], 0.5, atol=1e-6), "first 16 samples rendered"
    assert np.all(out[16:] == 0.0), "samples beyond max_block left untouched"


def test_render_2d_output_broadcasts_mono_to_channels():
    m = Mixer(max_block=32, channels=2)
    m.register(ConstSource(0.25, 32))
    out = np.zeros((32, 2), dtype=np.float32)
    m.render(out, 32)
    assert np.allclose(out, 0.25, atol=1e-6), (
        "mono mix must broadcast identically across both channels")


# ---------------------------------------------------------------------------
# 3. Scheduling — the drift gate (D2). event(offset) is the observable.
# ---------------------------------------------------------------------------
def test_schedule_future_event_fires_at_correct_offset():
    m = Mixer(max_block=128)
    fired: list[int] = []
    accepted = m.schedule(10, lambda off: fired.append(off))
    assert accepted is True, "a future event must be accepted"
    out = np.zeros(128, dtype=np.float32)
    t0 = m.clock                            # 0
    m.render(out, 64)
    assert fired == [10 - t0], (
        f"event scheduled at abs 10 should fire at offset {10 - t0}, got {fired}")


def test_schedule_past_event_drops_and_counts():
    """The headline late-event assertion: a past event is DROPPED, the
    dropped_events canary increments, and the event NEVER fires."""
    m = Mixer(max_block=128)
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 100)                      # clock -> 100
    fired: list[int] = []
    accepted = m.schedule(99, lambda off: fired.append(off))  # 99 < 100
    assert accepted is False, "a past event must be rejected by schedule()"
    assert m.dropped_events == 1, (
        f"dropped_events must increment on a past schedule, got {m.dropped_events}")
    m.render(out, 64)
    assert fired == [], "a dropped event must never fire (late != played-now)"


def test_boundary_event_belongs_to_next_block():
    """Half-open [t0, t0+frames): an event at exactly t0+frames waits for the
    next block and fires there at offset 0 — no double-fire on the seam."""
    m = Mixer(max_block=128)
    fired: list[int] = []
    m.schedule(64, lambda off: fired.append(off))
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 64)                       # covers [0, 64) — boundary excluded
    assert fired == [], (
        "event at t0+frames must NOT fire in the block that ends at it")
    assert m.pending_events() == 1, "boundary event should still be pending"
    m.render(out, 64)                       # covers [64, 128)
    assert fired == [0], (
        f"boundary event must fire at offset 0 of the next block, got {fired}")


def test_two_events_same_block_fire_in_ascending_sample_order():
    m = Mixer(max_block=128)
    log: list[tuple[str, int]] = []
    # Schedule out of order; fire order must be by sample, not schedule order.
    m.schedule(40, lambda off: log.append(("late", off)))
    m.schedule(10, lambda off: log.append(("early", off)))
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 64)
    assert log == [("early", 10), ("late", 40)], (
        f"two same-block events must fire ascending by sample, got {log}")


def test_same_sample_events_fire_in_schedule_order():
    """Tiebreaker: two events at the SAME abs_sample fire in the order they
    were scheduled (the seq tiebreaker), both at the same offset."""
    m = Mixer(max_block=128)
    log: list[str] = []
    m.schedule(20, lambda off: log.append("first"))
    m.schedule(20, lambda off: log.append("second"))
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 64)
    assert log == ["first", "second"], (
        f"same-sample events must fire in schedule order, got {log}")


@pytest.mark.parametrize("blocksize", [32, 50, 64, 128, 256])
def test_event_landing_is_blocksize_invariant(blocksize):
    """An event at absolute sample N fires at N no matter how the stream
    chunks its callbacks. Reconstruct the absolute landing sample as
    t0_at_fire + offset and assert it equals N for every blocksize."""
    target = 200
    m = Mixer(max_block=256)
    landed_offset: list[int] = []
    m.schedule(target, lambda off: landed_offset.append(off))
    out = np.zeros(blocksize, dtype=np.float32)
    fired_abs = None
    # Render blocks until the event fires (or we pass it).
    while m.clock <= target:
        t0 = m.clock
        before = len(landed_offset)
        m.render(out, blocksize)
        if len(landed_offset) > before:
            fired_abs = t0 + landed_offset[-1]
            break
    assert fired_abs == target, (
        f"blocksize {blocksize}: event scheduled at {target} fired at "
        f"absolute sample {fired_abs}")


def test_misbehaving_event_does_not_crash_render():
    """A handler that raises must be swallowed — the audio callback survives —
    and a well-behaved event scheduled alongside it still fires."""
    m = Mixer(max_block=128)
    fired: list[int] = []

    def boom(off):
        raise RuntimeError("event handler blew up")

    m.schedule(5, boom)
    m.schedule(20, lambda off: fired.append(off))
    out = np.zeros(128, dtype=np.float32)
    m.render(out, 64)                       # must not raise
    assert fired == [20], (
        f"a raising event must not prevent later events firing, got {fired}")


# ---------------------------------------------------------------------------
# 4. Source registry — lock-snapshot semantics
# ---------------------------------------------------------------------------
def test_register_returns_handle_and_counts():
    m = Mixer(max_block=32)
    src = ConstSource(0.1, 32)
    handle = m.register(src)
    assert handle is src, "register should return the source as its handle"
    assert m.active_sources() == 1


def test_register_is_idempotent_and_sums_once():
    m = Mixer(max_block=32)
    src = ConstSource(0.5, 32)
    m.register(src)
    m.register(src)                          # second register is a no-op
    assert m.active_sources() == 1, "re-registering must not duplicate the source"
    out = np.zeros(32, dtype=np.float32)
    m.render(out, 32)
    assert np.allclose(out, 0.5, atol=1e-6), (
        f"idempotent register must not double-sum, got {out[0]}")


def test_unregister_removes_source():
    m = Mixer(max_block=32)
    src = ConstSource(0.5, 32)
    m.register(src)
    m.unregister(src)
    assert m.active_sources() == 0
    out = np.full(32, 3.0, dtype=np.float32)
    m.render(out, 32)
    assert np.all(out == 0.0), "after unregister the source must not contribute"


def test_unregister_absent_source_is_silent():
    m = Mixer(max_block=32)
    # Never registered — must not raise (guards double-stop).
    m.unregister(ConstSource(0.5, 32))
    assert m.active_sources() == 0


# ---------------------------------------------------------------------------
# 5. Coordination surface — sounding_midis (D3 vote-exclude feed)
# ---------------------------------------------------------------------------
def test_sounding_midis_collects_pitched_excludes_unpitched():
    m = Mixer(max_block=64)
    m.register(ConstSource(0.1, 64, midi=60))   # pitched
    m.register(ConstSource(0.1, 64, midi=67))   # pitched
    m.register(ConstSource(0.1, 64, midi=None))  # unpitched (a click)
    assert m.sounding_midis() == frozenset({60, 67}), (
        f"sounding_midis must collect pitched notes only, got {m.sounding_midis()}")


def test_sounding_midis_empty_when_nothing_pitched():
    m = Mixer(max_block=64)
    m.register(ConstSource(0.1, 64, midi=None))
    assert m.sounding_midis() == frozenset(), "no pitched source -> empty set"


# ---------------------------------------------------------------------------
# 6. TestToneSource — the reference source
# ---------------------------------------------------------------------------
def test_testtone_produces_bounded_sine_energy():
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=512, gain=0.5)
    out = np.zeros(512, dtype=np.float32)
    src.render(out, 512, t0=0)
    assert np.any(np.abs(out) > 1e-3), "test tone should produce audible energy"
    assert np.max(np.abs(out)) <= 0.5 + 1e-6, (
        f"a gain-0.5 sine must stay within ±0.5, peak {np.max(np.abs(out))}")


def test_testtone_render_is_additive():
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=128, gain=0.3)
    baseline = np.full(128, 0.25, dtype=np.float32)
    out = baseline.copy()
    src.render(out, 128, t0=0)
    assert np.all(out >= baseline - 1e-6) or np.any(out != baseline), (
        "render must ADD to the buffer, not overwrite the existing content")
    # Stronger: the difference equals a fresh render into zeros.
    fresh = np.zeros(128, dtype=np.float32)
    src2 = TestToneSource(freq=440.0, samplerate=48000, max_block=128, gain=0.3)
    src2.render(fresh, 128, t0=0)
    assert np.allclose(out, baseline + fresh, atol=1e-6), (
        "additive render must equal baseline + the tone")


def test_testtone_gain_zero_is_silent():
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=64, gain=0.0)
    out = np.zeros(64, dtype=np.float32)
    src.render(out, 64, t0=0)
    assert np.all(out == 0.0), "gain 0 must emit silence"


@pytest.mark.parametrize("raw,expected", [
    pytest.param(0.5, 0.5, id="mid"),
    pytest.param(-1.0, 0.0, id="below_zero_clamped"),
    pytest.param(2.0, 1.0, id="above_one_clamped"),
    pytest.param(0.0, 0.0, id="zero"),
    pytest.param(1.0, 1.0, id="one"),
])
def test_testtone_set_gain_clamps_unit_interval(raw, expected):
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=64)
    src.set_gain(raw)
    assert src.gain == expected, (
        f"set_gain({raw}) should clamp to {expected}, got {src.gain}")


def test_testtone_active_midi_passthrough():
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=64, midi=69)
    assert src.active_midi == 69
    src2 = TestToneSource(freq=440.0, samplerate=48000, max_block=64)
    assert src2.active_midi is None, "default test tone is unpitched (None)"


# ---------------------------------------------------------------------------
# 6b. Click-free envelope (opt-in) + Mixer auto-reap of finished sources
# ---------------------------------------------------------------------------
def test_testtone_default_has_no_envelope():
    """The default tone is the byte-identical steady reference: full amplitude
    from the first sample, and never 'finished'."""
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=256, gain=0.5)
    assert src.finished is False
    out = np.zeros(256, dtype=np.float32)
    src.render(out, 256, t0=0)
    # No attack ramp: the sine swings within the first few samples, not held
    # near zero by an envelope.
    assert np.max(np.abs(out[:32])) > 1e-3


def test_testtone_enveloped_attacks_from_silence():
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=256, gain=0.5,
                         attack_ms=8.0, release_ms=60.0)
    out = np.zeros(256, dtype=np.float32)
    src.render(out, 256, t0=0)
    assert abs(float(out[0])) < 1e-4, "enveloped tone must start at ~0 (no click)"
    assert np.max(np.abs(out)) > 0.1, "should ramp up to audible within the block"


def test_testtone_release_fades_to_zero_and_finishes():
    sr, mb = 48000, 256
    src = TestToneSource(freq=440.0, samplerate=sr, max_block=mb, gain=0.5,
                         attack_ms=8.0, release_ms=40.0)
    out = np.zeros(mb, dtype=np.float32)
    for _ in range(8):                       # reach steady full gain
        out[:] = 0.0
        src.render(out, mb, t0=0)
    src.release()
    last = out
    for _ in range(40):                      # render past the 40 ms release
        out = np.zeros(mb, dtype=np.float32)
        src.render(out, mb, t0=0)
        last = out
    assert src.finished is True, "a fully-released tone must report finished"
    assert np.max(np.abs(last)) < 1e-4, "release must fade to ~0, not hard-cut"


def test_testtone_nonenveloped_release_marks_finished():
    """release() on a plain (non-enveloped) tone marks it finished at once, so
    the Mixer reaps it through the same path — no behavioural special-case."""
    src = TestToneSource(freq=440.0, samplerate=48000, max_block=64, gain=0.2)
    assert src.finished is False
    src.release()
    assert src.finished is True


def test_mixer_reaps_finished_sources():
    """A source reporting finished=True is dropped by the Mixer after render —
    the contract (MixerSource.finished) that lets a released tone/pad
    self-retire with a click-free fade instead of a hard unregister."""
    sr, mb = 48000, 128
    m = Mixer(max_block=mb)
    keep = TestToneSource(freq=440.0, samplerate=sr, max_block=mb, gain=0.2)
    transient = TestToneSource(freq=660.0, samplerate=sr, max_block=mb,
                               gain=0.2, attack_ms=4.0, release_ms=20.0)
    m.register(keep)
    m.register(transient)
    out = np.zeros(mb, dtype=np.float32)
    m.render(out, mb)
    assert m.active_sources() == 2
    transient.release()
    for _ in range(40):                      # render past the release
        out[:] = 0.0
        m.render(out, mb)
    assert m.active_sources() == 1, "the faded source must be reaped"
    assert keep in m._sources, "the steady (unfinished) source must survive"


def test_testtone_released_then_muted_still_finishes():
    """A released enveloped tone DUCKED to gain 0 mid-fade must still complete
    its fade and finish. The envelope is time-based, so it must advance even
    while muted — otherwise the release freezes, never reaches zero, finished
    never goes True, and the Mixer never reaps the source (it lingers forever).
    Regression for the gain==0 early-return skipping the envelope (uruk-hai)."""
    sr, mb = 48000, 256
    src = TestToneSource(freq=440.0, samplerate=sr, max_block=mb, gain=0.5,
                         attack_ms=8.0, release_ms=40.0)
    out = np.zeros(mb, dtype=np.float32)
    for _ in range(8):                       # reach steady full gain
        out[:] = 0.0
        src.render(out, mb, t0=0)
    src.release()                            # begin the release fade ...
    src.set_gain(0.0)                        # ... then duck to silence mid-fade
    for _ in range(40):                      # render past the 40 ms release
        out[:] = 0.0
        src.render(out, mb, t0=0)
    assert src.finished is True, (
        "a released tone muted mid-fade must still finish, not freeze forever")


# ---------------------------------------------------------------------------
# 7. Constructor validation
# ---------------------------------------------------------------------------
def test_mixer_rejects_nonpositive_max_block():
    with pytest.raises(ValueError):
        Mixer(max_block=0)


@pytest.mark.parametrize("sr,mb", [
    pytest.param(0, 64, id="zero_samplerate"),
    pytest.param(-48000, 64, id="negative_samplerate"),
    pytest.param(48000, 0, id="zero_max_block"),
])
def test_testtone_rejects_bad_args(sr, mb):
    with pytest.raises(ValueError):
        TestToneSource(freq=440.0, samplerate=sr, max_block=mb)


# ---------------------------------------------------------------------------
# 8. THE NO-ALLOC GATE (plan §2 — the stricter bar output is held to).
#
# Steady-state render (sources summing, no events due) must not allocate
# data buffers on the hot path. Calibration matters here: numpy SLICING
# (``self._acc[:frames]``, ``out[:frames]``) creates small transient view
# OBJECTS that tracemalloc counts even for a perfectly-conforming source —
# so an absolute byte threshold is the wrong instrument. Two robust gates
# instead:
#   (1) no RETAINED growth across many renders  -> catches real leaks
#   (2) the source under test allocates no MORE than a conforming,
#       fully-preallocated baseline source       -> catches per-block temps
# (2) isolates a source's own allocation from the render path's intrinsic
# view churn, and is stable across numpy/Python versions.
# ---------------------------------------------------------------------------
def _measure_render_alloc(source, *, frames=512, max_block=512, iters=200):
    m = Mixer(max_block=max_block)
    m.register(source)
    out = np.zeros(frames, dtype=np.float32)
    # Warm up: flush any one-time allocations (first-touch, lazy buffers).
    for _ in range(5):
        m.render(out, frames)
    tracemalloc.start()
    base_cur, _ = tracemalloc.get_traced_memory()
    for _ in range(iters):
        m.render(out, frames)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - base_cur, cur - base_cur


@pytest.mark.parametrize("make_source", [
    pytest.param(lambda: ConstSource(0.4, 8192), id="const_source"),
    pytest.param(
        lambda: TestToneSource(freq=440.0, samplerate=48000,
                               max_block=8192, gain=0.2),
        id="testtone_source"),
])
def test_render_does_not_leak(make_source):
    """No source may cause RETAINED memory growth across many renders — that
    would be a genuine leak (accumulating references), distinct from the
    transient view churn that frees each block."""
    _peak, retained = _measure_render_alloc(make_source(), frames=8192,
                                            max_block=8192)
    assert retained < 1024, (
        f"steady-state render retained {retained} bytes across 200 iters — "
        f"a real leak (transient per-block allocations should free, not "
        f"accumulate)")


@pytest.mark.parametrize("make_source", [
    pytest.param(lambda mb: ConstSource(0.4, mb), id="const_source"),
    pytest.param(
        lambda mb: TestToneSource(freq=440.0, samplerate=48000,
                                  max_block=mb, gain=0.2),
        id="testtone_source"),
])
def test_render_has_no_frames_proportional_allocation(make_source):
    """The no-alloc gate, calibrated to the actual invariant: NO allocation
    PROPORTIONAL TO BLOCK SIZE on the render hot path.

    Why non-scaling and not an absolute floor: numpy slicing
    (``self._acc[:frames]``, ``out[:frames]``, a source's ``buf[:frames]``)
    boxes small transient VIEW OBJECTS that tracemalloc's peak counts even
    for a perfectly-conforming, fully-preallocated source — a fixed ~300-400 B
    that does NOT vary with frames. A real per-block data temporary (the bug
    this gate hunts) allocates ~frames*itemsize and so scales linearly. So we
    measure peak at a small and a large blocksize and assert the difference
    is flat. (Credit: Orchestrator's frames-sweep showed the constant is
    irreducible numpy view-boxing, not source allocation.)

    KNOWN LIMITATION (logged, per "no silent caps"): tracemalloc tracks
    Python-level allocations; numpy's raw data buffers are allocated outside
    the Python allocator and are INVISIBLE here. A source that does an
    internal dtype cast (e.g. ``float64 -> float32 astype`` per block) still
    allocates a real data buffer this gate cannot see. That class of issue is
    caught by code review, not this test — see the note in the channel re:
    TestToneSource.render's astype at sax_mixer.py ~404. This gate enforces
    the detectable invariant; it does not certify literal zero malloc.
    """
    small_peak, _ = _measure_render_alloc(make_source(8192), frames=512,
                                          max_block=8192)
    large_peak, _ = _measure_render_alloc(make_source(8192), frames=8192,
                                          max_block=8192)
    growth = large_peak - small_peak
    assert growth < 256, (
        f"render allocation scales with block size: peak grew {growth} B "
        f"going from frames=512 to frames=8192 — a per-block data temporary "
        f"proportional to frames (the contract forbids hot-path allocation)")


def test_whole_buffer_fast_path_reduces_allocation():
    """Steady-state cleanliness (Gandalf's intent), asserted ROBUSTLY.

    An absolute floor on the steady-state peak is environment-fragile: it
    measures ~252 B here (numpy ufunc-dispatch object boxing) — only a few
    bytes under a 256 B line, which a different numpy/Python build can cross.
    Instead assert the INVARIANT the whole-buffer fast path exists to provide:
    rendering a FULL block (``frames == max_block`` — the whole-buffer path
    that creates no slice view) allocates materially LESS than a PARTIAL block
    (``frames < max_block``, which must slice ``_work[:n]`` / ``out[:n]``).
    This proves the fast path engages, and is stable across numpy versions
    because the irreducible per-call constant cancels in the difference.
    """
    mb = 512
    full_peak, _ = _measure_render_alloc(
        TestToneSource(freq=440.0, samplerate=48000, max_block=mb, gain=0.2),
        frames=mb, max_block=mb)
    partial_peak, _ = _measure_render_alloc(
        TestToneSource(freq=440.0, samplerate=48000, max_block=mb, gain=0.2),
        frames=mb - 1, max_block=mb)
    assert full_peak < partial_peak, (
        f"whole-buffer fast path should allocate less than the slicing "
        f"partial-block path: full={full_peak}B vs partial={partial_peak}B "
        f"(the frames==max_block steady state must avoid slice-view boxing)")


def test_testtone_steady_state_allocation_is_bounded():
    """Strict steady-state floor (Gandalf/Orchestrator consensus) at a
    cross-environment-ROBUST threshold.

    Gandalf's whole-buffer fast path makes the frames==max_block render
    genuinely allocation-light. Measured here: ~252 B peak. The originally
    proposed 256 B line passes — but by only ~4 B, which is one numpy/Python
    build away from flaking. So the bound is 1024 B: still far below the
    SMALLEST realistic per-block data temporary (a 256-sample float32 block is
    already 1 KB), so a genuine hot-path allocation still fails this gate,
    while numpy's irreducible per-call ufunc/view boxing passes with margin.
    The non-scaling and fast-path tests above give the PRECISE guarantees;
    this absolute floor is the coarse backstop the contract asked for.
    """
    peak, _ = _measure_render_alloc(
        TestToneSource(freq=440.0, samplerate=48000, max_block=512, gain=0.2),
        frames=512, max_block=512)
    assert peak < 1024, (
        f"steady-state (frames==max_block) render peaked {peak} B — a real "
        f"per-block data temporary would be >=1 KB. The reference source must "
        f"stay allocation-light on the hot path (plan §2).")


# ===========================================================================
# 9. GainGlide — the D3 duck de-zipper (mechanism half of the duck split).
#
# Glides an applied gain toward a coarsely-set target SAMPLE-BY-SAMPLE,
# slew-limited, so a coarse per-hop duck step (1.0 -> 0.30 in ~2 frames) is
# smoothed to a click-free ramp. Pure numpy — runs on every suite. (Lives in
# sax_mixer; embedded by the Sprint-3 drone source.)
# ===========================================================================
def _max_slew_step(samplerate, glide_ms):
    return 1.0 / max(1.0, samplerate * glide_ms / 1000.0)


def test_gainglide_rejects_bad_args():
    with pytest.raises(ValueError):
        GainGlide(0, 48000)
    with pytest.raises(ValueError):
        GainGlide(512, 0)


def test_gainglide_set_target_clamps_unit_interval():
    g = GainGlide(512, 48000)
    g.set_target(2.0); assert g.target == 1.0
    g.set_target(-1.0); assert g.target == 0.0
    g.set_target(0.3); assert g.target == pytest.approx(0.3)


def test_gainglide_downward_glide_respects_slew_cap():
    """A coarse 1.0 -> 0.30 target jump must produce a per-sample gain whose
    adjacent steps never exceed the slew cap (~0.004167/sample at 5ms@48k),
    measured ACROSS block boundaries too — that's the no-zipper guarantee."""
    sr, glide_ms, block = 48000, 5.0, 2048
    cap = _max_slew_step(sr, glide_ms)
    g = GainGlide(block, sr, glide_ms=glide_ms, initial=1.0)
    g.set_target(0.30)
    env = [1.0]   # seed with the pre-glide gain so the first step is checked
    for _ in range(40):
        b = np.ones(block, dtype=np.float32)   # ones -> b becomes the gain envelope
        g.apply(b, block)
        env.extend(b.tolist())
        if g.gain == pytest.approx(0.30):
            break
    diffs = np.abs(np.diff(np.array(env, dtype=np.float64)))
    assert diffs.max() <= cap + 1e-6, (
        f"max per-sample gain step {diffs.max()} exceeds slew cap {cap} (zipper)")
    assert g.gain == pytest.approx(0.30), "glide must converge exactly to target"


def test_gainglide_upward_glide_converges():
    sr, block = 48000, 2048
    g = GainGlide(block, sr, glide_ms=5.0, initial=0.30)
    g.set_target(1.0)
    for _ in range(40):
        b = np.ones(block, dtype=np.float32)
        g.apply(b, block)
        if g.gain == pytest.approx(1.0):
            break
    assert g.gain == pytest.approx(1.0)


def test_gainglide_snaps_when_within_one_step():
    """When the target is within one slew step, snap to it (no overshoot, no
    lingering sub-step drift)."""
    g = GainGlide(512, 48000, glide_ms=5.0, initial=0.3010)
    g.set_target(0.30)
    b = np.ones(512, dtype=np.float32)
    g.apply(b, 512)
    assert g.gain == pytest.approx(0.30)
    assert np.allclose(b, 0.30, atol=1e-6), "within-step target fills at target"


def test_gainglide_apply_scales_block_in_place_and_returns_gain():
    g = GainGlide(512, 48000, initial=1.0)   # already at unity
    b = np.full(512, 0.5, dtype=np.float32)
    ret = g.apply(b, 512)
    assert ret == pytest.approx(g.gain)
    assert np.allclose(b, 0.5, atol=1e-6), "unity gain leaves the block unchanged"


def test_gainglide_apply_has_no_frames_proportional_allocation():
    """Steady-state apply (gain == target) must not allocate proportionally to
    block size. Same non-scaling instrument as the mixer no-alloc gate:
    intrinsic numpy view-boxing is constant; a real per-block temp scales."""
    def peak(frames, mb=8192, iters=200):
        g = GainGlide(mb, 48000, initial=0.5)
        g.set_target(0.5)                      # already at target -> steady
        b = np.ones(frames, dtype=np.float32)
        for _ in range(5):
            g.apply(b, frames)
        tracemalloc.start()
        base, _ = tracemalloc.get_traced_memory()
        for _ in range(iters):
            g.apply(b, frames)
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pk - base
    growth = peak(8192) - peak(512)
    assert growth < 256, (
        f"GainGlide.apply allocation scales with block size (grew {growth} B) "
        f"— the per-sample envelope must stay in the preallocated buffer")

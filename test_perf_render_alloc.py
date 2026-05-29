"""PERF-RENDER-ALLOC (v1.1 hardening ledger): every ``MixerSource.render()`` is
allocation-free on the steady-state hot path — it adds no data buffer whose size
grows with the block.

This generalises the no-alloc gate that ``test_mixer.py`` applies to
TestToneSource/ConstSource and ``test_deck.py`` applies to DeckPlaybackSource,
extending it to the three sources that lacked one: PitchPipeSource, the metronome
ClickSource (active burst), and DroneSource's numpy downmix/duck path. The drone's
native synth fill is exempt by ruling (it writes a single preallocated buffer); a
fake synth here isolates the numpy portion so the gate needs no SF2 / PortAudio and
runs on system-python.

Technique (numpy-aware). tracemalloc cannot see numpy's own data-buffer allocator,
so an absolute byte floor is the wrong instrument: numpy slicing also boxes a small
CONSTANT view object per call (~300-400 B, flat in the block size). We therefore
measure peak alloc at a small and a large block and assert the DIFFERENCE is flat.
A real per-block data temporary scales ~frames*itemsize (≥ ~30 KB across this
sweep); a conforming preallocated source stays constant. A manual read of each
render() (cited) confirms none does an invisible numpy temporary (``astype`` /
``np.abs`` / arithmetic without ``out=``) that scaling alone could miss:

    TestToneSource    sax_mixer.py:508       np.multiply/np.sin out=, *=, +=, float32
    PitchPipeSource   sax_pitch_pipes.py:128 same in-place sine pattern
    ClickSource       sax_metronome.py:245   out[s:s+n] += buf[p:p+n]
    DroneSource       sax_drone.py:359       in-place downmix; max()/min() (no np.abs)
    DeckPlaybackSource sax_deck.py:245       preallocated resample scratch — gated in
                                             test_deck.py::test_playback_source_render_is_non_scaling_alloc
"""
from __future__ import annotations

import sys
import tracemalloc
import types

import numpy as np
import pytest

from sax_mixer import TestToneSource
from sax_metronome import ClickSource
import sax_drone

pp = pytest.importorskip("sax_pitch_pipes", reason="pitch pipes module")
PitchPipeSource = pp.PitchPipeSource

_SR = 48000
_MB = 8192


# --- source factories (data, not code) -------------------------------------
# Each maps a max_block -> a source rendering in (or warmed into) its steady
# state. Kept tiny and uniform so the measurement mechanism below is the only
# moving part.

class _ActiveClick:
    """Re-triggers a metronome click every render so the active
    ``out[:n] += buf[:n]`` path is exercised — a real click drains in well under
    one block, so a plain ClickSource would measure only the no-op steady state
    (already covered by test_metronome.py)."""

    def __init__(self) -> None:
        self._c = ClickSource(_SR)

    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        self._c.trigger(accent=True)
        self._c.render(out, frames, t0)


def _make_drone(max_block: int):
    """A DroneSource whose native synth is faked, isolating the numpy
    downmix/duck path (the SF2-backed generate is exempt by ruling and invisible
    to tracemalloc anyway). Keeps the gate SF2-/PortAudio-free."""
    fake = types.ModuleType("tinysoundfont")

    class _FakeSynth:
        def __init__(self, samplerate: int) -> None:
            pass

        def sfload(self, path):
            return 0

        def program_select(self, *a) -> None:
            pass

        def set_tuning(self, *a) -> None:
            pass

        def generate(self, n, mv) -> None:
            pass  # leave the source's preallocated stereo buffer untouched

    fake.Synth = _FakeSynth
    saved = sys.modules.get("tinysoundfont")
    sys.modules["tinysoundfont"] = fake
    try:
        return sax_drone.DroneSource(_SR, max_block, sf2_path="fake.sf2")
    finally:  # restore immediately; the instance keeps its fake _syn
        if saved is None:
            sys.modules.pop("tinysoundfont", None)
        else:
            sys.modules["tinysoundfont"] = saved


_FACTORIES = [
    pytest.param(lambda mb: TestToneSource(440.0, _SR, mb, gain=0.2), id="testtone"),
    pytest.param(lambda mb: PitchPipeSource(69, _SR, mb), id="pitchpipe"),
    pytest.param(lambda mb: _ActiveClick(), id="click_active"),
    pytest.param(_make_drone, id="drone_numpy"),
]


# --- measurement mechanism --------------------------------------------------

def _peak_alloc(make_source, frames: int, *, warmup: int = 8, iters: int = 200):
    """Peak tracemalloc bytes over ``iters`` direct renders at ``frames``, after
    a warmup that flushes one-time allocations and any attack envelope."""
    src = make_source(_MB)
    out = np.zeros(_MB, dtype=np.float32)
    for _ in range(warmup):
        out[:] = 0.0
        src.render(out, frames, 0)
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    for _ in range(iters):
        out[:] = 0.0
        src.render(out, frames, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - base


# Separation: a single float32 per-block temporary would grow by
# (_MB - 512) * 4 ≈ 30 KB across this sweep. numpy's per-call view boxing is a
# flat constant. 2 KB sits ~15x below the smallest real temp and well above the
# constant — a wide, non-flaky margin.
_SCALING_LIMIT = 2048


@pytest.mark.parametrize("make_source", _FACTORIES)
def test_render_has_no_frames_proportional_allocation(make_source):
    """No source allocates a data buffer proportional to the block size."""
    growth = _peak_alloc(make_source, _MB) - _peak_alloc(make_source, 512)
    assert growth < _SCALING_LIMIT, (
        f"render allocation scales with block size: peak grew {growth} B going "
        f"from frames=512 to frames={_MB} — a per-block data temporary "
        f"(~frames*itemsize) the zero-alloc contract forbids on the hot path")


@pytest.mark.parametrize("make_source", _FACTORIES)
def test_render_does_not_leak(make_source):
    """No source RETAINS memory across many renders — that would be a genuine
    leak (accumulating references), distinct from transient per-block churn that
    frees each block."""
    src = make_source(_MB)
    out = np.zeros(_MB, dtype=np.float32)
    for _ in range(8):
        out[:] = 0.0
        src.render(out, _MB, 0)
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    for _ in range(300):
        out[:] = 0.0
        src.render(out, _MB, 0)
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = cur - base
    assert retained < 1024, (
        f"steady-state render retained {retained} B across 300 iters — a real "
        f"leak (transient per-block allocations should free, not accumulate)")

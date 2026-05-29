"""AUDIO-TRANSIENT-REGRESSION (v1.1 hardening ledger): the click-free
start/stop behaviour of each audio source, locked as a unified regression net
using the sim_harness edge helpers (edge_level / has_fade_in / has_fade_out /
boundary_jump).

The transient WORK was done in v1.0/v1.0.1 (see the transient-audit). This file
closes the remaining gap: those helpers were unit-tested in test_sim.py but never
applied to the real sources, and the per-source transient locks were ad-hoc raw
asserts. Coverage map (cite, don't duplicate):

  * enveloped TestToneSource + PitchPipeSource — start/stop fades: locked HERE
    (no harness-based edge lock existed before).
  * metronome ClickSource — self-enveloped burst: onset/decay locked HERE; the
    multi-block resume is test_metronome.py::test_click_spanning_block_boundary_resumes
    and the no-truncation stop is test_metronome.py::test_stop_does_not_truncate_inflight_click.
  * DeckPlaybackSource — 5 ms edge fade: test_deck.py::
    test_playback_source_direct_path_body_is_faithful_with_edge_fades.
  * Drone duck de-zipper (GainGlide): test_mixer.py::
    test_gainglide_downward_glide_respects_slew_cap (slew cap across block seams)
    + test_drone.py (set_duck_target glide). The drone NOTE envelope is tsf's own.
  * Plain (non-enveloped) TestToneSource is INTENTIONALLY hard-edged — the
    byte-identical reference tone; its click-free variant is opt-in via
    attack_ms/release_ms, which is what this file exercises.
"""
from __future__ import annotations

import numpy as np
import pytest

import sim_harness as H
from sax_mixer import TestToneSource
from sax_metronome import ClickSource

pp = pytest.importorskip("sax_pitch_pipes", reason="pitch pipes module")
PitchPipeSource = pp.PitchPipeSource

SR = 48000
MB = 2048


def _render(src, blocks: int, frames: int = MB) -> np.ndarray:
    """Concatenate ``blocks`` renders of ``src`` into one signal. Each block is
    rendered additively into a freshly-zeroed buffer, then copied out."""
    parts = []
    buf = np.zeros(frames, dtype=np.float32)
    for _ in range(blocks):
        buf[:] = 0.0
        src.render(buf, frames, 0)
        parts.append(buf.copy())
    return np.concatenate(parts)


def _render_until_finished(src, max_blocks: int = 400, frames: int = MB) -> np.ndarray:
    """Render until the source reports ``finished`` (or a generous cap) — used to
    capture a full release tail."""
    parts = []
    buf = np.zeros(frames, dtype=np.float32)
    for _ in range(max_blocks):
        buf[:] = 0.0
        src.render(buf, frames, 0)
        parts.append(buf.copy())
        if getattr(src, "finished", False):
            break
    return np.concatenate(parts)


# Sources whose contract is "fade in on start, fade out on release".
_ENVELOPED = [
    pytest.param(
        lambda: TestToneSource(440.0, SR, MB, gain=0.3,
                               attack_ms=10.0, release_ms=50.0),
        id="testtone_enveloped"),
    pytest.param(lambda: PitchPipeSource(69, SR, MB), id="pitchpipe"),
]


@pytest.mark.parametrize("make", _ENVELOPED)
def test_enveloped_source_starts_click_free(make):
    """A fresh enveloped source attacks from silence — begins at ~0 and ramps up
    — so tap-on produces no click."""
    sig = _render(make(), blocks=2)
    assert H.edge_level(sig, head=True, n=1) < 1e-3, \
        "onset sample is not ~0 — a full-amplitude start clicks"
    assert H.has_fade_in(sig, sr=SR), \
        "no attack ramp detected — the envelope must fade in"


@pytest.mark.parametrize("make", _ENVELOPED)
def test_enveloped_source_releases_click_free(make):
    """release() ramps down to silence — ends at ~0 — so tap-off produces no
    click, then the source reports finished so the Mixer reaps it."""
    src = make()
    _render(src, blocks=3)               # reach the steady (fully-attacked) state
    src.release()
    sig = _render_until_finished(src)
    assert H.edge_level(sig, head=False, n=1) < 1e-3, \
        "final sample is not ~0 — a hard stop clicks"
    assert H.has_fade_out(sig, sr=SR), \
        "no release ramp detected — the envelope must fade out"
    assert src.finished, "a released enveloped source must finish (so it is reaped)"


def test_metronome_click_is_self_enveloped():
    """The click voice begins at ~0 (≈1 ms attack) and decays to silence (no hard
    cut) — a click-free tick. The ~38 ms burst fits within one 42 ms block."""
    c = ClickSource(SR)
    c.trigger(accent=True)
    sig = _render(c, blocks=1)
    assert H.edge_level(sig, head=True, n=1) < 1e-3, \
        "click onset is not ~0 — missing the attack ramp"
    assert H.has_fade_out(sig, sr=SR), \
        "click does not decay to silence — a hard cut clicks"

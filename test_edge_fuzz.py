"""EDGE-FUZZ-DSP-IMPORT (v1.1 hardening ledger): the import + DSP entry points
must survive degenerate / malformed / hostile input — never crash, gate or
reject cleanly, and never let a corrupt frame poison later results.

Coverage map (cite, don't duplicate):
  * yin_pitch degenerate inputs (silence/empty/single-sample/near-zero/hard-clip):
    test_yin_accuracy.py::test_yin_pitch_robust_to_degenerate_input + test_yin_baseline.py.
  * CSV import (NaN/inf/out-of-range/BOM/malformed rows): test_csv_import_paths.py
    (#28) + test_csv_bom_import.py.
  * config / customs hostile JSON: test_config_roundtrip.py (STAB-CONFIG-FUZZ).

Gaps closed HERE:
  * WAV import (DeckController.load_take, the public "load a take" path): it wraps
    read_wav in try/except and is documented "never raises" — locked now against a
    battery of malformed files.
  * Engine non-finite sample gating: feed_input_frames computes an rms and bails a
    frame whose rms is non-finite (sax_audio_engine.py:2000). Locked now: NaN/inf
    input emits nothing and does not raise, a clean tone after garbage still
    detects (no poisoning), and degenerate block sizes are tolerated.
"""
from __future__ import annotations

import wave

import numpy as np
import pytest

import sim_harness as H
from sax_audio_engine import AudioEngine
from sax_mixer import Mixer
import sax_deck
from sax_deck import DeckController, DeckState, write_wav


# ---------------------------------------------------------------------------
# 1. Malformed WAV import — load_take must report failure, never raise, never
#    wedge the deck into HAVE_TAKE.
# ---------------------------------------------------------------------------
def _write_truncated(p):
    """A valid WAV with its body cut off after the RIFF header."""
    write_wav(str(p), (np.sin(np.arange(4000) / 8.0) * 0.3).astype(np.float32), 48000)
    p.write_bytes(p.read_bytes()[:24])


def _write_eight_bit(p):
    """An 8-bit-PCM WAV — read_wav supports 16-bit only (raises ValueError)."""
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(48000)
        w.writeframes(bytes(1000))


_MALFORMED = [
    ("empty", lambda p: p.write_bytes(b"")),
    ("not_a_wav", lambda p: p.write_bytes(b"plainly not a RIFF/WAVE file, just text")),
    ("truncated_header", _write_truncated),
    ("eight_bit_pcm", _write_eight_bit),
    ("zero_frames", lambda p: write_wav(str(p), np.zeros(0, np.float32), 48000)),
    ("nonexistent", None),
]


@pytest.mark.parametrize("name,builder", _MALFORMED, ids=[c[0] for c in _MALFORMED])
def test_load_take_rejects_malformed_wav_without_raising(tmp_path, name, builder):
    p = tmp_path / f"{name}.wav"
    if builder is not None:
        builder(p)
    deck = DeckController(Mixer(max_block=1024), 48000)
    assert deck.load_take(str(p)) is False, f"{name}: load_take must report failure"
    assert deck.state == DeckState.IDLE, f"{name}: a failed load must not change deck state"


def test_load_take_accepts_valid_wav(tmp_path):
    """The malformed-input guards must not reject a well-formed take."""
    p = tmp_path / "good.wav"
    write_wav(str(p), (np.sin(np.arange(8000) / 8.0) * 0.3).astype(np.float32), 48000)
    deck = DeckController(Mixer(max_block=1024), 48000)
    assert deck.load_take(str(p)) is True
    assert deck.state == DeckState.HAVE_TAKE


# ---------------------------------------------------------------------------
# 2. Engine non-finite / degenerate input — the detection hot path must gate,
#    not crash, and must recover on the next clean frame.
# ---------------------------------------------------------------------------
_NB = H.DEFAULT_N


def _fresh_engine() -> AudioEngine:
    eng = AudioEngine()
    eng.set_filter_mode("normal")
    eng.set_a4(440.0)
    return eng


@pytest.mark.parametrize("fill", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_engine_gates_nonfinite_input_without_emitting(fill):
    """A buffer full of NaN/inf is gated (rms non-finite -> bail the frame) — no
    note emitted, and reaching this assert proves no exception was raised. Uses
    24 blocks (the same length the recover test confirms a clean tone DOES emit
    at) so 'no emit' is a meaningful contrast, not just too-few-hops."""
    res = H.feed_engine(_fresh_engine(), np.full(_NB * 24, fill, dtype=np.float32))
    assert not res.emitted, "non-finite input must be gated (no note emitted)"


def test_engine_recovers_after_nonfinite_frames():
    """A corrupt (NaN) frame is bailed per-frame and must NOT poison later
    detection — a clean A4 after the garbage still confirms."""
    eng = _fresh_engine()
    H.feed_engine(eng, np.full(_NB * 4, np.nan, dtype=np.float32))       # poison
    res = H.feed_engine(eng, H.sax_like(440.0, H.DEFAULT_SR, _NB * 24, 0.5))
    assert res.emitted and res.dominant_midi == 69, \
        "engine must detect a clean A4 after non-finite frames (no poisoning)"


@pytest.mark.parametrize("frames", [np.zeros(0, np.float32), np.array([0.5], np.float32)],
                         ids=["zero_frames", "single_sample"])
def test_engine_tolerates_degenerate_block_sizes(frames):
    """A zero-length or single-sample callback buffer must not crash the tap."""
    _fresh_engine().feed_input_frames(frames)

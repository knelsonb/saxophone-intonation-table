"""Mic input-gain (SETUP-MICGAIN) — engine-level behaviour.

The mic gain is applied in the input hot path to the silence-gate decision and
the level meter ONLY — never to the signal buffer — so the cents readout stays
scale-invariant and the tape-deck tap records the raw mic. These tests drive
the real input path device-free via AudioEngine.feed_input_frames():

  * set_mic_gain clamps to ~+/-24 dB and ignores non-finite / non-numeric input.
  * a sub-floor tone is GATED at 0 dB but CLEARS the floor at +12 dB (the point
    of the feature — let a quiet mic register).
  * the level meter (last_rms_db) scales by exactly the applied gain.

Pure numpy; runs on the always-runnable suite (no PyQt6 / no PortAudio).
"""
from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip('numpy')
import numpy as np  # noqa: E402

from sax_audio_engine import (  # noqa: E402
    AudioEngine, FILTER_PRESETS, DEFAULT_BLOCK_SIZE,
)

_SR = 44100
_N = DEFAULT_BLOCK_SIZE


def _sine(f: float, n: int, amp: float) -> np.ndarray:
    """Continuous float32 sine, amplitude *amp* (RMS = amp/sqrt(2))."""
    t = np.arange(n, dtype=np.float64)
    return (amp * np.sin(2.0 * math.pi * f * t / _SR)).astype(np.float32)


def _feed_full_ring(gain: float, amp: float, freq: float = 440.0) -> AudioEngine:
    """Feed a continuous tone in block-sized chunks until the engine's input
    ring is fully populated, so the measured RMS is amp/sqrt(2) regardless of
    the ring's internal 8x sizing. Returns the engine for inspection."""
    eng = AudioEngine()
    eng.set_mic_gain(gain)
    total = _sine(freq, _N * 10, amp)
    for i in range(10):
        eng.feed_input_frames(total[i * _N:(i + 1) * _N])
    return eng


def test_default_mic_gain_is_unity():
    assert AudioEngine().mic_gain == 1.0


def test_set_mic_gain_clamps_and_ignores_garbage():
    eng = AudioEngine()
    eng.set_mic_gain(2.0)
    assert abs(eng.mic_gain - 2.0) < 1e-9
    eng.set_mic_gain(1000.0)
    assert eng.mic_gain == 16.0          # clamp high (~+24 dB)
    eng.set_mic_gain(0.0)
    assert eng.mic_gain == 0.05          # clamp low (~-26 dB)
    # Non-finite / non-numeric must be ignored, leaving the last good value.
    eng.set_mic_gain(float('nan'))
    assert eng.mic_gain == 0.05
    eng.set_mic_gain(float('inf'))
    assert eng.mic_gain == 0.05
    eng.set_mic_gain("loud")
    assert eng.mic_gain == 0.05


def test_mic_gain_lifts_sub_floor_signal_past_silence_gate():
    floor = FILTER_PRESETS['normal']['rms_floor']
    # RMS at 60% of the floor: gated at unity, cleared by a >=+4.4 dB boost.
    amp = (floor * 0.6) * math.sqrt(2.0)
    # 0 dB: below the floor → gated as silence → last_freq stays 0.
    assert _feed_full_ring(1.0, amp).last_freq == 0.0
    # +12 dB (x3.98): rms_eff = 2.39x floor → gate cleared → YIN ran, so
    # last_freq is recorded (non-zero) even before the lock machinery.
    assert _feed_full_ring(3.981, amp).last_freq != 0.0


def test_mic_gain_does_not_gate_a_loud_signal_either_way():
    # A healthy tone clears the floor at any gain; the gain only moves the
    # meter, never the detected pitch (signal is never scaled).
    loud = _feed_full_ring(1.0, 0.05)
    assert loud.last_freq != 0.0


def test_mic_gain_scales_level_meter_by_exactly_the_gain():
    amp = 0.05  # well above the floor at both gains
    db1 = _feed_full_ring(1.0, amp).last_rms_db
    db2 = _feed_full_ring(2.0, amp).last_rms_db
    # A 2x linear gain is +6.02 dB on the meter.
    assert abs((db2 - db1) - 20.0 * math.log10(2.0)) < 0.05

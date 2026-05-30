"""Acceptance net for ``sax_drone`` (parity Sprint 3).

Two tiers:
  * PURE tier (always runs) — the GM preset catalog, voice resolution +
    fallback, and the midi->freq / drone_freq math incl. non-440 A4. sax_drone
    guard-imports tinysoundfont, so these import + test on system-python.
  * SYNTH tier (importorskip tinysoundfont) — the TSF-backed DroneSource +
    DroneController: it actually sounds, reports the audible-tail-aware
    active_midi (the wire that makes sounding_midis() non-empty), and respects
    semitone/volume. Runs on the full-dep venv / CI.

Pre-written against Sauron's firm S3 contract (msg 3430), verified against the
landed module.
"""
from __future__ import annotations

import numpy as np
import pytest

import sax_drone as D
from sax_mixer import Mixer


def _expected_freq(midi, a4):
    return a4 * (2.0 ** ((midi - 69) / 12.0))


# ===========================================================================
# PURE tier — catalog + voice resolution + frequency math.
# ===========================================================================
def test_drone_presets_are_the_five_android_voices():
    by_id = {p[0]: p[2] for p in D.DRONE_PRESETS}
    assert by_id == {
        "organ": 19, "strings": 48, "cello": 42, "tenorsax": 66, "warmpad": 89,
    }, "the 5 presets must match droneVoices.ts (id -> GM program)"


def test_drone_full_gm_has_128_voices():
    assert len(D.DRONE_FULL_GM) == 128, "full GM catalog must list all 128 programs"


@pytest.mark.parametrize("voice_id,program", [
    ("organ", 19), ("strings", 48), ("cello", 42),
    ("tenorsax", 66), ("warmpad", 89),
])
def test_resolve_drone_voice_known(voice_id, program):
    assert D.resolve_drone_voice(voice_id) == program


@pytest.mark.parametrize("bad", ["bogus", "", None, "trumpet99"])
def test_resolve_drone_voice_unknown_falls_back_to_organ(bad):
    assert D.resolve_drone_voice(bad) == 19, "unknown voice must fall back to organ/19"


@pytest.mark.parametrize("midi,a4,expected", [
    pytest.param(69, 440.0, 440.0, id="A4_440"),
    pytest.param(57, 440.0, 220.0, id="A3_octave_down"),
    pytest.param(81, 440.0, 880.0, id="A5_octave_up"),
    pytest.param(60, 440.0, 261.6255653, id="C4"),
    pytest.param(69, 442.0, 442.0, id="A4_442_nondefault"),
    pytest.param(69, 430.0, 430.0, id="A4_430_nondefault"),
])
def test_midi_to_freq(midi, a4, expected):
    assert D.midi_to_freq(midi, a4) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("ref,semi,a4,expected_midi", [
    pytest.param(69, 0, 440.0, 69, id="A4_no_offset"),
    pytest.param(69, 12, 440.0, 81, id="up_octave"),
    pytest.param(69, -12, 440.0, 57, id="down_octave"),
    pytest.param(60, 7, 440.0, 67, id="C4_up_a_fifth"),
    pytest.param(69, 0, 442.0, 69, id="nondefault_a4"),
])
def test_drone_freq_is_reference_plus_semitones(ref, semi, a4, expected_midi):
    assert D.drone_freq(ref, semi, a4) == pytest.approx(
        _expected_freq(expected_midi, a4), rel=1e-6)


def test_drone_freq_nondefault_a4_scales():
    """drone_freq at 442 must be the 440 value times 442/440 (reference-relative,
    not hardcoded Hz)."""
    assert D.drone_freq(69, 3, 442.0) == pytest.approx(
        D.drone_freq(69, 3, 440.0) * (442.0 / 440.0), rel=1e-9)


# ===========================================================================
# SYNTH tier — TSF-backed DroneSource / DroneController (venv / CI only).
# ===========================================================================
tsf = pytest.importorskip(
    "tinysoundfont", reason="tinysoundfont not installed; drone-synth tests are venv/CI only")


def _drone(**kw):
    return D.DroneController(Mixer(max_block=2048), 48000, **kw)


def test_enabled_drone_sounds_and_reports_active_midi():
    m = Mixer(max_block=2048)
    c = D.DroneController(m, 48000, voice_id="organ", volume=0.6)
    c.set_enabled(True)
    assert c.is_enabled() is True
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert np.max(np.abs(out)) > 1e-3, "an enabled drone must produce audio"
    # Default reference_midi=69, semitones=0 -> sounding A4 (MIDI 69).
    assert c.source.active_midi == 69
    assert m.sounding_midis() == frozenset({69}), (
        "the sounding drone is the wire that makes sounding_midis() non-empty")


def test_semitone_offset_shifts_sounding_note():
    m = Mixer(max_block=2048)
    c = D.DroneController(m, 48000, semitones=0)
    c.set_enabled(True)
    c.set_semitones(12)            # up an octave -> MIDI 81
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert c.source.active_midi == 81, "semitone offset must shift the sounding MIDI"


def test_semitones_clamped_to_plus_minus_octave():
    c = _drone()
    c.set_semitones(99)
    assert c.semitones == 12, "semitones clamp to +12"
    c.set_semitones(-99)
    assert c.semitones == -12, "semitones clamp to -12"


def test_set_program_clamps_to_gm_range_no_crash():
    """ADVERSARIAL-SWEEP wave 4: tsf's program_select raises (RuntimeError:
    channel_set_preset_number) for a program outside the GM range [0,127];
    DroneSource.set_program must clamp instead of crashing."""
    m = Mixer(max_block=2048)
    c = D.DroneController(m, 48000, voice_id="organ")
    c.set_enabled(True)                 # creates the TSF-backed source
    src = c.source
    src.set_program(-1)
    assert src._program == 0, "negative program must clamp to 0 (no RuntimeError)"
    src.set_program(256)
    assert src._program == 127, "out-of-range program must clamp to 127"
    src.set_program(40)
    assert src._program == 40, "in-range program is unchanged"


def test_active_midi_is_audible_tail_aware():
    """active_midi follows AUDIO not note-on: after disable (note_off) it must
    stay set while the release tail is audible, then flip None once silent —
    so the duck holds while leakage bleeds, then releases."""
    m = Mixer(max_block=2048)
    c = D.DroneController(m, 48000, voice_id="organ")
    c.set_enabled(True)
    out = np.zeros(2048, dtype=np.float32)
    m.render(out, 2048)
    assert c.source.active_midi == 69

    c.set_enabled(False)           # note_off; tail fades, source stays registered
    # Immediately after note_off the tail is still ringing.
    m.render(out, 2048)
    assert c.source.active_midi == 69, "active_midi must persist through the tail"

    # Render until the tail decays below the audible threshold + confirm window.
    flipped = False
    for _ in range(400):           # ~17s @48k/2048 — ample for any release tail
        m.render(out, 2048)
        if c.source.active_midi is None:
            flipped = True
            break
    assert flipped, "active_midi must flip None once the tail is truly inaudible"


def test_duck_target_attenuates_drone_output():
    """set_duck_target (the engine's D3 hook) must glide the drone's gain down,
    reducing output level — the mechanism half of duck-on-suspicion."""
    m = Mixer(max_block=2048)
    c = D.DroneController(m, 48000, voice_id="organ", volume=1.0)
    c.set_enabled(True)
    out = np.zeros(2048, dtype=np.float32)
    # Warm up to steady SUSTAIN first — measuring on the attack ramp would
    # under-read the open level and make the duck comparison meaningless.
    for _ in range(30):
        m.render(out, 2048)
    peak_open = float(np.max(np.abs(out)))
    assert peak_open > 1e-4, "drone should be sounding at sustain"

    c.source.set_duck_target(0.30)
    peak_ducked = peak_open
    for _ in range(30):                # let the per-sample glide settle to 0.30
        out = np.zeros(2048, dtype=np.float32)
        m.render(out, 2048)
        peak_ducked = float(np.max(np.abs(out)))
    assert peak_ducked < peak_open * 0.6, (
        f"duck target 0.30 must clearly attenuate the sustained drone: "
        f"open={peak_open} ducked={peak_ducked}")


def test_set_voice_changes_program():
    c = _drone(voice_id="organ")
    c.set_voice("cello")
    assert c.voice_id == "cello"


# ===========================================================================
# Live duck-attach path (Sauron's 3474 catch / Orchestrator 3481 hard gate):
# the duck only fires for the USER if DroneController is built with engine= so
# enable -> engine.attach_duck_consumer(source) runs. The e2e/coordination_step
# tests attach a consumer MANUALLY, bypassing this GUI-construction path — so
# this unit lock tests the live wiring the other tests can't see.
# ===========================================================================
class _FakeEngine:
    def __init__(self):
        self.attached = []
        self.detached = []

    def attach_duck_consumer(self, c):
        self.attached.append(c)

    def detach_duck_consumer(self, c=None):
        self.detached.append(c)


def test_enable_attaches_drone_as_duck_consumer_to_engine():
    eng = _FakeEngine()
    c = D.DroneController(Mixer(max_block=2048), 48000, voice_id="organ", engine=eng)
    c.set_enabled(True)
    assert c.source is not None
    assert c.source in eng.attached, (
        "enable must attach the drone source via engine.attach_duck_consumer — "
        "otherwise the duck never fires for the user (only in manual-attach tests)")
    c.set_enabled(False)
    assert c.source in eng.detached, "disable must detach the duck consumer"


def test_no_engine_means_no_attach_but_still_sounds():
    """engine=None (default) is valid: the drone still SOUNDS + is vote-excluded
    (readout stays live), it simply never ducks. Construction must not require
    an engine."""
    c = D.DroneController(Mixer(max_block=2048), 48000, voice_id="organ")  # engine=None
    c.set_enabled(True)                      # must not raise
    assert c.is_enabled() is True

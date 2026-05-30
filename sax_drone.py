"""Drone: a sustained reference voice via TinySoundFont (GeneralUser-GS) +
the D3 mic-coordination duck consumer.

Sprint-3 keystone (parity feature D1). The drone sounds a sustained pitched
note — the current reference note shifted by a signed semitone offset — through
TinySoundFont loading the bundled GeneralUser-GS.sf2, so the user can tune
against any of the 5 ported presets (Organ/Strings/Cello/Tenor Sax/Warm Pad =
GM 19/48/42/66/89) or the full 128-voice GM catalogue.

Because the drone is the first *pitched* output source (``active_midi`` != None),
it is what finally activates the Sprint-1 mic-coordination policy end-to-end:
its sounding MIDI flows through ``mixer.sounding_midis()`` ->
``engine.get_sounding_output_midis()`` -> ``OutputCoordinator.update()``, which
vote-excludes that MIDI from the incumbent pitch (so the tuner readout doesn't
chase the drone) and, on confirmed mic-bleed, ducks the drone. The duck is a
per-frame ``duck_level`` the engine pushes via ``set_duck_target``; an embedded
``GainGlide`` smooths it sample-by-sample so the gain change never clicks.

``active_midi`` follows AUDIO, not note-on: it stays set through the release
tail while the note is still audible and flips to ``None`` only once the output
is truly inaudible — so the duck holds while the tail still bleeds, then opens.

Pure helpers (voice resolution, the GM catalogue, the frequency math) import
without TinySoundFont, so the catalogue and freq math are testable on plain
Python; the TSF-backed source is exercised in the venv where tinysoundfont is
installed.
"""

from __future__ import annotations

import math
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

try:  # GainGlide is Gandalf's de-zipper (sax_mixer); guard so this module
    from sax_mixer import GainGlide as _MixerGainGlide  # imports even if absent.
except Exception:  # pragma: no cover
    _MixerGainGlide = None

try:  # frozen-aware asset resolution (Gandalf's packaging lane).
    from sax_assets import asset_path as _asset_path
except Exception:  # pragma: no cover - dev fallback if sax_assets is absent
    _asset_path = None

A4_DEFAULT = 440.0
_SF2_FILENAME = "GeneralUser-GS.sf2"

__all__ = [
    "A4_DEFAULT", "GM_NAMES", "DRONE_PRESETS", "DRONE_FULL_GM",
    "resolve_drone_voice", "midi_to_freq", "drone_freq",
    "DroneSource", "DroneController",
]

# General MIDI Level-1 instrument names, indexed by 0-based program number
# (== the preset number TinySoundFont's program_select takes).
GM_NAMES: List[str] = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavi", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
    "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass",
    "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin",
    "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings",
    "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2",
    "SynthStrings 1", "SynthStrings 2", "Choir Aahs", "Voice Oohs",
    "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba",
    "Muted Trumpet", "French Horn", "Brass Section", "SynthBrass 1",
    "SynthBrass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute",
    "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle",
    "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
    "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
    "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)",
    "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)",
    "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)",
    "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo",
    "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise",
    "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter",
    "Applause", "Gunshot",
]

# Five ported presets (id, display name, GM program) — verbatim from the
# Android droneVoices catalogue.
DRONE_PRESETS: List[Tuple[str, str, int]] = [
    ("organ", "Organ", 19),
    ("strings", "Strings", 48),
    ("cello", "Cello", 42),
    ("tenorsax", "Tenor Sax", 66),
    ("warmpad", "Warm Pad", 89),
]
_PRESET_PROGRAM = {pid: prog for pid, _name, prog in DRONE_PRESETS}

# Full 128-voice expander for the SETUP picker: id "gm<N>", GM name, program N.
DRONE_FULL_GM: List[Tuple[str, str, int]] = [
    (f"gm{i}", GM_NAMES[i], i) for i in range(128)
]

_DEFAULT_PROGRAM = 19  # Organ — the fallback for an unknown voice id.


def _default_sf2_path() -> str:
    """Frozen-aware path to the bundled SoundFont. Uses sax_assets.asset_path
    (resolves against sys._MEIPASS in a PyInstaller build) when available, else
    falls back to a dev-checkout path relative to this module."""
    if _asset_path is not None:
        return _asset_path("assets", _SF2_FILENAME)
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", _SF2_FILENAME)


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------
def resolve_drone_voice(voice_id) -> int:
    """Map a voice id to a GM program number (0-127).

    Accepts a preset id ("organ"...), a full-GM id ("gm0".."gm127"), a bare
    int / digit string, or anything else (-> Organn/19 fallback). Never raises.
    """
    if isinstance(voice_id, bool):
        return _DEFAULT_PROGRAM
    if isinstance(voice_id, int):
        return voice_id if 0 <= voice_id <= 127 else _DEFAULT_PROGRAM
    s = str(voice_id).strip().lower()
    if s in _PRESET_PROGRAM:
        return _PRESET_PROGRAM[s]
    if s.startswith("gm"):
        try:
            n = int(s[2:])
            if 0 <= n <= 127:
                return n
        except ValueError:
            pass
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 127:
            return n
    return _DEFAULT_PROGRAM


def midi_to_freq(midi: float, a4: float = A4_DEFAULT) -> float:
    """Frequency (Hz) of a MIDI note at the given A4 (inverse of freq_to_midi)."""
    return float(a4) * (2.0 ** ((float(midi) - 69.0) / 12.0))


def drone_freq(reference_midi: int, semitones: int,
               a4: float = A4_DEFAULT) -> float:
    """Frequency the drone sounds: reference note shifted by ``semitones``, at
    the current A4. Correct at any A4 (not just 440)."""
    return midi_to_freq(int(reference_midi) + int(semitones), a4)


def _a4_to_tuning_semitones(a4: float) -> float:
    """TinySoundFont ``set_tuning`` is in SEMITONES (validated: tuning=12 ->
    exact octave). Shifting concert pitch from 440 to ``a4`` is
    ``12*log2(a4/440)`` semitones."""
    return 12.0 * math.log2(float(a4) / 440.0)


def _clamp01(v: float) -> float:
    v = float(v)
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


# Fallback GainGlide (only used if sax_mixer.GainGlide isn't importable) so this
# module stays self-contained; mirrors the API Gandalf's class exposes.
if _MixerGainGlide is not None:
    GainGlide = _MixerGainGlide
else:  # pragma: no cover - exercised only without sax_mixer.GainGlide
    class GainGlide:  # type: ignore[no-redef]
        def __init__(self, max_block: int, samplerate: int,
                     glide_ms: float = 5.0, initial: float = 1.0):
            self.gain = float(initial)
            self.target = float(initial)
            self._step = 1.0 / max(1.0, glide_ms * samplerate / 1000.0)
            self._ramp = np.arange(int(max_block), dtype=np.float32)
            self._g = np.zeros(int(max_block), dtype=np.float32)

        def set_target(self, level: float) -> None:
            self.target = float(level)

        def apply(self, buf: np.ndarray, frames: int) -> None:
            if frames <= 0:
                return
            if self.gain == self.target:
                if self.gain != 1.0:
                    buf[:frames] *= np.float32(self.gain)
                return
            step = self._step if self.target > self.gain else -self._step
            g = self._g[:frames]
            np.multiply(self._ramp[:frames], np.float32(step), out=g)
            g += np.float32(self.gain)
            lo, hi = sorted((self.gain, self.target))
            np.clip(g, lo, hi, out=g)
            buf[:frames] *= g
            self.gain = float(g[frames - 1])
            if abs(self.gain - self.target) <= self._step:
                self.gain = self.target


# ---------------------------------------------------------------------------
# DroneSource — TSF-backed sustained voice, a MixerSource.
# ---------------------------------------------------------------------------
class DroneSource:
    """Sustained TinySoundFont voice summed into the mono mixer.

    Holds one ``tinysoundfont.Synth`` (SF2 loaded once), plays a single note at
    a time on channel 0, downmixes TSF's stereo output to mono into a
    preallocated buffer (zero-allocation Python glue per Gandalf's native-source
    ruling), applies the user volume and the embedded duck ``GainGlide``, and
    sums into the block. ``active_midi`` is audible-tail aware.
    """

    __test__ = False

    # Output peak below which the (released) voice is considered inaudible, and
    # how long it must stay there before active_midi flips to None.
    _SILENCE_PEAK = 1e-4

    def __init__(self, samplerate: int, max_block: int,
                 gm_program: int = _DEFAULT_PROGRAM, *, volume: float = 0.5,
                 a4: float = A4_DEFAULT, sf2_path: Optional[str] = None,
                 glide_ms: float = 5.0):
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        import tinysoundfont as tsf  # lazy: keeps the pure helpers TSF-free

        self.samplerate = int(samplerate)
        self._max_block = int(max_block)
        self._user_volume = _clamp01(volume)
        self._a4 = float(a4)
        self._program = int(gm_program)

        self._syn = tsf.Synth(samplerate=self.samplerate)
        self._sfid = self._syn.sfload(sf2_path or _default_sf2_path())
        self._syn.program_select(0, self._sfid, 0, self._program)
        self._syn.set_tuning(0, _a4_to_tuning_semitones(self._a4))

        # Preallocated buffers (no per-block allocation). TSF writes stereo
        # interleaved float32 into _stereo_buf; _stereo_np views it as float32.
        self._stereo_buf = bytearray(self._max_block * 2 * 4)
        self._stereo_mv = memoryview(self._stereo_buf)
        self._stereo_np = np.frombuffer(self._stereo_buf, dtype=np.float32)
        self._mono = np.zeros(self._max_block, dtype=np.float32)
        # Fixed strided views onto the interleaved buffer, built ONCE (they track
        # the buffer's live contents), so the per-block downmix allocates no
        # view objects. Preboxed scalars keep the in-place ops allocation-free.
        self._left = self._stereo_np[0::2]
        self._right = self._stereo_np[1::2]
        self._half32 = np.float32(0.5)
        self._vol32 = np.float32(self._user_volume)

        self._duck = GainGlide(self._max_block, self.samplerate,
                               glide_ms=glide_ms, initial=1.0)

        self._note: Optional[int] = None      # currently note-on MIDI, or None
        self._sounding: Optional[int] = None   # audible MIDI incl. tail, or None
        self._silent_blocks = 0
        # ~50 ms of consecutive silence confirms the tail is gone.
        self._silence_confirm = max(
            1, int(0.05 * self.samplerate / self._max_block))

    # -- control ------------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        self._user_volume = _clamp01(volume)
        self._vol32 = np.float32(self._user_volume)

    def set_program(self, gm_program: int) -> None:
        # tsf's program_select raises (RuntimeError: channel_set_preset_number)
        # for a program outside the GM range [0,127]; clamp. The controller's
        # resolve_drone_voice already yields a valid program — this guards
        # direct / API callers.
        self._program = max(0, min(127, int(gm_program)))
        self._syn.program_select(0, self._sfid, 0, self._program)
        if self._note is not None:           # re-voice a sounding note
            self._retrigger()

    def set_a4(self, a4: float) -> None:
        a4 = float(a4)
        if not math.isfinite(a4) or a4 <= 0.0:
            return            # a4 <= 0 -> _a4_to_tuning_semitones' log2 domain error
        self._a4 = a4
        self._syn.set_tuning(0, _a4_to_tuning_semitones(self._a4))
        if self._note is not None:
            self._retrigger()

    def set_duck_target(self, level: float) -> None:
        """Engine D3 hook: per-hop duck gain target (in [duck_depth, 1.0])."""
        self._duck.set_target(level)

    def note_on(self, midi: int, velocity: int = 100) -> None:
        midi = max(0, min(127, int(midi)))
        if self._note is not None:
            self._syn.noteoff(0, self._note)
        self._note = midi
        self._syn.noteon(0, midi, int(velocity))
        self._sounding = midi
        self._silent_blocks = 0

    def note_off(self) -> None:
        if self._note is not None:
            self._syn.noteoff(0, self._note)
            self._note = None
        # _sounding stays set until the release tail fades (audible-tail rule).

    def silence(self) -> None:
        """Force the audible-tail state to silent immediately (note off + kill
        any ringing tail), WITHOUT releasing the synth so the drone can be
        re-enabled. Used when the drone is disabled while no render will run to
        drain the tail (output stopped): otherwise active_midi (== _sounding)
        would stick non-None and the D3 coordinator would vote-exclude that MIDI
        forever."""
        if self._note is not None:
            try:
                self._syn.noteoff(0, self._note)
            except Exception:
                pass
            self._note = None
        try:
            self._syn.sounds_off()
        except Exception:
            pass
        self._sounding = None
        self._silent_blocks = 0

    def _retrigger(self) -> None:
        note = self._note
        if note is not None:
            self._syn.noteoff(0, note)
            self._syn.noteon(0, note, 100)
            self._sounding = note
            self._silent_blocks = 0

    @property
    def active_midi(self) -> Optional[int]:
        return self._sounding

    @property
    def duck_gain(self) -> float:
        """Current (glided) duck gain — observable for tests/telemetry."""
        return self._duck.gain

    def close(self) -> None:
        try:
            self._syn.sounds_off()
        except Exception:
            pass
        self._note = None
        self._sounding = None

    # -- MixerSource render -------------------------------------------------
    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        if frames <= 0:
            return
        n = self._max_block if frames >= self._max_block else frames

        # Native synth fills the preallocated stereo buffer (bounded, one buffer
        # per block — exempt from the strict numpy gate per the native ruling).
        self._syn.generate(n, self._stereo_mv[:n * 2 * 4])

        # Downmix interleaved L/R into the preallocated mono buffer, in place.
        # Use the whole prebuilt views/buffers when the block fills them (the
        # steady state) so no per-block view objects are allocated.
        if n == self._max_block:
            left, right, mono = self._left, self._right, self._mono
        else:
            left, right, mono = self._left[:n], self._right[:n], self._mono[:n]
        mono[:] = left
        mono += right
        mono *= self._half32

        # Audible-tail tracking on the RAW (pre-volume, pre-duck) synth output,
        # so a low user volume or a deep duck doesn't fool the tail detector.
        # max/min reductions avoid an np.abs temporary (zero per-block alloc).
        hi = float(mono.max())
        lo = float(mono.min())
        peak = hi if hi >= -lo else -lo
        if self._note is None:
            if peak < self._SILENCE_PEAK:
                self._silent_blocks += 1
                if self._silent_blocks >= self._silence_confirm:
                    self._sounding = None
            else:
                self._silent_blocks = 0
        else:
            self._silent_blocks = 0

        # User volume (the source's own gain) then the duck glide — both on the
        # drone's OWN buffer, never the shared accumulator, so the duck affects
        # only the drone and not other sources summed into `out`.
        if self._user_volume != 1.0:
            mono *= self._vol32
        self._duck.apply(mono, n)

        if out.shape[0] == n:
            out += mono
        else:
            out[:n] += mono


# ---------------------------------------------------------------------------
# DroneController — state + lifecycle on the mixer.
# ---------------------------------------------------------------------------
class DroneController:
    """Owns drone state and the (lazily created) DroneSource.

    The SF2 (32 MB) loads only on the first ``set_enabled(True)``, so
    constructing the controller at GUI start-up is cheap. The drone sounds
    ``reference_midi + semitones`` (clamped to MIDI range); ``set_reference_midi``
    is wired from the tuner's current target note. ``on_state_changed`` fires on
    any change so the GUI's TUNER drone bar and SETUP voice picker stay in sync.

    Duck wiring: when an ``engine`` exposing ``attach_duck_consumer`` /
    ``detach_duck_consumer`` is supplied, the source is attached on enable and
    detached on disable, so the engine's D3 wiring pushes ``set_duck_target``.
    """

    SEMITONE_LIMIT = 12

    def __init__(self, mixer, samplerate: int, *, voice_id: str = "organ",
                 volume: float = 0.5, semitones: int = 0,
                 reference_midi: int = 69, a4: float = A4_DEFAULT,
                 enabled: bool = False, engine=None,
                 sf2_path: Optional[str] = None,
                 on_state_changed: Optional[Callable[[], None]] = None):
        self._mixer = mixer
        self._samplerate = int(samplerate)
        self.voice_id = voice_id
        self.volume = _clamp01(volume)
        self.semitones = self._clamp_semi(semitones)
        self.reference_midi = int(reference_midi)
        self._a4 = float(a4)
        self._engine = engine
        self._sf2_path = sf2_path
        self._on_state_changed = on_state_changed
        self.enabled = False
        self._source: Optional[DroneSource] = None
        self._max_block = int(getattr(mixer, "max_block", 2048))
        # NOTE: requested enabled=True is honoured lazily — call set_enabled(True)
        # after the output stream is open (GUI honesty discipline).
        if enabled:
            self.set_enabled(True)

    # -- helpers ------------------------------------------------------------
    @classmethod
    def _clamp_semi(cls, n: int) -> int:
        return max(-cls.SEMITONE_LIMIT, min(cls.SEMITONE_LIMIT, int(n)))

    def _played_midi(self) -> int:
        return max(0, min(127, self.reference_midi + self.semitones))

    def _ensure_source(self) -> DroneSource:
        if self._source is None:
            self._max_block = int(getattr(self._mixer, "max_block", self._max_block))
            self._source = DroneSource(
                self._samplerate, self._max_block,
                resolve_drone_voice(self.voice_id), volume=self.volume,
                a4=self._a4, sf2_path=self._sf2_path)
        return self._source

    def _emit(self) -> None:
        cb = self._on_state_changed
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    @property
    def source(self) -> Optional[DroneSource]:
        return self._source

    def is_enabled(self) -> bool:
        return self.enabled

    # -- lifecycle ----------------------------------------------------------
    def set_enabled(self, on: bool) -> None:
        on = bool(on)
        if on == self.enabled:
            return
        self.enabled = on
        if on:
            src = self._ensure_source()
            self._mixer.register(src)
            src.note_on(self._played_midi())
            eng = self._engine
            if eng is not None and hasattr(eng, "attach_duck_consumer"):
                eng.attach_duck_consumer(src)
        else:
            if self._source is not None:
                self._source.note_off()   # tail fades; stays registered
                eng = self._engine
                if eng is not None and hasattr(eng, "detach_duck_consumer"):
                    eng.detach_duck_consumer(self._source)
                # If an engine's output stream is NOT running, render() never
                # runs to drain the audible tail, so _sounding (active_midi)
                # would stick and the D3 coordinator would vote-exclude that
                # MIDI forever. Force silence now. (When output IS running the
                # tail fades naturally over ~50ms, correctly vote-excluded while
                # audible; with no engine there's no D3 coordinator to get stuck,
                # so the audible-tail rule is left intact.)
                if eng is not None and not getattr(eng, "output_running", False):
                    self._source.silence()
        self._emit()

    def toggle(self) -> None:
        self.set_enabled(not self.enabled)

    def shutdown(self) -> None:
        """Stop and fully release the synth (call on teardown)."""
        if self._source is not None:
            self._source.note_off()
            self._mixer.unregister(self._source)
            self._source.close()
            self._source = None   # don't resurrect a closed synth on re-enable
        self.enabled = False
        self._emit()

    # -- setters ------------------------------------------------------------
    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id
        if self._source is not None:
            self._source.set_program(resolve_drone_voice(voice_id))
        self._emit()

    def set_volume(self, volume: float) -> None:
        self.volume = _clamp01(volume)
        if self._source is not None:
            self._source.set_volume(self.volume)
        self._emit()

    def set_semitones(self, semitones: int) -> None:
        self.semitones = self._clamp_semi(semitones)
        if self.enabled and self._source is not None:
            self._source.note_on(self._played_midi())
        self._emit()

    def set_reference_midi(self, midi: int) -> None:
        self.reference_midi = int(midi)
        if self.enabled and self._source is not None:
            self._source.note_on(self._played_midi())
        self._emit()

    def set_a4(self, a4: float) -> None:
        a4 = float(a4)
        if not math.isfinite(a4) or a4 <= 0.0:
            return            # match DroneSource.set_a4 + the engine: reject a4 <= 0
        self._a4 = a4
        if self._source is not None:
            self._source.set_a4(self._a4)
        self._emit()

    def set_samplerate(self, samplerate: int) -> None:
        """Rebuild the synth for a new output rate (call on a device switch).
        TinySoundFont binds its rate at construction, so the DroneSource is
        recreated (voice / volume / a4 / playing-note preserved). Without this,
        a rate change leaves the drone generating at the OLD rate while the DAC
        plays at the new one -- the tuning reference goes sharp/flat by
        12*log2(new/old) semitones (e.g. +2547 cents on a 44.1k -> 192k switch).
        Mirrors MetronomeController.set_samplerate; called from
        _reopen_output_bracketed alongside the metronome re-rate."""
        sr = int(samplerate)
        if sr <= 0 or sr == self._samplerate:
            return
        self._samplerate = sr
        if self._source is None:
            return  # next _ensure_source() builds at the new rate
        was_enabled = self.enabled
        eng = self._engine
        if eng is not None and hasattr(eng, "detach_duck_consumer"):
            try:
                eng.detach_duck_consumer(self._source)
            except Exception:
                pass
        try:
            self._mixer.unregister(self._source)
        except Exception:
            pass
        try:
            self._source.close()
        except Exception:
            pass
        self._source = None
        if was_enabled:
            src = self._ensure_source()
            self._mixer.register(src)
            src.note_on(self._played_midi())
            if eng is not None and hasattr(eng, "attach_duck_consumer"):
                eng.attach_duck_consumer(src)
        self._emit()

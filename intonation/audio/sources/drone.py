"""Drone: a sustained reference voice via TinySoundFont (GeneralUser-GS) +
the D3 mic-coordination duck consumer.

The drone sounds a sustained pitched
note — the current reference note shifted by a signed semitone offset — through
TinySoundFont loading the bundled GeneralUser-GS.sf2, so the user can tune
against any of the 5 ported presets (Organ/Strings/Cello/Tenor Sax/Warm Pad =
GM 19/48/42/66/89) or the full 128-voice GM catalogue.

Because the drone is the first *pitched* output source (``active_midi`` != None),
it is what activates the mic-coordination policy end-to-end:
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
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

from intonation.assets import asset_path
from intonation.audio.mixer import GainGlide
from intonation.core.pitch import A4_DEFAULT, midi_to_freq

_SF2_FILENAME = "GeneralUser-GS.sf2"

__all__ = [
    "A4_DEFAULT", "GM_NAMES", "DRONE_PRESETS", "DRONE_FULL_GM",
    "DRONE_SYNTH_VOICES", "SINE_VOICE_ID", "is_sine_voice",
    "resolve_drone_voice", "midi_to_freq", "drone_freq",
    "DroneSource", "SineDroneSource", "DroneController",
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

# Desktop-only SYNTHESISED voices — NOT part of the Android droneVoices catalog
# (DRONE_PRESETS, pinned to the five ported voices by test_drone). Same
# (id, display name, GM program) shape so the GUI picker treats them uniformly;
# the program is unused (these are generated, not loaded from the SF2). The pure
# sine is the drone's sub-cent, vibrato-free PRECISION tuning reference (the SF2
# voices carry musical vibrato/chorus — see test_virtual_bench §9/§16).
SINE_VOICE_ID = "sine"
DRONE_SYNTH_VOICES: List[Tuple[str, str, int]] = [
    (SINE_VOICE_ID, "Pure Sine", -1),
]


def _default_sf2_path() -> str:
    """Frozen-aware path to the bundled SoundFont: resolves against
    sys._MEIPASS in a PyInstaller build, else the source checkout."""
    return asset_path("assets", _SF2_FILENAME)


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
    if s.isdecimal():
        n = int(s)
        if 0 <= n <= 127:
            return n
    return _DEFAULT_PROGRAM


def is_sine_voice(voice_id) -> bool:
    """True if ``voice_id`` selects the pure-sine precision voice (a synthesised
    voice, NOT an SF2 patch). The controller branches on this to build a
    SineDroneSource instead of a DroneSource."""
    return str(voice_id).strip().lower() == SINE_VOICE_ID


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


# ---------------------------------------------------------------------------
# DroneSource — TSF-backed sustained voice, a MixerSource.
# ---------------------------------------------------------------------------
class DroneSource:
    """Sustained TinySoundFont voice summed into the mono mixer.

    Holds one ``tinysoundfont.Synth`` (SF2 loaded once), plays a single note at
    a time on channel 0, downmixes TSF's stereo output to mono into a
    preallocated buffer (zero-allocation Python glue around the native
    synth), applies the user volume and the embedded duck ``GainGlide``, and
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
        self.samplerate = int(samplerate)
        self._user_volume = _clamp01(volume)
        self._a4 = float(a4)
        self._program = int(gm_program)
        self._sf2_path = sf2_path or _default_sf2_path()
        self._note: Optional[int] = None      # currently note-on MIDI, or None
        self._sounding: Optional[int] = None   # audible MIDI incl. tail, or None
        self._silent_blocks = 0
        # Serialises the control side: the GUI's note / voice / A4 changes
        # against a format change on the audio-control thread, which swaps
        # the synth. render() never takes it (no render runs during a format
        # change).
        self._ctl = threading.RLock()

        self._syn, self._sfid = self._new_synth(self.samplerate)
        self._half32 = np.float32(0.5)
        self._vol32 = np.float32(self._user_volume)
        self._duck = GainGlide(int(max_block), self.samplerate,
                               glide_ms=glide_ms, initial=1.0)
        self._max_block = 0
        self._alloc(int(max_block))

    def _new_synth(self, samplerate: int):
        """A TinySoundFont synth at ``samplerate`` with the SF2, the current
        program and the A4 tuning loaded. The rate binds here and nowhere
        else, which is why a rate change builds a new one."""
        import tinysoundfont as tsf  # lazy: keeps the pure helpers TSF-free
        syn = tsf.Synth(samplerate=int(samplerate))
        sfid = syn.sfload(self._sf2_path)
        syn.program_select(0, sfid, 0, self._program)
        syn.set_tuning(0, _a4_to_tuning_semitones(self._a4))
        return syn, sfid

    def _alloc(self, max_block: int) -> None:
        """Preallocated buffers (no per-block allocation). TSF writes stereo
        interleaved float32 into _stereo_buf; _stereo_np views it as float32."""
        self._max_block = int(max_block)
        self._stereo_buf = bytearray(self._max_block * 2 * 4)
        self._stereo_mv = memoryview(self._stereo_buf)
        self._stereo_np = np.frombuffer(self._stereo_buf, dtype=np.float32)
        self._mono = np.zeros(self._max_block, dtype=np.float32)
        # Fixed strided views onto the interleaved buffer, built ONCE (they track
        # the buffer's live contents), so the per-block downmix allocates no
        # view objects. Preboxed scalars keep the in-place ops allocation-free.
        self._left = self._stereo_np[0::2]
        self._right = self._stereo_np[1::2]
        self._duck.reformat(self._max_block, self.samplerate)
        # ~50 ms of consecutive silence confirms the tail is gone.
        self._silence_confirm = max(
            1, int(0.05 * self.samplerate / self._max_block))

    def on_format_changed(self, samplerate: int, max_block: int) -> None:
        """Follow the output to another rate or block size. A new rate needs a
        new synth (TinySoundFont fixes it at construction); a held note is
        struck again on it, so the drone keeps sounding at the same pitch."""
        samplerate, max_block = int(samplerate), int(max_block)
        if samplerate <= 0 or max_block < 1:
            return
        rate_changed = samplerate != self.samplerate
        syn = sfid = None
        if rate_changed:
            # Loading the SoundFont is slow: do it before taking the lock.
            syn, sfid = self._new_synth(samplerate)
        old = None
        with self._ctl:
            if syn is not None:
                # Bring in what changed while it loaded, then swap.
                syn.program_select(0, sfid, 0, self._program)
                syn.set_tuning(0, _a4_to_tuning_semitones(self._a4))
                if self._note is not None:
                    syn.noteon(0, self._note, 100)
                old = self._syn
                self._syn, self._sfid = syn, sfid
                self.samplerate = samplerate
            if rate_changed or max_block != self._max_block:
                self._alloc(max_block)
        if old is not None:
            try:
                old.sounds_off()
            except Exception:
                pass

    # -- control ------------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        self._user_volume = _clamp01(volume)
        self._vol32 = np.float32(self._user_volume)

    def set_program(self, gm_program: int) -> None:
        # tsf's program_select raises (RuntimeError: channel_set_preset_number)
        # for a program outside the GM range [0,127]; clamp. The controller's
        # resolve_drone_voice already yields a valid program — this guards
        # direct / API callers.
        with self._ctl:
            self._program = max(0, min(127, int(gm_program)))
            self._syn.program_select(0, self._sfid, 0, self._program)
            if self._note is not None:           # re-voice a sounding note
                self._retrigger()

    def set_a4(self, a4: float) -> None:
        a4 = float(a4)
        if not math.isfinite(a4) or a4 <= 0.0:
            return            # a4 <= 0 -> _a4_to_tuning_semitones' log2 domain error
        with self._ctl:
            self._a4 = a4
            self._syn.set_tuning(0, _a4_to_tuning_semitones(self._a4))
            if self._note is not None:
                self._retrigger()

    def set_duck_target(self, level: float) -> None:
        """Engine D3 hook: per-hop duck gain target (in [duck_depth, 1.0])."""
        self._duck.set_target(level)

    def note_on(self, midi: int, velocity: int = 100) -> None:
        midi = max(0, min(127, int(midi)))
        with self._ctl:
            if self._note is not None:
                self._syn.noteoff(0, self._note)
            self._note = midi
            self._syn.noteon(0, midi, int(velocity))
            self._sounding = midi
            self._silent_blocks = 0

    def note_off(self) -> None:
        with self._ctl:
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
        with self._ctl:
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
        with self._ctl:
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
# SineDroneSource — pure-numpy sine, the vibrato-free PRECISION drone voice.
# ---------------------------------------------------------------------------
class SineDroneSource:
    """A single sustained pure sine — the drone's precision tuning reference.

    DroneSource plays sampled SF2 patches, which carry baked-in vibrato/chorus
    (a *musical* reference). This synthesises a numpy sine at exactly
    ``midi_to_freq(note, a4)``: no SoundFont, no vibrato, sub-cent stable — a
    reference you can hold a note against. It mirrors DroneSource's MixerSource +
    control API so DroneController drives either behind one interface, needs NO
    tinysoundfont (runs on system-python), is click-free (a short attack/release
    envelope over a continuous phase accumulator), and allocates nothing per
    block (preallocated phase / sine / mono buffers). ``active_midi`` is
    audible-tail aware (held through the release ramp) so the D3 mic-coordinator
    vote-excludes the sounding note exactly like the SF2 drone.
    """

    __test__ = False

    # Output peak below which a released voice is inaudible, and how long it must
    # stay there before active_midi flips to None (mirrors DroneSource).
    _SILENCE_PEAK = 1e-4
    _TWO_PI = 2.0 * math.pi

    def __init__(self, samplerate: int, max_block: int, *,
                 volume: float = 0.5, a4: float = A4_DEFAULT,
                 glide_ms: float = 5.0):
        if samplerate <= 0:
            raise ValueError(f"samplerate must be > 0, got {samplerate}")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        self.samplerate = int(samplerate)
        self._max_block = int(max_block)
        self._user_volume = _clamp01(volume)
        self._a4 = float(a4)

        # Phase accumulator: _inc = radians/sample of the current note (0 until a
        # note sounds); _phase wrapped to [0, 2pi) in float64 so a long sustained
        # hold keeps full precision.
        self._inc = 0.0
        self._phase = 0.0

        self._vol32 = np.float32(self._user_volume)

        # Note envelope (click-free attack/release) + the D3 duck — both glide.
        self._env = GainGlide(self._max_block, self.samplerate,
                              glide_ms=glide_ms, initial=0.0)
        self._duck = GainGlide(self._max_block, self.samplerate,
                               glide_ms=glide_ms, initial=1.0)

        self._note: Optional[int] = None      # currently note-on MIDI, or None
        self._sounding: Optional[int] = None   # audible MIDI incl. tail, or None
        self._silent_blocks = 0
        # Serialises pitch changes (GUI) against a rate change (audio-control
        # thread): both rewrite _inc from the rate. render() never takes it.
        self._ctl = threading.Lock()
        self._alloc(self._max_block)

    def _alloc(self, max_block: int) -> None:
        """Preallocated work buffers — zero per-block allocation. Phase + sine
        in float64 for accuracy; the mono output is float32 (the mixer's
        dtype)."""
        self._max_block = int(max_block)
        self._idx = np.arange(self._max_block, dtype=np.float64)
        self._phs = np.zeros(self._max_block, dtype=np.float64)
        self._sin = np.zeros(self._max_block, dtype=np.float64)
        self._mono = np.zeros(self._max_block, dtype=np.float32)
        self._env.reformat(self._max_block, self.samplerate)
        self._duck.reformat(self._max_block, self.samplerate)
        self._silence_confirm = max(
            1, int(0.05 * self.samplerate / self._max_block))

    def on_format_changed(self, samplerate: int, max_block: int) -> None:
        """Follow the output to another rate or block size, still sounding at
        the same frequency. The phase carries over, so there is no click."""
        samplerate, max_block = int(samplerate), int(max_block)
        if samplerate <= 0 or max_block < 1:
            return
        with self._ctl:
            rate_changed = samplerate != self.samplerate
            if rate_changed:
                # Radians per sample scale inversely with the rate; this also
                # keeps a releasing tail (no note) at its pitch.
                self._inc *= self.samplerate / samplerate
                self.samplerate = samplerate
            if rate_changed or max_block != self._max_block:
                self._alloc(max_block)

    # -- control ------------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        self._user_volume = _clamp01(volume)
        self._vol32 = np.float32(self._user_volume)

    def set_program(self, gm_program: int) -> None:
        # A pure sine has no GM patch — accept + ignore for API symmetry with
        # DroneSource. (DroneController rebuilds the source on an SF2<->sine
        # switch, so set_program is only ever a no-op self-select here.)
        return

    def set_a4(self, a4: float) -> None:
        a4 = float(a4)
        if not math.isfinite(a4) or a4 <= 0.0:
            return                 # match DroneSource.set_a4 + the engine
        with self._ctl:
            self._a4 = a4
            if self._note is not None:
                self._set_freq(self._note)   # phase stays continuous -> click-free

    def set_duck_target(self, level: float) -> None:
        """Engine D3 hook: per-hop duck gain target (in [duck_depth, 1.0])."""
        self._duck.set_target(level)

    def _set_freq(self, midi: int) -> None:
        freq = midi_to_freq(midi, self._a4)
        self._inc = self._TWO_PI * float(freq) / float(self.samplerate)

    def note_on(self, midi: int, velocity: int = 100) -> None:
        midi = max(0, min(127, int(midi)))
        with self._ctl:
            self._note = midi
            self._set_freq(midi)
        self._env.set_target(1.0)        # ramp up (click-free attack)
        self._sounding = midi
        self._silent_blocks = 0

    def note_off(self) -> None:
        self._note = None
        self._env.set_target(0.0)        # ramp down; _sounding clears on the tail

    def silence(self) -> None:
        """Force the audible-tail state to silent immediately WITHOUT precluding
        re-enable (mirrors DroneSource.silence): note off, release the envelope,
        and clear ``_sounding`` directly so active_midi can't stick non-None when
        no render runs to drain it (the D3-critical part). The envelope releases
        over its normal ~5 ms ramp rather than snapping — click-free, and moot
        when this is called with output stopped (nothing is rendering)."""
        self._note = None
        self._env.set_target(0.0)
        self._sounding = None
        self._silent_blocks = 0

    def close(self) -> None:
        self._note = None
        self._sounding = None
        self._env.set_target(0.0)

    @property
    def active_midi(self) -> Optional[int]:
        return self._sounding

    @property
    def duck_gain(self) -> float:
        """Current (glided) duck gain — observable for tests/telemetry."""
        return self._duck.gain

    # -- MixerSource render -------------------------------------------------
    def render(self, out: np.ndarray, frames: int, t0: int) -> None:
        if frames <= 0:
            return
        n = self._max_block if frames >= self._max_block else frames

        if n == self._max_block:
            idx, phs, sin64, mono = self._idx, self._phs, self._sin, self._mono
        else:
            idx, phs = self._idx[:n], self._phs[:n]
            sin64, mono = self._sin[:n], self._mono[:n]

        # sine[i] = sin(_phase + i*_inc), generated in float64, cast into the
        # float32 mono buffer in place (no per-block allocation).
        np.multiply(idx, self._inc, out=phs)
        phs += self._phase
        np.sin(phs, out=sin64)
        mono[:] = sin64
        # Advance + wrap the running phase so the next block continues seamlessly
        # (and a freq change between blocks stays phase-continuous -> no click).
        self._phase = math.fmod(self._phase + n * self._inc, self._TWO_PI)

        # Envelope (attack/release) FIRST, so a released note actually decays to
        # silence before the audible-tail check reads its peak.
        self._env.apply(mono, n)

        # Audible-tail tracking on the post-envelope, pre-volume/pre-duck signal
        # (max/min avoid an np.abs temporary — zero per-block alloc), mirroring
        # DroneSource so D3 sees the same active_midi lifecycle.
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

        # User volume then the duck glide — both on the drone's OWN buffer, never
        # the shared accumulator (the duck must affect only the drone).
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
        self._source: "Optional[DroneSource | SineDroneSource]" = None
        self._max_block = int(getattr(mixer, "max_block", 2048))
        # Set when a SoundFont voice could not load and the pure-sine voice
        # was used instead (see _ensure_source_or_sine).
        self.synth_fallback = False
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

    def _format(self) -> tuple[int, int]:
        """(rate, block) to build a source at: the mixer's output format once
        one is known, else what this controller was constructed with. A built
        source follows later format changes through the mixer."""
        sr = int(getattr(self._mixer, "samplerate", 0) or 0)
        if sr <= 0:
            return self._samplerate, self._max_block
        return sr, int(getattr(self._mixer, "max_block", self._max_block))

    def _ensure_source(self) -> "DroneSource | SineDroneSource":
        if self._source is None:
            sr, mb = self._format()
            if is_sine_voice(self.voice_id):
                self._source = SineDroneSource(
                    sr, mb, volume=self.volume, a4=self._a4)
            else:
                self._source = DroneSource(
                    sr, mb, resolve_drone_voice(self.voice_id),
                    volume=self.volume, a4=self._a4, sf2_path=self._sf2_path)
        return self._source

    def _ensure_source_or_sine(self) -> "DroneSource | SineDroneSource":
        """Build the selected voice, falling back to the pure-sine voice when
        the SoundFont synth cannot load (tinysoundfont not installed, or the
        .sf2 missing/unreadable). ``synth_fallback`` tells the GUI to explain
        why the voice changed."""
        try:
            return self._ensure_source()
        except (ImportError, OSError, RuntimeError):
            self._source = None
            self.voice_id = SINE_VOICE_ID
            self.synth_fallback = True
            return self._ensure_source()

    def _rebuild_source(self) -> None:
        """Tear the live source down and rebuild it from the current voice /
        volume / a4, preserving enabled + the playing note + mixer registration
        + duck attach. Needed when the source TYPE changes (SF2 DroneSource <->
        SineDroneSource are different classes); a new output rate or block
        size reaches the live source through the mixer instead
        (``on_format_changed``). Assumes a live source exists; callers guard."""
        was_enabled = self.enabled
        eng = self._engine
        if self._source is not None:
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
            src = self._ensure_source_or_sine()
            self._mixer.register(src)
            src.note_on(self._played_midi())
            if eng is not None and hasattr(eng, "attach_duck_consumer"):
                eng.attach_duck_consumer(src)

    def _emit(self) -> None:
        cb = self._on_state_changed
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    @property
    def source(self) -> "Optional[DroneSource | SineDroneSource]":
        return self._source

    def is_enabled(self) -> bool:
        return self.enabled

    # -- lifecycle ----------------------------------------------------------
    def set_enabled(self, on: bool) -> None:
        on = bool(on)
        if on == self.enabled:
            return
        if on:
            src = self._ensure_source_or_sine()
            self.enabled = True
            self._mixer.register(src)
            src.note_on(self._played_midi())
            eng = self._engine
            if eng is not None and hasattr(eng, "attach_duck_consumer"):
                eng.attach_duck_consumer(src)
        else:
            self.enabled = False
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
        # An SF2<->sine switch changes the source CLASS, which can't be
        # reprogrammed in place — rebuild it. A patch change within the SF2
        # catalog just reselects the program on the live synth.
        type_changed = is_sine_voice(voice_id) != is_sine_voice(self.voice_id)
        self.voice_id = voice_id
        if self._source is not None:
            if type_changed:
                self._rebuild_source()
            else:
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

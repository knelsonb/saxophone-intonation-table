"""
User-level configuration and custom-instrument storage.

Both live in the data directory (see sax_paths):
* ``config.json`` — per-user preferences: the history opt-in, the welcome
  flag, devices, the last session's instrument, A4, tab, tempo and so on.
* ``custom_instruments.json`` — list of user-defined instruments (key,
  transposition, names). Appended whenever the user adds a Custom… entry
  from the instrument combo.

The settings schema is declared once, on ``AppConfig``: every field carries
its default and the coercer that turns whatever the file holds into a valid
value. Loading and saving are generated from that declaration.

Damage is handled per kind. A missing file means "first launch". A file that
cannot be read (permissions, a Windows sharing lock) is left alone for the
whole session; ``ConfigStore`` remembers that. A file that can be read but
not understood is unrecoverable and defaults are saved over it.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from sax_atomic import atomic_write_json
from sax_paths import HISTORY_FILENAME, AppPaths
from sax_theme import coerce_theme_name

DEFAULT_LOG_FILENAME = HISTORY_FILENAME


# ---------------------------------------------------------------------------
# Helpers — tolerant coercion. A corrupt config field should fall back to
# the default, never crash the GUI on startup.
# ---------------------------------------------------------------------------
def _as_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off', ''):
            return False
    return default


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float('inf')) — Python's json.load accepts the
        # non-standard Infinity / NaN tokens by default, so a hand-edited or
        # corrupt config can hand us a non-finite float here. Degrade, don't crash.
        return default


def _as_int_list(v) -> list[int]:
    """Coerce a JSON value into a list of ints. Anything not iterable, or
    with non-numeric entries, degrades to an empty list (the GUI then
    falls back to its built-in default splitter sizes)."""
    if not isinstance(v, (list, tuple)):
        return []
    out: list[int] = []
    for entry in v:
        try:
            out.append(int(entry))
        except (TypeError, ValueError, OverflowError):
            # OverflowError guards int(float('inf')); a single bad entry
            # degrades the whole list to empty (GUI uses its default sizes).
            return []
    return out


_SAMPLERATE_PREF_ALLOWED = frozenset(
    ("auto", "44100", "48000", "88200", "96000", "192000")
)


def _as_samplerate_pref(v) -> str:
    """Coerce the saved value into one of SAMPLERATE_PREF_VALUES. A stale
    or malformed config silently degrades to 'auto' so the engine always
    has a usable policy."""
    if v is None:
        return "auto"
    s = str(v).strip().lower()
    if s in _SAMPLERATE_PREF_ALLOWED:
        return s
    return "auto"


# Allowed nav-shell tabs. The TUNER centerpiece is the safe default any
# stale / unknown value degrades to.
TAB_VALUES = frozenset(("tuner", "metro", "deck", "setup"))


def _as_active_tab(v) -> str:
    """Coerce the saved tab id into one of TAB_VALUES. A stale or malformed
    config silently degrades to 'tuner' so the app always opens on the
    tuner centerpiece. Same allowlist shape as _as_samplerate_pref."""
    if v is None:
        return "tuner"
    s = str(v).strip().lower()
    if s in TAB_VALUES:
        return s
    return "tuner"


def _as_float(v, default: float) -> float:
    """Coerce a JSON value to float, falling back to default. Callers clamp
    the result to the field's valid range (e.g. click_volume to [0, 1])."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    # Reject NaN/inf — a corrupt value must degrade to the default, not
    # propagate a non-finite gain into the audio path.
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


# Allowed metronome time signatures. Common time is the safe default any
# stale / unknown value degrades to. Exported so the GUI's time-sig
# selector and this coercer can never drift apart.
TIME_SIG_VALUES = frozenset(("2/4", "3/4", "4/4", "6/8"))


def _as_time_sig(v) -> str:
    """Coerce the saved time signature into one of TIME_SIG_VALUES, degrading
    to '4/4' for anything stale or malformed. Same allowlist shape as
    _as_active_tab / _as_samplerate_pref."""
    if v is None:
        return "4/4"
    s = str(v).strip()
    if s in TIME_SIG_VALUES:
        return s
    return "4/4"


# Pitch-detection response presets. Must match FILTER_PRESETS in
# sax_audio_engine (test_v121_regressions checks); kept here so loading the
# config does not import the audio engine.
FILTER_MODE_VALUES = frozenset(("fast", "normal", "slow"))


def _as_filter_mode(v) -> str:
    """Coerce the saved response preset into FILTER_MODE_VALUES. Anything
    else degrades to 'normal', the mode the engine would use anyway."""
    s = "" if v is None else str(v).strip().lower()
    return s if s in FILTER_MODE_VALUES else "normal"


DRONE_VOICE_DEFAULT = "organ"


def _as_drone_voice_id(v) -> str:
    """Coerce a saved drone voice id to one the drone catalog knows, else the
    organ preset. The catalog is imported lazily so loading the config does
    not pull in the synth module; if it cannot be imported at all, the value
    is kept (the drone controller falls back to organ for unknown ids)."""
    if v is None or isinstance(v, bool):
        return DRONE_VOICE_DEFAULT
    s = str(v).strip().lower()
    if not s:
        return DRONE_VOICE_DEFAULT
    try:
        from sax_drone import DRONE_FULL_GM, DRONE_PRESETS, DRONE_SYNTH_VOICES
    except Exception:
        return s
    known = {vid for vid, _name, _prog in
             (*DRONE_PRESETS, *DRONE_SYNTH_VOICES, *DRONE_FULL_GM)}
    return s if s in known else DRONE_VOICE_DEFAULT


def _as_str(v, default: str) -> str:
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Schema building blocks. A coercer receives the raw JSON value (None when the
# key is absent) and the field's default, and returns a valid value.
# ---------------------------------------------------------------------------
Coercer = Callable[[Any, Any], Any]


def _int_in(lo: Optional[int] = None, hi: Optional[int] = None) -> Coercer:
    def coerce(v, default):
        n = _as_int(v, default)
        if lo is not None:
            n = max(lo, n)
        if hi is not None:
            n = min(hi, n)
        return n
    return coerce


def _float_in(lo: float, hi: float) -> Coercer:
    def coerce(v, default):
        return max(lo, min(hi, _as_float(v, default)))
    return coerce


def _value_only(fn: Callable[[Any], Any]) -> Coercer:
    """Adapt a coercer that carries its own fallback (the allow-list ones)."""
    return lambda v, _default: fn(v)


def _setting(default: Any, coerce: Coercer):
    """Declare one config.json key: its default and its coercer. A list
    default is copied per instance, as dataclasses require."""
    meta = {"coerce": coerce}
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata=meta)
    return field(default=default, metadata=meta)


# The field order is the key order of config.json; append new fields at the
# end so files stay diffable across versions.
@dataclass(slots=True)
class AppConfig:
    welcome_shown: bool = _setting(False, _as_bool)
    persistence_enabled: bool = _setting(False, _as_bool)
    # Where the JSONL log lives when persistence is enabled. Empty = the
    # default history file in the data directory; the user can override.
    log_path: str = _setting("", _as_str)
    # Whether to accept measurements that fall outside the current
    # instrument's nominal fingered range. ON = overtones / altissimo /
    # accidentals get their own cells and appear in the matrix.
    # OFF = the audio callback silently drops them before they reach
    # `stats`, so the table stays bounded to the instrument's range.
    allow_out_of_range: bool = _setting(True, _as_bool)
    # Extra octave columns to show in matrix mode beyond the half-step-
    # beyond auto-rule. 0 = minimal (just the half-step rule); 1 = one
    # extra column of context on each side; etc. Power-user knob. The upper
    # bound keeps a hand-edited value from allocating millions of table
    # cells; 4 extra octaves already exceeds any instrument.
    matrix_extra_octaves: int = _setting(0, _int_in(0, 4))
    # 'auto' (width-driven), 'single' (force list), 'matrix' (force grid).
    layout_mode_preference: str = _setting('auto', _as_str)
    # Pitch-detection response. 'fast' = minimal smoothing, snappy tuner.
    # 'normal' = balanced (default). 'slow' = aggressive smoothing,
    # ideal for long tones / tuning analysis. See FILTER_PRESETS in
    # sax_audio_engine.py for the parameter values.
    filter_mode: str = _setting('normal', _value_only(_as_filter_mode))
    # Mic input gain in dB (default 0 = no change). Applied to the
    # silence-gate decision + level meter only (NOT the signal), so a quiet
    # mic can clear the detection floor without changing the cents readout.
    # See AudioEngine.set_mic_gain / the input callback in sax_audio_engine.py.
    mic_gain_db: float = _setting(0.0, _float_in(-24.0, 24.0))
    # Minimum measurement count for a note to appear in the table.
    # Hides notes you only blipped accidentally so the analysis only
    # shows notes you actually held. Defaults to 5 to match the
    # threshold the autotune feature already uses.
    min_n_visible: int = _setting(5, _int_in(0))
    # Show the live spectrogram + diagnostics panel beneath the tuner.
    # Off by default so casual users get the uncluttered tuner-only view;
    # power users flip it on to inspect FFT and runtime metrics.
    show_diagnostics: bool = _setting(False, _as_bool)

    # ---- v0.5.4 audio device selection -----------------------------------
    # Persistence-friendly device identifier. Index is intentionally NOT
    # stored — it shifts every time the user plugs in a USB hub.
    audio_device_name: str = _setting("", _as_str)
    audio_device_host_api: str = _setting("", _as_str)
    # 0 = auto-negotiate (try 44100 → device default → fallback list).
    audio_device_samplerate: int = _setting(0, _int_in(0))
    # v0.5.6: user-facing sample rate policy. 'auto' walks
    # SAMPLERATE_CANDIDATES in order (48k first); a specific value pins that
    # rate and surfaces UNSUPPORTED_RATE if the device refuses. Anything
    # unknown degrades to 'auto'.
    audio_samplerate_pref: str = _setting("auto", _value_only(_as_samplerate_pref))
    # First-run-only banner telling the user we're running off 44100.
    audio_sr_notice_shown: bool = _setting(False, _as_bool)
    # Power-user toggle: show every host API row in the picker instead of
    # collapsing duplicates by device name.
    show_all_host_apis: bool = _setting(False, _as_bool)
    # Opt-in low-latency exclusive mode. Off by default because WDM-KS is
    # the source of the GLE 0xAA busy crash on Windows.
    prefer_wdmks: bool = _setting(False, _as_bool)

    # ---- v0.6.3 audio OUTPUT device selection ------------------------
    # Mirror of the input-device fields above, for the new sd.OutputStream
    # that feeds the mixer (metronome / drone / pitch pipes / test tone).
    # Index is intentionally NOT stored — it reshuffles on every USB-hub
    # replug, exactly as for the input device.
    output_device_name: str = _setting("", _as_str)
    output_device_host_api: str = _setting("", _as_str)
    # 0 = auto-negotiate (try 44100 → device default → fallback list),
    # mirroring audio_device_samplerate.
    output_device_samplerate: int = _setting(0, _int_in(0))
    # D5: separate sd.OutputStream is the primary path; this opt-in enables
    # a same-device full-duplex sd.Stream where input == output device and
    # the user wants the lower-latency win. Off by default.
    output_prefer_duplex: bool = _setting(False, _as_bool)

    # ---- v0.5.5 session-state save-on-exit -------------------------------
    # QMainWindow.saveGeometry/saveState return QByteArrays; we base64-
    # encode them as plain ASCII strings so the config file round-trips
    # through json without binary noise. Empty strings = "no saved state,
    # use defaults" (first launch or wiped config).
    window_geometry: str = _setting("", _as_str)
    window_state: str = _setting("", _as_str)
    # Horizontal splitter widths between the tuner pane (left) and the
    # intonation table pane (right). Empty list = use built-in default.
    splitter_sizes: list[int] = _setting([], _value_only(_as_int_list))
    # The instrument the user last had selected, restored on the next
    # launch so they don't redo the picker every time.
    last_instrument_key: str = _setting("bb_tenor", _as_str)
    # The nickname text the user last typed. Persisted so the same horn
    # keeps its tag across sessions.
    last_nickname: str = _setting("", _as_str)
    # 'griff' (fingered notation, default) or 'klingend' (sounding pitch).
    last_display_mode: str = _setting("griff", _as_str)
    # Concert-A reference frequency, 430..450 Hz.
    last_a4_hz: int = _setting(440, _int_in(430, 450))
    # UI language. 'de' or 'en'. First launch picks from the system locale;
    # subsequent launches honor the user's last explicit choice.
    last_lang: str = _setting("en", _as_str)
    # v0.6.3: the nav-shell tab the user last had open, restored on the
    # next launch. One of TAB_VALUES; anything else degrades to the TUNER
    # centerpiece.
    last_active_tab: str = _setting("tuner", _value_only(_as_active_tab))

    # ---- v0.8.0 metronome -----------------------------------------------
    # Last tempo in BPM, restored on launch. Clamped to [30, 300] (the
    # metronome's supported range); default 100 matches the Android app.
    last_bpm: int = _setting(100, _int_in(30, 300))
    # Last time signature. One of TIME_SIG_VALUES; anything else degrades
    # to common time. Drives the accent-on-downbeat grouping.
    last_time_sig: str = _setting("4/4", _value_only(_as_time_sig))
    # Metronome click volume, 0.0 (silent) .. 1.0 (full).
    click_volume: float = _setting(1.0, _float_in(0.0, 1.0))

    # ---- v0.10.0 tape deck ----------------------------------------------
    # Max single-take recording length in seconds. Bounds the engine's
    # preallocated mic-capture buffer (np.zeros(int(s*sr), f32)) so a
    # forgotten record can't grow without limit; 300s ~= 53 MB f32 @44.1k.
    # Clamped to [1, 600]. Matches the engine's bounded-prealloc input tap.
    deck_max_seconds: int = _setting(300, _int_in(1, 600))
    # Full path of the WAV the user last EXPORTED a take to. Persisted so the
    # export file dialog reopens at that directory/filename. Empty = no
    # export yet (dialog opens at the platform default).
    last_take_path: str = _setting("", _as_str)
    # Directory for the deck's working/scratch take, if the controller spills
    # to disk rather than holding the take in RAM. Empty = the OS temp dir.
    # Kept distinct from last_take_path (export destination) so the scratch
    # lifecycle and the user's chosen save location never collide.
    deck_scratch_dir: str = _setting("", _as_str)

    # ---- v0.11.0 theme switching ------------------------------------------
    # UI theme: 'dark' (default, the original look), 'night' (red-shifted,
    # low-blue for dark-hall tuning), or 'light'. Coerced to a known theme
    # via sax_theme.coerce_theme_name, so a stale/unknown value degrades to
    # dark rather than leaving the app unstyled.
    theme: str = _setting("dark", _value_only(coerce_theme_name))

    # ---- drone (TUNER drone bar + SETUP voice picker) --------------------
    # Voice id as sax_drone understands it: a preset ("organ"...), the pure
    # sine ("sine") or a full-GM id ("gm0".."gm127"). Unknown ids degrade to
    # the organ preset.
    drone_voice_id: str = _setting(DRONE_VOICE_DEFAULT, _value_only(_as_drone_voice_id))
    # Drone level, 0.0 (silent) .. 1.0. Zero is a real setting and survives.
    drone_volume: float = _setting(0.5, _float_in(0.0, 1.0))
    # Offset from the reference note in semitones, clamped to [-12, 12].
    drone_semitones: int = _setting(0, _int_in(-12, 12))

    def effective_log_path(self, paths: Optional[AppPaths] = None) -> Optional[Path]:
        """The history file these settings ask for, or None when history is
        off. SAX_INTONATION_LOG_PATH is applied on top by AppPaths.history."""
        if not self.persistence_enabled:
            return None
        if self.log_path:
            return Path(self.log_path).expanduser()
        return (paths if paths is not None else active_store().paths).default_history


_FIELDS = fields(AppConfig)
FIELD_NAMES = tuple(f.name for f in _FIELDS)


def _default_of(f) -> Any:
    return f.default if f.default is not MISSING else f.default_factory()


def config_from_dict(data: dict) -> AppConfig:
    """Build a config from a parsed config.json. Missing or malformed keys
    fall back to their defaults; unknown keys are ignored here."""
    return AppConfig(**{
        f.name: f.metadata["coerce"](data.get(f.name), _default_of(f))
        for f in _FIELDS
    })


def config_to_dict(cfg: AppConfig) -> dict:
    """The config.json payload, keys in declaration order."""
    return asdict(cfg)


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


@dataclass
class CustomInstrument:
    key: str
    transp: int
    name_de: str
    name_en: str
    nickname: str = ""   # user-friendly tag, used by CSV slicing


# ---------------------------------------------------------------------------
# The store: config.json and custom_instruments.json in one data directory
# ---------------------------------------------------------------------------
class ConfigStore:
    """Reads and writes the settings and custom-instrument files, and
    remembers what went wrong with them during this session.

    State after ``load()``:
    * ``unreadable`` — config.json exists but could not be read. ``save()``
      leaves it alone for the rest of the session, so a transient lock cannot
      replace the user's settings with defaults.

    Keys this build does not know (written by a newer release) are kept and
    written back after the known ones, so opening an older build does not
    erase a newer build's settings.
    """

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.unreadable = False
        self._unknown_keys: dict[str, Any] = {}

    # -- config.json ------------------------------------------------------
    def load(self) -> AppConfig:
        self.unreadable = False
        self._unknown_keys = {}
        path = self.paths.config
        try:
            with path.open("rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return AppConfig()
        except OSError:
            self.unreadable = True
            return AppConfig()
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            # JSONDecodeError and UnicodeDecodeError are both ValueErrors.
            data = None
        if not isinstance(data, dict):
            return AppConfig()
        self._unknown_keys = {k: v for k, v in data.items()
                              if k not in FIELD_NAMES and _json_safe(v)}
        return config_from_dict(data)

    def save(self, cfg: AppConfig) -> bool:
        """Persist the settings atomically (temp file + os.replace), so two
        instances racing on shutdown cannot leave a half-written file. Returns
        False when nothing was written: the file was unreadable at load, or
        the write failed."""
        if self.unreadable:
            return False
        payload = config_to_dict(cfg)
        for key, value in self._unknown_keys.items():
            payload.setdefault(key, value)
        return atomic_write_json(self.paths.config, payload, tmp_prefix=".config.")

    # -- custom_instruments.json -----------------------------------------
    def load_customs(self, strict: bool = False) -> list[CustomInstrument]:
        """Load saved custom instruments.

        With ``strict=True`` a read failure (OSError) is raised instead of
        returning ``[]``. Callers that rewrite the file from the loaded list
        must use strict mode, otherwise one failed read followed by a save
        would delete every saved instrument."""
        path = self.paths.customs
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except OSError:
            if strict:
                raise
            return []
        except ValueError:
            # Corrupt JSON or undecodable bytes: nothing recoverable to keep.
            return []
        # A corrupt/hand-edited file may hold a non-list top-level (null, a
        # number, a bool); iterating it would raise on startup.
        if not isinstance(raw, list):
            return []
        out: list[CustomInstrument] = []
        for entry in raw:
            try:
                out.append(CustomInstrument(
                    key=str(entry["key"]),
                    transp=int(entry["transp"]),
                    name_de=str(entry.get("name_de", entry.get("name", entry["key"]))),
                    name_en=str(entry.get("name_en", entry.get("name", entry["key"]))),
                    nickname=str(entry.get("nickname", "")),
                ))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return out

    def save_customs(self, customs: list[CustomInstrument]) -> bool:
        """Persist custom instruments atomically — a crash mid-write must not
        nuke the user's instrument DB."""
        return atomic_write_json(
            self.paths.customs, [asdict(c) for c in customs],
            tmp_prefix=".customs.")


# ---------------------------------------------------------------------------
# The session's store. The window creates one from its AppPaths at start-up
# and installs it with use_store(); the module-level functions below are the
# entry points the rest of the app (and older call sites) use.
# ---------------------------------------------------------------------------
_active: Optional[ConfigStore] = None


def active_store() -> ConfigStore:
    """The store installed by use_store(), or one for the current HOME,
    created on first use."""
    global _active
    if _active is None:
        _active = ConfigStore(AppPaths.from_environment())
    return _active


def use_store(store: Optional[ConfigStore]) -> None:
    global _active
    _active = store


def load_config() -> AppConfig:
    return active_store().load()


def save_config(cfg: AppConfig) -> None:
    active_store().save(cfg)


def load_customs(strict: bool = False) -> list[CustomInstrument]:
    return active_store().load_customs(strict)


def save_customs(customs: list[CustomInstrument]) -> None:
    active_store().save_customs(customs)


def __getattr__(name: str):
    # Paths that used to be module constants computed at import. They now
    # follow the active store, so nothing is resolved before start-up.
    paths = {"CONFIG_DIR": lambda p: p.data_dir,
             "CONFIG_PATH": lambda p: p.config,
             "CUSTOMS_PATH": lambda p: p.customs}
    if name in paths:
        return paths[name](active_store().paths)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

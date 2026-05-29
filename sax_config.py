"""
User-level configuration and custom-instrument storage.

Both live under ``~/.intonation_analyzer/``:
* ``config.json`` — per-user preferences. Currently holds the persistent-log
  opt-in choice and a flag marking that the first-boot welcome dialog has
  been shown so it doesn't fire on every launch.
* ``custom_instruments.json`` — list of user-defined instruments (key,
  transposition, names). Appended whenever the user adds a Custom… entry
  from the instrument combo.

All operations are best-effort: a corrupt or unwritable file is treated as
"no config yet" rather than crashing the GUI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from sax_atomic import atomic_write_json
from sax_theme import coerce_theme_name


CONFIG_DIR = Path.home() / ".intonation_analyzer"
CONFIG_PATH = CONFIG_DIR / "config.json"
CUSTOMS_PATH = CONFIG_DIR / "custom_instruments.json"
DEFAULT_LOG_FILENAME = "log.jsonl"


@dataclass
class AppConfig:
    welcome_shown: bool = False
    persistence_enabled: bool = False
    # Where the JSONL log lives when persistence is enabled. Defaults to
    # ~/.intonation_analyzer/log.jsonl but the user can override.
    log_path: str = ""
    # Whether to accept measurements that fall outside the current
    # instrument's nominal fingered range. ON = overtones / altissimo /
    # accidentals get their own cells and appear in the matrix.
    # OFF = the audio callback silently drops them before they reach
    # `stats`, so the table stays bounded to the instrument's range.
    allow_out_of_range: bool = True
    # Extra octave columns to show in matrix mode beyond the half-step-
    # beyond auto-rule. 0 = minimal (just the half-step rule); 1 = one
    # extra column of context on each side; etc. Power-user knob.
    matrix_extra_octaves: int = 0
    # 'auto' (width-driven), 'single' (force list), 'matrix' (force grid).
    layout_mode_preference: str = 'auto'
    # Pitch-detection response. 'fast' = minimal smoothing, snappy tuner.
    # 'normal' = balanced (default). 'slow' = aggressive smoothing,
    # ideal for long tones / tuning analysis. See FILTER_PRESETS in
    # sax_audio_engine.py for the parameter values.
    filter_mode: str = 'normal'
    # Mic input gain in dB (default 0 = no change). Applied to the
    # silence-gate decision + level meter only (NOT the signal), so a quiet
    # mic can clear the detection floor without changing the cents readout.
    # See AudioEngine.set_mic_gain / the input callback in sax_audio_engine.py.
    mic_gain_db: float = 0.0
    # Minimum measurement count for a note to appear in the table.
    # Hides notes you only blipped accidentally so the analysis only
    # shows notes you actually held. Defaults to 5 to match the
    # threshold the autotune feature already uses.
    min_n_visible: int = 5
    # Show the live spectrogram + diagnostics panel beneath the tuner.
    # Off by default so casual users get the uncluttered tuner-only view;
    # power users flip it on to inspect FFT and runtime metrics.
    show_diagnostics: bool = False

    # ---- v0.5.4 audio device selection -----------------------------------
    # Persistence-friendly device identifier. Index is intentionally NOT
    # stored — it shifts every time the user plugs in a USB hub.
    audio_device_name: str = ""
    audio_device_host_api: str = ""
    # 0 = auto-negotiate (try 44100 → device default → fallback list).
    audio_device_samplerate: int = 0
    # v0.5.6: user-facing sample rate policy. 'auto' walks
    # SAMPLERATE_CANDIDATES highest-first; a specific value pins that
    # rate and surfaces UNSUPPORTED_RATE if the device refuses. Tolerant
    # default — see load_config below.
    audio_samplerate_pref: str = "auto"
    # First-run-only banner telling the user we're running off 44100.
    audio_sr_notice_shown: bool = False
    # Power-user toggle: show every host API row in the picker instead of
    # collapsing duplicates by device name.
    show_all_host_apis: bool = False
    # Opt-in low-latency exclusive mode. Off by default because WDM-KS is
    # the source of the GLE 0xAA busy crash on Windows.
    prefer_wdmks: bool = False

    # ---- v0.6.3 audio OUTPUT device selection (parity sprint) ------------
    # Mirror of the input-device fields above, for the new sd.OutputStream
    # that feeds the mixer (metronome / drone / pitch pipes / test tone).
    # Index is intentionally NOT stored — it reshuffles on every USB-hub
    # replug, exactly as for the input device.
    output_device_name: str = ""
    output_device_host_api: str = ""
    # 0 = auto-negotiate (try 44100 → device default → fallback list),
    # mirroring audio_device_samplerate.
    output_device_samplerate: int = 0
    # D5: separate sd.OutputStream is the primary path; this opt-in enables
    # a same-device full-duplex sd.Stream where input == output device and
    # the user wants the lower-latency win. Off by default.
    output_prefer_duplex: bool = False

    # ---- v0.5.5 session-state save-on-exit -------------------------------
    # QMainWindow.saveGeometry/saveState return QByteArrays; we base64-
    # encode them as plain ASCII strings so the config file round-trips
    # through json without binary noise. Empty strings = "no saved state,
    # use defaults" (first launch or wiped config).
    window_geometry: str = ""
    window_state: str = ""
    # Horizontal splitter widths between the tuner pane (left) and the
    # intonation table pane (right). Empty list = use built-in default.
    splitter_sizes: list[int] = field(default_factory=list)
    # The instrument the user last had selected, restored on the next
    # launch so they don't redo the picker every time.
    last_instrument_key: str = "bb_tenor"
    # The nickname text the user last typed. Persisted so the same horn
    # keeps its tag across sessions.
    last_nickname: str = ""
    # 'griff' (fingered notation, default) or 'klingend' (sounding pitch).
    last_display_mode: str = "griff"
    # Concert-A reference frequency, 430..450 Hz.
    last_a4_hz: int = 440
    # UI language. 'de' or 'en'. First launch picks from the system locale;
    # subsequent launches honor the user's last explicit choice.
    last_lang: str = "en"
    # v0.6.3 (parity sprint): the nav-shell tab the user last had open,
    # restored on the next launch. One of TAB_VALUES; anything else
    # degrades to the TUNER centerpiece. METRO/DECK/SETUP tabs land in
    # later sprints but the field is restored from Sprint 1 onward.
    last_active_tab: str = "tuner"

    # ---- v0.8.0 metronome (parity Sprint 2) ------------------------------
    # Last tempo in BPM, restored on launch. Clamped to [30, 300] (the
    # metronome's supported range); default 100 matches the Android app.
    last_bpm: int = 100
    # Last time signature. One of TIME_SIG_VALUES; anything else degrades
    # to common time. Drives the accent-on-downbeat grouping.
    last_time_sig: str = "4/4"
    # Metronome click volume, 0.0 (silent) .. 1.0 (full). Clamped on load.
    click_volume: float = 1.0

    # ---- v0.10.0 tape deck (parity Sprint 4) -----------------------------
    # Max single-take recording length in seconds. Bounds the engine's
    # preallocated mic-capture buffer (np.zeros(int(s*sr), f32)) so a
    # forgotten record can't grow without limit; 300s ~= 53 MB f32 @44.1k.
    # Clamped to [1, 600] on load. Matches Gandalf's bounded-prealloc tap.
    deck_max_seconds: int = 300
    # Full path of the WAV the user last EXPORTED a take to. Persisted so the
    # export file dialog reopens at that directory/filename. Empty = no
    # export yet (dialog opens at the platform default).
    last_take_path: str = ""
    # Directory for the deck's working/scratch take, if the controller spills
    # to disk rather than holding the take in RAM. Empty = the OS temp dir.
    # Kept distinct from last_take_path (export destination) so the scratch
    # lifecycle and the user's chosen save location never collide.
    deck_scratch_dir: str = ""

    # ---- v0.11.0 theme switching (parity Sprint 5) -----------------------
    # UI theme: 'dark' (default, the original look), 'night' (red-shifted,
    # low-blue for dark-hall tuning), or 'light'. Coerced to a known theme on
    # load via sax_theme.coerce_theme_name, so a stale/unknown value degrades
    # to dark rather than leaving the app unstyled.
    theme: str = "dark"

    def effective_log_path(self) -> Optional[Path]:
        if not self.persistence_enabled:
            return None
        if self.log_path:
            return Path(self.log_path).expanduser()
        return CONFIG_DIR / DEFAULT_LOG_FILENAME


@dataclass
class CustomInstrument:
    key: str
    transp: int
    name_de: str
    name_en: str
    nickname: str = ""   # user-friendly tag, used by CSV slicing


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


def _as_str(v, default: str) -> str:
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Config load / save
# ---------------------------------------------------------------------------
def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    if not isinstance(data, dict):
        return AppConfig()
    # Tolerate missing or malformed keys for forward/backward compat.
    return AppConfig(
        welcome_shown=_as_bool(data.get("welcome_shown"), False),
        persistence_enabled=_as_bool(data.get("persistence_enabled"), False),
        log_path=_as_str(data.get("log_path"), ""),
        allow_out_of_range=_as_bool(data.get("allow_out_of_range"), True),
        matrix_extra_octaves=_as_int(data.get("matrix_extra_octaves"), 0),
        layout_mode_preference=_as_str(
            data.get("layout_mode_preference"), "auto"),
        filter_mode=_as_str(data.get("filter_mode"), "normal"),
        mic_gain_db=max(-24.0, min(24.0, _as_float(data.get("mic_gain_db"), 0.0))),
        min_n_visible=max(0, _as_int(data.get("min_n_visible"), 5)),
        show_diagnostics=_as_bool(data.get("show_diagnostics"), False),
        audio_device_name=_as_str(data.get("audio_device_name"), ""),
        audio_device_host_api=_as_str(data.get("audio_device_host_api"), ""),
        audio_device_samplerate=max(
            0, _as_int(data.get("audio_device_samplerate"), 0)),
        audio_samplerate_pref=_as_samplerate_pref(
            data.get("audio_samplerate_pref")),
        audio_sr_notice_shown=_as_bool(
            data.get("audio_sr_notice_shown"), False),
        show_all_host_apis=_as_bool(data.get("show_all_host_apis"), False),
        prefer_wdmks=_as_bool(data.get("prefer_wdmks"), False),
        output_device_name=_as_str(data.get("output_device_name"), ""),
        output_device_host_api=_as_str(
            data.get("output_device_host_api"), ""),
        output_device_samplerate=max(
            0, _as_int(data.get("output_device_samplerate"), 0)),
        output_prefer_duplex=_as_bool(
            data.get("output_prefer_duplex"), False),
        window_geometry=_as_str(data.get("window_geometry"), ""),
        window_state=_as_str(data.get("window_state"), ""),
        splitter_sizes=_as_int_list(data.get("splitter_sizes")),
        last_instrument_key=_as_str(
            data.get("last_instrument_key"), "bb_tenor"),
        last_nickname=_as_str(data.get("last_nickname"), ""),
        last_display_mode=_as_str(
            data.get("last_display_mode"), "griff"),
        last_a4_hz=max(430, min(450, _as_int(data.get("last_a4_hz"), 440))),
        last_lang=_as_str(data.get("last_lang"), "en"),
        last_active_tab=_as_active_tab(data.get("last_active_tab")),
        last_bpm=max(30, min(300, _as_int(data.get("last_bpm"), 100))),
        last_time_sig=_as_time_sig(data.get("last_time_sig")),
        click_volume=max(0.0, min(1.0, _as_float(data.get("click_volume"), 1.0))),
        deck_max_seconds=max(1, min(600, _as_int(data.get("deck_max_seconds"), 300))),
        last_take_path=_as_str(data.get("last_take_path"), ""),
        deck_scratch_dir=_as_str(data.get("deck_scratch_dir"), ""),
        theme=coerce_theme_name(data.get("theme")),
    )


def save_config(cfg: AppConfig) -> None:
    """Persist the user config atomically.

    v0.5.7.8: switched from open-write-in-place to tempfile + os.replace
    so two app instances racing on shutdown can't leave a half-written
    JSON file behind. The temp file is created in CONFIG_DIR so
    os.replace stays a same-volume rename on Windows."""
    atomic_write_json(CONFIG_PATH, asdict(cfg), tmp_prefix=".config.")


# ---------------------------------------------------------------------------
# Custom instruments
# ---------------------------------------------------------------------------
def load_customs() -> list[CustomInstrument]:
    if not CUSTOMS_PATH.exists():
        return []
    try:
        with CUSTOMS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
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
            except (KeyError, TypeError, ValueError):
                continue
        return out
    except (OSError, json.JSONDecodeError):
        return []


def save_customs(customs: list[CustomInstrument]) -> None:
    """Persist custom instruments atomically (tempfile + os.replace), same
    pattern as save_config — a crash mid-write must not nuke the user's
    instrument DB."""
    atomic_write_json(
        CUSTOMS_PATH, [asdict(c) for c in customs], tmp_prefix=".customs.")

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
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


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
    except (TypeError, ValueError):
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
        except (TypeError, ValueError):
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
    )


def save_config(cfg: AppConfig) -> None:
    """Persist the user config atomically.

    v0.5.7.8: switched from open-write-in-place to tempfile + os.replace
    so two app instances racing on shutdown can't leave a half-written
    JSON file behind (mirrors the pattern in
    sax_instruments._write_overrides_atomic). The temp file is created in
    CONFIG_DIR so os.replace stays a same-volume rename on Windows."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".config.", suffix=".tmp", dir=str(CONFIG_DIR))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        os.replace(tmp_path, CONFIG_PATH)
        tmp_path = None
    except OSError:
        pass
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CUSTOMS_PATH.open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in customs], f, indent=2)
    except OSError:
        pass

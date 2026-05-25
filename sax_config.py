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
    # ideal for long tones / tuning analysis. See _FILTER_PRESETS in
    # sax_intonation_gui.py for the parameter values.
    filter_mode: str = 'normal'
    # Minimum measurement count for a note to appear in the table.
    # Hides notes you only blipped accidentally so the analysis only
    # shows notes you actually held. Defaults to 5 to match the
    # threshold the autotune feature already uses.
    min_n_visible: int = 5

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
# Config load / save
# ---------------------------------------------------------------------------
def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Tolerate missing keys for forward/backward compat.
        return AppConfig(
            welcome_shown=bool(data.get("welcome_shown", False)),
            persistence_enabled=bool(data.get("persistence_enabled", False)),
            log_path=str(data.get("log_path", "")),
            allow_out_of_range=bool(data.get("allow_out_of_range", True)),
            matrix_extra_octaves=int(data.get("matrix_extra_octaves", 0)),
            layout_mode_preference=str(
                data.get("layout_mode_preference", "auto")),
            filter_mode=str(data.get("filter_mode", "normal")),
            min_n_visible=max(0, int(data.get("min_n_visible", 5))),
        )
    except (OSError, json.JSONDecodeError):
        return AppConfig()


def save_config(cfg: AppConfig) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
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

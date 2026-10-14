"""Where the app keeps its files.

Everything the app writes for the user lives in one data directory,
``~/.intonation_analyzer/``:

* ``config.json`` — settings (persistence.config.ConfigStore);
* ``custom_instruments.json`` — user-defined instruments
  (persistence.config.ConfigStore);
* ``instrument_ranges.json`` — per-instrument range overrides
  (catalog.instruments);
* ``log.jsonl`` — the measurement history, when the user opted in, with older
  months beside it as ``log-YYYY-MM.jsonl`` (intonation.persistence.history);
* ``crash.log`` — tracebacks of unexpected errors (intonation.crash).

An ``AppPaths`` is resolved once at start-up from the environment (HOME, or
USERPROFILE on Windows, and ``SAX_INTONATION_LOG_PATH``) and handed to the
objects that read and write those files. Resolving it touches no file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DATA_DIR_NAME = ".intonation_analyzer"
CONFIG_FILENAME = "config.json"
CUSTOMS_FILENAME = "custom_instruments.json"
RANGES_FILENAME = "instrument_ranges.json"
HISTORY_FILENAME = "log.jsonl"
CRASH_LOG_FILENAME = "crash.log"

# Names a history file that always wins over the settings; a power-user and
# test hook kept from the first releases.
HISTORY_OVERRIDE_ENV = "SAX_INTONATION_LOG_PATH"


@dataclass(frozen=True)
class AppPaths:
    """The data directory and the history override, plus the file names
    derived from them."""

    data_dir: Path
    history_override: Optional[Path] = None

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None
                         ) -> "AppPaths":
        """Resolve from the current HOME and SAX_INTONATION_LOG_PATH.

        Called at start-up rather than at import, so a process (or a test)
        that changes HOME before building the window gets its own files."""
        env = os.environ if environ is None else environ
        override = env.get(HISTORY_OVERRIDE_ENV) or None
        return cls(
            data_dir=Path.home() / DATA_DIR_NAME,
            history_override=Path(override).expanduser() if override else None,
        )

    @property
    def config(self) -> Path:
        return self.data_dir / CONFIG_FILENAME

    @property
    def customs(self) -> Path:
        return self.data_dir / CUSTOMS_FILENAME

    @property
    def ranges(self) -> Path:
        return self.data_dir / RANGES_FILENAME

    @property
    def default_history(self) -> Path:
        return self.data_dir / HISTORY_FILENAME

    @property
    def crash_log(self) -> Path:
        return self.data_dir / CRASH_LOG_FILENAME

    def history(self, chosen: Optional[Path]) -> Optional[Path]:
        """The history file to append to: the environment override if set,
        else ``chosen`` (the settings' choice; None keeps history in memory)."""
        return self.history_override if self.history_override is not None else chosen

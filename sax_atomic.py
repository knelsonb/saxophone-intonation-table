"""Atomic JSON write helper shared by config / customs / instrument-range
persistence. One implementation; three call sites.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2,
                      tmp_prefix: str = ".tmp.") -> bool:
    """Write `payload` as JSON to `path` atomically (tempfile + os.replace)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=tmp_prefix, suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp_path, path)
        tmp_path = None
        return True
    except OSError:
        return False
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

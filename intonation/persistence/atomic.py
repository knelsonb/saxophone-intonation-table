"""Atomic write helpers.

``atomic_write_json`` is shared by config / customs / instrument-range
persistence and reports failure as False. ``atomic_open`` is for user-facing
exports, where the caller needs the real error to explain it.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Optional


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
            # allow_nan=False: never persist NaN/Infinity. They are non-standard
            # JSON (external tools choke) and would round-trip back as a
            # non-finite float. A non-finite slipping into the payload raises
            # ValueError here -> caught below -> the save fails safely (the old
            # file is left intact by the atomic replace) rather than writing
            # garbage or crashing the caller.
            json.dump(payload, f, indent=indent, allow_nan=False)
        os.replace(tmp_path, path)
        tmp_path = None
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@contextmanager
def atomic_open(path: Path | str, *, binary: bool = False,
                encoding: str = "utf-8",
                newline: Optional[str] = None) -> Iterator[IO]:
    """Open a temporary file beside ``path`` for writing and move it over
    ``path`` when the block finishes without an exception.

    An export that fails halfway (disk full, the old file locked by another
    program) leaves the previous file as it was. The temporary file is
    created with "x" rather than mkstemp so it gets the user's normal
    permissions, which the replaced file then keeps. Errors propagate: the
    caller turns them into a message the user can act on.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    if binary:
        f = open(tmp, "xb")
    else:
        f = open(tmp, "x", encoding=encoding, newline=newline)
    try:
        with f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

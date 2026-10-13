"""A tiny undo stack for temporary attribute and environment changes.

The harness has to be usable from any test (and from an ad-hoc script), so it
cannot lean on pytest's ``monkeypatch`` fixture. Every change made while an app
is alive goes through a ``PatchStack`` and is reverted, in reverse order, when
the app closes.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_MISSING = object()


class PatchStack:
    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        old = obj.__dict__.get(name, _MISSING) if hasattr(obj, "__dict__") else _MISSING
        if old is _MISSING:
            # Inherited (or slot / sip) attribute: remember "was not own".
            self._undo.append(("delattr", obj, name))
        else:
            self._undo.append(("setattr", obj, name, old))
        setattr(obj, name, value)

    def setenv(self, name: str, value: str | None) -> None:
        self._undo.append(("env", name, os.environ.get(name)))
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def hook(self, fn) -> None:
        """Register an arbitrary callable to run on undo."""
        self._undo.append(("call", fn))

    def undo(self) -> None:
        while self._undo:
            item = self._undo.pop()
            kind = item[0]
            try:
                if kind == "setattr":
                    setattr(item[1], item[2], item[3])
                elif kind == "delattr":
                    try:
                        delattr(item[1], item[2])
                    except (AttributeError, TypeError):
                        pass
                elif kind == "env":
                    if item[2] is None:
                        os.environ.pop(item[1], None)
                    else:
                        os.environ[item[1]] = item[2]
                elif kind == "call":
                    item[1]()
            except Exception as exc:  # pragma: no cover - best effort restore
                print(f"harness: undo failed for {item[:2]}: {exc}", file=sys.stderr)

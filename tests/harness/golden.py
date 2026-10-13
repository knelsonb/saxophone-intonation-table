"""Golden-file comparison.

``assert_golden(actual, name)`` compares ``actual`` to ``tests/golden/<name>``.
Run with ``UPDATE_GOLDEN=1`` to (re)write the files instead of comparing, then
read the diff before committing: a golden file is a promise about app output.

``actual`` may be a ``Path`` (file content is compared), ``str``/``bytes`` or a
JSON-able object (rendered with sorted keys, 2-space indent).

PDFs are not compared byte for byte (they embed timestamps and ids) and the venv
has no PDF text extractor; the harness exports the *structured data handed to
the PDF renderer* instead (``app.export.pdf_model``) and that is what is golden.
Charts likewise: the render model, never pixels.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable, Union

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"

Normalizer = Union[Callable[[str], str], Iterable[tuple]]

# Wall-clock text that legitimately differs on every run.
DATE_DMY = (re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}"), "<DD.MM.YYYY HH:MM>")
DATE_ISO_MIN = (re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"), "<YYYY-MM-DD HH:MM>")
STANDARD = (DATE_DMY, DATE_ISO_MIN)


def fixture_path(name: str) -> Path:
    """Path of an input fixture under tests/golden/."""
    p = GOLDEN_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"golden fixture {name!r} not found at {p}")
    return p


def _to_text(actual) -> str:
    if isinstance(actual, Path):
        raw = actual.read_bytes()
    elif isinstance(actual, bytes):
        raw = actual
    elif isinstance(actual, str):
        raw = actual.encode("utf-8")
    else:
        raw = (json.dumps(actual, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    text = raw.decode("utf-8")
    return text.replace("\r\n", "\n")


def _normalize(text: str, normalize: Normalizer | None) -> str:
    if normalize is None:
        return text
    if callable(normalize):
        return normalize(text)
    for pattern, repl in normalize:
        text = pattern.sub(repl, text) if hasattr(pattern, "sub") else text.replace(pattern, repl)
    return text


def assert_golden(actual, name: str, normalize: Normalizer | None = STANDARD) -> None:
    text = _normalize(_to_text(actual), normalize)
    target = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        return
    if not target.exists():
        raise AssertionError(
            f"golden file {name!r} does not exist. Run with UPDATE_GOLDEN=1 to create it, "
            f"then review {target}.")
    expected = target.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text != expected:
        diff = "".join(difflib.unified_diff(
            expected.splitlines(True), text.splitlines(True),
            fromfile=f"golden/{name}", tofile="actual", n=2))
        raise AssertionError(
            f"output differs from golden file {name!r} "
            f"(UPDATE_GOLDEN=1 rewrites it):\n{diff[:4000]}")

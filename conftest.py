"""
Root conftest.py — shared pytest fixtures and path setup.

Pytest options live in pyproject.toml ([tool.pytest.ini_options]). Only add
things here that are genuinely shared; per-file setup stays in the test file
itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so test files can import production
# modules (intonation.audio.engine, intonation.persistence.history, …) without
# installing the package or setting PYTHONPATH manually.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_log_path(tmp_path: Path) -> Path:
    """Return a ``pathlib.Path`` pointing to a writable temporary file.

    The file does not exist yet — callers can write to it or pass it to
    ``MeasurementLog(path=...)`` directly.  ``tmp_path`` is a pytest
    built-in that is cleaned up automatically after each test.
    """
    return tmp_path / "test_log.json"


@pytest.fixture(autouse=True)
def _fresh_persistence_stores(monkeypatch):
    """Give every test fresh range-override and crash-log stores.

    The stores remember what happened to their files: RangeOverrides caches
    what it read. Shared across tests, one test's state would leak into the
    next and the result would depend on test order. Each store is created
    again on first use, from the HOME the test runs with. (Settings need no
    reset: every ConfigStore is created by whoever uses it.)
    """
    from intonation import crash
    from intonation.catalog import instruments
    monkeypatch.setattr(instruments, "_active_overrides", None)
    monkeypatch.setattr(crash, "_log_path", None)


@pytest.fixture()
def load_engine_with_sounddevice(monkeypatch):
    """Return ``load(fake_sd)``: a private copy of ``intonation.audio.engine``
    executed against a fake ``sounddevice`` module.

    The engine decides at import time whether audio is available, so a fake
    device layer has to be in ``sys.modules`` while the module body runs. The
    copy is installed through monkeypatch and removed again after the test; the
    real ``intonation.audio.engine`` that every other test imported is never
    reloaded, so its classes and enums keep a single identity across the session.
    The copy also replaces the ``engine`` attribute of ``intonation.audio``:
    ``from intonation.audio import engine`` reads that attribute, not
    ``sys.modules``.
    """
    import intonation.audio

    def load(fake_sd):
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
        spec = importlib.util.spec_from_file_location(
            "intonation.audio.engine", _REPO_ROOT / "intonation" / "audio" / "engine.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "intonation.audio.engine", module)
        monkeypatch.setattr(intonation.audio, "engine", module, raising=False)
        spec.loader.exec_module(module)
        return module

    return load

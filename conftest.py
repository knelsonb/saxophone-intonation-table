"""
Root conftest.py — shared pytest fixtures and path setup.

Pytest options live in pyproject.toml ([tool.pytest.ini_options]). Only add
things here that are genuinely shared; per-file setup stays in the test file
itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so test files can import production
# modules (sax_audio_engine, sax_intonation_log, …) without installing the
# package or setting PYTHONPATH manually.
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
def _config_unreadable_starts_false(monkeypatch):
    """Reset sax_config's "last load found an unreadable file" flag per test.

    load_config() records an unreadable config in a module global, and
    save_config() refuses to write while it is set so a locked file is never
    overwritten. A test that provokes that state would otherwise make every
    later save_config() in the session a silent no-op, and the result would
    depend on test order.
    """
    import sax_config
    monkeypatch.setattr(sax_config, "_config_unreadable", False)

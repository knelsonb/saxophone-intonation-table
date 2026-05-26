"""
Root conftest.py — shared pytest fixtures and path setup.

Kept deliberately minimal: no pytest.ini / pyproject.toml yet (Phase 1).
Only add things here that are genuinely shared; per-file setup stays in
the test file itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
import tempfile
import pytest


@pytest.fixture()
def tmp_log_path(tmp_path: Path) -> Path:
    """Return a ``pathlib.Path`` pointing to a writable temporary file.

    The file does not exist yet — callers can write to it or pass it to
    ``MeasurementLog(path=...)`` directly.  ``tmp_path`` is a pytest
    built-in that is cleaned up automatically after each test.
    """
    return tmp_path / "test_log.json"

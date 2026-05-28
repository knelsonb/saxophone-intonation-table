"""Suite integration for ``scripts/gui_smoke.py`` (co-owned: Frodo authors the
GUI assertions, Treebeard folds it into the pytest suite / CI gate).

The smoke BUILDS a real ``MainWindow`` under offscreen Qt, so it is run as an
isolated SUBPROCESS rather than imported: the script sets ``HOME`` to a
throwaway dir, monkeypatches ``QMessageBox.information``, and constructs a
``QApplication`` — all process-global side effects we do not want leaking into
the pytest interpreter or other tests. A subprocess gives us a clean room and a
single, unambiguous pass/fail signal (exit 0/1).

Skips cleanly when PyQt6 is absent (the always-runnable numpy/logic suite on
system python); runs for real on the PyQt6-equipped venv / CI runner. Because
the subprocess reuses ``sys.executable``, the guard and the child interpreter
agree on whether PyQt6 is present.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Only meaningful where PyQt6 is installed; otherwise the child build can't run.
pytest.importorskip('PyQt6', reason='PyQt6 not installed; skipping GUI smoke')

_SMOKE = Path(__file__).parent / "scripts" / "gui_smoke.py"


def test_gui_smoke_construction():
    """The offscreen MainWindow construction smoke must exit 0 (all nav-shell
    assertions pass). Surfaces the script's own output on failure so a broken
    assertion is readable in the pytest report."""
    assert _SMOKE.exists(), f"gui_smoke script missing at {_SMOKE}"
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, str(_SMOKE)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, (
        "GUI smoke failed (exit "
        f"{proc.returncode}).\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )

"""Regression tests for the v1.2.1 fixes.

Pure-logic cases run in-process. Cases that need a real ``MainWindow`` build
it in a subprocess under offscreen Qt, as test_gui_smoke does: the window
reads ``~/.intonation_analyzer`` at import time, so each case gets its own
throwaway HOME holding the config it needs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import sax_config
from sax_intonation_log import MeasurementLog

_REPO = Path(__file__).resolve().parent
_HAS_QT = importlib.util.find_spec("PyQt6") is not None
requires_qt = pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not installed")


# ---------------------------------------------------------------------------
# Subprocess MainWindow harness
# ---------------------------------------------------------------------------
_GUI_PRELUDE = '''\
import json, os, sys
home, repo = sys.argv[1], sys.argv[2]
os.environ["HOME"] = home
os.environ["USERPROFILE"] = home
sys.path.insert(0, repo)
import sax_intonation_gui as G
from PyQt6.QtWidgets import QApplication
G.QMessageBox.information = staticmethod(lambda *a, **k: None)
app = QApplication.instance() or QApplication([])
win = G.MainWindow()
out = {}
'''

_GUI_EPILOGUE = '''
sys.stdout.write("RESULT " + json.dumps(out) + "\\n")
sys.stdout.flush()
# Skip interpreter teardown: the engine's poller threads and Qt objects are
# irrelevant to the result and only slow the exit down.
os._exit(0)
'''


def _run_gui(tmp_path: Path, body: str, config: dict | None = None,
             files: dict[str, str] | None = None) -> dict:
    """Build a MainWindow in a fresh process, run ``body`` against ``win``
    and return the ``out`` dict it fills in."""
    home = tmp_path / "home"
    cfg_dir = home / ".intonation_analyzer"
    cfg_dir.mkdir(parents=True)
    cfg = {"welcome_shown": True}
    cfg.update(config or {})
    (cfg_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for name, text in (files or {}).items():
        (cfg_dir / name).write_text(text, encoding="utf-8")
    script = tmp_path / "gui_case.py"
    script.write_text(_GUI_PRELUDE + textwrap.dedent(body) + _GUI_EPILOGUE,
                      encoding="utf-8")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, str(script), str(home), str(_REPO)],
        capture_output=True, text=True, env=env, timeout=120)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"GUI case failed (exit {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return json.loads(lines[-1][len("RESULT "):])


# ---------------------------------------------------------------------------
# 1.1  Saved A4 and nickname reach the first run and every A4 consumer
# ---------------------------------------------------------------------------
def test_clearing_label_on_current_run():
    log = MeasurementLog()
    log.start_run(instrument="eb_alto", a4_hz=440.0, label="MarkVI")
    log.set_current_run_metadata(label="")
    assert log.current_run().label == ""


def test_metadata_without_label_keeps_label():
    log = MeasurementLog()
    log.start_run(instrument="eb_alto", a4_hz=440.0, label="MarkVI")
    log.set_current_run_metadata(maker="Selmer")
    run = log.current_run()
    assert run.label == "MarkVI" and run.maker == "Selmer"


@requires_qt
def test_first_run_carries_restored_a4_and_nickname(tmp_path):
    out = _run_gui(tmp_path, '''
        if not G.AUDIO_OK:
            out["skip"] = True
        else:
            run = win._log.current_run()
            out["run_a4"] = run.a4_hz
            out["run_label"] = run.label
        out["engine_a4"] = win._engine.a4
        out["drone_a4"] = (win._drone_ctrl._a4
                           if win._drone_ctrl is not None else None)
        out["a4_combo"] = win._a4_combo.currentData()
        out["setup_a4_combo"] = win._setup_a4_combo.currentData()
    ''', config={"last_a4_hz": 442, "last_nickname": "MarkVI"})
    assert out["engine_a4"] == 442.0
    assert out["a4_combo"] == 442
    assert out["setup_a4_combo"] == 442
    if out["drone_a4"] is not None:
        assert out["drone_a4"] == 442.0, "drone still tuned to the default A4"
    if not out.get("skip"):
        assert out["run_a4"] == 442.0, "first run stamped before the A4 restore"
        assert out["run_label"] == "MarkVI"


@requires_qt
def test_nickname_change_starts_a_new_run(tmp_path):
    out = _run_gui(tmp_path, '''
        if not G.AUDIO_OK:
            out["skip"] = True
        else:
            first = win._log.current_run()
            # One reading on the first horn, so the run is not coalesced.
            win._log.add_measurement(69, 69, 0.0, 440.0)
            win._nick_edit.setText("Selmer")
            win._on_nickname_changed()
            second = win._log.current_run()
            # A focus change with the same text must not open another run.
            win._on_nickname_changed()
            out["same_after_refocus"] = (
                win._log.current_run().run_id == second.run_id)
            win._log.add_measurement(69, 69, 0.0, 440.0)
            win._nick_edit.setText("")
            win._on_nickname_changed()
            third = win._log.current_run()
            runs = {r.run_id: r.label for r in win._log.runs()}
            out["first_label_kept"] = runs.get(first.run_id)
            out["new_run"] = second.run_id != first.run_id
            out["second_label"] = second.label
            out["third_label"] = third.label
            out["third_new"] = third.run_id != second.run_id
    ''', config={"last_nickname": "MarkVI"})
    if out.get("skip"):
        pytest.skip("no audio backend: runs are not opened")
    assert out["new_run"], "nickname change relabelled the existing run"
    assert out["first_label_kept"] == "MarkVI"
    assert out["second_label"] == "Selmer"
    assert out["same_after_refocus"]
    assert out["third_new"] and out["third_label"] == ""


# ---------------------------------------------------------------------------
# 1.2  Matrix columns follow the live octave span; right-click uses it
# ---------------------------------------------------------------------------
@requires_qt
def test_matrix_span_follows_display_and_played_notes(tmp_path):
    out = _run_gui(tmp_path, '''
        win.resize(1400, 900)
        win.show()
        app.processEvents()
        ctrl = win._table_ctrl
        win._refresh_table()
        out["mode"] = win._layout_mode
        out["griff_span"] = list(ctrl.painted_octave_span())
        # Display toggle: fingered -> sounding shifts tenor down 14 semitones.
        win._disp_combo.setCurrentIndex(1)
        win._refresh_table()
        out["klingend_live"] = list(ctrl._matrix_octave_range())
        out["klingend_painted"] = list(ctrl.painted_octave_span())
        # A played note far below the range widens the live span.
        st = G.NoteStats()
        for _ in range(6):
            st.add(0.0, 55.0)
        with win._lock:
            win.stats[33] = st
        win._refresh_table()
        lo, hi = ctrl.painted_octave_span()
        out["played_live"] = list(ctrl._matrix_octave_range())
        out["played_painted"] = [lo, hi]
        out["columns"] = win._table.columnCount()

        # Right-click a played cell: the menu must name that note.
        captured = []

        class _Menu(G.QMenu):
            def exec(self, *a, **k):
                captured.append(self.actions()[0].data())
                return None

        G.QMenu = _Menu
        midi = 50   # D3 sounding, inside the tenor's sounding range
        st2 = G.NoteStats()
        for _ in range(6):
            st2.add(0.0, 146.8)
        with win._lock:
            win.stats[midi] = st2
        win._refresh_table()
        lo, _hi = ctrl.painted_octave_span()
        row, col = midi % 12, (midi // 12 - 1) - lo
        rect = win._table.visualRect(win._table.model().index(row, col))
        win._on_table_context_menu(rect.center())
        out["menu_midi"] = captured[0] if captured else None
    ''', config={"layout_mode_preference": "matrix",
                 "last_instrument_key": "bb_tenor",
                 "last_display_mode": "griff"})
    assert out["mode"] == "matrix"
    assert out["klingend_painted"] == out["klingend_live"]
    assert out["klingend_painted"] != out["griff_span"]
    assert out["played_painted"] == out["played_live"]
    assert out["played_painted"][0] <= 33 // 12 - 1
    lo, hi = out["played_painted"]
    assert out["columns"] == hi - lo + 1
    assert out["menu_midi"] == 50

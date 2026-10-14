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
from sax_paths import AppPaths

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
from sax_dialogs import AudioPickerDialog, RangeEditorDialog
from sax_stats import NoteStats
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
        out["drone_a4"] = (win._drone_bar.controller._a4
                           if win._drone_bar.controller is not None else None)
        out["a4_combo"] = win._toolbar._a4_combo.currentData()
        out["setup_a4_combo"] = win._setup_tab._a4_combo.currentData()
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
            win._toolbar._nick_edit.setText("Selmer")
            win._toolbar._nick_edit.editingFinished.emit()
            second = win._log.current_run()
            # A focus change with the same text must not open another run.
            win._toolbar._nick_edit.editingFinished.emit()
            out["same_after_refocus"] = (
                win._log.current_run().run_id == second.run_id)
            win._log.add_measurement(69, 69, 0.0, 440.0)
            win._toolbar._nick_edit.setText("")
            win._toolbar._nick_edit.editingFinished.emit()
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
        ctrl = win._tuner_tab.table_controller
        win._tuner_tab.refresh_table()
        out["mode"] = win._tuner_tab.layout_mode
        out["griff_span"] = list(ctrl.painted_octave_span())
        # Display toggle: fingered -> sounding shifts tenor down 14 semitones.
        win._toolbar._disp_combo.setCurrentIndex(1)
        win._tuner_tab.refresh_table()
        out["klingend_live"] = list(ctrl._matrix_octave_range())
        out["klingend_painted"] = list(ctrl.painted_octave_span())
        # A played note far below the range widens the live span.
        st = NoteStats()
        for _ in range(6):
            st.add(0.0, 55.0)
        win._session.stats[33] = st
        win._tuner_tab.refresh_table()
        lo, hi = ctrl.painted_octave_span()
        out["played_live"] = list(ctrl._matrix_octave_range())
        out["played_painted"] = [lo, hi]
        out["columns"] = win._tuner_tab._table.columnCount()

        # Right-click a played cell: the menu must name that note.
        captured = []

        import sax_tab_tuner

        class _Menu(sax_tab_tuner.QMenu):
            def exec(self, *a, **k):
                captured.append(self.actions()[0].data())
                return None

        sax_tab_tuner.QMenu = _Menu
        midi = 50   # D3 sounding, inside the tenor's sounding range
        st2 = NoteStats()
        for _ in range(6):
            st2.add(0.0, 146.8)
        win._session.stats[midi] = st2
        win._tuner_tab.refresh_table()
        lo, _hi = ctrl.painted_octave_span()
        row, col = midi % 12, (midi // 12 - 1) - lo
        rect = win._tuner_tab._table.visualRect(win._tuner_tab._table.model().index(row, col))
        win._tuner_tab._on_context_menu(rect.center())
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


# ---------------------------------------------------------------------------
# 1.3  Range overrides: strict save, honest editor, parser accepts its output
# ---------------------------------------------------------------------------
@pytest.fixture()
def ranges_file(tmp_path, monkeypatch):
    import sax_instruments
    path = tmp_path / "instrument_ranges.json"
    monkeypatch.setattr(sax_instruments, "_active_overrides",
                        sax_instruments.RangeOverrides(path))
    return path


def _two_overrides(path: Path) -> str:
    text = json.dumps({"version": 1,
                       "ranges": {"eb_alto": [50, 80], "bb_tenor": [46, 78]}})
    path.write_text(text, encoding="utf-8")
    return text


def test_save_override_keeps_other_keys(ranges_file):
    import sax_instruments
    _two_overrides(ranges_file)
    sax_instruments.save_range_override("eb_bari", 45, 77)
    data = json.loads(ranges_file.read_text(encoding="utf-8"))["ranges"]
    assert data == {"eb_alto": [50, 80], "bb_tenor": [46, 78],
                    "eb_bari": [45, 77]}
    assert sax_instruments.fingered_range("eb_bari") == (45, 77)


def test_save_override_never_rewrites_after_failed_read(ranges_file,
                                                        monkeypatch):
    import sax_instruments
    original = _two_overrides(ranges_file)
    real_open = Path.open

    def locked(self, *args, **kwargs):
        if self == ranges_file:
            raise PermissionError(13, "locked", str(self))
        return real_open(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "open", locked)
        with pytest.raises(sax_instruments.RangeOverridesUnreadable):
            sax_instruments.save_range_override("eb_bari", 45, 77)
        with pytest.raises(sax_instruments.RangeOverridesUnreadable):
            sax_instruments.clear_range_override("eb_alto")
    assert ranges_file.read_text(encoding="utf-8") == original


def test_save_override_never_rewrites_corrupt_file(ranges_file):
    import sax_instruments
    ranges_file.write_text("{not json", encoding="utf-8")
    with pytest.raises(sax_instruments.RangeOverridesUnreadable):
        sax_instruments.save_range_override("eb_bari", 45, 77)
    assert ranges_file.read_text(encoding="utf-8") == "{not json"


def test_save_override_reports_write_failure(ranges_file, monkeypatch):
    import sax_instruments
    _two_overrides(ranges_file)
    monkeypatch.setattr(sax_instruments, "atomic_write_json",
                        lambda *a, **k: False)
    with pytest.raises(OSError):
        sax_instruments.save_range_override("eb_bari", 45, 77)


def test_save_override_rejects_out_of_midi_range(ranges_file):
    import sax_instruments
    with pytest.raises(ValueError):
        sax_instruments.save_range_override("eb_bari", 40, 148)
    assert not ranges_file.exists()


@requires_qt
@pytest.mark.parametrize("text,midi", [
    ("A#/Bb3", 58), ("Bb3", 58), ("A#3", 58), ("H3", 59), ("h4", 71),
    ("B3", 59), ("C-1", 0), ("G9", 127), ("F#/Gb4", 66),
])
def test_note_name_parser_accepts_display_forms(text, midi):
    from sax_pitch import note_name_to_midi
    assert note_name_to_midi(text) == midi


@requires_qt
@pytest.mark.parametrize("text", ["A#/Cb3", "X3", "Bb", "3", "G#10", ""])
def test_note_name_parser_rejects_garbage(text):
    from sax_pitch import note_name_to_midi
    assert note_name_to_midi(text) is None


@requires_qt
def test_note_name_parser_round_trips_every_displayed_name():
    from sax_pitch import midi_note_name, note_name_to_midi
    bad = [m for m in range(128) if note_name_to_midi(midi_note_name(m)) != m]
    assert not bad, f"names the editor shows but cannot read back: {bad}"


@requires_qt
def test_range_editor_stays_open_when_nothing_was_saved(tmp_path):
    out = _run_gui(tmp_path, '''
        import sax_instruments
        def make():
            return RangeEditorDialog(
                win, win._t, "eb_bari", "Bari", 45, 77, 45, 77, True,
                display="klingend", transp=-21)
        # Sounding 127 is fingered 148: outside MIDI, must not save.
        dlg = make()
        dlg._hi_spin.setValue(127)
        dlg._on_save()
        out["oor_result"] = dlg.result()
        out["oor_error"] = dlg._error_lbl.isVisibleTo(dlg)
        # Unreadable overrides file: the save fails and the editor says so.
        dlg = make()
        dlg._hi_spin.setValue(50)
        dlg._on_save()
        out["fail_result"] = dlg.result()
        out["fail_error"] = dlg._error_lbl.isVisibleTo(dlg)
        out["file"] = sax_instruments._RANGES_FILE.read_text(encoding="utf-8")
    ''', files={"instrument_ranges.json": "{not json"})
    assert out["oor_result"] == 0 and out["oor_error"]
    assert out["fail_result"] == 0 and out["fail_error"]
    assert out["file"] == "{not json"


# ---------------------------------------------------------------------------
# 1.4  Drone voice, volume and semitones persist; volume 0 survives
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path):
    cfg_dir = tmp_path / ".intonation_analyzer"
    cfg_dir.mkdir(parents=True)
    return sax_config.ConfigStore(AppPaths(cfg_dir))


@pytest.fixture()
def isolated_config(store):
    return store.paths.config


def test_drone_settings_round_trip(store):
    store.save(sax_config.AppConfig(
        drone_voice_id="cello", drone_volume=0.0, drone_semitones=-5))
    cfg = store.load()
    assert (cfg.drone_voice_id, cfg.drone_volume, cfg.drone_semitones) == (
        "cello", 0.0, -5)


@pytest.mark.parametrize("raw,expected", [
    ("sine", "sine"), ("GM42", "gm42"), ("warmpad", "warmpad"),
    ("gm128", "organ"), ("bogus", "organ"), (17, "organ"), (None, "organ"),
    ("", "organ"),
])
def test_drone_voice_id_coercion(store, isolated_config, raw, expected):
    isolated_config.write_text(json.dumps({"drone_voice_id": raw}),
                               encoding="utf-8")
    assert store.load().drone_voice_id == expected


@pytest.mark.parametrize("raw,expected", [
    (0.0, 0.0), (0, 0.0), (0.25, 0.25), (2.0, 1.0), (-1, 0.0),
    ("loud", 0.5), (None, 0.5),
])
def test_drone_volume_coercion(store, isolated_config, raw, expected):
    isolated_config.write_text(json.dumps({"drone_volume": raw}),
                               encoding="utf-8")
    assert store.load().drone_volume == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    (3, 3), (-12, -12), (30, 12), (-30, -12), ("x", 0), (None, 0),
])
def test_drone_semitones_coercion(store, isolated_config, raw, expected):
    isolated_config.write_text(json.dumps({"drone_semitones": raw}),
                               encoding="utf-8")
    assert store.load().drone_semitones == expected


@requires_qt
def test_window_restores_saved_drone_settings(tmp_path):
    out = _run_gui(tmp_path, '''
        out["volume"] = win._drone_bar._volume
        out["slider"] = win._drone_bar._vol_slider.value()
        out["semitones"] = win._drone_bar.semitones
        out["voice"] = win._drone_bar.voice_id
        out["setup_voice"] = win._setup_tab._voice_combo.currentData()
    ''', config={"drone_voice_id": "cello", "drone_volume": 0.0,
                 "drone_semitones": -3})
    assert out["volume"] == 0.0 and out["slider"] == 0
    assert out["semitones"] == -3
    assert out["voice"] == "cello" and out["setup_voice"] == "cello"


# ---------------------------------------------------------------------------
# 1.5  CSV formula injection
# ---------------------------------------------------------------------------
import csv  # noqa: E402

_HOSTILE = {
    "label": '=HYPERLINK("http://x","y")',
    "maker": "+cmd|' /C calc'!A0",
    "model": "@SUM(A1:A2)",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hostile_log() -> MeasurementLog:
    log = MeasurementLog()
    log.start_run(instrument="eb_alto", a4_hz=440.0, **_HOSTILE)
    log.add_measurement(69, 78, -3.25, 439.2)
    return log


def test_raw_export_neutralises_text_cells(tmp_path):
    out = tmp_path / "raw.csv"
    _hostile_log().export_csv(out, mode="raw")
    row = _read_rows(out)[0]
    assert row["nickname"] == "'" + _HOSTILE["label"]
    assert row["maker"] == "'" + _HOSTILE["maker"]
    assert row["model"] == "'" + _HOSTILE["model"]
    # Numbers stay numbers: a negative cents value is not a formula.
    assert row["cents"] == "-3.250"
    assert row["instrument"] == "eb_alto"


@pytest.mark.parametrize("mode,column", [
    ("per_run_note", "nickname"), ("per_run_note", "maker"),
    ("per_instrument_note", "nicknames"), ("per_nickname_note", "nickname"),
])
def test_aggregate_exports_neutralise_text_cells(tmp_path, mode, column):
    out = tmp_path / f"{mode}.csv"
    _hostile_log().export_csv(out, mode=mode)
    cell = _read_rows(out)[0][column]
    assert cell.startswith("'"), f"{mode}.{column} exported as {cell!r}"


@pytest.mark.parametrize("value", [
    "=1+1", "+1", "-1", "@x", "\t=1", "\r=1", "'=1", "''=1",
])
def test_csv_safe_round_trips(value):
    from sax_intonation_log import _csv_import_text, _csv_safe
    escaped = _csv_safe(value)
    assert escaped.startswith("'")
    # Import strips control characters, so compare against that form.
    assert _csv_import_text(escaped) == value.replace("\t", "").replace("\r", "")


def test_csv_safe_leaves_plain_text_alone():
    from sax_intonation_log import _csv_import_text, _csv_safe
    for value in ("MarkVI", "'quoted", "2026-10-13T10:00:00", ""):
        assert _csv_safe(value) == value
        assert _csv_import_text(value) == value


def test_export_import_export_is_stable_and_safe(tmp_path):
    first = tmp_path / "first.csv"
    _hostile_log().export_csv(first, mode="raw")
    reloaded = MeasurementLog()
    runs, rows, skipped = reloaded.import_raw_csv(first)
    assert (runs, rows, skipped) == (1, 1, 0)
    run = reloaded.runs()[0]
    assert (run.label, run.maker, run.model) == (
        _HOSTILE["label"], _HOSTILE["maker"], _HOSTILE["model"])
    second = tmp_path / "second.csv"
    reloaded.export_csv(second, mode="raw")
    assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")


def test_import_strips_control_characters(tmp_path):
    header = MeasurementLog.RAW_CSV_HEADER
    row = ["2026-10-13T10:00:00", "abc123", "2026-10-13T10:00:00", "eb_alto",
           "=evil\x07\x1b[31m", "440.00", "69", "A4", "78", "F#5", "1.000",
           "440.250", "Sel\x00mer", "Mark\x85VI"]
    path = tmp_path / "in.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header, row])
    log = MeasurementLog()
    assert log.import_raw_csv(path)[:2] == (1, 1)
    run = log.runs()[0]
    assert run.label == "=evil[31m"
    assert (run.maker, run.model) == ("Selmer", "MarkVI")
    out = tmp_path / "out.csv"
    log.export_csv(out, mode="raw")
    assert _read_rows(out)[0]["nickname"] == "'=evil[31m"


# ---------------------------------------------------------------------------
# 1.7  One threshold rule, applied to the rounded value the user sees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cents,band", [
    (0.0, "in"), (4.5, "in"), (5.4, "in"), (-5.4, "in"), (5.5, "close"),
    (8.0, "close"), (12.4, "close"), (-12.5, "close"), (12.6, "off"),
    (15.0, "off"), (-40.0, "off"), (float("nan"), "off"), (None, "off"),
])
def test_cents_band(cents, band):
    from sax_theme import cents_band
    assert cents_band(cents) == band


@requires_qt
def test_cents_band_agrees_with_printed_value():
    from sax_pitch import format_cents
    from sax_theme import cents_band
    for tenth in range(-300, 301):
        value = tenth / 10.0
        printed = int(format_cents(value, 440.0, 48000))
        assert cents_band(value) == cents_band(printed), value


@requires_qt
def test_bar_delegate_and_table_use_shared_thresholds(tmp_path):
    out = _run_gui(tmp_path, '''
        from PyQt6.QtCore import Qt, QRect
        from PyQt6.QtGui import QImage, QPainter, QStandardItemModel, QStandardItem
        from PyQt6.QtWidgets import QStyleOptionViewItem

        import sax_widgets as W
        roles = []
        real_themed = W._themed

        def spy(role, alpha=255):
            roles.append(role)
            return real_themed(role, alpha)

        W._themed = spy
        delegate = W.CentBarDelegate()
        model = QStandardItemModel(1, 1)
        img = QImage(200, 30, QImage.Format.Format_ARGB32)
        bars = {}
        for cents in (8.0, 15.0):
            item = QStandardItem()
            item.setData({"cents": cents, "freq": 440.0},
                         Qt.ItemDataRole.UserRole)
            model.setItem(0, 0, item)
            opt = QStyleOptionViewItem()
            opt.rect = QRect(0, 0, 200, 30)
            roles.clear()
            p = QPainter(img)
            delegate.paint(p, opt, model.index(0, 0))
            p.end()
            bars[str(cents)] = sorted(set(roles))
        W._themed = real_themed
        out["bars"] = bars

        # Single-column table: +5.4 prints "+5" and must be green.
        win._cfg.layout_mode_preference = "single"
        win._cfg.min_n_visible = 0
        st = NoteStats()
        for _ in range(5):
            st.add(5.4, 440.0)
        win._session.stats.clear()
        win._session.stats[69] = st
        win._tuner_tab.refresh_table()
        out["mean_text"] = win._tuner_tab._table.item(0, 2).text()
        out["mean_color"] = win._tuner_tab._table.item(0, 2).foreground().color().name()
    ''')
    assert "warn" in out["bars"]["8.0"] and "bad" not in out["bars"]["8.0"]
    assert "bad" in out["bars"]["15.0"]
    assert out["mean_text"] == "+5"
    assert out["mean_color"] == "#3a9e5f"


# ---------------------------------------------------------------------------
# 1.8  Unhandled exceptions are logged and reported instead of aborting
# ---------------------------------------------------------------------------
def test_crash_log_is_capped_at_line_boundary(tmp_path):
    import sax_crash
    log = tmp_path / "crash.log"
    for i in range(200):
        sax_crash.append_crash_log(f"Traceback {i}\n  line\nError: {i}",
                                   path=log, max_bytes=2048)
    data = log.read_bytes()
    assert len(data) <= 2048
    # Truncation drops whole lines only: no partial line at the start.
    import re
    for line in data.decode("utf-8").splitlines():
        assert re.fullmatch(r"=== .* ===|Traceback \d+|  line|Error: \d+|",
                            line), line
    assert b"Error: 199" in data and b"Error: 0\n" not in data


def test_excepthook_logs_and_reports(tmp_path, monkeypatch):
    import sax_crash
    monkeypatch.setattr(sax_crash, "_log_path", tmp_path / "cfg" / "crash.log")
    shown = []
    hook = sax_crash.make_excepthook(lambda path, details: shown.append(
        (path, details)))
    try:
        raise ValueError("boom-in-slot")
    except ValueError:
        hook(*sys.exc_info())
    log = tmp_path / "cfg" / "crash.log"
    assert "boom-in-slot" in log.read_text(encoding="utf-8")
    assert shown and shown[0][0] == log and "boom-in-slot" in shown[0][1]


def test_excepthook_survives_unwritable_log(tmp_path, monkeypatch):
    import sax_crash
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sax_crash, "_log_path", blocker / "crash.log")
    shown = []
    hook = sax_crash.make_excepthook(lambda path, details: shown.append(path))
    try:
        raise ValueError("boom")
    except ValueError:
        hook(*sys.exc_info())
    assert shown == [None]


def _run_script(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "case.py"
    script.write_text(
        "import os, sys\n"
        f"os.environ['HOME'] = {str(home)!r}\n"
        f"os.environ['USERPROFILE'] = {str(home)!r}\n"
        f"sys.path.insert(0, {str(_REPO)!r})\n" + textwrap.dedent(body),
        encoding="utf-8")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, env=env, timeout=120)


def test_excepthook_before_qapplication_logs(tmp_path):
    proc = _run_script(tmp_path, '''
        import sax_crash
        sax_crash.install()
        raise RuntimeError("boom-before-qt")
    ''')
    log = tmp_path / "home" / ".intonation_analyzer" / "crash.log"
    assert proc.returncode == 1
    assert "boom-before-qt" in log.read_text(encoding="utf-8")


def test_thread_exceptions_are_logged(tmp_path):
    proc = _run_script(tmp_path, '''
        import threading, sax_crash
        sax_crash.install()
        t = threading.Thread(target=lambda: 1 / 0, name="worker-x")
        t.start(); t.join()
    ''')
    log = tmp_path / "home" / ".intonation_analyzer" / "crash.log"
    assert proc.returncode == 0
    text = log.read_text(encoding="utf-8")
    assert "worker-x" in text and "ZeroDivisionError" in text


@requires_qt
def test_exception_in_slot_shows_dialog_instead_of_aborting(tmp_path):
    proc = _run_script(tmp_path, '''
        import sax_crash
        sax_crash.install()
        sax_crash.set_lang_source(lambda: "de")
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication, QMessageBox
        shown = []
        QMessageBox.exec = lambda self: shown.append(
            (self.windowTitle(), self.text())) or 0
        app = QApplication([])

        def bad():
            raise RuntimeError("boom-in-slot")

        QTimer.singleShot(0, bad)
        for _ in range(5):
            app.processEvents()
        print("SHOWN", shown, flush=True)
        os._exit(0)
    ''')
    assert proc.returncode == 0, proc.stderr
    assert "Unerwarteter Fehler" in proc.stdout
    assert "crash.log" in proc.stdout
    log = tmp_path / "home" / ".intonation_analyzer" / "crash.log"
    assert "boom-in-slot" in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1.9  Auto rate prefers 48k/44.1k; hidden diagnostics stop their timers
# ---------------------------------------------------------------------------
class _AcceptAll:
    @staticmethod
    def check_input_settings(**kwargs):
        return None

    @staticmethod
    def check_output_settings(**kwargs):
        return None


def _probe_engine(monkeypatch, pref):
    import sax_audio_engine as E
    monkeypatch.setattr(E, "sd", _AcceptAll, raising=False)
    eng = E.AudioEngine()
    eng.set_samplerate_pref(pref)
    dev = E.DeviceInfo(index=0, name="Mic", host_api="ALSA",
                       max_input_channels=1, default_samplerate=192000.0)
    return eng, dev


def test_auto_rate_tries_48k_then_44k_first(monkeypatch):
    eng, dev = _probe_engine(monkeypatch, "auto")
    assert eng._negotiate_sample_rate(dev)[:2] == [48000, 44100]
    assert eng._negotiate_output_sample_rate(dev)[:2] == [48000, 44100]


def test_pinned_rate_is_still_honoured(monkeypatch):
    eng, dev = _probe_engine(monkeypatch, "192000")
    assert eng._negotiate_sample_rate(dev) == [192000]


@requires_qt
def test_diagnostics_timers_run_only_while_visible(tmp_path):
    out = _run_gui(tmp_path, '''
        win.resize(1400, 900)
        win.show()
        app.processEvents()
        def state():
            return [win._tuner_tab._spectro._timer.isActive(),
                    win._tuner_tab._data_panel._timer.isActive()]
        out["hidden"] = state()
        win._tuner_tab._cb_diag.setChecked(True)
        app.processEvents()
        out["shown"] = state()
        win._tabs.setCurrentIndex(win._tab_keys.index("setup"))
        app.processEvents()
        out["other_tab"] = state()
        win._tabs.setCurrentIndex(win._tab_keys.index("tuner"))
        app.processEvents()
        out["back"] = state()
        win._tuner_tab._cb_diag.setChecked(False)
        app.processEvents()
        out["off"] = state()
    ''', config={"show_diagnostics": False})
    assert out["hidden"] == [False, False]
    assert out["shown"] == [True, True]
    assert out["other_tab"] == [False, False]
    assert out["back"] == [True, True]
    assert out["off"] == [False, False]


# ---------------------------------------------------------------------------
# 1.11  Picker cursor, language toggle, unknown filter mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("fast", "fast"), ("SLOW", "slow"), ("turbo", "normal"), (None, "normal"),
    (3, "normal"),
])
def test_filter_mode_coercion(store, isolated_config, raw, expected):
    isolated_config.write_text(json.dumps({"filter_mode": raw}),
                               encoding="utf-8")
    assert store.load().filter_mode == expected


def test_filter_mode_values_match_engine_presets():
    from sax_audio_engine import FILTER_PRESETS
    assert sax_config.FILTER_MODE_VALUES == set(FILTER_PRESETS)


@requires_qt
def test_response_combo_shows_the_engine_mode(tmp_path):
    out = _run_gui(tmp_path, '''
        from sax_controls import response_combo
        out["engine"] = win._engine.filter_mode
        out["toolbar"] = win._toolbar._filter_combo.currentData()
        out["setup"] = win._setup_tab._response_combo.currentData()
        # Even with a bad value in memory, a freshly filled combo shows the
        # mode the engine runs.
        win._cfg.filter_mode = "turbo"
        combo = response_combo(win._tr, win._engine.filter_mode)
        out["fresh"] = combo.currentData()
    ''', config={"filter_mode": "turbo"})
    assert out["engine"] == "normal"
    assert out["toolbar"] == out["setup"] == out["fresh"] == "normal"


@requires_qt
def test_clicking_active_language_keeps_it_checked(tmp_path):
    out = _run_gui(tmp_path, '''
        from PyQt6.QtWidgets import QDialog, QPushButton, QVBoxLayout
        dlg = QDialog(win)
        win._add_dialog_lang_toggle(dlg, QVBoxLayout(dlg))
        btns = {b.text(): b for b in dlg.findChildren(QPushButton)}
        btns[win.lang.upper()].click()
        out["lang"] = win.lang
        out["checked"] = sorted(t for t, b in btns.items() if b.isChecked())
    ''', config={"last_lang": "en"})
    assert out["lang"] == "en"
    assert out["checked"] == ["EN"]


@requires_qt
def test_picker_restores_wait_cursor_when_closed_mid_reopen(tmp_path):
    out = _run_gui(tmp_path, '''
        import threading
        from PyQt6.QtGui import QGuiApplication
        from sax_audio_engine import DeviceInfo
        release = threading.Event()
        dev = DeviceInfo(index=0, name="Mic", host_api="ALSA",
                         max_input_channels=1, default_samplerate=48000.0)
        win._engine.get_active_device = lambda: dev
        win._engine.stop = lambda: None
        win._engine.open_device = lambda *a, **k: release.wait(10)
        dlg = AudioPickerDialog(win, win._t, win._engine, win._cfg, None, win._store)
        idx = dlg._sr_combo.findData("48000")
        dlg._sr_combo.setCurrentIndex(idx)
        app.processEvents()
        out["busy_while_reopening"] = QGuiApplication.overrideCursor() is not None
        dlg.reject()
        app.processEvents()
        out["after_close"] = QGuiApplication.overrideCursor() is not None
        release.set()
    ''')
    assert out["busy_while_reopening"]
    assert not out["after_close"], "wait cursor left on after the dialog closed"

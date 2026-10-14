"""Unit contracts for the persistence layer: the settings schema and store,
the data-directory paths and the month-split history files.

The player-facing behaviour (a damaged or locked settings file, last year's
history moving out of the live log, damaged history records) is covered by
tests/acceptance; these tests pin the mechanics underneath, which a scenario
cannot reach: rollback of a failed rotation, a torn last line, the open-file
count per reading, and that importing the modules touches no file.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from intonation.persistence import history as hist
from intonation.persistence.config import AppConfig, ConfigStore
from intonation.persistence.history import Measurement, MeasurementLog, RunMeta
from intonation.persistence.paths import AppPaths

_REPO = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Settings schema
# ---------------------------------------------------------------------------
def test_every_setting_declares_a_coercer():
    from dataclasses import fields
    assert all(callable(f.metadata.get("coerce")) for f in fields(AppConfig))


def test_assigning_an_undeclared_setting_fails_loudly():
    """A typo in a setting name used to create an attribute that was never
    saved; with slots it raises."""
    with pytest.raises(AttributeError):
        AppConfig().drone_volumme = 0.3


def test_saved_keys_keep_the_release_order(tmp_path):
    """config.json lists the keys in the order every release wrote them; new
    settings are appended at the end."""
    v12 = json.loads((_REPO / "tests/golden/config_v1.2.json").read_text(encoding="utf-8"))
    store = ConfigStore(AppPaths(tmp_path / "data"))
    store.save(AppConfig())
    saved = list(json.loads(store.paths.config.read_text(encoding="utf-8")))
    assert saved[:len(v12)] == list(v12)
    assert saved[len(v12):] == ["drone_voice_id", "drone_volume", "drone_semitones"]


def test_settings_from_a_newer_release_survive_a_save(tmp_path):
    """Opening the file with an older build keeps keys it does not know."""
    store = ConfigStore(AppPaths(tmp_path / "data"))
    store.paths.config.parent.mkdir(parents=True)
    store.paths.config.write_text(json.dumps({
        "last_bpm": 90, "future_toggle": True, "future_list": [1, "a"],
        "future_nan": float("nan")}), encoding="utf-8")
    cfg = store.load()
    assert not hasattr(cfg, "future_toggle")
    cfg.last_bpm = 120
    assert store.save(cfg)
    saved = json.loads(store.paths.config.read_text(encoding="utf-8"))
    assert saved["last_bpm"] == 120
    assert (saved["future_toggle"], saved["future_list"]) == (True, [1, "a"])
    # A value strict JSON cannot hold is dropped rather than blocking the save.
    assert "future_nan" not in saved
    assert list(saved)[-2:] == ["future_toggle", "future_list"]


def test_a_damaged_settings_file_is_kept_once_and_replaced(tmp_path):
    store = ConfigStore(AppPaths(tmp_path / "data"))
    store.paths.config.parent.mkdir(parents=True)
    store.paths.config.write_bytes(b'{"last_bpm": 90,,}')
    assert store.load() == AppConfig()
    backup = store.corrupt_backup
    assert backup is not None and backup.name.startswith("config.json.corrupt-")
    assert backup.read_bytes() == b'{"last_bpm": 90,,}'
    assert not store.unreadable and store.save(AppConfig(last_bpm=95))

    again = ConfigStore(AppPaths(tmp_path / "data"))
    assert again.load().last_bpm == 95 and again.corrupt_backup is None


def test_a_damaged_settings_file_that_cannot_be_copied_is_left_alone(tmp_path, monkeypatch):
    store = ConfigStore(AppPaths(tmp_path / "data"))
    store.paths.config.parent.mkdir(parents=True)
    store.paths.config.write_bytes(b"[]")
    real_open = Path.open

    def no_new_files(self, mode="r", *args, **kwargs):
        if "x" in mode:
            raise PermissionError(13, "read-only folder", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_new_files)
    store.load()
    assert store.unreadable and store.corrupt_backup is None
    assert not store.save(AppConfig())
    assert store.paths.config.read_bytes() == b"[]"


def test_paths_follow_home_and_the_history_override(tmp_path):
    paths = AppPaths.from_environment({"SAX_INTONATION_LOG_PATH": str(tmp_path / "h.jsonl")})
    assert paths.data_dir == Path.home() / ".intonation_analyzer"
    assert paths.history(None) == tmp_path / "h.jsonl"
    plain = AppPaths(tmp_path)
    assert plain.history(None) is None
    assert AppConfig(persistence_enabled=True).effective_log_path(plain) \
        == tmp_path / "log.jsonl"
    assert AppConfig(persistence_enabled=True, log_path=str(tmp_path / "x.jsonl")) \
        .effective_log_path(plain) == tmp_path / "x.jsonl"


def test_importing_the_persistence_modules_touches_no_file(monkeypatch):
    """Paths are resolved when the window starts; importing reads nothing."""
    touched = []
    for name in ("open", "exists", "is_file", "stat", "read_bytes", "read_text"):
        real = getattr(Path, name)

        def spy(self, *args, _real=real, _name=name, **kwargs):
            touched.append((_name, str(self)))
            return _real(self, *args, **kwargs)
        monkeypatch.setattr(Path, name, spy)
    for module in ("intonation.persistence.paths", "intonation.persistence.config",
                   "intonation.catalog.instruments", "intonation.persistence.history",
                   "intonation.crash"):
        spec = importlib.util.spec_from_file_location(
            f"_fresh_{module.replace('.', '_')}", _REPO / f"{module.replace('.', '/')}.py")
        fresh = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, fresh)   # dataclasses look it up
        spec.loader.exec_module(fresh)
    assert touched == []


# ---------------------------------------------------------------------------
# History: record validation
# ---------------------------------------------------------------------------
def _measurement_line(**over) -> str:
    rec = {"kind": "measurement", "run_id": "r1", "timestamp": "2026-03-02T18:00:00",
           "instrument": "eb_alto", "a4_hz": 440.0, "midi_sounding": 69,
           "midi_fingered": 78, "cents": 1.5, "freq_hz": 440.4}
    rec.update(over)
    return json.dumps(rec)


@pytest.mark.parametrize("over", [
    {"cents": "sharp"}, {"cents": float("nan")}, {"cents": None},
    {"midi_sounding": True}, {"midi_sounding": 128}, {"midi_sounding": 69.5},
    {"midi_fingered": 69 + 128}, {"timestamp": None}, {"run_id": 7},
    {"a4_hz": 0.0}, {"freq_hz": float("inf")},
])
def test_a_mistyped_reading_is_rejected(over):
    assert hist.parse_record(_measurement_line(**over)) is None


def test_a_contrabass_reading_fingered_above_midi_is_kept():
    """Fingered notes leave 0..127 for very low instruments; that is real data."""
    rec = hist.parse_record(_measurement_line(midi_sounding=100, midi_fingered=133,
                                              midi_unknown_field=1))
    assert isinstance(rec, Measurement) and rec.midi_fingered == 133


def test_a_run_without_a_start_time_is_rejected():
    run = {"kind": "run", "run_id": "r1", "started_at": None,
           "instrument": "eb_alto", "a4_hz": 440.0}
    assert hist.parse_record(json.dumps(run)) is None
    run["started_at"] = "2026-03-02T18:00:00"
    assert isinstance(hist.parse_record(json.dumps(run)), RunMeta)


# ---------------------------------------------------------------------------
# History: monthly files
# ---------------------------------------------------------------------------
def _run_line(run_id: str, started: str) -> str:
    return RunMeta(run_id=run_id, started_at=started, instrument="eb_alto",
                   a4_hz=440.0).to_jsonl()


def _meas_line(run_id: str, ts: str, midi: int = 69) -> str:
    return Measurement(run_id=run_id, timestamp=ts, instrument="eb_alto", a4_hz=440.0,
                       midi_sounding=midi, midi_fingered=midi + 9, cents=1.0,
                       freq_hz=440.0).to_jsonl()


@pytest.fixture()
def october(monkeypatch):
    monkeypatch.setattr(hist, "current_month", lambda: "2026-10")


def _mixed_history(path: Path) -> list[str]:
    lines = [
        _run_line("jan", "2026-01-31T23:50:00"),
        _meas_line("jan", "2026-01-31T23:59:00"),
        "garbage after january",
        _meas_line("jan", "2026-02-01T00:01:00"),       # the run crossed midnight
        _run_line("mar", "2026-03-05T10:00:00"),
        _meas_line("mar", "2026-03-05T10:00:01"),
        _run_line("oct", "2026-10-02T09:00:00"),
        _meas_line("oct", "2026-10-02T09:00:01"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def test_rotation_moves_each_run_to_the_month_it_started(tmp_path, october):
    live = tmp_path / "log.jsonl"
    lines = _mixed_history(live)
    log = MeasurementLog(live)

    assert live.read_text(encoding="utf-8").splitlines() == lines[6:]
    assert (tmp_path / "log-2026-01.jsonl").read_text(encoding="utf-8").splitlines() \
        == lines[:4]
    assert (tmp_path / "log-2026-03.jsonl").read_text(encoding="utf-8").splitlines() \
        == lines[4:6]
    assert [m.run_id for m in log.measurements()] == ["jan", "jan", "mar", "oct"]
    assert sorted(r.run_id for r in log.runs()) == ["jan", "mar", "oct"]


def test_rotation_reads_only_the_first_record_when_nothing_is_old(tmp_path, october):
    live = tmp_path / "log.jsonl"
    live.write_text(_run_line("oct", "2026-10-02T09:00:00") + "\n"
                    + _meas_line("old", "2025-01-01T00:00:00") + "\n", encoding="utf-8")
    before = live.read_bytes()
    MeasurementLog(live)
    assert live.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["log.jsonl"]


def test_a_failed_rotation_changes_nothing(tmp_path, october, monkeypatch):
    live = tmp_path / "log.jsonl"
    _mixed_history(live)
    earlier = tmp_path / "log-2026-03.jsonl"
    earlier.write_text(_run_line("mar0", "2026-03-01T08:00:00") + "\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def refuse(src, dst):
        raise PermissionError(13, "in use by another program", str(dst))
    monkeypatch.setattr(hist.os, "replace", refuse)
    log = MeasurementLog(live)

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert len(log.measurements()) == 4


def test_a_torn_last_line_does_not_swallow_the_next_reading(tmp_path, october):
    live = tmp_path / "log.jsonl"
    live.write_text(_run_line("oct", "2026-10-02T09:00:00") + "\n"
                    + '{"kind": "measurement", "run_id": "oc', encoding="utf-8")
    log = MeasurementLog(live)
    log.start_run("eb_alto", 440.0)
    log.add_measurement(69, 78, 0.5, 440.1)
    log.close()
    assert len(MeasurementLog(live).measurements()) == 1


def test_the_history_file_is_opened_once_per_session(tmp_path, october, monkeypatch):
    live = tmp_path / "log.jsonl"
    opened = []
    real_open = Path.open

    def counting(self, mode="r", *args, **kwargs):
        if self == live and "a" in mode:
            opened.append(mode)
        return real_open(self, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counting)

    log = MeasurementLog(live)
    log.start_run("eb_alto", 440.0)
    for i in range(50):
        log.add_measurement(69, 78, i / 10, 440.0)
    # Every reading is on disk before the next one arrives.
    assert live.read_text(encoding="utf-8").count('"measurement"') == 50
    log.close()
    assert opened == ["a"]


def test_an_import_of_an_earlier_month_goes_to_that_months_file(tmp_path, october):
    live = tmp_path / "log.jsonl"
    log = MeasurementLog(live)
    log.start_run("eb_alto", 440.0)
    log.add_measurement(69, 78, 0.5, 440.1)
    live_before = live.read_bytes()

    source = MeasurementLog()
    source._runs["mar"] = RunMeta("mar", "2026-03-05T10:00:00", "eb_alto", 440.0)
    source._measurements.append(Measurement("mar", "2026-03-05T10:00:01", "eb_alto",
                                            440.0, 69, 78, 2.0, 440.5))
    exported = tmp_path / "march.csv"
    source.export_csv(exported)
    assert log.import_raw_csv(exported) == (1, 1, 0)
    log.close()

    assert live.read_bytes() == live_before
    assert '"mar"' in (tmp_path / "log-2026-03.jsonl").read_text(encoding="utf-8")
    again = [m.run_id for m in MeasurementLog(live).measurements()]
    assert len(again) == 2 and again[0] == "mar"     # earlier month first


def test_a_csv_export_that_fails_keeps_the_previous_file(tmp_path, monkeypatch):
    log = MeasurementLog()
    log.start_run("eb_alto", 440.0)
    log.add_measurement(69, 78, 0.5, 440.1)
    target = tmp_path / "history.csv"
    target.write_text("last week\n", encoding="utf-8")

    def broken(w, measurements, runs):
        w.writerow(["half"])
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(hist, "_write_raw", broken)
    with pytest.raises(OSError):
        log.export_csv(target)
    assert target.read_text(encoding="utf-8") == "last week\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history.csv"]


def test_aggregates_match_the_in_memory_path(tmp_path, october):
    """Streaming from disk and holding everything in memory give the same
    numbers (the CSV goldens pin the format; this pins the source)."""
    live = tmp_path / "log.jsonl"
    on_disk = MeasurementLog(live)
    in_memory = MeasurementLog()
    for log in (on_disk, in_memory):
        log.start_run("eb_alto", 440.0)
        for i in range(20):
            log.add_measurement(69 + i % 3, 78 + i % 3, math.sin(i) * 10, 440.0)
    on_disk.close()
    reread = MeasurementLog(live)
    out = {}
    for name, log in (("disk", reread), ("memory", in_memory)):
        path = tmp_path / f"{name}.csv"
        log.export_csv(path, mode="overall_per_note")
        with path.open(encoding="utf-8", newline="") as f:
            out[name] = list(csv.reader(f))
    assert out["disk"] == out["memory"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

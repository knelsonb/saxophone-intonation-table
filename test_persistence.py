"""Unit contracts for the persistence layer: the settings schema and store
and the data-directory paths.

The player-facing behaviour (a damaged or locked settings file) is covered by
tests/acceptance; these tests pin the mechanics underneath, which a scenario
cannot reach: the backup of a damaged file, a backup that cannot be written,
and that importing the modules touches no file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sax_config import AppConfig, ConfigStore
from sax_paths import AppPaths

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
    for module in ("sax_paths", "sax_config", "sax_instruments",
                   "sax_intonation_log", "sax_crash"):
        spec = importlib.util.spec_from_file_location(
            f"_fresh_{module}", _REPO / f"{module}.py")
        fresh = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, fresh)   # dataclasses look it up
        spec.loader.exec_module(fresh)
    assert touched == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Round-trip + tolerant-coercion tests for ``sax_config``.

Phase-0 discovery flagged the gap this file fills: before the parity
sprint there was NO test exercising ``save_config`` → ``load_config`` or
the ``_as_*`` coercers directly, even though every field is read back from
a user-editable ``~/.intonation_analyzer/config.json`` that can be stale,
partial, or corrupt.  The governing contract (sax_config.py module
docstring) is *a corrupt or missing field falls back to its default,
never crashes the GUI on startup* — these tests lock exactly that.

No PyQt6 dependency; pure config logic.

Conventions (matching test_coerce_range_entry.py):
* ``pytest.mark.parametrize`` for the per-field coercion tables, one named
  case per row.
* Every assert carries an expected-vs-got failure message.
* ``CONFIG_PATH`` / ``CONFIG_DIR`` are redirected into ``tmp_path`` via
  monkeypatch so tests never touch the real user home directory.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

import sax_config
from sax_config import AppConfig, load_config, save_config


# ---------------------------------------------------------------------------
# Fixture: redirect the module-level CONFIG_PATH/CONFIG_DIR into a tmp dir so
# save_config writes there and load_config reads it back. atomic_write_json
# creates parents, so the dir need not pre-exist.
# ---------------------------------------------------------------------------
@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".intonation_analyzer"
    cfg_path = cfg_dir / "config.json"
    monkeypatch.setattr(sax_config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(sax_config, "CONFIG_PATH", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------------
# 1. Full round-trip — every field survives save → load unchanged.
# ---------------------------------------------------------------------------
def test_roundtrip_preserves_all_fields(isolated_config):
    """A fully-populated, non-default config must survive save→load byte
    for byte at the dataclass level (asdict equality)."""
    original = AppConfig(
        welcome_shown=True,
        persistence_enabled=True,
        log_path="/tmp/somewhere/log.jsonl",
        allow_out_of_range=False,
        matrix_extra_octaves=2,
        layout_mode_preference="matrix",
        filter_mode="slow",
        min_n_visible=12,
        show_diagnostics=True,
        audio_device_name="FIIO DSP Audio",
        audio_device_host_api="Windows WASAPI",
        audio_device_samplerate=48000,
        audio_samplerate_pref="48000",
        audio_sr_notice_shown=True,
        show_all_host_apis=True,
        prefer_wdmks=True,
        output_device_name="Speakers (Realtek)",
        output_device_host_api="Windows WASAPI",
        output_device_samplerate=44100,
        output_prefer_duplex=True,
        window_geometry="QmFzZTY0R2VvbQ==",
        window_state="QmFzZTY0U3RhdGU=",
        splitter_sizes=[420, 620],
        last_instrument_key="eb_alto",
        last_nickname="MarkVI",
        last_display_mode="klingend",
        last_a4_hz=442,
        last_lang="de",
        last_active_tab="setup",
    )

    save_config(original)
    assert isolated_config.exists(), "save_config did not write the file"

    loaded = load_config()
    assert asdict(loaded) == asdict(original), (
        "round-trip mutated the config:\n"
        f"  saved : {asdict(original)}\n"
        f"  loaded: {asdict(loaded)}"
    )


# ---------------------------------------------------------------------------
# 2. The new output-device fields specifically round-trip.
# ---------------------------------------------------------------------------
def test_output_device_fields_roundtrip(isolated_config):
    cfg = AppConfig(
        output_device_name="Headphones (FIIO)",
        output_device_host_api="MME",
        output_device_samplerate=96000,
        output_prefer_duplex=True,
        last_active_tab="metro",
    )
    save_config(cfg)
    loaded = load_config()
    assert loaded.output_device_name == "Headphones (FIIO)"
    assert loaded.output_device_host_api == "MME"
    assert loaded.output_device_samplerate == 96000
    assert loaded.output_prefer_duplex is True
    assert loaded.last_active_tab == "metro"


# ---------------------------------------------------------------------------
# 3. Tolerant fallbacks — the never-crash contract.
# ---------------------------------------------------------------------------
def test_missing_file_returns_defaults(isolated_config):
    """No config file yet → pristine defaults, no exception."""
    assert not isolated_config.exists()
    cfg = load_config()
    assert cfg == AppConfig(), "missing file must yield default AppConfig()"


def test_corrupt_json_returns_defaults(isolated_config):
    """A syntactically broken file must degrade to defaults, not raise."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("{not valid json,,,", encoding="utf-8")
    cfg = load_config()
    assert cfg == AppConfig(), "corrupt JSON must yield default AppConfig()"


def test_non_dict_json_returns_defaults(isolated_config):
    """Valid JSON that isn't an object (e.g. a list) → defaults."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("[1, 2, 3]", encoding="utf-8")
    cfg = load_config()
    assert cfg == AppConfig(), "non-dict JSON must yield default AppConfig()"


def test_partial_config_fills_new_field_defaults(isolated_config):
    """A config written by an OLDER build (no output-device keys) must load
    with the new fields at their defaults — backward compatibility."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        '{"audio_device_name": "Old Mic", "last_lang": "de"}',
        encoding="utf-8")
    cfg = load_config()
    # Old field preserved …
    assert cfg.audio_device_name == "Old Mic"
    assert cfg.last_lang == "de"
    # … new parity-sprint fields fall back to defaults, no crash.
    assert cfg.output_device_name == ""
    assert cfg.output_device_host_api == ""
    assert cfg.output_device_samplerate == 0
    assert cfg.output_prefer_duplex is False
    assert cfg.last_active_tab == "tuner"


def test_unknown_keys_ignored(isolated_config):
    """Keys from a NEWER build than this one must be ignored, not crash —
    forward compatibility."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        '{"output_device_name": "Spk", "a_field_from_the_future": 99}',
        encoding="utf-8")
    cfg = load_config()
    assert cfg.output_device_name == "Spk"
    assert not hasattr(cfg, "a_field_from_the_future")


# ---------------------------------------------------------------------------
# 4. Per-field coercion — output_device_samplerate clamps to >= 0.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    pytest.param(48000, 48000, id="valid_int"),
    pytest.param("44100", 44100, id="numeric_string"),
    pytest.param(-1, 0, id="negative_clamped_to_zero"),
    pytest.param(-48000, 0, id="large_negative_clamped"),
    pytest.param("garbage", 0, id="non_numeric_defaults_zero"),
    pytest.param(None, 0, id="missing_defaults_zero"),
    pytest.param(0, 0, id="zero_means_auto_negotiate"),
])
def test_output_device_samplerate_coercion(isolated_config, raw, expected):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    import json
    isolated_config.write_text(
        json.dumps({"output_device_samplerate": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.output_device_samplerate == expected, (
        f"output_device_samplerate({raw!r}) -> {cfg.output_device_samplerate}, "
        f"expected {expected}")


# ---------------------------------------------------------------------------
# 5. Per-field coercion — output_prefer_duplex tolerant bool.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    pytest.param(True, True, id="bool_true"),
    pytest.param(False, False, id="bool_false"),
    pytest.param("true", True, id="str_true"),
    pytest.param("on", True, id="str_on"),
    pytest.param("1", True, id="str_one"),
    pytest.param(1, True, id="int_one"),
    pytest.param("false", False, id="str_false"),
    pytest.param("", False, id="empty_string_false"),
    pytest.param("nonsense", False, id="garbage_defaults_false"),
    pytest.param(None, False, id="missing_defaults_false"),
])
def test_output_prefer_duplex_coercion(isolated_config, raw, expected):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    import json
    isolated_config.write_text(
        json.dumps({"output_prefer_duplex": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.output_prefer_duplex is expected, (
        f"output_prefer_duplex({raw!r}) -> {cfg.output_prefer_duplex}, "
        f"expected {expected}")


# ---------------------------------------------------------------------------
# 6. Per-field coercion — last_active_tab allowlist, degrade to 'tuner'.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    pytest.param("tuner", "tuner", id="tuner"),
    pytest.param("metro", "metro", id="metro"),
    pytest.param("deck", "deck", id="deck"),
    pytest.param("setup", "setup", id="setup"),
    pytest.param("METRO", "metro", id="case_insensitive"),
    pytest.param("  deck  ", "deck", id="whitespace_stripped"),
    pytest.param("nonexistent", "tuner", id="unknown_degrades_to_tuner"),
    pytest.param("", "tuner", id="empty_degrades_to_tuner"),
    pytest.param(None, "tuner", id="missing_degrades_to_tuner"),
    pytest.param(42, "tuner", id="wrong_type_degrades_to_tuner"),
])
def test_last_active_tab_coercion(isolated_config, raw, expected):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    import json
    isolated_config.write_text(
        json.dumps({"last_active_tab": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.last_active_tab == expected, (
        f"last_active_tab({raw!r}) -> {cfg.last_active_tab!r}, "
        f"expected {expected!r}")


def test_last_active_tab_values_match_allowlist():
    """The coercer's allowlist must equal the exported TAB_VALUES set so
    Frodo's nav shell and the coercer can never drift apart."""
    for tab in sax_config.TAB_VALUES:
        assert sax_config._as_active_tab(tab) == tab, (
            f"TAB_VALUES member {tab!r} not accepted by _as_active_tab")

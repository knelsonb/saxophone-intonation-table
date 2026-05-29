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
        mic_gain_db=12.0,
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
        last_bpm=132,
        last_time_sig="6/8",
        click_volume=0.65,
        deck_max_seconds=180,
        last_take_path="/home/u/takes/last.wav",
        deck_scratch_dir="/var/tmp/deckscratch",
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
    # v0.8.0 metronome fields also default cleanly for an old config.
    assert cfg.last_bpm == 100
    assert cfg.last_time_sig == "4/4"
    assert cfg.click_volume == 1.0
    # v0.10.0 tape-deck fields default cleanly for an old config.
    assert cfg.deck_max_seconds == 300
    assert cfg.last_take_path == ""
    assert cfg.deck_scratch_dir == ""


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


# ---------------------------------------------------------------------------
# 7. v0.8.0 metronome fields — coercion + clamp (parity Sprint 2).
# ---------------------------------------------------------------------------
def test_metro_fields_roundtrip(isolated_config):
    cfg = AppConfig(last_bpm=144, last_time_sig="3/4", click_volume=0.4)
    save_config(cfg)
    loaded = load_config()
    assert loaded.last_bpm == 144
    assert loaded.last_time_sig == "3/4"
    assert loaded.click_volume == pytest.approx(0.4)


@pytest.mark.parametrize("raw,expected", [
    pytest.param(100, 100, id="default_mid"),
    pytest.param(30, 30, id="low_bound"),
    pytest.param(300, 300, id="high_bound"),
    pytest.param(29, 30, id="below_clamps_up"),
    pytest.param(0, 30, id="zero_clamps_up"),
    pytest.param(-50, 30, id="negative_clamps_up"),
    pytest.param(301, 300, id="above_clamps_down"),
    pytest.param(99999, 300, id="far_above_clamps_down"),
    pytest.param("120", 120, id="numeric_string"),
    pytest.param("junk", 100, id="garbage_defaults_then_in_range"),
    pytest.param(None, 100, id="missing_defaults"),
])
def test_last_bpm_clamped_to_range(isolated_config, raw, expected):
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"last_bpm": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.last_bpm == expected, (
        f"last_bpm({raw!r}) -> {cfg.last_bpm}, expected {expected} (clamp 30-300)")


# ---------------------------------------------------------------------------
# v0.11.0 theme field (Sprint 5) -- coerced via sax_theme.coerce_theme_name.
# ---------------------------------------------------------------------------
def test_theme_roundtrip(isolated_config):
    for name in ("dark", "night", "light"):
        save_config(AppConfig(theme=name))
        assert load_config().theme == name, f"theme {name!r} did not round-trip"


@pytest.mark.parametrize("raw,expected", [
    pytest.param("dark", "dark", id="dark"),
    pytest.param("night", "night", id="night"),
    pytest.param("light", "light", id="light"),
    pytest.param("LIGHT", "light", id="uppercase_normalised"),
    pytest.param("  Night ", "night", id="whitespace_stripped"),
    pytest.param("bogus", "dark", id="unknown_degrades"),
    pytest.param("", "dark", id="empty_degrades"),
    pytest.param(None, "dark", id="missing_degrades"),
    pytest.param(123, "dark", id="wrong_type_degrades"),
])
def test_theme_coercion(isolated_config, raw, expected):
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"theme": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.theme == expected, (
        f"theme({raw!r}) -> {cfg.theme}, expected {expected}")


@pytest.mark.parametrize("raw,expected", [
    pytest.param("2/4", "2/4", id="two_four"),
    pytest.param("3/4", "3/4", id="three_four"),
    pytest.param("4/4", "4/4", id="four_four"),
    pytest.param("6/8", "6/8", id="six_eight"),
    pytest.param(" 3/4 ", "3/4", id="whitespace_stripped"),
    pytest.param("7/8", "4/4", id="unsupported_degrades"),
    pytest.param("", "4/4", id="empty_degrades"),
    pytest.param(None, "4/4", id="missing_degrades"),
    pytest.param(44, "4/4", id="wrong_type_degrades"),
])
def test_last_time_sig_coercion(isolated_config, raw, expected):
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"last_time_sig": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.last_time_sig == expected, (
        f"last_time_sig({raw!r}) -> {cfg.last_time_sig!r}, expected {expected!r}")


@pytest.mark.parametrize("raw,expected", [
    pytest.param(0.5, 0.5, id="mid"),
    pytest.param(0.0, 0.0, id="silent"),
    pytest.param(1.0, 1.0, id="full"),
    pytest.param(-0.5, 0.0, id="below_clamps_to_zero"),
    pytest.param(2.0, 1.0, id="above_clamps_to_one"),
    pytest.param("0.25", 0.25, id="numeric_string"),
    pytest.param("loud", 1.0, id="garbage_defaults"),
    pytest.param(None, 1.0, id="missing_defaults"),
])
def test_click_volume_clamped_to_unit_interval(isolated_config, raw, expected):
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"click_volume": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.click_volume == pytest.approx(expected), (
        f"click_volume({raw!r}) -> {cfg.click_volume}, expected {expected}")


def test_click_volume_rejects_non_finite():
    """NaN/inf in the config must degrade to the default (1.0), never reach
    the audio path as a non-finite gain. JSON can't encode inf/nan natively,
    so feed them via the coercer directly (the path a hand-edited or
    programmatically-built dict could take)."""
    assert sax_config._as_float(float("nan"), 1.0) == 1.0
    assert sax_config._as_float(float("inf"), 1.0) == 1.0
    assert sax_config._as_float(float("-inf"), 1.0) == 1.0


def test_time_sig_values_match_allowlist():
    """The coercer must accept every exported TIME_SIG_VALUES member, so the
    GUI selector and the coercer can't drift apart."""
    for sig in sax_config.TIME_SIG_VALUES:
        assert sax_config._as_time_sig(sig) == sig, (
            f"TIME_SIG_VALUES member {sig!r} not accepted by _as_time_sig")


# ---------------------------------------------------------------------------
# 8. v0.10.0 tape-deck fields — round-trip + clamp/coercion (parity Sprint 4).
# deck_max_seconds bounds the engine's preallocated mic-capture buffer, so an
# out-of-range / corrupt value must clamp into [1, 600] — never propagate a
# 0-or-negative length (np.zeros(0) = a dead recorder) or a runaway size.
# ---------------------------------------------------------------------------
def test_deck_fields_roundtrip(isolated_config):
    cfg = AppConfig(
        deck_max_seconds=120,
        last_take_path="/home/u/takes/solo.wav",
        deck_scratch_dir="/var/tmp/deck",
    )
    save_config(cfg)
    loaded = load_config()
    assert loaded.deck_max_seconds == 120
    assert loaded.last_take_path == "/home/u/takes/solo.wav"
    assert loaded.deck_scratch_dir == "/var/tmp/deck"


@pytest.mark.parametrize("raw,expected", [
    pytest.param(300, 300, id="default_mid"),
    pytest.param(1, 1, id="low_bound"),
    pytest.param(600, 600, id="high_bound"),
    pytest.param(0, 1, id="zero_clamps_up"),
    pytest.param(-30, 1, id="negative_clamps_up"),
    pytest.param(601, 600, id="above_clamps_down"),
    pytest.param(99999, 600, id="far_above_clamps_down"),
    pytest.param("90", 90, id="numeric_string"),
    pytest.param("junk", 300, id="garbage_defaults_then_in_range"),
    pytest.param(None, 300, id="missing_defaults"),
])
def test_deck_max_seconds_clamped_to_range(isolated_config, raw, expected):
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"deck_max_seconds": raw}), encoding="utf-8")
    cfg = load_config()
    assert cfg.deck_max_seconds == expected, (
        f"deck_max_seconds({raw!r}) -> {cfg.deck_max_seconds}, "
        f"expected {expected} (clamp 1-600)")


@pytest.mark.parametrize("field_name", ["last_take_path", "deck_scratch_dir"])
@pytest.mark.parametrize("raw,expected", [
    pytest.param("/some/take.wav", "/some/take.wav", id="valid_path"),
    pytest.param("", "", id="empty_string"),
    pytest.param(None, "", id="missing_defaults_empty"),
])
def test_deck_path_fields_coerce_to_str(isolated_config, field_name, raw, expected):
    """last_take_path / deck_scratch_dir are tolerant string fields: a present
    path round-trips, a null/missing value degrades to '' (no export yet / OS
    temp dir)."""
    import json
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({field_name: raw}), encoding="utf-8")
    cfg = load_config()
    assert getattr(cfg, field_name) == expected, (
        f"{field_name}({raw!r}) -> {getattr(cfg, field_name)!r}, "
        f"expected {expected!r}")


# ---------------------------------------------------------------------------
# Non-finite hardening: a corrupt/hand-edited config can carry the non-standard
# Infinity / NaN tokens (Python's json.load accepts them by default). The
# coercers must DEGRADE to defaults (never crash on startup — int(float('inf'))
# raises OverflowError), and atomic_write_json must never PERSIST a non-finite.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")],
                         ids=["inf", "-inf", "nan"])
def test_as_int_degrades_on_non_finite(bad):
    # int(float('inf')) raises OverflowError; int(nan) raises ValueError. Both
    # must be caught and fall back to the default, not propagate up the load.
    assert sax_config._as_int(bad, 7) == 7


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")],
                         ids=["inf", "-inf", "nan"])
def test_as_int_list_degrades_on_non_finite(bad):
    assert sax_config._as_int_list([10, bad, 30]) == []


def test_load_config_with_infinity_tokens_returns_defaults(isolated_config):
    """A config.json carrying Infinity / NaN (json.load accepts these tokens)
    must load to safe defaults, not crash the GUI on startup."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        '{"last_bpm": Infinity, "splitter_sizes": [1, NaN, 3], '
        '"last_a4_hz": -Infinity}', encoding="utf-8")
    cfg = load_config()                       # the bug was: this raised
    defaults = AppConfig()
    assert cfg.last_bpm == defaults.last_bpm, "last_bpm must degrade to default"
    assert cfg.last_a4_hz == defaults.last_a4_hz, "last_a4_hz must degrade"
    assert isinstance(cfg.splitter_sizes, list), "splitter_sizes must not crash"


def test_atomic_write_rejects_non_finite_payload(tmp_path):
    """allow_nan=False: a non-finite slipping into a payload fails the write
    safely (returns False, leaves no file) rather than persisting garbage."""
    from sax_atomic import atomic_write_json
    path = tmp_path / "x.json"
    assert atomic_write_json(path, {"v": float("inf")}) is False
    assert not path.exists(), "no garbage file may be left behind"


def test_atomic_write_accepts_finite_payload(tmp_path):
    from sax_atomic import atomic_write_json
    path = tmp_path / "x.json"
    assert atomic_write_json(path, {"v": 1.5, "n": 3}) is True
    assert path.exists()

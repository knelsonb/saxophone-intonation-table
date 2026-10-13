"""Golden-file checks: what the app writes, pinned against reviewed files.

Inputs live in tests/golden/ (see README.md there). Expected outputs live in
tests/golden/expected/ and are rewritten with ``UPDATE_GOLDEN=1``; read the diff
before committing, a changed golden is a changed promise to users.
"""
from __future__ import annotations

import json

import pytest

from tests.acceptance._scenarios import play_known_table
from tests.harness import HeadlessApp, assert_golden, fixture_path, i18n_catalog

CSV_MODES = ["raw", "per_run_note", "per_instrument_note", "per_nickname_note",
             "instrument_avg", "overall_per_note", ]


@pytest.fixture()
def history_app(make_app):
    """An app whose saved history is the committed log.jsonl fixture."""
    return make_app(
        config={"persistence_enabled": True, "last_instrument_key": "eb_alto"},
        history=fixture_path("log.jsonl"),
        customs=fixture_path("custom_instruments.json"),
        ranges=fixture_path("instrument_ranges.json"))


# ------------------------------------------------------------------------ CSV
@pytest.mark.parametrize("mode", CSV_MODES)
def test_csv_export_mode_matches_golden(history_app, mode):
    history_app.launch()
    assert history_app.history_counts == (3, 96)
    kw = {"instrument": "eb_alto"} if mode == "instrument_avg" else {}
    path = history_app.export.csv(mode, **kw)
    assert_golden(path, f"expected/csv_{mode}.csv")


def test_raw_csv_export_round_trips_into_a_new_installation(history_app, make_app, tmp_path,
                                                            host, clock):
    history_app.launch()
    exported = history_app.export.csv("raw")
    history_app.close()

    other = HeadlessApp(tmp_path / "other-home", host=host, clock=clock)
    try:
        other.launch()
        assert other.history_counts == (0, 0)
        result = other.import_csv(exported)
        assert (result.status, result.runs, result.measurements, result.skipped) == \
            ("imported", 3, 96, 0)
        again = other.import_csv(exported)
        assert (again.status, again.measurements, again.skipped) == ("nothing_new", 0, 96)
        assert_golden(other.export.csv("raw"), "expected/csv_raw.csv")
    finally:
        other.close()


# ---------------------------------------------------------------- TXT / PDF / chart
def _table_app(make_app, **kw):
    app = make_app(**kw)
    app.launch()
    play_known_table(app)
    return app


def test_txt_export_matches_golden(make_app):
    app = _table_app(make_app)
    path = app.export.txt(maker="Selmer", model="Mark VI")
    assert_golden(path, "expected/txt_known_table_en.txt")


def test_txt_export_in_german_keeps_markup_characters_literal(make_app):
    app = _table_app(make_app, lang="de")
    path = app.export.txt(maker="Müller & Söhne", model="<Mark VI> #2")
    assert_golden(path, "expected/txt_known_table_de_markup.txt")


def test_pdf_render_model_matches_golden(make_app):
    app = _table_app(make_app)
    app.export.pdf(maker="Selmer", model="Mark VI")
    assert_golden(app.export.pdf_model, "expected/pdf_model_known_table_en.json")


def test_pdf_render_model_escapes_markup_in_maker_and_model(make_app):
    app = _table_app(make_app, lang="de")
    app.export.pdf(maker="Müller & Söhne", model="<b>Mark VI</b>")
    assert_golden(app.export.pdf_model, "expected/pdf_model_known_table_de_markup.json")


def test_chart_render_model_matches_golden(make_app):
    app = _table_app(make_app)
    app.export.chart(maker="Selmer", model="Mark VI")
    assert_golden(app.export.chart_model, "expected/chart_model_known_table_en.json")


# --------------------------------------------------------------------- configs
VOLATILE_CONFIG_KEYS = ("window_geometry", "window_state", "splitter_sizes")

# What each release's saved settings must restore (and how a saved file differs
# after the current version has re-saved it).
CONFIG_CASES = [
    pytest.param("config_v0.6.json", dict(instrument="bb_tenor", a4=443, display="sounding",
                                          lang="de"), id="v0.6"),
    pytest.param("config_v1.0.json", dict(instrument="eb_alto", a4=442, display="fingered",
                                          lang="en"), id="v1.0"),
    pytest.param("config_v1.1.json", dict(instrument="eb_bari", a4=440, display="sounding",
                                          lang="de"), id="v1.1"),
    pytest.param("config_v1.2.json", dict(instrument="custom_mezzo_soprano", a4=441,
                                          display="fingered", lang="en"), id="v1.2"),
]


def _without_volatile(text: str) -> str:
    data = json.loads(text)
    for k in VOLATILE_CONFIG_KEYS:
        data[k] = "<varies>"
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("fixture, expect", CONFIG_CASES)
def test_settings_saved_by_each_release_are_restored(make_app, fixture, expect):
    app = make_app(config=fixture_path(fixture), customs=fixture_path("custom_instruments.json"))
    app.launch()
    assert app.instrument == expect["instrument"]
    assert app.a4 == expect["a4"]
    assert app.engine_a4 == float(expect["a4"])
    assert app.display == expect["display"]
    assert app.ui_language == expect["lang"]
    assert app.table.rows() == []          # settings restore, measurements do not
    app.close()

    after = fixture.replace("config_", "config_after_")
    assert_golden(app.file("config"), f"expected/{after}", normalize=_without_volatile)


def test_every_release_config_is_saved_back_with_the_current_field_set(make_app):
    saved = {}
    for fixture in ("config_v0.6.json", "config_v1.0.json", "config_v1.1.json",
                    "config_v1.2.json"):
        app = make_app(config=fixture_path(fixture),
                       customs=fixture_path("custom_instruments.json"))
        app.launch()
        app.close()
        saved[fixture] = sorted(app.read_config())
    keys = saved["config_v1.2.json"]
    assert all(v == keys for v in saved.values()), "field sets differ after re-save"
    assert_golden(keys, "expected/config_keys.json")


# ------------------------------------------------------------------------ i18n
def test_i18n_snapshot_matches_golden():
    catalog = i18n_catalog()
    snapshot = {lang: {"keys": sorted(table),
                       "placeholders": {k: v for k, v in sorted(table.items()) if v}}
                for lang, table in sorted(catalog.items())}
    assert_golden(snapshot, "expected/i18n_snapshot.json")


def test_every_ui_string_exists_in_both_languages_with_the_same_placeholders():
    catalog = i18n_catalog()
    de, en = catalog["de"], catalog["en"]
    assert sorted(de) == sorted(en), (
        f"keys only in de: {sorted(set(de) - set(en))}; only in en: {sorted(set(en) - set(de))}")
    mismatched = {k: (de[k], en[k]) for k in de if de[k] != en[k]}
    assert not mismatched, f"placeholder sets differ between languages: {mismatched}"

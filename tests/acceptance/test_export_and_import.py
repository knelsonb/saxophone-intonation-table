"""Export, import and history: getting my measurements out of the app and back in.

Scenarios 8-11 of the acceptance plan:
  8  TXT / PDF / chart exports with played and blank rows, markup characters in
     the maker and model, English and German;
  9  CSV slices exported and imported into a new installation, in the encodings
     spreadsheets save (UTF-8, UTF-8 with BOM, Windows-1252), with the confirm
     prompt and duplicate runs;
  10 broken or hostile CSV files: a readable message and nothing changed;
  11 a damaged history log: every good line still loads.
"""
from __future__ import annotations

import csv
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.acceptance._scenarios import KNOWN_TABLE, play_known_table
from tests.harness import HeadlessApp, Tone, assert_golden, fixture_path

AGGREGATE_MODES = ["per_run_note", "per_instrument_note", "per_nickname_note",
                   "instrument_avg", "overall_per_note"]

RAW_HEADER = ("timestamp,run_id,run_started_at,instrument,nickname,a4_hz,midi_sounding,"
              "sounding_note,midi_fingered,fingered_note,cents,freq_hz,maker,model")

# Library and OS wording that must never reach a player as the whole explanation.
RAW_ERROR_MARKERS = ("Errno", "Permission denied", "Failed to save", "may not be supported",
                     "field larger than field limit", "Traceback")


# ------------------------------------------------------------------- helpers
def _history_app(make_app, lang: str = "en", history=None) -> HeadlessApp:
    """An installation with history saving on and the committed three-run log."""
    return make_app(lang=lang,
                    config={"persistence_enabled": True, "last_instrument_key": "eb_alto"},
                    history=history if history is not None else fixture_path("log.jsonl"))


@contextmanager
def _new_installation(tmp_path, host, clock, **kw):
    """A second computer: its own home directory, nothing saved yet."""
    other = HeadlessApp(tmp_path / "new-installation", host=host, clock=clock, **kw)
    other.launch()
    try:
        yield other
    finally:
        other.close()


def _export_raw_and_close(app: HeadlessApp) -> Path:
    app.launch()
    path = app.export.csv("raw")
    app.close()
    return path


def _csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _raw_line(run_id: str, *, nickname: str = "", a4: str = "440.00", cents: str = "1.000",
              maker: str = "", model: str = "") -> str:
    """One raw-export data row: an Eb alto sounding A4 (fingered F#5)."""
    return (f"2026-04-01T10:00:00,{run_id},2026-04-01T10:00:00,eb_alto,{nickname},{a4},"
            f"69,A4,78,F#/Gb5,{cents},440.254,{maker},{model}")


def _log_lines() -> list[bytes]:
    return fixture_path("log.jsonl").read_bytes().splitlines(keepends=True)


def _snapshot(app: HeadlessApp):
    """Everything an import must leave alone: the history and the table."""
    return (app.history_counts, app.file("history").read_bytes(), app.table.rows())


def _locked_folder_skip():
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores folder permissions")


# ======================================================= 8: TXT / PDF / chart
def test_exports_list_the_notes_i_played_and_leave_out_the_blank_range_rows(app):
    """The table shows my alto's whole range; the reports list only what I played."""
    play_known_table(app)
    rows = app.table.rows(played_only=False)
    played = [r for r in rows if r.played]
    assert [r.sounding for r in played] == [note for note, _ in KNOWN_TABLE]
    assert len(rows) > len(played)                      # blank rows are on screen

    assert_golden(app.export.txt(maker="Selmer", model="Mark VI"),
                  "expected/txt_known_table_en.txt")

    app.export.pdf()
    table = next(f["table"] for f in app.export.pdf_model["flowables"] if "table" in f)
    assert [cells[1] for cells in table[1:]] == [r.sounding for r in played]

    app.export.chart()
    assert [n[0] for n in app.export.chart_model["notes"]] == [r.fingered for r in played]


def test_ampersand_and_angle_brackets_in_maker_and_model_print_literally_in_both_languages(app):
    """'Müller & Söhne' and '<Mark VI> #2' come out exactly as typed, in English and German."""
    maker, model = "Müller & Söhne", "<Mark VI> #2"
    play_known_table(app)
    for lang in ("en", "de"):
        app.settings.language(lang)

        txt = app.export.txt(maker=maker, model=model).read_text(encoding="utf-8")
        assert txt.splitlines()[0] == app.text("txt_header")
        assert maker in txt and model in txt

        app.export.pdf(maker=maker, model=model)
        plain = [f["plain"] for f in app.export.pdf_model["flowables"] if "plain" in f]
        assert plain[0] == app.text("pdf_title")
        assert app.text("pdf_maker", maker=maker) in plain
        assert app.text("pdf_model", model=model) in plain

        app.export.chart(maker=maker, model=model)
        subtitle = app.export.chart_model["subtitle"]
        assert maker in subtitle and model in subtitle
        assert app.export.chart_model["title"] == app.text("chart_title")


def test_chart_and_csv_say_there_is_nothing_to_export_before_i_play(app):
    """Exporting before playing anything gets a plain 'no data' message and no file."""
    assert app.export.chart(expect_failure=True) is None
    assert app.dialogs.last_message().text == app.text("chart_no_data")
    assert app.export.csv(expect_failure=True) is None
    assert app.dialogs.last_message().text == app.text("csv_no_data")


@pytest.mark.xfail(strict=True, reason="Frodo-18: TXT export with nothing played writes "
                                       "an empty report instead of saying so")
def test_txt_export_says_there_is_nothing_to_export_before_i_play(app):
    """Same as the chart and CSV: no readings, no empty report on disk."""
    assert app.export.txt(expect_failure=True) is None
    assert app.dialogs.last_message().kind == "information"


@pytest.mark.xfail(strict=True, reason="Frodo-18: a TXT or PDF name typed without an "
                                       "extension is saved without one")
def test_a_report_name_typed_without_extension_gets_one(app):
    """Typing 'my-table' in the save dialog gives me my-table.txt / my-table.pdf."""
    play_known_table(app)
    folder = app.home / "exports"
    app.export.txt(path=folder / "my-table", expect_failure=True)
    app.export.pdf(path=folder / "my-report", expect_failure=True)
    assert (folder / "my-table.txt").exists()
    assert (folder / "my-report.pdf").exists()


@pytest.mark.xfail(strict=True, reason="Frodo-18: the TXT report dates are always "
                                       "DD.MM.YYYY; the chart of the same session uses YYYY-MM-DD")
def test_the_txt_report_and_the_chart_write_the_date_the_same_way(app):
    """Two files from one session should not disagree about how to write today."""
    play_known_table(app)
    app.export.chart()
    chart_date = re.search(r"\d{4}-\d{2}-\d{2}", app.export.chart_model["subtitle"]).group(0)
    txt = app.export.txt().read_text(encoding="utf-8")
    assert chart_date in txt


@pytest.mark.xfail(strict=True, reason="Aragorn-7: '.png' is added after the save dialog, "
                                       "so an existing chart is replaced without asking")
def test_saving_a_chart_over_last_weeks_picture_asks_first(app):
    """I type 'my-chart'; my-chart.png already exists; I say No; it is still there."""
    play_known_table(app)
    folder = app.home / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    existing = folder / "my-chart.png"
    existing.write_bytes(b"last week's chart")
    app.dialogs.expect("question", "No", optional=True)

    app.export.chart(path=folder / "my-chart", expect_failure=True)

    assert existing.read_bytes() == b"last week's chart"


@pytest.mark.xfail(strict=True, reason="Frodo-9: export errors show raw English OS / library "
                                       "text in the German UI")
def test_a_locked_folder_is_explained_in_german_for_every_export(make_app, tmp_path):
    """Saving into a folder I may not write to says so in my language, naming the file."""
    _locked_folder_skip()
    app = make_app(lang="de")
    app.launch()
    play_known_table(app)
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        shown = {}
        for kind, name in (("txt", "table.txt"), ("pdf", "table.pdf"),
                           ("csv", "history.csv"), ("chart", "chart.png")):
            assert getattr(app.export, kind)(path=locked / name, expect_failure=True) is None
            shown[name] = app.dialogs.last_message()
    finally:
        os.chmod(locked, 0o700)
    for name, msg in shown.items():
        assert msg.kind in ("critical", "warning"), name
        assert name in msg.text, name
        assert not any(m in msg.text for m in RAW_ERROR_MARKERS), msg.text


# =========================================== 8/history: what the app promises
@pytest.mark.xfail(strict=True, reason="Frodo-7: the welcome text promises charts from past "
                                       "data, but the chart reads only the live table")
def test_the_chart_can_be_drawn_from_saved_history_as_the_welcome_promises(make_app):
    """With three saved runs and nothing played yet today, the chart still has data."""
    app = _history_app(make_app)
    app.launch()
    assert app.history_counts == (3, 96)
    assert app.table.rows() == []
    assert app.export.chart().exists()


@pytest.mark.xfail(strict=True, reason="Frodo-7: closing the window throws away unsaved "
                                       "readings without asking")
def test_closing_with_readings_that_were_never_saved_asks_first(app):
    """History is off and I have not exported: closing should warn me."""
    assert not app.settings.history_enabled
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    app.dialogs.expect("question", "Yes", optional=True)
    mark = len(app.dialogs.messages)

    app.close()

    assert any(m.kind == "question" for m in app.dialogs.messages[mark:])


# ========================================================= 9: CSV round trips
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"],
                         ids=["utf8", "utf8-bom", "windows-1252"])
def test_a_raw_csv_resaved_by_a_spreadsheet_imports_into_a_new_installation(
        make_app, tmp_path, host, clock, encoding):
    """Export on one computer, open and save in Excel, import on another: same data."""
    exported = _export_raw_and_close(_history_app(make_app))
    resaved = tmp_path / f"resaved-{encoding}.csv"
    resaved.write_bytes(exported.read_bytes().decode("utf-8").encode(encoding))

    with _new_installation(tmp_path, host, clock) as other:
        result = other.import_csv(resaved)
        assert (result.status, result.runs, result.measurements, result.skipped) == \
            ("imported", 3, 96, 0)
        assert other.history_counts == (3, 96)
        assert_golden(other.export.csv("raw"), "expected/csv_raw.csv")


def test_summary_csv_slices_are_refused_on_import_with_the_reason(make_app, tmp_path,
                                                                  host, clock):
    """Only the raw slice can come back in; each summary slice is refused, nothing changes."""
    app = _history_app(make_app)
    app.launch()
    files = {mode: app.export.csv(mode, **({"instrument": "eb_alto"}
                                           if mode == "instrument_avg" else {}))
             for mode in AGGREGATE_MODES}
    app.close()

    with _new_installation(tmp_path, host, clock) as other:
        for mode, path in files.items():
            result = other.import_csv(path)
            assert result.status == "rejected", mode
            assert result.message.startswith(other.text("csv_import_badhdr")), mode
            assert other.history_counts == (0, 0), mode


def test_import_asks_first_and_importing_the_same_file_twice_adds_nothing(
        make_app, tmp_path, host, clock):
    """Cancel imports nothing; OK imports everything; the same file again is all duplicates."""
    exported = _export_raw_and_close(_history_app(make_app))

    with _new_installation(tmp_path, host, clock) as other:
        assert other.import_csv(exported, confirm=False).status == "cancelled"
        prompt = [m for m in other.dialogs.messages if m.kind == "question"][-1]
        assert exported.name in prompt.text
        assert other.history_counts == (0, 0)

        first = other.import_csv(exported)
        assert (first.status, first.runs, first.measurements, first.skipped) == \
            ("imported", 3, 96, 0)
        assert first.message == other.text("csv_import_saved", runs=3, meas=96, skipped=0)

        again = other.import_csv(exported)
        assert (again.status, again.runs, again.measurements, again.skipped) == \
            ("nothing_new", 0, 0, 96)
        assert other.history_counts == (3, 96)


@pytest.mark.xfail(strict=True, reason="Ent-5: one non-UTF-8 byte makes the whole file "
                                       "decode as Windows-1252 and garbles every umlaut")
def test_one_stray_byte_does_not_garble_the_names_in_the_rest_of_the_file(
        make_app, tmp_path, host, clock):
    """A hand-added row typed in a Windows editor must not turn 'Müller' into 'MÃ¼ller'."""
    exported = _export_raw_and_close(_history_app(make_app))
    patched = tmp_path / "hand-edited.csv"
    extra = _raw_line("bbbb00000001", maker="Caf\xe9").encode("cp1252") + b"\r\n"
    patched.write_bytes(exported.read_bytes() + extra)

    with _new_installation(tmp_path, host, clock) as other:
        assert other.import_csv(patched).status == "imported"
        nicknames = {r["nickname"] for r in _csv_rows(other.export.csv("raw"))}
        assert "Müller Bariton" in nicknames


def test_every_reading_from_a_custom_horn_survives_export_and_import(make_app, tmp_path,
                                                                     host, clock):
    """A custom instrument three octaves down: what I export is what I get back."""
    app = make_app()
    app.launch()
    app.settings.add_custom_instrument("Contra Test", transpose=-36)
    app.settings.filter_to_range(False)
    app.play(Tone(note="B6", seconds=1.0))
    assert app.table.row("B6", by="sounding").n > 0
    _runs, measured = app.history_counts
    exported = app.export.csv("raw")
    app.close()

    with _new_installation(tmp_path, host, clock) as other:
        result = other.import_csv(exported)
        assert (result.measurements, result.skipped) == (measured, 0)


@pytest.mark.xfail(strict=True, reason="Frodo-8: after an import the window looks exactly "
                                       "as before, so it looks like nothing happened")
def test_after_an_import_i_can_see_something_changed(make_app, tmp_path, host, clock):
    """Three runs came in; the main window should show that, not stay blank."""
    exported = _export_raw_and_close(_history_app(make_app))

    with _new_installation(tmp_path, host, clock) as other:
        caption = other.table.caption
        assert other.import_csv(exported).status == "imported"
        assert other.table.rows() or other.table.caption != caption


@pytest.mark.xfail(strict=True, reason="Frodo-8: with history on an import is permanent "
                                       "(no undo) but the prompt is the same as without")
def test_the_import_prompt_warns_when_history_will_keep_the_import(make_app, tmp_path):
    """With history saving on, the confirm prompt should tell me this cannot be undone."""
    small = tmp_path / "small.csv"
    small.write_text(RAW_HEADER + "\n" + _raw_line("dddd00000001") + "\n", encoding="utf-8")

    prompts = []
    for history in (False, True):
        app = make_app(config={"persistence_enabled": history})
        app.launch()
        assert app.settings.history_enabled is history
        app.import_csv(small, confirm=False)
        prompts.append([m for m in app.dialogs.messages if m.kind == "question"][-1].text)
        app.close()

    assert prompts[0] != prompts[1]


# ==================================================== 9: maker/model on CSV
def test_maker_and_model_typed_at_csv_export_appear_on_the_exported_readings(app):
    """The maker/model prompt before a CSV export labels the readings in the file."""
    app.pick_instrument("eb_alto")
    app.play(Tone(note="C4", seconds=1.0))
    app.play(Tone(note="E4", seconds=1.0))

    rows = _csv_rows(app.export.csv("raw", maker="Selmer", model="Mark VI"))

    assert rows
    assert {(r["maker"], r["model"]) for r in rows} == {("Selmer", "Mark VI")}


@pytest.mark.xfail(strict=True, reason="Frodo-11: CSV maker/model go only onto the current "
                                       "run, not onto readings before an A4 change")
def test_maker_and_model_reach_readings_from_before_a_concert_pitch_change(app):
    """Same horn all evening; changing A4 halfway must not split the label."""
    app.pick_instrument("eb_alto")
    app.play(Tone(note="C4", seconds=1.0))
    app.set_a4(442)
    app.play(Tone(note="E4", a4=442.0, seconds=1.0))

    rows = _csv_rows(app.export.csv("raw", maker="Selmer", model="Mark VI"))

    assert {r["a4_hz"] for r in rows} == {"440.00", "442.00"}
    assert {(r["maker"], r["model"]) for r in rows} == {("Selmer", "Mark VI")}


@pytest.mark.xfail(strict=True, reason="Frodo-11: maker/model given at CSV export are kept "
                                       "in memory only and gone after a restart")
def test_maker_and_model_given_at_export_stay_on_those_readings_after_a_restart(make_app):
    """I labelled yesterday's readings 'Selmer Mark VI'; today's export still says so."""
    app = make_app(config={"persistence_enabled": True})
    app.launch()
    app.pick_instrument("eb_alto")
    app.play(Tone(note="C4", seconds=1.0))
    first = _csv_rows(app.export.csv("raw", maker="Selmer", model="Mark VI"))
    assert {(r["maker"], r["model"]) for r in first} == {("Selmer", "Mark VI")}

    app.restart()
    again = _csv_rows(app.export.csv("raw"))

    assert len(again) == len(first)
    assert {(r["maker"], r["model"]) for r in again} == {("Selmer", "Mark VI")}


# ================================================================ 10: hostile
def _write_hostile(path: Path, kind: str) -> None:
    if kind == "wrong-header":
        path.write_text("Note,Cents,Comment\nA4,+3,good\n", encoding="utf-8")
    elif kind == "nan-and-infinite-numbers":
        path.write_text("\n".join([RAW_HEADER, _raw_line("eeee00000001", cents="nan"),
                                   _raw_line("eeee00000002", a4="inf"),
                                   _raw_line("eeee00000003", cents="1e400")]) + "\n",
                        encoding="utf-8")
    elif kind == "oversized-field":
        path.write_text(RAW_HEADER + "\n" + _raw_line("eeee00000004", model="x" * 200_000)
                        + "\n", encoding="utf-8")
    elif kind == "utf-16":
        path.write_text(RAW_HEADER + "\n" + _raw_line("eeee00000005") + "\n",
                        encoding="utf-16")
    else:
        raise ValueError(kind)


@pytest.mark.parametrize("kind", [
    "wrong-header",
    "nan-and-infinite-numbers",
    pytest.param("oversized-field", marks=pytest.mark.xfail(
        strict=True, reason="Ent-5: a huge cell shows the raw csv-library error, "
                            "untranslated")),
    pytest.param("utf-16", marks=pytest.mark.xfail(
        strict=True, reason="Ent-5: a UTF-16 file is called 'not a raw export' and the "
                            "message dumps the garbled header")),
])
def test_a_broken_csv_gets_a_readable_message_in_my_language_and_changes_nothing(
        make_app, tmp_path, kind):
    """Each bad file is turned away (or its bad rows skipped) with the app's own words,
    in German and in English, and my history and table stay exactly as they were."""
    path = tmp_path / f"{kind}.csv"
    _write_hostile(path, kind)
    app = _history_app(make_app, lang="de")
    app.launch()
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", seconds=1.0))
    before = _snapshot(app)

    messages = {}
    for lang in ("de", "en"):
        app.settings.language(lang)
        result = app.import_csv(path)
        assert result.status in ("rejected", "nothing_new"), result
        assert result.measurements == 0
        assert _snapshot(app) == before
        messages[lang] = result.message

    for text in messages.values():
        assert text.strip()
        assert "\\x00" not in text and "\x00" not in text
        assert not any(m in text for m in RAW_ERROR_MARKERS), text
    # The first line is the app's own sentence, so it changes with the language.
    assert messages["de"].splitlines()[0] != messages["en"].splitlines()[0]


def test_spreadsheet_formulas_in_an_imported_file_are_exported_as_plain_text(
        make_app, tmp_path, host, clock):
    """Cells starting with = + @ come back out quoted, so a spreadsheet will not run them,
    and a second export / import cycle changes nothing."""
    nickname, maker, model = '=HYPERLINK("http://x","click")', "+cmd|' /C calc'!A0", "@SUM(1+1)"
    crafted = tmp_path / "crafted.csv"
    with open(crafted, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER.split(","))
        w.writerow(["2026-04-01T10:00:00", "ffff00000001", "2026-04-01T10:00:00", "eb_alto",
                    nickname, "440.00", "69", "A4", "78", "F#/Gb5", "-3.000", "439.238",
                    maker, model])

    app = make_app()
    app.launch()
    result = app.import_csv(crafted)
    assert (result.status, result.runs, result.measurements) == ("imported", 1, 1)
    first = app.export.csv("raw")
    row = _csv_rows(first)[0]
    assert (row["nickname"], row["maker"], row["model"]) == \
        ("'" + nickname, "'" + maker, "'" + model)
    assert row["cents"] == "-3.000"                     # numbers are left alone
    app.close()

    with _new_installation(tmp_path, host, clock) as other:
        assert other.import_csv(first).measurements == 1
        assert other.export.csv("raw").read_bytes() == first.read_bytes()


# ============================================================ 11: history log
def test_a_damaged_history_log_still_loads_every_good_line(make_app):
    """Garbage bytes, half-written and non-record lines are skipped; the three runs load,
    and new readings keep being added and survive a restart."""
    good = _log_lines()
    damage = [b"\xff\xfe\x00 not json at all\n", b"[1, 2, 3]\n", b"42\n", b"null\n",
              b'"just a string"\n', b'{"kind": "measurement", "run_id": "aaaa0000\n',
              b'{"kind": "measurement", "run_id": "aaaa00000001"}\n',
              b'{"kind": "something else"}\n', b"\n"]
    damaged = good[:1] + damage[:4] + good[1:40] + damage[4:] + good[40:]
    app = _history_app(make_app, history=b"".join(damaged))
    app.launch()

    assert app.history_counts == (3, 96)
    assert_golden(app.export.csv("raw"), "expected/csv_raw.csv")

    app.play(Tone(note="A4", seconds=1.0))
    runs, measured = app.history_counts
    assert (runs, measured > 96) == (4, True)
    app.restart()
    assert app.history_counts == (runs, measured)


def test_history_records_with_wrong_field_types_are_skipped_not_fatal(make_app):
    """A run with no start time and a reading whose cents is text are dropped; the rest
    of my history exports exactly as before."""
    mistyped = [b'{"kind": "run", "run_id": "cccc00000001", "started_at": null, '
                b'"instrument": "eb_alto", "a4_hz": 440.0}\n',
                b'{"kind": "measurement", "run_id": "aaaa00000001", '
                b'"timestamp": "2026-03-02T18:59:00", "instrument": "eb_alto", "a4_hz": 440.0, '
                b'"midi_sounding": 69, "midi_fingered": 78, "cents": "sharp", "freq_hz": 440.0}\n']
    app = _history_app(make_app, history=b"".join(_log_lines() + mistyped))
    app.launch()

    assert app.history_counts == (3, 96)
    assert_golden(app.export.csv("raw"), "expected/csv_raw.csv")


def test_a_csv_export_that_fails_halfway_leaves_my_previous_file_intact(make_app):
    """Saving over last week's export: either the new file is complete or the old one
    is untouched, never half of the new one."""
    bad_reading = (b'{"kind": "measurement", "run_id": "aaaa00000003", '
                   b'"timestamp": "2026-03-16T19:59:00", "instrument": "eb_alto", '
                   b'"a4_hz": 440.0, "midi_sounding": 69, "midi_fingered": 78, '
                   b'"cents": "sharp", "freq_hz": 440.0}\n')
    app = _history_app(make_app, history=b"".join(_log_lines() + [bad_reading]))
    app.launch()
    target = app.home / "exports" / "history.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("last week's export\n", encoding="utf-8")

    app.export.csv("raw", path=target, keep_existing=True)

    if app.dialogs.last_message().kind == "critical":
        assert target.read_text(encoding="utf-8") == "last week's export\n"
    else:
        assert len(_csv_rows(target)) == app.export.csv_rows

"""Unit tests for intonation.export.report: the pure TXT / PDF renderers over a SessionSnapshot.

The acceptance suite drives the same renderers through the window; these
cover cases a scenario cannot reach cheaply (a piccolo header without
playing a note, a write that fails halfway) and pin the snapshot contract.
"""
from __future__ import annotations

import datetime
from collections import namedtuple

import pytest

from intonation.export import report as report_module
from intonation.i18n.strings import STRINGS
from intonation.persistence.atomic import atomic_open

Stat = namedtuple("Stat", "mean std n")
CREATED = datetime.datetime(2026, 10, 13, 16, 44)


def _snapshot(instrument="eb_alto", stats=None, **kw):
    return report_module.take_snapshot(
        instrument=instrument, instrument_label=instrument, a4=440.0,
        display="griff", sample_rate=44100,
        stats={} if stats is None else stats, created=CREATED, **kw)


def _header_line(report: str) -> str:
    return next(line for line in report.splitlines() if line.startswith("Transpos."))


def test_snapshot_lists_played_notes_only_in_pitch_order():
    snap = _snapshot(stats={72: Stat(-3.0, 1.0, 4), 60: Stat(2.0, 0.5, 3),
                            65: Stat(0.0, 0.0, 0)})
    assert [r.midi_sounding for r in snap.rows] == [60, 72]
    # Eb alto: sounding C4 is fingered A4.
    assert (snap.rows[0].sounding, snap.rows[0].fingered) == ("C4", "A4")
    assert snap.transposition == -9
    assert snap.total == 7


@pytest.mark.parametrize("instrument, key", [
    ("flute_piccolo", "txt_transp_8va"),
    ("double_bass", "txt_transp_8vb"),
])
@pytest.mark.parametrize("lang", ["en", "de"])
def test_an_octave_transposer_says_it_sounds_an_octave_away(instrument, key, lang):
    report = report_module.render_txt(_snapshot(instrument), lang)
    assert _header_line(report) == STRINGS[lang][key]
    assert STRINGS[lang]["txt_no_transp"] not in report


def test_a_transposition_of_several_octaves_names_the_count(monkeypatch):
    monkeypatch.setattr(report_module.instruments, "transp_map",
                        lambda: {"contra_test": -36})
    report = report_module.render_txt(_snapshot("contra_test"), "en")
    assert _header_line(report) == STRINGS["en"]["txt_transp_octaves_down"].format(n=3)


def test_other_transpositions_keep_the_fingered_c_wording():
    report = report_module.render_txt(_snapshot("eb_alto"), "en")
    assert _header_line(report) == STRINGS["en"]["txt_transp"].format(note="D#/Eb")


@pytest.mark.parametrize("lang, stamp", [("en", "2026-10-13 16:44"),
                                         ("de", "13.10.2026 16:44")])
def test_the_txt_date_is_written_the_way_the_language_writes_it(lang, stamp):
    report = report_module.render_txt(_snapshot(), lang)
    assert STRINGS[lang]["txt_date"].format(dt=stamp) in report.splitlines()


def test_a_failed_txt_write_leaves_the_previous_report_and_no_temp_file(tmp_path, monkeypatch):
    target = tmp_path / "report.txt"
    target.write_text("last week\n", encoding="utf-8")

    def boom(snapshot, lang):
        raise RuntimeError("renderer failed")
    monkeypatch.setattr(report_module, "render_txt", boom)

    with pytest.raises(RuntimeError):
        report_module.write_txt(_snapshot(), "en", target)
    assert target.read_text(encoding="utf-8") == "last week\n"
    assert [p.name for p in tmp_path.iterdir()] == ["report.txt"]


def test_write_txt_replaces_an_existing_report(tmp_path):
    target = tmp_path / "report.txt"
    target.write_text("last week\n", encoding="utf-8")
    snap = _snapshot(stats={60: Stat(2.0, 0.5, 3)})
    report_module.write_txt(snap, "en", target)
    assert target.read_text(encoding="utf-8") == report_module.render_txt(snap, "en")
    assert [p.name for p in tmp_path.iterdir()] == ["report.txt"]


def test_atomic_open_reports_an_unwritable_folder(tmp_path):
    missing = tmp_path / "no-such-folder" / "out.png"
    with pytest.raises(OSError):
        with atomic_open(missing, binary=True) as f:
            f.write(b"x")
    assert not missing.parent.exists()

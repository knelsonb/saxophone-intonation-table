"""
Phase-0 safety net for import_raw_csv — covers paths not exercised by the
existing test_csv_bom_import.py (which tests only the current-header + BOM
case).

NOTE: import_raw_csv is a method on MeasurementLog, not a module-level
function.  RAW_CSV_HEADER_LEGACY is a class attribute on MeasurementLog, not a
module-level constant.  The task brief asked for
    from sax_intonation_log import MeasurementLog, import_raw_csv, RAW_CSV_HEADER_LEGACY
but that import form would raise ImportError because neither name exists at
module scope.  Tests here use MeasurementLog.import_raw_csv and
MeasurementLog.RAW_CSV_HEADER_LEGACY instead.

Conflict policy (observed, locked here): DROP-ROW.  When a subsequent row
under an already-seen run_id disagrees on instrument or a4_hz, that individual
row is skipped via `continue`; the run itself (and its earlier accepted rows)
are kept.  The whole run is NOT dropped.

Legacy nickname: when importing a legacy CSV (no nickname column),
RunMeta.label is set to "" (empty string).
"""

from __future__ import annotations

import csv

import pytest

from sax_intonation_log import MeasurementLog

# Convenient aliases so test bodies read cleanly.
HEADER = MeasurementLog.RAW_CSV_HEADER
HEADER_LEGACY = MeasurementLog.RAW_CSV_HEADER_LEGACY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    """Write *header* + *rows* to *path* with utf-8, no BOM."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _fresh_log():
    return MeasurementLog(path=None)


# Minimal valid current-format row fields (all strings, as they appear in CSV).
# Callers may override individual fields.
_BASE_CURRENT = {
    "timestamp":      "2026-01-01T10:00:00",
    "run_id":         "run1",
    "run_started_at": "2026-01-01T10:00:00",
    "instrument":     "bb_tenor",
    "nickname":       "TestNick",
    "a4_hz":          "440.0",
    "midi_sounding":  "62",
    "sounding_note":  "D4",
    "midi_fingered":  "74",
    "fingered_note":  "D5",
    "cents":          "-5.0",
    "freq_hz":        "292.0",
    "maker":          "Yamaha",
    "model":          "YTS-280",
}

_BASE_LEGACY = {
    "timestamp":      "2026-01-01T10:00:00",
    "run_id":         "run1",
    "run_started_at": "2026-01-01T10:00:00",
    "instrument":     "bb_tenor",
    "a4_hz":          "440.0",
    "midi_sounding":  "62",
    "sounding_note":  "D4",
    "midi_fingered":  "74",
    "fingered_note":  "D5",
    "cents":          "-5.0",
    "freq_hz":        "292.0",
    "maker":          "Yamaha",
    "model":          "YTS-280",
}


def _current_row(**overrides):
    d = dict(_BASE_CURRENT)
    d.update(overrides)
    return [d[k] for k in (
        "timestamp", "run_id", "run_started_at", "instrument", "nickname",
        "a4_hz", "midi_sounding", "sounding_note", "midi_fingered",
        "fingered_note", "cents", "freq_hz", "maker", "model",
    )]


def _legacy_row(**overrides):
    d = dict(_BASE_LEGACY)
    d.update(overrides)
    return [d[k] for k in (
        "timestamp", "run_id", "run_started_at", "instrument",
        "a4_hz", "midi_sounding", "sounding_note", "midi_fingered",
        "fingered_note", "cents", "freq_hz", "maker", "model",
    )]


# ---------------------------------------------------------------------------
# Test 1: legacy header round-trip — 3 valid rows, 1 run
# ---------------------------------------------------------------------------

def test_legacy_header_three_rows(tmp_path):
    """A RAW_CSV_HEADER_LEGACY file with 3 consistent rows under one run_id
    should import as 1 run and 3 measurements."""
    rows = [
        _legacy_row(midi_sounding="60", cents="-3.0", freq_hz="261.0"),
        _legacy_row(midi_sounding="62", cents="-5.0", freq_hz="292.0"),
        _legacy_row(midi_sounding="64", cents="+2.0", freq_hz="328.0"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER_LEGACY, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert (runs_added, meas_added) == (1, 3)

    measurements = log.measurements()
    assert len(measurements) == 3

    # All 3 measurements belong to run1.
    assert all(m.run_id == "run1" for m in measurements)

    # Legacy path sets label (nickname) to empty string.
    runs = log.runs()
    assert len(runs) == 1
    assert runs[0].label == ""

    # Verify instrument and a4_hz were captured correctly.
    assert runs[0].instrument == "bb_tenor"
    assert runs[0].a4_hz == pytest.approx(440.0)

    # Spot-check one measurement's numeric fields.
    midi_values = {m.midi_sounding for m in measurements}
    assert midi_values == {60, 62, 64}


# ---------------------------------------------------------------------------
# Test 2: legacy header — nickname field is absent; label defaults to ""
# (Covers the specific "lock current behaviour" requirement for legacy nickname.)
# ---------------------------------------------------------------------------

def test_legacy_header_nickname_is_empty_string(tmp_path):
    """Legacy CSV has no nickname column; imported run's label must be ''."""
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER_LEGACY, [_legacy_row()])

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert (runs_added, meas_added) == (1, 1)
    runs = log.runs()
    assert runs[0].label == ""


# ---------------------------------------------------------------------------
# Test 3: intra-run instrument conflict — drop-row policy
# ---------------------------------------------------------------------------

def test_intra_run_instrument_conflict_drops_conflicting_row(tmp_path):
    """Row 1: instrument='bb_tenor'.  Row 2: instrument='eb_alto' (conflict).
    Row 3: instrument='bb_tenor' (consistent with row 1).

    Expected: row 2 is silently dropped.
    meas_added == 2 (rows 1 and 3 accepted).
    No measurement in the log has instrument='eb_alto'.
    The run itself survives with instrument='bb_tenor'.
    """
    rows = [
        _current_row(midi_sounding="60", instrument="bb_tenor"),
        _current_row(midi_sounding="62", instrument="eb_alto"),   # conflict
        _current_row(midi_sounding="64", instrument="bb_tenor"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    # Policy: drop-row, not drop-run.
    assert runs_added == 1
    assert meas_added == 2

    measurements = log.measurements()
    assert len(measurements) == 2

    instruments_in_log = {m.instrument for m in measurements}
    assert "eb_alto" not in instruments_in_log
    assert "bb_tenor" in instruments_in_log

    midi_values = {m.midi_sounding for m in measurements}
    assert midi_values == {60, 64}


# ---------------------------------------------------------------------------
# Test 4: intra-run A4 conflict — drop-row policy
# ---------------------------------------------------------------------------

def test_intra_run_a4_conflict_drops_conflicting_row(tmp_path):
    """Row 1: a4_hz=440.0.  Row 2: a4_hz=441.0 (conflict > 0.01 tolerance).
    Row 3: a4_hz=440.0 (consistent).

    Expected: row 2 dropped; meas_added == 2.
    """
    rows = [
        _current_row(midi_sounding="60", a4_hz="440.0"),
        _current_row(midi_sounding="62", a4_hz="441.0"),   # conflict
        _current_row(midi_sounding="64", a4_hz="440.0"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert runs_added == 1
    assert meas_added == 2

    measurements = log.measurements()
    assert len(measurements) == 2

    # No measurement should carry the conflicting a4_hz.
    a4_values = {m.a4_hz for m in measurements}
    assert 441.0 not in a4_values
    assert 440.0 in a4_values

    midi_values = {m.midi_sounding for m in measurements}
    assert midi_values == {60, 64}


# ---------------------------------------------------------------------------
# Test 4b: A4 within tolerance boundary (0.005 Hz difference) is NOT dropped
# ---------------------------------------------------------------------------

def test_intra_run_a4_within_tolerance_is_accepted(tmp_path):
    """a4_hz difference of 0.005 is within the 0.01 tolerance and must NOT be
    treated as a conflict.  Both rows should be accepted."""
    rows = [
        _current_row(midi_sounding="60", a4_hz="440.000"),
        _current_row(midi_sounding="62", a4_hz="440.005"),  # within tolerance
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert (runs_added, meas_added) == (1, 2)


# ---------------------------------------------------------------------------
# Test 5: duplicate run_id against a pre-populated log — rows are skipped
# ---------------------------------------------------------------------------

def test_duplicate_run_id_against_existing_log_is_skipped(tmp_path):
    """Pre-populate the log with run_id='run1'.  Import a CSV that also
    contains run_id='run1'.  All rows for that run_id must be skipped.
    meas_added must be 0; existing measurements are unchanged."""
    # Pre-populate log with one run and one measurement for "run1".
    log = _fresh_log()
    log._runs["run1"] = __import__(
        "sax_intonation_log", fromlist=["RunMeta"]
    ).RunMeta(
        run_id="run1",
        started_at="2025-01-01T09:00:00",
        instrument="bb_tenor",
        a4_hz=440.0,
    )
    # Add a measurement directly so we can verify it is untouched.
    from sax_intonation_log import Measurement
    existing_m = Measurement(
        run_id="run1",
        timestamp="2025-01-01T09:00:01",
        instrument="bb_tenor",
        a4_hz=440.0,
        midi_sounding=60,
        midi_fingered=72,
        cents=0.0,
        freq_hz=261.6,
    )
    log._measurements.append(existing_m)

    # CSV contains the same run_id with two rows.
    rows = [
        _current_row(midi_sounding="62"),
        _current_row(midi_sounding="64"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    runs_added, meas_added = log.import_raw_csv(p)

    assert (runs_added, meas_added) == (0, 0)

    # Original measurement must still be the only one in the log.
    measurements = log.measurements()
    assert len(measurements) == 1
    assert measurements[0].midi_sounding == 60


# ---------------------------------------------------------------------------
# Test 6: malformed row (non-numeric cents) — bad row skipped, rest accepted
# ---------------------------------------------------------------------------

def test_malformed_row_non_numeric_cents_is_skipped(tmp_path):
    """A row with a non-numeric cents value must be silently skipped.
    The surrounding valid rows must still be imported."""
    rows = [
        _current_row(midi_sounding="60", cents="-3.0"),
        _current_row(midi_sounding="62", cents="NOT_A_NUMBER"),  # bad
        _current_row(midi_sounding="64", cents="+1.5"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    # The bad row is silently skipped; 2 of 3 measurements make it through.
    # The run is still created because two valid rows share run1.
    assert runs_added == 1
    assert meas_added == 2

    midi_values = {m.midi_sounding for m in log.measurements()}
    assert midi_values == {60, 64}


# ---------------------------------------------------------------------------
# Test 6b: malformed row — missing column (wrong column count)
# ---------------------------------------------------------------------------

def test_malformed_row_missing_column_is_skipped(tmp_path):
    """A row that is shorter than the expected column count must be skipped."""
    p = tmp_path / "in.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        # Good row first so the run gets created.
        w.writerow(_current_row(midi_sounding="60"))
        # Truncated row — only 5 columns instead of 14.
        w.writerow(["2026-01-01T10:00:01", "run1", "2026-01-01T10:00:00", "bb_tenor", "TestNick"])
        # Good row last.
        w.writerow(_current_row(midi_sounding="64"))

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert runs_added == 1
    assert meas_added == 2

    midi_values = {m.midi_sounding for m in log.measurements()}
    assert midi_values == {60, 64}


# ---------------------------------------------------------------------------
# Test 7: header-only CSV (no data rows) — returns (0, 0), no exception
# ---------------------------------------------------------------------------

def test_empty_csv_header_only(tmp_path):
    """A CSV that contains only the header line must return (0, 0) without
    raising."""
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, [])

    log = _fresh_log()
    result = log.import_raw_csv(p)

    assert result == (0, 0)
    assert log.measurements() == []
    assert log.runs() == []


def test_empty_csv_legacy_header_only(tmp_path):
    """Same as above but with the legacy header — still (0, 0)."""
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER_LEGACY, [])

    log = _fresh_log()
    result = log.import_raw_csv(p)

    assert result == (0, 0)


# ---------------------------------------------------------------------------
# Test 7b: completely empty file (not even a header)
# ---------------------------------------------------------------------------

def test_completely_empty_file(tmp_path):
    """A zero-byte file must return (0, 0) without raising."""
    p = tmp_path / "in.csv"
    p.write_text("", encoding="utf-8")

    log = _fresh_log()
    result = log.import_raw_csv(p)

    assert result == (0, 0)


# ---------------------------------------------------------------------------
# Test 8: wrong header — must raise ValueError
# ---------------------------------------------------------------------------

def test_wrong_header_raises_value_error(tmp_path):
    """A CSV whose header matches neither RAW_CSV_HEADER nor
    RAW_CSV_HEADER_LEGACY must raise ValueError."""
    p = tmp_path / "in.csv"
    _write_csv(p, ["col_a", "col_b", "col_c"], [["x", "y", "z"]])

    log = _fresh_log()
    with pytest.raises(ValueError, match="does not match a known raw export"):
        log.import_raw_csv(p)


def test_wrong_header_per_run_note_slice_raises_value_error(tmp_path):
    """A per_run_note aggregated export CSV must also raise — its header is
    neither of the two raw headers."""
    per_run_header = [
        "run_id", "run_started_at", "instrument", "nickname", "a4_hz",
        "midi_sounding", "sounding_note", "fingered_note",
        "n", "mean_cents", "std_cents", "min_cents", "max_cents",
        "maker", "model",
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, per_run_header, [])

    log = _fresh_log()
    with pytest.raises(ValueError):
        log.import_raw_csv(p)


# ---------------------------------------------------------------------------
# Test 9: two distinct run_ids in one file
# ---------------------------------------------------------------------------

def test_two_runs_in_one_file(tmp_path):
    """A CSV with rows belonging to two different run_ids must create two runs
    and the correct measurement count for each."""
    rows = [
        _current_row(run_id="run_a", midi_sounding="60"),
        _current_row(run_id="run_a", midi_sounding="62"),
        _current_row(run_id="run_b", midi_sounding="65",
                     instrument="eb_alto", run_started_at="2026-01-02T10:00:00",
                     timestamp="2026-01-02T10:00:01"),
    ]
    p = tmp_path / "in.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)

    assert runs_added == 2
    assert meas_added == 3

    run_ids = {r.run_id for r in log.runs()}
    assert run_ids == {"run_a", "run_b"}


# ---------------------------------------------------------------------------
# Test 10: import refuses files larger than MAX_IMPORT_BYTES (v0.6 cap)
# ---------------------------------------------------------------------------

def test_file_size_cap_refuses_outsized_files(tmp_path, monkeypatch):
    """A CSV whose on-disk size exceeds MAX_IMPORT_BYTES must raise
    ValueError before any reader I/O happens.  Monkeypatch the cap down
    so we don't have to write 50 MB to disk."""
    monkeypatch.setattr(MeasurementLog, "MAX_IMPORT_BYTES", 100)
    rows = [_current_row(midi_sounding=str(60 + i % 12)) for i in range(50)]
    p = tmp_path / "big.csv"
    _write_csv(p, HEADER, rows)
    assert p.stat().st_size > 100  # sanity — the test file IS over the cap

    log = _fresh_log()
    with pytest.raises(ValueError, match="too large"):
        log.import_raw_csv(p)
    assert len(list(log.runs())) == 0  # nothing was imported


# ---------------------------------------------------------------------------
# Test 11: import refuses files with more rows than MAX_IMPORT_ROWS (v0.6 cap)
# ---------------------------------------------------------------------------

def test_row_count_cap_refuses_too_many_rows(tmp_path, monkeypatch):
    """Once the row counter exceeds MAX_IMPORT_ROWS, the loop must raise
    ValueError.  Monkeypatch the cap to 10 so the test stays cheap."""
    monkeypatch.setattr(MeasurementLog, "MAX_IMPORT_ROWS", 10)
    rows = [_current_row(midi_sounding=str(60 + i % 12)) for i in range(20)]
    p = tmp_path / "many.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    with pytest.raises(ValueError, match="exceeds"):
        log.import_raw_csv(p)


# ---------------------------------------------------------------------------
# Test 12: imports at the row-cap boundary succeed
# ---------------------------------------------------------------------------

def test_row_count_at_cap_is_accepted(tmp_path, monkeypatch):
    """Exactly MAX_IMPORT_ROWS rows must be accepted; the cap is a strict
    > comparison, not >=."""
    monkeypatch.setattr(MeasurementLog, "MAX_IMPORT_ROWS", 5)
    rows = [_current_row(midi_sounding=str(60 + i)) for i in range(5)]
    p = tmp_path / "edge.csv"
    _write_csv(p, HEADER, rows)

    log = _fresh_log()
    runs_added, meas_added = log.import_raw_csv(p)
    assert runs_added == 1
    assert meas_added == 5

"""v0.5.7.7: a UTF-8 BOM on the front of a re-imported CSV (Excel's
default on Windows) must not break import_raw_csv.
"""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

from sax_intonation_log import MeasurementLog


def test_import_raw_csv_accepts_utf8_bom() -> None:
    header = MeasurementLog.RAW_CSV_HEADER
    row = (
        "2026-05-25T12:00:00",  # timestamp
        "run-bom-1",            # run_id
        "2026-05-25T12:00:00",  # run_started_at
        "Alto",                  # instrument
        "bom-nickname",         # nickname
        "440.0",                 # a4_hz
        "69",                    # midi_sounding
        "A4",                    # sounding_note
        "57",                    # midi_fingered
        "A3",                    # fingered_note
        "0.0",                   # cents
        "440.0",                 # freq_hz
        "TestMaker",             # maker
        "TestModel",             # model
    )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerow(row)
    payload = buf.getvalue()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "excel-roundtrip.csv"
        # Excel's "CSV UTF-8 (Comma delimited)" prepends U+FEFF.
        p.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))

        log = MeasurementLog(path=None)
        runs_added, meas_added = log.import_raw_csv(p)

    assert (runs_added, meas_added) == (1, 1), (runs_added, meas_added)
    print("test_csv_bom_import: OK")


if __name__ == '__main__':
    test_import_raw_csv_accepts_utf8_bom()

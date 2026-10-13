"""Regenerate log.jsonl, the small measurement history the CSV goldens read.

    python tests/golden/make_log_fixture.py

Deterministic: fixed run ids, timestamps and a formula for the cents. Three runs
on two instruments, one of them with a name containing a comma, quotes and an
umlaut, so the CSV quoting and encoding are part of what the goldens pin.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    # run_id, started_at, instrument, a4, maker, model, label, transposition, notes
    ("aaaa00000001", "2026-03-02T18:00:00", "eb_alto", 440.0, "Selmer",
     'Mark VI, "gold"', "Selmer Mark VI", -9, range(63, 73)),
    ("aaaa00000002", "2026-03-09T18:05:00", "bb_tenor", 442.0, "Yamaha",
     "YTS-62", "Yamaha YTS-62", -14, range(54, 62)),
    ("aaaa00000003", "2026-03-16T19:30:00", "eb_alto", 440.0, "", "",
     "Müller Bariton", -9, range(64, 70)),
]


def main() -> None:
    lines = []
    for n, (rid, started, instr, a4, maker, model, label, transp, notes) in enumerate(RUNS):
        lines.append(json.dumps({"kind": "run", "run_id": rid, "started_at": started,
                                 "instrument": instr, "a4_hz": a4, "maker": maker,
                                 "model": model, "label": label}))
        for i, midi in enumerate(notes):
            for k in range(3 + (midi + n) % 3):
                cents = float(((midi * 7 + k * 5 + n * 3) % 31) - 15)
                freq = a4 * 2.0 ** ((midi - 69) / 12.0 + cents / 1200.0)
                minute = (i * 4 + k) % 60
                lines.append(json.dumps({
                    "kind": "measurement", "run_id": rid,
                    "timestamp": f"{started[:11]}{18 + n}:{minute:02d}:{(k * 7) % 60:02d}",
                    "instrument": instr, "a4_hz": a4, "midi_sounding": midi,
                    "midi_fingered": midi - transp, "cents": cents,
                    "freq_hz": round(freq, 3)}))
    (HERE / "log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote log.jsonl ({len(lines)} lines)")


if __name__ == "__main__":
    main()

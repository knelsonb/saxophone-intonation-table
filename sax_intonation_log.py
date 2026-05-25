"""
Measurement logging and CSV export for the Saxophone Intonation Analyzer.

Every detected note is recorded as one row in an in-memory log. The GUI can
then export filtered/aggregated subsets as CSV.

Persistence is OPT-IN. If a `path` is passed to `MeasurementLog`, every run
and measurement is also appended as a JSON Lines record so history can
survive between sessions. The default is in-memory only — the GUI enables
persistence when the environment variable `SAX_INTONATION_LOG_PATH` is set.

Run semantics
-------------
A "run" is a contiguous span of recording with one instrument and one concert
pitch A. The GUI opens a new run on app start, on instrument change, on A4
change, and when recording resumes from a pause. Each run records its
instrument and A4 at the moment it opens; measurements added to that run
inherit those values, so an in-flight audio callback arriving milliseconds
after a UI change still attributes to the run that was active when it fired.

Note names are universal music theory and live in this module. Mapping from
sounding to fingered MIDI requires per-instrument transposition, which the
GUI knows; callers therefore supply both MIDI numbers to `add_measurement`.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


_CHROMA = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F',
           'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def midi_note_name(m: int) -> str:
    return f"{_CHROMA[m % 12]}{m // 12 - 1}"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class RunMeta:
    run_id: str
    started_at: str          # ISO-8601 local time
    instrument: str          # instrument key, e.g. "eb_alto"
    a4_hz: float
    maker: str = ""
    model: str = ""
    label: str = ""          # optional free-form label

    def to_jsonl(self) -> str:
        return json.dumps({"kind": "run", **asdict(self)})


@dataclass
class Measurement:
    run_id: str
    timestamp: str           # ISO-8601 local time, second precision
    instrument: str          # latched from the run at write time
    a4_hz: float             # latched from the run at write time
    midi_sounding: int
    midi_fingered: int       # caller supplies — depends on instrument transp.
    cents: float
    freq_hz: float

    def to_jsonl(self) -> str:
        return json.dumps({"kind": "measurement", **asdict(self)})


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------
class MeasurementLog:
    """Thread-safe in-memory log with optional JSONL persistence.

    The audio callback runs on a non-Qt thread; `add_measurement` is the only
    method called from there. Everything else is invoked from the UI thread.
    """

    def __init__(self, path: Optional[Path | str] = None):
        self._lock = threading.Lock()
        self._runs: dict[str, RunMeta] = {}
        self._measurements: list[Measurement] = []
        self._current_run_id: Optional[str] = None
        # Count of measurements in the current run; used to coalesce empties.
        self._current_run_count: int = 0
        self._path = Path(path) if path else None
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = obj.pop("kind", None)
                    if kind == "run":
                        try:
                            run = RunMeta(**obj)
                        except TypeError:
                            continue
                        self._runs[run.run_id] = run
                    elif kind == "measurement":
                        try:
                            m = Measurement(**obj)
                        except TypeError:
                            continue
                        self._measurements.append(m)
        except OSError:
            # Logging is best-effort; never block the app on disk issues.
            pass

    def _append_line(self, line: str) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # -- run management ---------------------------------------------------
    def start_run(self, instrument: str, a4_hz: float,
                  maker: str = "", model: str = "",
                  label: str = "") -> str:
        """Open a new run. If the *current* run has no measurements yet, it is
        replaced rather than left behind as an empty run record."""
        run = RunMeta(
            run_id=uuid.uuid4().hex[:12],
            started_at=datetime.datetime.now().isoformat(timespec="seconds"),
            instrument=instrument,
            a4_hz=float(a4_hz),
            maker=maker,
            model=model,
            label=label,
        )
        with self._lock:
            prev_id = self._current_run_id
            if (prev_id is not None and self._current_run_count == 0
                    and prev_id in self._runs):
                # Coalesce: drop the empty predecessor so A4-scrubbing
                # doesn't litter the log with one-second runs.
                del self._runs[prev_id]
            self._runs[run.run_id] = run
            self._current_run_id = run.run_id
            self._current_run_count = 0
        self._append_line(run.to_jsonl())
        return run.run_id

    def end_run(self) -> None:
        with self._lock:
            self._current_run_id = None
            self._current_run_count = 0

    def set_current_run_metadata(self, *, maker: str = "",
                                  model: str = "", label: str = "") -> None:
        """Stamp maker/model/label onto the currently-active run, in memory.

        Persistence is deliberately not re-issued (the original `run` record
        is already on disk). The in-memory copy is what feeds CSV export, so
        the next export picks the new values up; the on-disk record carries
        whatever was known at run-open time."""
        with self._lock:
            run_id = self._current_run_id
            if run_id is None or run_id not in self._runs:
                return
            run = self._runs[run_id]
            if maker:
                run.maker = maker
            if model:
                run.model = model
            if label:
                run.label = label

    @property
    def current_run_id(self) -> Optional[str]:
        with self._lock:
            return self._current_run_id

    # -- measurement intake ----------------------------------------------
    def add_measurement(self, midi_sounding: int, midi_fingered: int,
                        cents: float, freq_hz: float) -> None:
        """Record one measurement against the currently-active run.

        The run's instrument and A4 are latched onto the row at write time,
        so the value reflects what was active when `start_run` was called —
        not what `self.instrument` might have changed to milliseconds later.
        """
        with self._lock:
            run_id = self._current_run_id
            if run_id is None or run_id not in self._runs:
                return
            run = self._runs[run_id]
            m = Measurement(
                run_id=run_id,
                timestamp=datetime.datetime.now().isoformat(
                    timespec="seconds"),
                instrument=run.instrument,
                a4_hz=run.a4_hz,
                midi_sounding=int(midi_sounding),
                midi_fingered=int(midi_fingered),
                cents=float(cents),
                freq_hz=float(freq_hz),
            )
            self._measurements.append(m)
            self._current_run_count += 1
        self._append_line(m.to_jsonl())

    # -- inspection -------------------------------------------------------
    def runs(self) -> list[RunMeta]:
        with self._lock:
            return sorted(self._runs.values(),
                          key=lambda r: r.started_at, reverse=True)

    def instruments(self) -> list[str]:
        with self._lock:
            seen = {r.instrument for r in self._runs.values()}
            seen.update(m.instrument for m in self._measurements)
        return sorted(seen)

    def measurements(self) -> list[Measurement]:
        with self._lock:
            return list(self._measurements)

    # -- import -----------------------------------------------------------
    RAW_CSV_HEADER = (
        "timestamp", "run_id", "run_started_at", "instrument", "nickname",
        "a4_hz", "midi_sounding", "sounding_note", "midi_fingered",
        "fingered_note", "cents", "freq_hz", "maker", "model",
    )
    # Header from before the nickname column was added; accepted on import
    # so old exports still round-trip cleanly.
    RAW_CSV_HEADER_LEGACY = (
        "timestamp", "run_id", "run_started_at", "instrument", "a4_hz",
        "midi_sounding", "sounding_note", "midi_fingered", "fingered_note",
        "cents", "freq_hz", "maker", "model",
    )

    def import_raw_csv(self, path: Path | str) -> tuple[int, int]:
        """Load a previously-exported `raw` CSV into the log as historical
        runs. Returns (runs_added, measurements_added).

        Only the `raw` slice mode round-trips — aggregated modes drop the
        timestamp and per-measurement frequency, so they can't be restored.

        Imported runs become historical (no active-run pointer is touched).
        Existing `run_id`s in the log are skipped to avoid duplicates from
        re-importing the same file.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = tuple(next(reader))
            except StopIteration:
                return (0, 0)
            if header == self.RAW_CSV_HEADER:
                legacy = False
            elif header == self.RAW_CSV_HEADER_LEGACY:
                legacy = True
            else:
                raise ValueError(
                    "CSV header does not match a known raw export format. "
                    f"Got {header!r}.")

            with self._lock:
                existing_runs = set(self._runs.keys())

            new_runs: dict[str, RunMeta] = {}
            new_measurements: list[Measurement] = []

            for row in reader:
                expected_len = (len(self.RAW_CSV_HEADER_LEGACY) if legacy
                                else len(self.RAW_CSV_HEADER))
                if len(row) != expected_len:
                    continue
                if legacy:
                    (timestamp, run_id, run_started_at, instrument, a4_hz_s,
                     midi_sounding_s, _sounding, midi_fingered_s, _fingered,
                     cents_s, freq_s, maker, model) = row
                    nickname = ""
                else:
                    (timestamp, run_id, run_started_at, instrument, nickname,
                     a4_hz_s, midi_sounding_s, _sounding, midi_fingered_s,
                     _fingered, cents_s, freq_s, maker, model) = row
                if not run_id or run_id in existing_runs:
                    continue
                try:
                    a4_hz = float(a4_hz_s)
                    midi_sounding = int(midi_sounding_s)
                    midi_fingered = int(midi_fingered_s)
                    cents = float(cents_s)
                    freq_hz = float(freq_s)
                except ValueError:
                    continue

                if run_id not in new_runs:
                    new_runs[run_id] = RunMeta(
                        run_id=run_id,
                        started_at=run_started_at or timestamp,
                        instrument=instrument,
                        a4_hz=a4_hz,
                        maker=maker,
                        model=model,
                        label=nickname,
                    )
                else:
                    # Reject rows whose instrument or A4 disagrees with the
                    # first row of the same run_id. A hand-edited file with
                    # mixed instruments under one run_id would otherwise
                    # split silently across aggregators.
                    head = new_runs[run_id]
                    if (instrument != head.instrument
                            or abs(a4_hz - head.a4_hz) > 0.01):
                        continue
                new_measurements.append(Measurement(
                    run_id=run_id,
                    timestamp=timestamp,
                    instrument=instrument,
                    a4_hz=a4_hz,
                    midi_sounding=midi_sounding,
                    midi_fingered=midi_fingered,
                    cents=cents,
                    freq_hz=freq_hz,
                ))

        # Persist imported records too, so the user's running log subsumes
        # imports if persistence is enabled.
        with self._lock:
            for run in new_runs.values():
                self._runs[run.run_id] = run
            self._measurements.extend(new_measurements)
        # Bulk-append: a 50k-row import opening the JSONL 50k times is wasteful
        # and slow on Windows. One open() covers the whole batch.
        if self._path and (new_runs or new_measurements):
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    for run in new_runs.values():
                        f.write(run.to_jsonl() + "\n")
                    for m in new_measurements:
                        f.write(m.to_jsonl() + "\n")
            except OSError:
                pass
        return (len(new_runs), len(new_measurements))

    # -- export -----------------------------------------------------------
    SLICE_MODES = (
        "raw",                    # one row per measurement
        "per_run_note",           # one row per (run, note) — mean/std/n
        "per_instrument_note",    # aggregated across runs, per instrument+note
        "per_nickname_note",      # aggregated across runs sharing a nickname
        "instrument_avg",         # one instrument, aggregated per note
        "overall_per_note",       # aggregated across everything, per note
    )

    def export_csv(self, path: Path | str, *,
                   mode: str = "raw",
                   run_id: Optional[str] = None,
                   instrument: Optional[str] = None,
                   nickname: Optional[str] = None) -> int:
        """Write a CSV slice. Returns the number of data rows written.

        `nickname` filter: when set, only runs whose `label` matches exactly
        are included. Only meaningful in modes that aggregate across runs."""
        if mode not in self.SLICE_MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "instrument_avg" and not instrument:
            raise ValueError("instrument_avg mode requires an instrument")

        with self._lock:
            measurements = list(self._measurements)
            runs = dict(self._runs)

        if run_id is not None:
            measurements = [m for m in measurements if m.run_id == run_id]
        if instrument is not None:
            measurements = [m for m in measurements
                            if m.instrument == instrument]
        if nickname is not None:
            keep_ids = {rid for rid, r in runs.items() if r.label == nickname}
            measurements = [m for m in measurements if m.run_id in keep_ids]

        path = Path(path)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if mode == "raw":
                return _write_raw(w, measurements, runs)
            if mode == "per_run_note":
                return _write_per_run_note(w, measurements, runs)
            if mode == "per_instrument_note":
                return _write_per_instrument_note(w, measurements, runs)
            if mode == "per_nickname_note":
                return _write_per_nickname_note(w, measurements, runs)
            if mode == "instrument_avg":
                return _write_instrument_avg(w, measurements, instrument)
            return _write_overall(w, measurements)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _agg_stats(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std, min(values), max(values)


def _write_raw(w, measurements: list[Measurement],
               runs: dict[str, RunMeta]) -> int:
    w.writerow([
        "timestamp", "run_id", "run_started_at", "instrument", "nickname",
        "a4_hz", "midi_sounding", "sounding_note", "midi_fingered",
        "fingered_note", "cents", "freq_hz", "maker", "model",
    ])
    n = 0
    for m in measurements:
        run = runs.get(m.run_id)
        w.writerow([
            m.timestamp, m.run_id,
            run.started_at if run else "",
            m.instrument,
            run.label if run else "",
            f"{m.a4_hz:.2f}",
            m.midi_sounding, midi_note_name(m.midi_sounding),
            m.midi_fingered, midi_note_name(m.midi_fingered),
            f"{m.cents:.3f}", f"{m.freq_hz:.3f}",
            run.maker if run else "",
            run.model if run else "",
        ])
        n += 1
    return n


def _aggregate(measurements: Iterable[Measurement], key_fn):
    buckets: dict = {}
    for m in measurements:
        buckets.setdefault(key_fn(m), []).append(m.cents)
    return buckets


def _write_per_run_note(w, measurements: list[Measurement],
                         runs: dict[str, RunMeta]) -> int:
    w.writerow([
        "run_id", "run_started_at", "instrument", "nickname", "a4_hz",
        "midi_sounding", "sounding_note", "fingered_note",
        "n", "mean_cents", "std_cents", "min_cents", "max_cents",
        "maker", "model",
    ])
    # Key: (run_id, midi_sounding). We need a representative measurement to
    # recover midi_fingered (it can differ between runs of the same note if
    # the instrument changes).
    buckets: dict = {}
    repr_measurement: dict = {}
    for m in measurements:
        key = (m.run_id, m.midi_sounding)
        buckets.setdefault(key, []).append(m.cents)
        repr_measurement.setdefault(key, m)
    n_rows = 0
    for (run_id, midi), vals in sorted(buckets.items(),
                                        key=lambda kv: (kv[0][0], kv[0][1])):
        mean, std, mn, mx = _agg_stats(vals)
        run = runs.get(run_id)
        rep = repr_measurement[(run_id, midi)]
        w.writerow([
            run_id,
            run.started_at if run else "",
            rep.instrument,
            run.label if run else "",
            f"{rep.a4_hz:.2f}",
            midi, midi_note_name(midi),
            midi_note_name(rep.midi_fingered),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
            run.maker if run else "",
            run.model if run else "",
        ])
        n_rows += 1
    return n_rows


def _write_per_instrument_note(w, measurements: list[Measurement],
                                 runs: dict[str, RunMeta]) -> int:
    w.writerow([
        "instrument", "nicknames", "midi_sounding", "sounding_note",
        "fingered_note", "runs", "n",
        "mean_cents", "std_cents", "min_cents", "max_cents",
    ])
    buckets: dict = {}
    runs_seen: dict = {}
    fingered: dict = {}
    for m in measurements:
        key = (m.instrument, m.midi_sounding)
        buckets.setdefault(key, []).append(m.cents)
        runs_seen.setdefault(key, set()).add(m.run_id)
        fingered.setdefault(key, m.midi_fingered)
    n_rows = 0
    for (instrument, midi), vals in sorted(buckets.items()):
        mean, std, mn, mx = _agg_stats(vals)
        nick_set = sorted({
            runs[rid].label for rid in runs_seen[(instrument, midi)]
            if rid in runs and runs[rid].label
        })
        w.writerow([
            instrument,
            ";".join(nick_set),
            midi, midi_note_name(midi),
            midi_note_name(fingered[(instrument, midi)]),
            len(runs_seen[(instrument, midi)]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_per_nickname_note(w, measurements: list[Measurement],
                              runs: dict[str, RunMeta]) -> int:
    """One row per (nickname, note). Runs without a nickname are bucketed
    under an empty-string nickname so they're still visible."""
    w.writerow([
        "nickname", "instrument", "midi_sounding", "sounding_note",
        "fingered_note", "runs", "n",
        "mean_cents", "std_cents", "min_cents", "max_cents",
    ])
    buckets: dict = {}
    runs_seen: dict = {}
    fingered: dict = {}
    instrument_seen: dict = {}
    for m in measurements:
        nick = runs[m.run_id].label if m.run_id in runs else ""
        key = (nick, m.midi_sounding)
        buckets.setdefault(key, []).append(m.cents)
        runs_seen.setdefault(key, set()).add(m.run_id)
        fingered.setdefault(key, m.midi_fingered)
        instrument_seen.setdefault(key, set()).add(m.instrument)
    n_rows = 0
    for (nick, midi), vals in sorted(buckets.items()):
        mean, std, mn, mx = _agg_stats(vals)
        w.writerow([
            nick or "(unnamed)",
            ";".join(sorted(instrument_seen[(nick, midi)])),
            midi, midi_note_name(midi),
            midi_note_name(fingered[(nick, midi)]),
            len(runs_seen[(nick, midi)]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_instrument_avg(w, measurements: list[Measurement],
                           instrument: str) -> int:
    w.writerow([
        "instrument", "midi_sounding", "sounding_note", "fingered_note",
        "runs", "n", "mean_cents", "std_cents", "min_cents", "max_cents",
    ])
    buckets: dict = {}
    runs_seen: dict = {}
    fingered: dict = {}
    for m in measurements:
        buckets.setdefault(m.midi_sounding, []).append(m.cents)
        runs_seen.setdefault(m.midi_sounding, set()).add(m.run_id)
        fingered.setdefault(m.midi_sounding, m.midi_fingered)
    n_rows = 0
    for midi, vals in sorted(buckets.items()):
        mean, std, mn, mx = _agg_stats(vals)
        w.writerow([
            instrument, midi,
            midi_note_name(midi),
            midi_note_name(fingered[midi]),
            len(runs_seen[midi]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_overall(w, measurements: list[Measurement]) -> int:
    w.writerow([
        "midi_sounding", "sounding_note",
        "instruments", "runs", "n",
        "mean_cents", "std_cents", "min_cents", "max_cents",
    ])
    buckets: dict = {}
    runs_seen: dict = {}
    instruments_seen: dict = {}
    for m in measurements:
        buckets.setdefault(m.midi_sounding, []).append(m.cents)
        runs_seen.setdefault(m.midi_sounding, set()).add(m.run_id)
        instruments_seen.setdefault(m.midi_sounding, set()).add(m.instrument)
    n_rows = 0
    for midi, vals in sorted(buckets.items()):
        mean, std, mn, mx = _agg_stats(vals)
        w.writerow([
            midi, midi_note_name(midi),
            ";".join(sorted(instruments_seen[midi])),
            len(runs_seen[midi]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows

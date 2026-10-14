"""
Measurement logging and CSV export for the Saxophone Intonation Analyzer.

Every detected note is recorded as one row in the measurement log. The GUI
can then export filtered/aggregated subsets as CSV.

Persistence is OPT-IN. If a `path` is passed to `MeasurementLog`, every run
and measurement is also appended as a JSON Lines record so history can
survive between sessions. The default is in-memory only — the GUI enables
persistence when the user opts in on the welcome dialog, or when the
environment variable `SAX_INTONATION_LOG_PATH` names a file.

On disk the history is split by month (see `HistoryFiles`): the live file
holds this month's runs and earlier months sit beside it as
`log-YYYY-MM.jsonl`. Earlier sessions are read on demand by exports and
imports, streamed record by record, so start-up cost does not grow with the
history. Every record read back is validated; a damaged one is skipped.

Run semantics
-------------
A "run" is a contiguous span of recording with one instrument and one concert
pitch A. The GUI opens a new run on app start, on instrument change, on A4
change, on nickname change, and when recording resumes from a pause. Each run
records its instrument, A4 and nickname at the moment it opens; measurements
added to that run inherit those values, so an in-flight audio callback arriving
milliseconds after a UI change still attributes to the run that was active when
it fired.

Note names are universal music theory and live in this module. Mapping from
sounding to fingered MIDI requires per-instrument transposition, which the
GUI knows; callers therefore supply both MIDI numbers to `add_measurement`.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

from intonation.core.pitch import midi_note_name


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

# A spreadsheet that opens a CSV runs any cell starting with one of these as
# a formula (=HYPERLINK(...), +cmd|..., @SUM(...)); TAB and CR are on the list
# because some importers strip them and then see the next character.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _needs_quote_prefix(text: str) -> bool:
    """True when ``text`` starts with a formula trigger, possibly behind
    quote characters. Values that already start with quotes are covered too,
    so the prefix can be removed again on import without ambiguity."""
    return text.lstrip("'").startswith(_FORMULA_TRIGGERS)


def _csv_safe(value) -> str:
    """Text cell for CSV export: prefix a single quote when a spreadsheet
    would otherwise evaluate it. Numeric columns are written unchanged."""
    text = "" if value is None else str(value)
    return "'" + text if _needs_quote_prefix(text) else text


def _csv_import_text(value: str) -> str:
    """Undo _csv_safe and drop control characters from an imported text
    cell, so an export -> import -> export cycle is stable and a crafted
    file cannot smuggle control characters into the log."""
    if value.startswith("'") and _needs_quote_prefix(value[1:]):
        value = value[1:]
    return _CONTROL_CHARS.sub("", value)


def _read_csv_text(path: Path) -> str:
    """Decode an import file. UTF-8 (with or without the BOM Excel adds) is
    tried first; a file that is not valid UTF-8 is read as Windows-1252, the
    encoding Excel uses for "CSV" on Western-European Windows."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")

# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------
# A transposition is sounding minus fingered MIDI. One wider than the whole
# MIDI range is not an instrument, so it marks a damaged record. Fingered
# notes themselves may leave 0..127: a contrabass sounding high is fingered
# above 127, and those readings are real.
_MAX_TRANSPOSITION = 127


def plausible_reading(a4_hz: float, midi_sounding: int, midi_fingered: int,
                      cents: float, freq_hz: float) -> bool:
    """The numeric rules every stored reading meets, whether it comes from the
    history file or from a CSV import. float() happily parses 'inf', 'nan' and
    '1e400', and int() any magnitude, so a damaged or crafted record could
    poison the log: a non-finite cents makes a run's mean NaN, an impossible
    MIDI number breaks note naming. Every comparison with NaN is False, so the
    range tests also reject non-finite values."""
    return (1.0 <= a4_hz <= 20000.0             # plausible concert A4
            and 1.0 <= freq_hz <= 20000.0       # audible, finite
            and -100000.0 <= cents <= 100000.0  # finite deviation
            and 0 <= midi_sounding <= 127
            and abs(midi_sounding - midi_fingered) <= _MAX_TRANSPOSITION)


def _json_number(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except OverflowError:       # an int too large for a float
        return None


def _json_midi(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _run_from_record(obj: dict) -> Optional[RunMeta]:
    texts = [obj.get(k) for k in ("run_id", "started_at", "instrument")]
    texts += [obj.get(k, "") for k in ("maker", "model", "label")]
    if not all(isinstance(t, str) for t in texts):
        return None
    a4_hz = _json_number(obj.get("a4_hz"))
    if a4_hz is None or not (1.0 <= a4_hz <= 20000.0):
        return None
    run_id, started_at, instrument, maker, model, label = texts
    return RunMeta(run_id=run_id, started_at=started_at, instrument=instrument,
                   a4_hz=a4_hz, maker=maker, model=model, label=label)


def _measurement_from_record(obj: dict) -> Optional[Measurement]:
    texts = [obj.get(k) for k in ("run_id", "timestamp", "instrument")]
    if not all(isinstance(t, str) for t in texts):
        return None
    a4_hz, cents, freq_hz = (_json_number(obj.get(k))
                             for k in ("a4_hz", "cents", "freq_hz"))
    midi_sounding, midi_fingered = (_json_midi(obj.get(k))
                                    for k in ("midi_sounding", "midi_fingered"))
    if None in (a4_hz, cents, freq_hz, midi_sounding, midi_fingered):
        return None
    if not plausible_reading(a4_hz, midi_sounding, midi_fingered, cents, freq_hz):
        return None
    run_id, timestamp, instrument = texts
    return Measurement(run_id=run_id, timestamp=timestamp, instrument=instrument,
                       a4_hz=a4_hz, midi_sounding=midi_sounding,
                       midi_fingered=midi_fingered, cents=cents, freq_hz=freq_hz)


def _json_object(line: str | bytes) -> Optional[dict]:
    """The JSON object on one history line, or None for blank, torn, foreign
    or undecodable lines; those are skipped, never fatal."""
    if isinstance(line, bytes):
        # A stray non-UTF-8 byte damages its own line only.
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_record(line: str | bytes) -> RunMeta | Measurement | None:
    """One history line as a validated record, or None when it is not one."""
    obj = _json_object(line)
    if obj is None:
        return None
    kind = obj.get("kind")
    if kind == "run":
        return _run_from_record(obj)
    if kind == "measurement":
        return _measurement_from_record(obj)
    return None


# ---------------------------------------------------------------------------
# History files
# ---------------------------------------------------------------------------
_MONTH_PREFIX = re.compile(r"\d{4}-\d{2}")


def _month_of(stamp) -> Optional[str]:
    """'YYYY-MM' of an ISO-8601 timestamp, or None."""
    if isinstance(stamp, str) and _MONTH_PREFIX.match(stamp):
        return stamp[:7]
    return None


def current_month() -> str:
    # Records carry local time, so the month boundary is local too.
    return datetime.date.today().strftime("%Y-%m")


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            return True
        f.seek(-1, 2)
        return f.read(1) == b"\n"


class HistoryFiles:
    """The history on disk, one JSON record per line.

    The live file (``log.jsonl`` by default) holds the runs started in the
    current month. When the app starts in a new month, the runs of earlier
    months move to ``<stem>-YYYY-MM<suffix>`` beside it (``log-2026-03.jsonl``),
    so the live file, which is the only one written during a session, stays
    small. Lines are copied byte for byte; the record format does not change.
    Readers go through every file, oldest month first, so the split is
    invisible to exports and imports.

    Appends go through one handle that stays open for the session and is
    flushed after every write, so the UI thread never opens or closes the file
    per reading, and a crash loses nothing that was already reported."""

    def __init__(self, live: Path) -> None:
        self.live = Path(live)
        self._month_name = re.compile(
            re.escape(self.live.stem) + r"-(\d{4}-\d{2})" + re.escape(self.live.suffix))
        self._handle = None
        self._lock = threading.Lock()

    def month_file(self, month: str) -> Path:
        return self.live.with_name(f"{self.live.stem}-{month}{self.live.suffix}")

    def files(self) -> list[Path]:
        """Every history file, oldest month first and the live file last."""
        try:
            entries = list(self.live.parent.iterdir())
        except OSError:
            entries = []
        months = sorted(p for p in entries
                        if self._month_name.fullmatch(p.name) and p.is_file())
        return months + ([self.live] if self.live.is_file() else [])

    def lines(self) -> Iterator[bytes]:
        """Every line of every history file. A file that cannot be read is
        skipped: history is best-effort and must never stop an export."""
        for path in self.files():
            try:
                with path.open("rb") as f:
                    yield from f
            except OSError:
                continue

    # -- writing --------------------------------------------------------------
    def append(self, lines: list[str]) -> bool:
        """Append records to the live file. Returns False if the disk refused;
        the next append tries again with a fresh handle."""
        with self._lock:
            try:
                if self._handle is None:
                    self._handle = self._open_live()
                self._handle.write("".join(line + "\n" for line in lines))
                self._handle.flush()
                return True
            except OSError:
                self._close_handle()
                return False

    def _open_live(self):
        self.live.parent.mkdir(parents=True, exist_ok=True)
        torn = self.live.is_file() and not _ends_with_newline(self.live)
        handle = self.live.open("a", encoding="utf-8")
        if torn:
            # The last write of an earlier session was cut off; start on a
            # fresh line so the next record is not glued onto the torn one.
            handle.write("\n")
        return handle

    def append_to_month(self, month: str, lines: list[str]) -> bool:
        """Append records of an earlier month straight to that month's file
        (used by imports, so the live file keeps holding this month only)."""
        target = self.month_file(month)
        try:
            with target.open("a", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in lines))
            return True
        except OSError:
            return False

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    def close(self) -> None:
        with self._lock:
            self._close_handle()

    # -- monthly rotation ---------------------------------------------------
    def rotate(self, month: str) -> None:
        """Move every run started before ``month`` out of the live file.

        Only the first record of the live file is read to decide: the live
        file is written in time order and imports of earlier months go to
        their month files, so if the first record is from this month, all of
        them are. A full split happens once per month."""
        with self._lock:
            if self._handle is not None or not self._first_record_before(month):
                return
            try:
                with self.live.open("rb") as f:
                    lines = f.readlines()
            except OSError:
                return
            keep, moved = self._split(lines, month)
            if moved:
                self._move(keep, moved)

    def _first_record_before(self, month: str) -> bool:
        try:
            with self.live.open("rb") as f:
                # A damaged head is skipped, but a file of nothing but damage
                # must not be scanned to the end on every start.
                for _, line in zip(range(1000), f):
                    obj = _json_object(line)
                    if obj is None:
                        continue
                    first = _month_of(obj.get("started_at") or obj.get("timestamp"))
                    if first is not None:
                        return first < month
        except OSError:
            pass
        return False

    @staticmethod
    def _split(lines: list[bytes], month: str
               ) -> tuple[list[bytes], dict[str, list[bytes]]]:
        """Group lines by the month their run started. A measurement follows
        its run; one whose run record is missing goes by its own timestamp.
        Lines that are not records stay next to the line before them."""
        run_month: dict[str, str] = {}
        keep: list[bytes] = []
        moved: dict[str, list[bytes]] = {}
        here = month
        for line in lines:
            if not line.endswith(b"\n"):
                line += b"\n"
            obj = _json_object(line)
            if obj is not None:
                run_id = obj.get("run_id")
                if obj.get("kind") == "run":
                    found = _month_of(obj.get("started_at"))
                    if found is not None and isinstance(run_id, str):
                        run_month[run_id] = found
                else:
                    found = (run_month.get(run_id) if isinstance(run_id, str) else None) \
                        or _month_of(obj.get("timestamp"))
                if found is not None:
                    here = found
            if here < month:
                moved.setdefault(here, []).append(line)
            else:
                keep.append(line)
        return keep, moved

    def _move(self, keep: list[bytes], moved: dict[str, list[bytes]]) -> None:
        """Append the old lines to their month files, then replace the live
        file with what stays. If the replace fails, the appends are undone,
        so no reading is ever stored twice or lost."""
        grown: dict[Path, Optional[int]] = {}
        tmp: Optional[str] = None
        try:
            for old_month, chunk in sorted(moved.items()):
                target = self.month_file(old_month)
                size = target.stat().st_size if target.is_file() else None
                grown[target] = size
                with target.open("ab") as f:
                    if size and not _ends_with_newline(target):
                        f.write(b"\n")
                    f.writelines(chunk)
            fd, tmp = tempfile.mkstemp(prefix=".history.", suffix=".tmp",
                                       dir=str(self.live.parent))
            with os.fdopen(fd, "wb") as f:
                f.writelines(keep)
            os.replace(tmp, self.live)
            tmp = None
        except OSError:
            for target, size in grown.items():
                try:
                    if size is None:
                        target.unlink()
                    else:
                        os.truncate(target, size)
                except OSError:
                    pass
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


class MeasurementLog:
    """The measurement history: this session's runs in memory, earlier
    sessions on disk.

    Without a ``path`` everything stays in memory. With one, every run and
    measurement is also appended to the history files (see HistoryFiles), and
    the history of earlier sessions is read from them on demand, by exports
    and imports, never at start-up.

    All methods run on the UI thread. The lock keeps the in-memory state
    consistent should a reading ever arrive from another thread.
    """

    def __init__(self, path: Optional[Path | str] = None):
        self._lock = threading.Lock()
        # This session's runs (started or imported here) and measurements.
        # Records of earlier sessions stay on disk.
        self._runs: dict[str, RunMeta] = {}
        self._measurements: list[Measurement] = []
        self._current_run_id: Optional[str] = None
        # Count of measurements in the current run; used to coalesce empties.
        self._current_run_count: int = 0
        # Tracks which runs have already been appended to the JSONL file. We
        # defer the run-record append until the first measurement lands so
        # that a run coalesced before it ever recorded leaves zero footprint
        # on disk.
        self._persisted_runs: set[str] = set()
        self._path = Path(path) if path else None
        self._files = HistoryFiles(self._path) if self._path else None
        if self._files is not None:
            self._files.rotate(current_month())

    @property
    def path(self) -> Optional[Path]:
        """The live history file, or None when the history is in memory only."""
        return self._path

    def close(self) -> None:
        """Release the history file. Safe to call more than once; a later
        append reopens it."""
        if self._files is not None:
            self._files.close()

    # -- reading the history ---------------------------------------------
    def _disk_records(self, skip_runs: set[str], runs_only: bool = False):
        """Validated records of earlier sessions, oldest first. Records of
        ``skip_runs`` (this session's runs, which are kept in memory and may
        carry newer metadata) are left out."""
        if self._files is None:
            return
        for line in self._files.lines():
            # Measurement lines vastly outnumber run lines; a run scan skips
            # them before parsing. Every run record contains this token.
            if runs_only and b'"run"' not in line:
                continue
            rec = parse_record(line)
            if rec is None or rec.run_id in skip_runs:
                continue
            if runs_only and not isinstance(rec, RunMeta):
                continue
            yield rec

    def _session(self) -> tuple[dict[str, RunMeta], list[Measurement]]:
        with self._lock:
            return dict(self._runs), list(self._measurements)

    def _all_runs(self) -> dict[str, RunMeta]:
        session_runs, _ = self._session()
        runs: dict[str, RunMeta] = {}
        for run in self._disk_records(set(session_runs), runs_only=True):
            runs[run.run_id] = run
        runs.update(session_runs)
        return runs

    def _all_measurements(self) -> Iterator[Measurement]:
        session_runs, session_measurements = self._session()
        for rec in self._disk_records(set(session_runs)):
            if isinstance(rec, Measurement):
                yield rec
        yield from session_measurements

    # -- run management ---------------------------------------------------
    def start_run(self, instrument: str, a4_hz: float,
                  maker: str = "", model: str = "",
                  label: str = "") -> str:
        """Open a new run. If the *current* run has no measurements yet, it is
        replaced rather than left behind as an empty run record.

        No JSONL append happens here — the run record is written the first
        time `add_measurement` lands a row for it. That way coalesced empty
        runs (the result of A4 / family / instrument scrubs) leave no
        footprint on disk."""
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
                # Coalesce: drop the empty predecessor so scrubbing controls
                # doesn't litter the log with one-second runs.
                del self._runs[prev_id]
            self._runs[run.run_id] = run
            self._current_run_id = run.run_id
            self._current_run_count = 0
        return run.run_id

    def end_run(self) -> None:
        with self._lock:
            self._current_run_id = None
            self._current_run_count = 0

    def set_current_run_metadata(self, *, maker: str = "",
                                  model: str = "",
                                  label: Optional[str] = None) -> None:
        """Stamp maker/model/label onto the currently-active run, in memory.

        An empty maker or model leaves the stored value alone. ``label`` uses
        ``None`` for "leave alone" so that an empty string can clear it: a
        user who deletes the nickname expects the run to lose its label.

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
            if label is not None:
                run.label = label

    def current_run(self) -> Optional[RunMeta]:
        """A copy of the active run's metadata, or None when no run is open."""
        with self._lock:
            run = self._runs.get(self._current_run_id or "")
            return replace(run) if run is not None else None

    # -- measurement intake ----------------------------------------------
    def add_measurement(self, midi_sounding: int, midi_fingered: int,
                        cents: float, freq_hz: float) -> None:
        """Record one measurement against the currently-active run.

        The run's instrument and A4 are latched onto the row at write time,
        so the value reflects what was active when `start_run` was called —
        not what `self.instrument` might have changed to milliseconds later.
        """
        # Snapshot the run + measurement JSONL strings inside the lock so a
        # concurrent set_current_run_metadata() can't mutate maker/model/
        # label between the snapshot and the disk write. File IO stays
        # outside the lock so a slow disk never holds up a reader of the
        # in-memory state.
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
            needs_run_persist = run_id not in self._persisted_runs
            if needs_run_persist:
                self._persisted_runs.add(run_id)
                run_line = run.to_jsonl()
            else:
                run_line = None
            measurement_line = m.to_jsonl()
        # Append the run record on the first measurement so coalesced empty
        # runs leave no disk footprint.
        if self._files is not None:
            self._files.append([run_line, measurement_line] if run_line is not None
                               else [measurement_line])

    # -- inspection -------------------------------------------------------
    # These read the whole history from disk on every call; they serve
    # exports and dialogs, never the per-reading path.
    def runs(self) -> list[RunMeta]:
        return sorted(self._all_runs().values(),
                      key=lambda r: r.started_at, reverse=True)

    def instruments(self) -> list[str]:
        seen = {r.instrument for r in self._all_runs().values()}
        seen.update(m.instrument for m in self._all_measurements())
        return sorted(seen)

    def measurements(self) -> list[Measurement]:
        return list(self._all_measurements())

    def measurement_count(self) -> int:
        return sum(1 for _ in self._all_measurements())

    def has_measurements(self) -> bool:
        return next(self._all_measurements(), None) is not None

    # -- import -----------------------------------------------------------
    # Caps to prevent a malicious or accidental gigabyte CSV from blowing
    # out memory before the row loop notices.  50 MB / 500 k rows is well
    # above any plausible legitimate practice-session export.
    MAX_IMPORT_BYTES = 50 * 1024 * 1024
    MAX_IMPORT_ROWS = 500_000

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

    def import_raw_csv(self, path: Path | str) -> tuple[int, int, int]:
        """Load a previously-exported `raw` CSV into the log as historical
        runs. Returns (runs_added, measurements_added, skipped).

        ``skipped`` counts candidate data rows the importer rejected:
        duplicate run_id (already in the log or already imported earlier in
        the same file with a conflicting head row), malformed (wrong column
        count or non-numeric numeric fields), or intra-run instrument / A4
        conflicts. A header-only or empty file reports ``skipped == 0``.

        Only the `raw` slice mode round-trips — aggregated modes drop the
        timestamp and per-measurement frequency, so they can't be restored.

        Imported runs become historical (no active-run pointer is touched).
        Existing `run_id`s in the log are skipped to avoid duplicates from
        re-importing the same file.
        """
        path = Path(path)
        # v0.6: refuse outsized files before we open them — protects the
        # user from a gigabyte CSV self-DoS.  stat() failure (e.g. file
        # doesn't exist) is silently let through; the open() below will
        # raise the same FileNotFoundError it always has.
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > self.MAX_IMPORT_BYTES:
            raise ValueError(
                f"CSV too large ({size:,} bytes); import limit is "
                f"{self.MAX_IMPORT_BYTES:,} bytes.")
        # v0.5.7.7: utf-8-sig transparently strips a UTF-8 BOM if
        # present and behaves identically to utf-8 otherwise. Excel
        # adds a BOM on Save As CSV on Windows; without this, a
        # round-trip through Excel turns "run_id" into "﻿run_id"
        # and the header-match check raises. Symmetric: export still
        # writes plain utf-8 (no BOM) -- be conservative in what we
        # send, liberal in what we accept.
        with io.StringIO(_read_csv_text(path), newline="") as f:
            reader = csv.reader(f)
            try:
                header = tuple(next(reader))
            except StopIteration:
                return (0, 0, 0)
            if header == self.RAW_CSV_HEADER:
                legacy = False
            elif header == self.RAW_CSV_HEADER_LEGACY:
                legacy = True
            else:
                raise ValueError(
                    "CSV header does not match a known raw export format. "
                    f"Got {header!r}.")

            existing_runs = set(self._all_runs())

            new_runs: dict[str, RunMeta] = {}
            new_measurements: list[Measurement] = []
            rows_seen = 0
            skipped = 0

            for row in reader:
                rows_seen += 1
                if rows_seen > self.MAX_IMPORT_ROWS:
                    raise ValueError(
                        f"CSV exceeds {self.MAX_IMPORT_ROWS:,} rows; "
                        "import refused.")
                expected_len = (len(self.RAW_CSV_HEADER_LEGACY) if legacy
                                else len(self.RAW_CSV_HEADER))
                if len(row) != expected_len:
                    skipped += 1
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
                (timestamp, run_id, run_started_at, instrument, nickname,
                 maker, model) = (
                    _csv_import_text(v) for v in (
                        timestamp, run_id, run_started_at, instrument,
                        nickname, maker, model))
                if not run_id or run_id in existing_runs:
                    skipped += 1
                    continue
                try:
                    a4_hz = float(a4_hz_s)
                    midi_sounding = int(midi_sounding_s)
                    midi_fingered = int(midi_fingered_s)
                    cents = float(cents_s)
                    freq_hz = float(freq_s)
                except ValueError:
                    skipped += 1
                    continue

                if not plausible_reading(a4_hz, midi_sounding, midi_fingered,
                                         cents, freq_hz):
                    skipped += 1
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
                        skipped += 1
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
                self._persisted_runs.add(run.run_id)
            self._measurements.extend(new_measurements)
        if self._files is not None and new_runs:
            self._persist_imported(new_runs, new_measurements)
        return (len(new_runs), len(new_measurements), skipped)

    def _persist_imported(self, runs: dict[str, RunMeta],
                          measurements: list[Measurement]) -> None:
        """Write imported runs to the file of the month each started in, so
        the live file keeps holding this month only. One write per file: a
        50k-row import must not open the history 50k times."""
        this_month = current_month()
        by_month: dict[str, list[str]] = {}
        month_of_run: dict[str, str] = {}
        for run in runs.values():
            month = _month_of(run.started_at) or this_month
            month = month if month < this_month else this_month
            month_of_run[run.run_id] = month
            by_month.setdefault(month, []).append(run.to_jsonl())
        for m in measurements:
            by_month[month_of_run[m.run_id]].append(m.to_jsonl())
        for month, lines in sorted(by_month.items()):
            if month == this_month:
                self._files.append(lines)
            else:
                self._files.append_to_month(month, lines)

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

        # The history is streamed: the writers make one pass over the
        # measurements, and only the aggregates are held in memory.
        runs = self._all_runs()
        measurements: Iterable[Measurement] = self._all_measurements()
        if run_id is not None:
            measurements = (m for m in measurements if m.run_id == run_id)
        if instrument is not None:
            measurements = (m for m in measurements
                            if m.instrument == instrument)
        if nickname is not None:
            keep_ids = {rid for rid, r in runs.items() if r.label == nickname}
            measurements = (m for m in measurements if m.run_id in keep_ids)

        path = Path(path)
        # Written beside the target and renamed over it, so an export that
        # fails halfway leaves the previous file as it was. Opened with "x"
        # rather than mkstemp so the file gets the user's normal permissions.
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tmp.open("x", encoding="utf-8", newline="") as f:
                n = _write_slice(csv.writer(f), mode, measurements, runs, instrument)
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return n


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write_slice(w, mode: str, measurements: Iterable[Measurement],
                 runs: dict[str, RunMeta], instrument: Optional[str]) -> int:
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


def _write_raw(w, measurements: Iterable[Measurement],
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
            _csv_safe(m.timestamp), _csv_safe(m.run_id),
            _csv_safe(run.started_at if run else ""),
            _csv_safe(m.instrument),
            _csv_safe(run.label if run else ""),
            f"{m.a4_hz:.2f}",
            m.midi_sounding, midi_note_name(m.midi_sounding),
            m.midi_fingered, midi_note_name(m.midi_fingered),
            f"{m.cents:.3f}", f"{m.freq_hz:.3f}",
            _csv_safe(run.maker if run else ""),
            _csv_safe(run.model if run else ""),
        ])
        n += 1
    return n


def _write_per_run_note(w, measurements: Iterable[Measurement],
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
            _csv_safe(run_id),
            _csv_safe(run.started_at if run else ""),
            _csv_safe(rep.instrument),
            _csv_safe(run.label if run else ""),
            f"{rep.a4_hz:.2f}",
            midi, midi_note_name(midi),
            midi_note_name(rep.midi_fingered),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
            _csv_safe(run.maker if run else ""),
            _csv_safe(run.model if run else ""),
        ])
        n_rows += 1
    return n_rows


def _write_per_instrument_note(w, measurements: Iterable[Measurement],
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
            _csv_safe(instrument),
            _csv_safe(";".join(nick_set)),
            midi, midi_note_name(midi),
            midi_note_name(fingered[(instrument, midi)]),
            len(runs_seen[(instrument, midi)]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_per_nickname_note(w, measurements: Iterable[Measurement],
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
            _csv_safe(nick or "(unnamed)"),
            _csv_safe(";".join(sorted(instrument_seen[(nick, midi)]))),
            midi, midi_note_name(midi),
            midi_note_name(fingered[(nick, midi)]),
            len(runs_seen[(nick, midi)]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_instrument_avg(w, measurements: Iterable[Measurement],
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
            _csv_safe(instrument), midi,
            midi_note_name(midi),
            midi_note_name(fingered[midi]),
            len(runs_seen[midi]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows


def _write_overall(w, measurements: Iterable[Measurement]) -> int:
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
            _csv_safe(";".join(sorted(instruments_seen[midi]))),
            len(runs_seen[midi]),
            len(vals),
            f"{mean:.3f}", f"{std:.3f}", f"{mn:.3f}", f"{mx:.3f}",
        ])
        n_rows += 1
    return n_rows

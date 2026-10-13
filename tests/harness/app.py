"""HeadlessApp: the whole app, driven the way a person drives it.

Scenario tests import this and nothing else about the app::

    app = HeadlessApp(tmp_path, host=default_host(clock), clock=clock)
    app.launch()
    app.pick_instrument("eb_alto")
    app.play(Tone(note="A4", timbre="sax", seconds=1.5))
    assert app.table.row("A4", by="sounding").mean == 0

Every observable is plain data (``model.py``); every action is something a user
can do. How that maps onto widgets lives in ``driver.py``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from . import driver as drv
from .capture import OutputCapture
from .clock import ManualClock
from .dialogs import Dialogs
from .fake_audio import FakeAudioHost, HarnessError
from .model import (
    AudioStatus,
    AutotuneResult,
    DeckState,
    DroneState,
    ImportResult,
    MetronomeState,
    Reading,
    Row,
)
from .tones import Tone

DEFAULT_INPUT = "Test Mic"
DEFAULT_OUTPUT = "Test Speakers"

# Present in every config the harness writes unless a test overrides it: the
# welcome dialog is already answered, the UI language is the one under test, and
# the table uses the plain list layout (rows are read straight off the widget).
BASE_CONFIG = {"welcome_shown": True, "layout_mode_preference": "single"}


def default_host(clock: Optional[ManualClock] = None) -> FakeAudioHost:
    """A host with one microphone (44.1 kHz only) and one speaker (48 kHz only)."""
    host = FakeAudioHost(clock=clock)
    host.add_input(DEFAULT_INPUT, rates=(44100,), default_rate=44100)
    host.add_output(DEFAULT_OUTPUT, rates=(48000,), default_rate=48000)
    return host


class _Part:
    def __init__(self, app: "HeadlessApp") -> None:
        self._app = app

    @property
    def _d(self) -> drv.AppDriver:
        return self._app._require_driver()

    def _done(self, value=None):
        self._app._after()
        return value


# --------------------------------------------------------------------- welcome
class Welcome(_Part):
    """The first-launch dialog. It appears while the app starts, so script the
    answer BEFORE ``launch()`` (or ``restart()``)."""

    def _script(self, answer: str, **fields) -> None:
        self._app.dialogs.expect("welcome", answer, **fields)

    def decide_later(self, language: Optional[str] = None) -> None:
        """Dismiss it; the question comes back on the next launch."""
        self._script("reject", **({"language": language} if language else {}))

    def opt_in(self, language: Optional[str] = None) -> None:
        """Tick 'save history' and continue."""
        self._script("accept", history=True, **({"language": language} if language else {}))

    def continue_without_history(self, language: Optional[str] = None) -> None:
        self._script("accept", history=False, **({"language": language} if language else {}))

    @property
    def shown(self) -> bool:
        """Was the dialog shown during the most recent launch?"""
        return "welcome" in self._app.dialogs.forms_shown[self._app._forms_mark:]


# ----------------------------------------------------------------------- table
class Table(_Part):
    def rows(self, played_only: bool = True) -> list[Row]:
        """The table as displayed. By default only notes that were played and
        passed the minimum-count filter; ``played_only=False`` adds the blank
        rows that show the instrument's range."""
        return self._d.table_rows(played_only)

    def get(self, name: str, by: Optional[str] = None) -> Optional[Row]:
        found = [r for r in self.rows() if r.is_named(name, by)]
        if len(found) > 1:
            raise HarnessError(f"{name!r} matches several rows; say by='fingered' or 'sounding'")
        return found[0] if found else None

    def row(self, name: str, by: Optional[str] = None) -> Row:
        """The played row for ``name`` (fingered or sounding spelling); fails with
        the list of rows present when there is none."""
        r = self.get(name, by)
        if r is None:
            shown = ", ".join(f"{x.fingered}/{x.sounding}" for x in self.rows()) or "none"
            raise AssertionError(f"no row for {name!r} in the table (rows: {shown})")
        return r

    def discard(self, name: str, confirm: bool = True) -> None:
        """Right-click the note's row (list layout) or cell (matrix layout)
        and choose discard."""
        self._d.table_discard(name, confirm)
        self._done()

    def reset(self, confirm: bool = True) -> None:
        self._d.table_reset(confirm)
        self._done()

    @property
    def caption(self) -> str:
        return self._d.table_caption()

    @property
    def columns(self) -> list[str]:
        """Column headers as shown: the list layout's six titles, or one
        octave label per column of the matrix layout."""
        return self._d.table_columns()

    @property
    def total_measurements(self) -> int:
        return self._d.stats_total_n()


class Tuner(_Part):
    def reading(self) -> Reading:
        return self._d.tuner_reading()


class Autotune(_Part):
    def run(self, accept: bool = True) -> AutotuneResult:
        return self._done(self._d.autotune(accept))


# ---------------------------------------------------------------------- export
class Export(_Part):
    """File exports. Each returns the written ``Path``; when the app refuses
    (nothing to export) it raises ``ExportFailed`` unless ``expect_failure=True``,
    which returns ``None`` and leaves the app's message in
    ``app.dialogs.last_message()``."""

    def __init__(self, app: "HeadlessApp") -> None:
        super().__init__(app)
        self._n = 0

    def _path(self, path, stem: str, ext: str) -> Path:
        if path is not None:
            return Path(path)
        self._n += 1
        return self._app.home / "exports" / f"{stem}-{self._n}.{ext}"

    def txt(self, path=None, *, maker: str = "", model: str = "", expect_failure=False):
        return self._done(self._d.export_txt(self._path(path, "table", "txt"),
                                             maker, model, expect_failure))

    def pdf(self, path=None, *, maker: str = "", model: str = "", expect_failure=False):
        return self._done(self._d.export_pdf(self._path(path, "table", "pdf"),
                                             maker, model, expect_failure))

    def chart(self, path=None, *, maker: str = "", model: str = "", expect_failure=False):
        return self._done(self._d.export_chart(self._path(path, "chart", "png"),
                                               maker, model, expect_failure))

    def csv(self, mode: str = "raw", path=None, *, instrument: Optional[str] = None,
            run_id: Optional[str] = None, nickname: Optional[str] = None,
            maker: str = "", model: str = "", expect_failure=False):
        return self._done(self._d.export_csv(self._path(path, f"history-{mode}", "csv"),
                                             mode, instrument, run_id, nickname,
                                             maker, model, expect_failure))

    @property
    def pdf_model(self) -> Optional[dict]:
        """Structured content handed to the PDF renderer by the last PDF export."""
        return self._app._require_driver().models.pdf

    @property
    def chart_model(self) -> Optional[dict]:
        """Arguments handed to the chart renderer by the last chart export."""
        return self._app._require_driver().models.chart

    @property
    def csv_rows(self) -> Optional[int]:
        return self._app._require_driver().models.csv_rows


# ------------------------------------------------------------------- metronome
class Metronome(_Part):
    def start(self, bpm: Optional[int] = None, time_sig: Optional[str] = None) -> MetronomeState:
        self._d.metro_start(bpm, time_sig)
        return self._done(self.state)

    def stop(self) -> MetronomeState:
        self._d.metro_stop()
        return self._done(self.state)

    def set_volume(self, percent: int) -> None:
        self._d.metro_set_volume(percent)
        self._done()

    def tap(self) -> MetronomeState:
        self._d.metro_tap()
        return self._done(self.state)

    @property
    def state(self) -> MetronomeState:
        return self._d.metro_state()


class Drone(_Part):
    def on(self, note: str = "A4", voice: Optional[str] = None) -> DroneState:
        """Sound ``note`` (A3..A5) with ``voice`` (an id such as 'sine', 'organ')."""
        self._d.drone_on(note, voice)
        return self._done(self.state)

    def off(self) -> DroneState:
        self._d.drone_off()
        return self._done(self.state)

    @property
    def state(self) -> DroneState:
        return self._d.drone_state()


class Pipes(_Part):
    def press(self, note: str) -> None:
        self._d.pipes_press(note)
        self._done()

    def release(self, note: str) -> None:
        self._d.pipes_release(note)
        self._done()

    def close(self) -> None:
        self._d.pipes_close()
        self._done()

    @property
    def active(self) -> list[int]:
        """MIDI numbers of the pads currently sounding."""
        return self._d.pipes_active()

    @property
    def status(self) -> Optional[str]:
        return self._d.pipes_status()


class Deck(_Part):
    def record(self, seconds: float, tone: Optional[Tone] = None) -> DeckState:
        """Record ``seconds`` of microphone (``tone`` if given, else silence)."""
        self._d.deck_record()
        self._app._after()
        if self.state.state == "recording":
            if tone is not None:
                self._app.play(tone, release=0)
            else:
                self._app.silence(seconds)
            self._d.deck_stop()
        return self._done(self.state)

    def stop(self) -> DeckState:
        self._d.deck_stop()
        return self._done(self.state)

    def play(self, wait: bool = False) -> DeckState:
        """Start playback. With ``wait`` the clock runs until it ends."""
        self._d.deck_play()
        self._app._after()
        if wait:
            for _ in range(2000):
                if self.state.state != "playing":
                    break
                self._app.advance(0.1)
        return self.state

    def export(self, path=None) -> Optional[Path]:
        path = path or (self._app.home / "exports" / "take.wav")
        return self._done(self._d.deck_export(path))

    @property
    def state(self) -> DeckState:
        return self._d.deck_state()

    @property
    def can_record(self) -> bool:
        return self._d.deck_can_record()


# -------------------------------------------------------------------- settings
class Settings(_Part):
    def language(self, lang: str) -> None:
        self._d.set_language(lang)
        self._done()

    def theme(self, name: str) -> None:
        self._d.set_theme(name)
        self._done()

    def response(self, mode: str) -> None:
        self._d.set_response(mode)
        self._done()

    def min_n(self, n: int) -> None:
        self._d.set_min_n(n)
        self._done()

    def filter_to_range(self, on: bool) -> None:
        self._d.set_filter_to_range(on)
        self._done()

    def nickname(self, text: str) -> None:
        self._d.set_nickname(text)
        self._done()

    def mic_gain(self, db: int) -> None:
        self._d.set_mic_gain(db)
        self._done()

    def output_device(self, name: Optional[str]) -> None:
        """Pick an output device in SETUP (None = system default)."""
        self._d.select_output_device(name)
        self._done()

    def test_tone(self, on: bool = True) -> None:
        self._d.test_tone(on)
        self._done()

    @property
    def test_tone_playing(self) -> bool:
        return self._d.test_tone_playing()

    @property
    def test_tone_error(self) -> Optional[str]:
        return self._d.test_tone_status()

    def edit_range(self, lo=None, hi=None, *, restore: bool = False, accept: bool = True) -> bool:
        """Edit the active instrument's range (notes in the displayed notation).
        Returns whether the editor accepted; ``app.range_error`` says why not."""
        return self._done(self._d.edit_range(lo, hi, restore, accept))

    def add_custom_instrument(self, name: str, transpose: int = 0, accept: bool = True):
        """Add a custom instrument; returns its key (None if not added)."""
        return self._done(self._d.add_custom_instrument(name, transpose, accept))

    @property
    def range_error(self) -> Optional[str]:
        return self._d.last_form_note

    @property
    def history_enabled(self) -> bool:
        return self._d.history_enabled()

    @property
    def history_path(self) -> Optional[Path]:
        """Where measurements are being appended (None = in memory only)."""
        return self._d.history_path()


class Audio(_Part):
    def status(self) -> AudioStatus:
        return self._d.audio_status()

    def retry(self) -> AudioStatus:
        self._d.audio_retry()
        return self._done(self.status())


# ------------------------------------------------------------------------- app
class HeadlessApp:
    def __init__(self, home, lang: str = "en", host: Optional[FakeAudioHost] = None, *,
                 clock: Optional[ManualClock] = None, first_launch: bool = False,
                 config=None, customs=None, ranges=None, history=None) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.lang = lang
        self.clock = clock or (host.clock if host is not None and host.clock else ManualClock())
        self._owns_host = host is None
        self.host = host if host is not None else default_host(self.clock)
        self.dialogs = Dialogs()
        self._driver: Optional[drv.AppDriver] = None
        self.leaked_streams: list[str] = []
        self._forms_mark = 0
        self.welcome = Welcome(self)
        self.table = Table(self)
        self.tuner = Tuner(self)
        self.autotune = Autotune(self)
        self.export = Export(self)
        self.metronome = Metronome(self)
        self.drone = Drone(self)
        self.pipes = Pipes(self)
        self.deck = Deck(self)
        self.settings = Settings(self)
        self.audio = Audio(self)
        self._seed(config, customs, ranges, history, first_launch)

    # ----------------------------------------------------------------- seeding
    def _seed(self, config, customs, ranges, history, first_launch: bool) -> None:
        if isinstance(config, dict) or config is None:
            if config is None and first_launch:
                config = None
            else:
                base = {} if first_launch else {**BASE_CONFIG, "last_lang": self.lang}
                config = {**base, **(config or {})}
        for logical, value in (("config", config), ("customs", customs),
                               ("ranges", ranges), ("history", history)):
            if value is not None:
                self._write(drv.data_file(self.home, logical), value)

    @staticmethod
    def _write(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, Path):
            shutil.copyfile(value, path)
        elif isinstance(value, bytes):
            path.write_bytes(value)
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def file(self, logical: str) -> Path:
        """Path of a user data file: 'config', 'customs', 'ranges' or 'history'."""
        return drv.data_file(self.home, logical)

    def read_config(self) -> dict:
        """The saved settings file as written by the app (empty if none)."""
        p = self.file("config")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # --------------------------------------------------------------- lifecycle
    def launch(self) -> "HeadlessApp":
        if self._driver is not None:
            raise HarnessError("already launched")
        self._forms_mark = len(self.dialogs.forms_shown)
        d = drv.AppDriver(self.home, self.host, self.clock, self.dialogs, self.lang)
        self._driver = d
        try:
            d.start()
        except BaseException:
            self._driver = None
            raise
        return self

    def _stop_driver(self) -> list[str]:
        leaked: list[str] = []
        if self._driver is not None:
            d, self._driver = self._driver, None
            d.stop()
            leaked = d.streams_open_at_close
        self.leaked_streams = leaked
        return leaked

    def close(self) -> list[str]:
        """Close the window like a user (the app saves its settings) and release
        everything. Returns the streams the app left open, which a clean exit
        would not have. Raises if the app showed an unscripted dialog."""
        leaked = self._stop_driver()
        if self._owns_host:
            self.host.close()
        self.dialogs.assert_clean()
        self.dialogs.assert_none_pending()
        return leaked

    def restart(self) -> "HeadlessApp":
        """Quit and start again with the same home directory and audio host."""
        self._stop_driver()
        self.dialogs.assert_clean()
        return self.launch()

    def __enter__(self) -> "HeadlessApp":
        return self.launch() if self._driver is None else self

    def __exit__(self, *exc) -> None:
        self.close()

    def _require_driver(self) -> drv.AppDriver:
        if self._driver is None:
            raise HarnessError("the app is not running; call launch() first")
        return self._driver

    def _after(self) -> None:
        d = self._require_driver()
        d.settle()
        d.check()

    # ------------------------------------------------------------------ actions
    def pick_instrument(self, key: str) -> None:
        self._require_driver().pick_instrument(key)
        self._after()

    def set_a4(self, hz: int) -> None:
        self._require_driver().set_a4(hz)
        self._after()

    def set_display(self, mode: str) -> None:
        self._require_driver().set_display(mode)
        self._after()

    def import_csv(self, path, confirm: bool = True) -> ImportResult:
        res = self._require_driver().import_csv(path, confirm)
        self._after()
        return res

    # --------------------------------------------------------------------- time
    def _feed(self, samples: np.ndarray, sr: int, block: int) -> int:
        delivered = 0
        for i in range(0, samples.shape[0], block):
            chunk = samples[i:i + block]
            delivered += self.host.pump_input(chunk)
            self.clock.advance(chunk.shape[0] / float(sr))
        return delivered

    def _input_format(self) -> tuple[int, int]:
        s = self.host.input_stream
        return (s.samplerate, s.blocksize) if s is not None else (44100, 2048)

    def play(self, tone: Tone, release: float = 0.4) -> int:
        """Play ``tone`` into the microphone, then ``release`` seconds of silence
        (the detector holds back a note's last readings until the sound ends).
        Time advances as the audio is delivered. Returns frames the app received
        (0 when no input stream is open)."""
        self._require_driver()
        sr, block = self._input_format()
        delivered = self._feed(tone.render(sr), sr, block)
        if release:
            delivered += self._feed(np.zeros(int(round(release * sr)), np.float32), sr, block)
        self._after()
        return delivered

    def silence(self, seconds: float) -> int:
        self._require_driver()
        sr, block = self._input_format()
        delivered = self._feed(np.zeros(int(round(seconds * sr)), np.float32), sr, block)
        self._after()
        return delivered

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` pass: timers tick and output audio is pulled."""
        self._require_driver()
        self.clock.advance(seconds)
        self._after()

    def capture_output(self, seconds: float) -> OutputCapture:
        """Run ``seconds`` and return exactly what the speaker received in them."""
        self._require_driver()
        self.host.clear_output()
        self.advance(seconds)
        return self.host.output_capture()

    # -------------------------------------------------------------------- state
    @property
    def instrument(self) -> str:
        return self._require_driver().instrument()

    @property
    def a4(self) -> int:
        return self._require_driver().a4()

    @property
    def engine_a4(self) -> float:
        """The reference frequency the pitch detector is actually using."""
        return self._require_driver().engine_a4()

    @property
    def display(self) -> str:
        return self._require_driver().display()

    @property
    def ui_language(self) -> str:
        return self._require_driver().language()

    @property
    def window_title(self) -> str:
        return self._require_driver().window_title()

    def text(self, key: str, **kw) -> str:
        """The app's own wording for a UI string key, in the current language."""
        return self._require_driver().t(key, **kw)

    @property
    def history_counts(self) -> tuple[int, int]:
        """(runs with measurements, measurements) in the history log."""
        return self._require_driver().history_counts()

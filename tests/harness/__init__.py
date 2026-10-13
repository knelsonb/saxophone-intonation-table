"""Headless acceptance harness for the Intonation Analyzer.

Scenario tests import from this package only. ``driver.py`` is the single module
that knows how the app is built; see its docstring for the contract.
"""
import os

# Must be set before the first QApplication is created in this process.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from .app import DEFAULT_INPUT, DEFAULT_OUTPUT, HeadlessApp, default_host  # noqa: E402
from .capture import OutputCapture  # noqa: E402
from .clock import ManualClock  # noqa: E402
from .dialogs import Dialogs, Message  # noqa: E402
from .driver import i18n_catalog  # noqa: E402
from .fake_audio import FakeAudioHost, FakeDevice, HarnessError, PortAudioError  # noqa: E402
from .golden import STANDARD, assert_golden, fixture_path  # noqa: E402
from .model import (  # noqa: E402
                    AudioStatus,
                    AutotuneResult,
                    DeckState,
                    DroneState,
                    ExportFailed,
                    ImportResult,
                    MetronomeState,
                    Reading,
                    Row,
)
from .tones import Tone, midi_to_freq, note_to_midi  # noqa: E402

__all__ = [
    "DEFAULT_INPUT", "DEFAULT_OUTPUT", "HeadlessApp", "default_host", "OutputCapture",
    "ManualClock", "Dialogs", "Message", "i18n_catalog", "FakeAudioHost", "FakeDevice",
    "HarnessError", "PortAudioError", "STANDARD", "assert_golden", "fixture_path",
    "AudioStatus", "AutotuneResult", "DeckState", "DroneState", "ExportFailed",
    "ImportResult", "MetronomeState", "Reading", "Row", "Tone", "midi_to_freq",
    "note_to_midi",
]

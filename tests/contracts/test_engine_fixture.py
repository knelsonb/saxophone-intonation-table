"""The fake-sounddevice engine loader must leave no trace in sys.modules.

test_hotplug_recovery.py and test_output_stream.py run the engine against a fake
``sounddevice``. If that engine copy replaced the real module, every other test
file would see two engine modules with different enum classes. This pins the
conftest fixture's contract.
"""
from __future__ import annotations

import importlib
import sys
import types


def test_loader_installs_a_private_engine_and_restores_the_real_one(
        load_engine_with_sounddevice, monkeypatch):
    from intonation.audio import engine as real
    sounddevice_before = sys.modules.get("sounddevice")
    fake = types.ModuleType("sounddevice")

    private = load_engine_with_sounddevice(fake)

    assert private is not real
    assert sys.modules["intonation.audio.engine"] is private
    assert sys.modules["intonation.audio"].engine is private
    assert private.sd is fake and private.AUDIO_OK
    assert private.AudioEngineError is not real.AudioEngineError

    monkeypatch.undo()          # what fixture teardown does at the end of a test

    assert sys.modules["intonation.audio.engine"] is real
    assert sys.modules["intonation.audio"].engine is real
    assert sys.modules.get("sounddevice") is sounddevice_before
    assert importlib.import_module("intonation.audio.engine").AudioEngineError is (
        real.AudioEngineError)
    assert real.sd is not fake

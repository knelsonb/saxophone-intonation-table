"""Regression tests for the v1.2.0 review fixes.

Each test pins one defect found in the pre-release review: corrupt or
unreadable files on start-up, stale sample rates in the audio sources, and a
metronome restart race. They run headless with the real Mixer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sax_config
import sax_pitch_pipes as pp
import sim_harness as H
from sax_drone import _DEFAULT_PROGRAM, resolve_drone_voice
from sax_intonation_log import MeasurementLog, _read_csv_text
from sax_metronome import MetronomeController
from sax_mixer import Mixer


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".intonation_analyzer"
    monkeypatch.setattr(sax_config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(sax_config, "CONFIG_PATH", cfg_dir / "config.json")
    monkeypatch.setattr(sax_config, "CUSTOMS_PATH", cfg_dir / "custom_instruments.json")
    cfg_dir.mkdir(parents=True)
    return cfg_dir


def _fail_reads_of(mp, target: Path) -> None:
    """Make opening ``target`` raise OSError, as a Windows sharing lock would."""
    real_open = Path.open

    def guarded(self, *args, **kwargs):
        if self == target:
            raise PermissionError(13, "locked", str(self))
        return real_open(self, *args, **kwargs)

    mp.setattr(Path, "open", guarded)


# --- configuration -----------------------------------------------------------
def test_config_with_non_utf8_bytes_loads_defaults(isolated_config):
    sax_config.CONFIG_PATH.write_bytes(b'{"log_path": "C:\\\\M\xe4rz"}')
    assert sax_config.load_config() == sax_config.AppConfig()


def test_unreadable_config_is_not_overwritten_on_save(isolated_config, monkeypatch):
    original = json.dumps({"persistence_enabled": True, "log_path": "keep-me"})
    sax_config.CONFIG_PATH.write_text(original, encoding="utf-8")
    with monkeypatch.context() as mp:
        _fail_reads_of(mp, sax_config.CONFIG_PATH)
        sax_config.save_config(sax_config.load_config())
    assert sax_config.CONFIG_PATH.read_text(encoding="utf-8") == original


def test_corrupt_config_is_replaced_on_save(isolated_config):
    sax_config.CONFIG_PATH.write_text("{not json", encoding="utf-8")
    sax_config.save_config(sax_config.load_config())
    assert json.loads(sax_config.CONFIG_PATH.read_text(encoding="utf-8"))


def test_matrix_extra_octaves_is_clamped(isolated_config):
    sax_config.CONFIG_PATH.write_text('{"matrix_extra_octaves": 1000000000}',
                                      encoding="utf-8")
    assert sax_config.load_config().matrix_extra_octaves == 4


# --- custom instruments ------------------------------------------------------
def test_customs_overflowing_transposition_is_skipped(isolated_config):
    sax_config.CUSTOMS_PATH.write_text(
        '[{"key": "bad", "transp": 1e400}, {"key": "ok", "transp": -2}]',
        encoding="utf-8")
    assert [c.key for c in sax_config.load_customs()] == ["ok"]


def test_customs_strict_load_raises_on_read_failure(isolated_config, monkeypatch):
    sax_config.CUSTOMS_PATH.write_text('[{"key": "a", "transp": 0}]', encoding="utf-8")
    _fail_reads_of(monkeypatch, sax_config.CUSTOMS_PATH)
    assert sax_config.load_customs() == []
    with pytest.raises(OSError):
        sax_config.load_customs(strict=True)


# --- measurement log ---------------------------------------------------------
def test_log_skips_undecodable_and_non_object_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'42\nnull\n[1]\n{"kind": "run", \xff}\n')
    log = MeasurementLog(path=str(path))
    assert log is not None


def test_csv_import_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_bytes("nickname\nL\u00e4rm\n".encode("cp1252"))
    assert "L\u00e4rm" in _read_csv_text(path)


def test_csv_import_strips_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfrun_id\n")
    assert _read_csv_text(path).startswith("run_id")


# --- audio sources follow the output rate --------------------------------------
@pytest.mark.parametrize("sr", [48000, 96000])
def test_pitch_pipes_follow_output_rate(sr):
    """The controller is built before an output stream exists; set_output
    must make new pads sound at the target pitch on the real stream."""
    block = 2048
    mixer = Mixer(max_block=block)
    ctrl = pp.PitchPipesController(mixer, 44100, max_block=1024, a4=440.0)
    ctrl.set_output(sr, block)
    ctrl.toggle(69)
    out = np.zeros(block * 12, dtype=np.float32)
    for i in range(12):
        mixer.render(out[i * block:(i + 1) * block], block)
    err = H.cents(H.fft_peak_hz(out[block * 4:], sr), 440.0)
    assert abs(err) < 5.0, f"A4 pad off by {err:+.1f} cents at {sr} Hz"


def test_pitch_pipes_set_output_releases_stale_pads():
    mixer = Mixer(max_block=2048)
    ctrl = pp.PitchPipesController(mixer, 44100, a4=440.0)
    ctrl.toggle(60)
    ctrl.set_output(48000, 2048)
    assert ctrl.active_midis() == frozenset()


def test_drone_voice_superscript_digit_falls_back():
    assert resolve_drone_voice("\u00b2") == _DEFAULT_PROGRAM


# --- metronome ---------------------------------------------------------------
def test_metronome_quick_restart_keeps_one_beat_chain():
    sr, block = 48000, 1000
    mixer = Mixer(max_block=block)
    ctrl = MetronomeController(mixer, sr, bpm=120)
    out = np.zeros(block, dtype=np.float32)
    ctrl.start()
    for _ in range(9):                 # beat 0 fires; beat 1 is pending at 24000
        mixer.render(out, block)
    ctrl.stop()
    ctrl.start()                       # restart before the old beat 1 fires
    # The old run's beat stays queued until its onset (24000) and must then
    # die without scheduling a successor; afterwards exactly one chain remains.
    for _ in range(60):
        mixer.render(out, block)
    assert mixer.pending_events() == 1, "a second beat chain survived the restart"


# --- drone without the SoundFont synth ------------------------------------------
def test_drone_falls_back_to_sine_when_synth_cannot_load(monkeypatch):
    import sax_drone

    def no_synth(*_a, **_k):
        raise ImportError("No module named 'tinysoundfont'")

    monkeypatch.setattr(sax_drone, "DroneSource", no_synth)
    ctrl = sax_drone.DroneController(Mixer(max_block=1024), 48000, voice_id="organ")
    ctrl.set_enabled(True)
    assert ctrl.enabled and ctrl.synth_fallback
    assert ctrl.voice_id == sax_drone.SINE_VOICE_ID


# --- catalog -------------------------------------------------------------------
@pytest.mark.parametrize("key, transp", [
    ("eb_alto", -9), ("bb_tenor", -14), ("horn_eb_alto", -9),
    ("rec_sopranino_f", 12), ("rec_soprano_c", 12),
    ("double_bass", -12), ("bass_guitar", -12),
])
def test_catalog_transpositions(key, transp):
    import sax_instruments
    assert sax_instruments.transp_map()[key] == transp


@pytest.mark.parametrize("key, lowest_sounding", [
    ("double_bass", 28),       # E1, open low string
    ("bass_guitar", 28),       # E1
    ("rec_soprano_c", 72),     # C5
    ("rec_sopranino_f", 77),   # F5
])
def test_catalog_range_covers_lowest_note(key, lowest_sounding):
    import sax_instruments
    lo, _hi = sax_instruments.fingered_range(key)
    assert lo + sax_instruments.transp_map()[key] <= lowest_sounding


# --- interface text -------------------------------------------------------------
@pytest.mark.parametrize("lang", ["de", "en"])
def test_transposition_line_has_single_sign(lang):
    from sax_i18n import STRINGS
    text = STRINGS[lang]["pdf_transp"].format(note="A#/Bb", n=-14)
    assert "-14" in text and "+-" not in text


# --- table frequencies behind a concert-pitch change ----------------------------
def test_notestats_keeps_frequencies_for_a4_rederive():
    from sax_intonation_gui import NoteStats
    st = NoteStats()
    st.add(3.0, 440.76)
    st.add(-1.0)
    assert list(st.freqs) == [440.76] and st.n == 2

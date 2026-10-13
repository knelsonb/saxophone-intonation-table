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
from sax_intonation_log import Measurement, MeasurementLog, RunMeta, _read_csv_text
from sax_metronome import MetronomeController
from sax_mixer import Mixer
from sax_paths import AppPaths


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".intonation_analyzer"
    monkeypatch.setattr(sax_config, "_active", sax_config.ConfigStore(AppPaths(cfg_dir)))
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
def _jsonl_run(run_id: str, started_at: str) -> bytes:
    run = RunMeta(run_id=run_id, started_at=started_at, instrument="eb_alto",
                  a4_hz=440.0)
    return run.to_jsonl().encode("utf-8")


def _jsonl_measurement(run_id: str, midi: int) -> bytes:
    m = Measurement(run_id=run_id, timestamp="2026-10-13T10:00:00",
                    instrument="eb_alto", a4_hz=440.0, midi_sounding=midi,
                    midi_fingered=midi + 9, cents=1.5, freq_hz=440.0)
    return m.to_jsonl().encode("utf-8")


def test_log_recovers_valid_lines_around_damaged_ones(tmp_path):
    """A history file with a torn write, a stray binary byte and foreign JSON
    in the middle still yields every intact record, in order, and nothing
    from the damaged lines."""
    lines = [
        _jsonl_run("r1", "2026-10-13T10:00:00"),
        _jsonl_measurement("r1", 60),
        b"42",                                        # JSON, but not an object
        b"null",
        b"[1]",
        b'{"kind": "run", \xff}',                     # undecodable and malformed
        b"not json at all",
        _jsonl_measurement("r1", 62),
        b'{"kind": "measurement", "run_id": "r1"}',   # fields missing
        b'{"kind": "run", "run_id": "r9", "bogus": 1}',   # unknown field
        b'{"kind": "mystery", "x": 1}',               # unknown record kind
        b"",
        b"   ",
        _jsonl_run("r2", "2026-10-13T11:00:00"),
        _jsonl_measurement("r2", 64),
        b'{"kind": "measurement", "run_id": "r2"',    # torn final line, no newline
    ]
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"\n".join(lines))

    log = MeasurementLog(path=str(path))

    assert [r.run_id for r in log.runs()] == ["r2", "r1"]      # newest first
    assert [m.midi_sounding for m in log.measurements()] == [60, 62, 64]
    assert [m.run_id for m in log.measurements()] == ["r1", "r1", "r2"]


def test_log_keeps_working_after_recovering_from_damage(tmp_path):
    """The recovered log accepts new records and a later start reads both the
    old and the new ones."""
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"\n".join([
        _jsonl_run("r1", "2026-10-13T10:00:00"), _jsonl_measurement("r1", 60),
        b"garbage \xff\xfe",
    ]) + b"\n")

    log = MeasurementLog(path=str(path))
    log.start_run("bb_tenor", 442.0)
    log.add_measurement(midi_sounding=62, midi_fingered=76, cents=-2.0, freq_hz=293.0)

    again = MeasurementLog(path=str(path))
    assert len(again.runs()) == 2
    assert sorted(m.midi_sounding for m in again.measurements()) == [60, 62]


def test_log_with_only_damaged_lines_loads_empty(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'42\nnull\n[1]\n{"kind": "run", \xff}\n')
    log = MeasurementLog(path=str(path))
    assert log.runs() == [] and log.measurements() == []


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
    """The controller is built before an output stream exists; a pad pressed
    once the output has opened must sound at the target pitch on it."""
    block = 2048
    mixer = Mixer(max_block=1024, samplerate=44100)
    ctrl = pp.PitchPipesController(mixer, 44100, max_block=1024, a4=440.0)
    mixer.set_format(sr, block)
    ctrl.toggle(69)
    out = np.zeros(block * 12, dtype=np.float32)
    for i in range(12):
        mixer.render(out[i * block:(i + 1) * block], block)
    err = H.cents(H.fft_peak_hz(out[block * 4:], sr), 440.0)
    assert abs(err) < 5.0, f"A4 pad off by {err:+.1f} cents at {sr} Hz"


def test_a_held_pitch_pipe_follows_an_output_rate_change():
    """A pad held across an output switch keeps sounding, in tune, on the new
    stream, filling every block."""
    mixer = Mixer(max_block=1024, samplerate=48000)
    ctrl = pp.PitchPipesController(mixer, 48000, a4=440.0)
    ctrl.toggle(69)
    for _ in range(8):
        mixer.render(np.zeros(1024, dtype=np.float32), 1024)
    mixer.set_format(96000, 4458)
    assert ctrl.active_midis() == frozenset({69})
    out = np.zeros(4458 * 8, dtype=np.float32)
    for i in range(8):
        mixer.render(out[i * 4458:(i + 1) * 4458], 4458)
    err = H.cents(H.fft_peak_hz(out, 96000), 440.0)
    assert abs(err) < 5.0, f"held A4 pad off by {err:+.1f} cents after the switch"
    assert np.abs(out[-200:]).max() > 0.1, "the end of every block is filled"


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


def test_transp_map_sees_a_custom_registered_after_first_use(monkeypatch):
    # The window, table and exports all read transp_map() at use time; a
    # custom instrument added mid-session must reach every one of them.
    import sax_instruments
    monkeypatch.setattr(sax_instruments, "_CUSTOMS", [])
    assert "custom_mezzo" not in sax_instruments.transp_map()
    sax_instruments.register_custom("custom_mezzo", -7, "Mezzo", "Mezzo")
    assert sax_instruments.transp_map()["custom_mezzo"] == -7
    with pytest.raises(TypeError):
        sax_instruments.transp_map()["eb_alto"] = 0     # read-only: shared cache


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
    from sax_stats import NoteStats
    st = NoteStats()
    st.add(3.0, 440.76)
    st.add(-1.0)
    assert list(st.freqs) == [440.76] and st.n == 2

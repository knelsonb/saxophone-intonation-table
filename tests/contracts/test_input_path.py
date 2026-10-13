"""Input path contract: the callback copies, the analysis thread detects.

The PortAudio input callback only appends samples to the stream's ring (and
the deck tap); a per-stream analysis thread runs the detector once per hop and
emits the readings. These tests run the real thread against the harness's fake
audio host, whose ``pump_input`` plays the PortAudio thread's part.

The ring itself is checked directly: a consumer that falls behind loses whole
hops, never analyses a window the producer overwrote mid-copy.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import sim_harness as sim
from tests.harness.fake_audio import FakeAudioHost

_WAIT_S = 5.0


class _Sink:
    """Stand-in for the engine's Qt signals: records each reading and the
    thread that emitted it."""

    def __init__(self) -> None:
        self.events: list = []
        self.threads: set[str] = set()
        self.got = threading.Event()
        sink = self

        class _NoteEvent:
            @staticmethod
            def emit(ev) -> None:
                sink.events.append(ev)
                sink.threads.add(threading.current_thread().name)
                sink.got.set()

        class _NoteDetected:
            @staticmethod
            def emit(midi, freq, cents) -> None:
                pass

        self.note_event = _NoteEvent()
        self.note_detected = _NoteDetected()

    def wait_for(self, n: int) -> bool:
        deadline = time.monotonic() + _WAIT_S
        while len(self.events) < n and time.monotonic() < deadline:
            self.got.wait(0.05)
            self.got.clear()
        return len(self.events) >= n


@pytest.fixture()
def threaded_engine(load_engine_with_sounddevice):
    host = FakeAudioHost()
    host.add_input("Mic", rates=(48000,))
    mod = load_engine_with_sounddevice(host.sounddevice_module())
    assert mod.ANALYSIS_IN_THREAD, "the shipped default runs detection off the callback"
    eng = mod.AudioEngine()
    sink = _Sink()
    eng.signals = sink
    eng.start()
    assert eng.state == mod.AudioEngineState.RUNNING
    yield mod, eng, host, sink
    eng.stop()
    host.close()


def test_readings_are_detected_and_emitted_off_the_callback_thread(threaded_engine):
    _mod, eng, host, sink = threaded_engine
    tone = sim.sax_like(440.0, 48000, 48000 * 2, 0.5)
    hop = eng.hop_size
    for i in range(0, tone.shape[0] - hop + 1, hop):
        host.pump_input(tone[i:i + hop])        # the test thread plays PortAudio
        time.sleep(0.002)                       # roughly paced, as a device would be
    assert sink.wait_for(3), "the analysis thread never emitted a reading"
    assert threading.current_thread().name not in sink.threads
    assert sink.threads == {"input-analysis"}
    ev = sink.events[0]
    assert ev.midi == 69 and abs(ev.cents) < 1.0
    assert (ev.a4, ev.a4_generation) == (eng.a4, eng.a4_generation)


def test_the_callback_runs_no_detection_when_the_thread_owns_it(threaded_engine):
    mod, eng, host, _sink = threaded_engine
    calls: list[str] = []
    real = eng._detector.process

    def spy(*a, **k):
        calls.append(threading.current_thread().name)
        return real(*a, **k)

    eng._detector.process = spy
    host.pump_input(np.zeros(eng.hop_size * 4, np.float32))
    deadline = time.monotonic() + _WAIT_S
    while len(calls) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == 4, "one analysis per delivered block"
    assert set(calls) == {"input-analysis"}


def test_stopping_the_stream_stops_its_analysis_thread(threaded_engine):
    _mod, eng, _host, _sink = threaded_engine
    thread = eng._analysis_thread
    assert thread is not None and thread.is_alive()
    eng.stop()
    thread.join(_WAIT_S)
    assert not thread.is_alive()
    assert eng._analysis_thread is None


def test_a_new_stream_brings_its_own_ring_at_its_own_rate(threaded_engine):
    mod, eng, host, _sink = threaded_engine
    first = eng._ring
    host.add_input("Interface", rates=(96000,))
    eng.open_device(mod.DeviceSelection(name="Interface"))
    assert eng.state == mod.AudioEngineState.RUNNING
    ring = eng._ring
    assert ring is not first
    assert (ring.samplerate, ring.hop, ring.block) == (96000, eng.hop_size, eng.block_size)
    assert host.input_stream.blocksize == ring.hop


# ---------------------------------------------------------------- the ring
def _ring(block=8, hop=2):
    from sax_audio_engine import _InputRing
    return _InputRing(block, hop, 48000, inline=False)


def test_ring_gives_one_window_per_delivered_block_in_order():
    ring = _ring()
    for k in range(5):
        ring.push(np.full(2, k + 1, np.float32))
    ends = []
    while (end := ring.pop_mark()) is not None:
        assert ring.read_window(end)
        ends.append((end, ring.window.copy()))
    assert [e for e, _w in ends] == [2, 4, 6, 8, 10]
    # The window ending at frame 10 is the last 8 frames: blocks 2..5.
    assert ends[-1][1].tolist() == [2, 2, 3, 3, 4, 4, 5, 5]
    # Before 8 frames exist, the window starts with the ring's zeros.
    assert ends[0][1].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_ring_drops_hops_a_lagging_consumer_can_no_longer_read_intact():
    ring = _ring()                              # capacity: 8 + 16 * 2 = 40 frames
    for k in range(30):                         # 60 frames, nothing consumed
        ring.push(np.full(2, k, np.float32))
    analysed = []
    while (end := ring.pop_mark()) is not None:
        if ring.read_window(end):
            analysed.append(end)
            # An intact window holds exactly the 8 frames before `end`.
            first_block = (end - 8) // 2
            assert ring.window.tolist() == [float(b) for b in
                                            range(first_block, first_block + 4)
                                            for _ in range(2)]
    assert analysed, "the newest windows are still intact"
    assert analysed[0] > 8, "the oldest windows were overwritten and skipped"
    assert analysed[-1] == 60


def test_ring_keeps_only_the_newest_samples_of_an_oversized_block():
    ring = _ring()
    big = np.arange(100, dtype=np.float32)
    ring.push(big)
    end = ring.pop_mark()
    assert end == 100 and ring.read_window(end)
    assert ring.window.tolist() == big[-8:].tolist()


# ------------------------------------------------- A4 generation on readings
def _engine_with_sink():
    from sax_audio_engine import AudioEngine
    eng = AudioEngine()
    sink = _Sink()
    eng.signals = sink
    return eng, sink.events


def test_readings_carry_the_a4_they_were_graded_against():
    eng, events = _engine_with_sink()
    eng.set_a4(442.0)
    gen = eng.a4_generation
    sim.feed_engine(eng, sim.sax_like(440.0, 44100, 16384 * 12, 0.5), capture=False)
    assert events and all((e.a4, e.a4_generation) == (442.0, gen) for e in events)
    # 440 Hz against A4 = 442 Hz is 7.85 cents flat.
    assert all(abs(e.cents + 7.85) < 0.2 for e in events)


def test_an_a4_change_discards_readings_waiting_from_the_old_a4():
    """Readings held back by the release guard were graded against the old A4;
    after a change none of them may come out, even when the note continues."""
    eng, events = _engine_with_sink()
    tone = sim.sax_like(440.0, 44100, 16384 * 30, 0.5)
    for i in range(12):
        eng.feed_input_frames(tone[i * 2048:(i + 1) * 2048])
    held = len(events)
    assert held, "the note was measured before the change"
    eng.set_a4(443.0)
    new_gen = eng.a4_generation
    for i in range(12, 40):
        eng.feed_input_frames(tone[i * 2048:(i + 1) * 2048])
    after = events[held:]
    assert after, "the continuing note is measured at the new A4"
    assert all(e.a4_generation == new_gen and e.a4 == 443.0 for e in after)


def test_the_window_drops_a_reading_graded_against_a_replaced_a4():
    """A reading already queued to the GUI thread when A4 changes still carries
    the old generation; the window must not record it."""
    from types import SimpleNamespace

    import sax_intonation_gui as gui
    from sax_detector import NoteEvent

    recorded = []
    window = SimpleNamespace(_engine=SimpleNamespace(a4_generation=3),
                             _on_note=lambda *a: recorded.append(a))
    gui.MainWindow._on_note_event(window, NoteEvent(69, 440.0, 0.0, 440.0, 2))
    gui.MainWindow._on_note_event(window, NoteEvent(69, 442.0, 1.0, 442.0, 3))
    assert recorded == [(69, 442.0, 1.0)]

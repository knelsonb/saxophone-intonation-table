"""Audio-control contract: one thread owns device lifecycle commands.

Opens, closes, retries and the hot-plug poll run on the engine's
audio-control thread, in the order requested. A request returns at once, even
while a driver hangs, and the outcome arrives as a signal emitted from that
thread. The acceptance scenarios run these commands inline
(``CONTROL_IN_THREAD = False`` in the harness); this file covers the real
thread, against the harness's fake audio host.
"""
from __future__ import annotations

import threading
import time

import pytest

from sax_audio_control import ControlWorker
from tests.harness.fake_audio import FakeAudioHost

_WAIT_S = 5.0
_TIMEOUT_S = 0.2


# ----------------------------------------------------------------- the worker
def test_commands_run_in_order_on_one_thread_off_the_caller():
    worker = ControlWorker(threaded=True)
    ran: list = []
    for i in range(20):
        worker.submit(lambda i=i: ran.append((i, threading.current_thread().name)))
    assert worker.wait_idle(_WAIT_S)
    assert [i for i, _ in ran] == list(range(20))
    assert {name for _, name in ran} == {"audio-control"}
    worker.shutdown()


def test_submit_returns_while_a_command_blocks():
    worker = ControlWorker(threaded=True)
    release = threading.Event()
    done = threading.Event()
    t0 = time.perf_counter()
    worker.submit(release.wait, _WAIT_S)
    worker.submit(done.set)
    assert time.perf_counter() - t0 < 0.1, "submit must not wait for the command"
    assert not done.is_set(), "the second command waits for the first"
    release.set()
    assert done.wait(_WAIT_S)
    worker.shutdown()


def test_a_failing_command_is_reported_and_the_next_one_still_runs(monkeypatch):
    reported: list = []
    monkeypatch.setattr(threading, "excepthook", lambda args: reported.append(args))
    worker = ControlWorker(threaded=True)
    after = threading.Event()

    def boom():
        raise RuntimeError("driver exploded")
    worker.submit(boom)
    worker.submit(after.set)
    assert after.wait(_WAIT_S)
    assert worker.errors == 1
    assert [type(a.exc_value) for a in reported] == [RuntimeError]
    worker.shutdown()


def test_shutdown_runs_what_is_queued_then_refuses_more():
    worker = ControlWorker(threaded=True)
    ran: list = []
    worker.submit(ran.append, 1)
    assert worker.shutdown(_WAIT_S)
    assert ran == [1]
    assert worker.submit(ran.append, 2) is False
    assert ran == [1]


def test_inline_mode_runs_in_the_caller_and_raises_there():
    worker = ControlWorker(threaded=False)
    ran: list = []
    assert worker.submit(lambda: ran.append(threading.current_thread()))
    assert ran == [threading.current_thread()]
    with pytest.raises(ValueError):
        worker.submit(int, "not a number")


# -------------------------------------------------------- the engine on top
class _Signal:
    def __init__(self, sink: "_Sink", name: str) -> None:
        self._sink, self._name = sink, name

    def emit(self, *args) -> None:
        with self._sink.lock:
            self._sink.events.append((self._name, threading.current_thread().name, args))


class _Sink:
    """Stand-in for the engine's Qt signals: records (signal, thread, args)."""

    NAMES = ("state_changed", "devices_changed", "note_detected", "note_event",
             "interface_appeared", "output_state_changed", "output_devices_changed")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list = []
        for name in self.NAMES:
            setattr(self, name, _Signal(self, name))

    def of(self, name: str) -> list:
        with self.lock:
            return [e for e in self.events if e[0] == name]

    def wait_for(self, name: str, pred, timeout: float = _WAIT_S):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for e in self.of(name):
                if pred(e[2]):
                    return e
            time.sleep(0.005)
        return None


@pytest.fixture()
def threaded(load_engine_with_sounddevice):
    host = FakeAudioHost()
    host.add_input("Mic", rates=(44100,))
    host.add_output("Speakers", rates=(48000,))
    mod = load_engine_with_sounddevice(host.sounddevice_module())
    assert mod.CONTROL_IN_THREAD, "the shipped default runs device commands off the GUI thread"
    mod.HOST_API_OPEN_TIMEOUT_S = _TIMEOUT_S
    eng = mod.AudioEngine()
    sink = _Sink()
    eng.signals = sink
    yield mod, eng, host, sink
    eng.shutdown()
    host.close()


def test_a_start_request_returns_before_a_hung_mic_times_out(threaded):
    mod, eng, host, sink = threaded
    host.set_fail("Mic", "hang")
    t0 = time.perf_counter()
    eng.request_start()
    assert time.perf_counter() - t0 < _TIMEOUT_S / 2, "the request must not wait on the driver"

    failed = sink.wait_for("state_changed", lambda a: a[0] == mod.AudioEngineState.FAILED)
    assert failed is not None, "the timeout must come back as a state change"
    _name, thread, (_state, err, _msg, device) = failed
    assert thread == "audio-control"
    assert err == mod.AudioEngineError.HOSTAPI_FAILURE and device == "Mic"
    host.release_hangs()


def test_a_hung_mic_costs_one_timeout_not_one_per_rate(threaded):
    mod, eng, host, sink = threaded
    host.set_rates("Mic", (44100, 48000, 96000))
    host.set_fail("Mic", "hang")
    eng.request_start()
    assert sink.wait_for("state_changed", lambda a: a[0] == mod.AudioEngineState.FAILED)
    # Every abandoned open is still blocked in the driver; let them finish so
    # each one is logged, then count them.
    host.release_hangs()
    attempts = [a for a in host.open_attempts if a[0] == "input"]
    assert len(attempts) == 1, f"one hung open per device, got {attempts}"


def test_output_opens_on_the_control_thread_and_reports_running(threaded):
    mod, eng, host, sink = threaded
    eng.request_open_output(None)
    running = sink.wait_for("output_state_changed",
                            lambda a: a[0].state == mod.OutputState.RUNNING)
    assert running is not None
    assert running[1] == "audio-control"
    assert running[2][0].device == "Speakers"
    assert eng.output_running and eng.mixer.samplerate == 48000


def test_the_poll_notices_an_unplugged_output_and_reopens_it(threaded):
    mod, eng, host, sink = threaded
    eng.request_open_output(None)
    assert sink.wait_for("output_state_changed",
                         lambda a: a[0].state == mod.OutputState.RUNNING)
    host.unplug("Speakers")
    eng.request_poll()
    lost = sink.wait_for("output_state_changed",
                         lambda a: a[0].state == mod.OutputState.DISCONNECTED)
    assert lost is not None and lost[2][0].device == "Speakers"
    host.plug("Speakers")
    eng.request_poll()
    assert eng._control.wait_idle(_WAIT_S)
    states = [e[2][0].state for e in sink.of("output_state_changed")]
    assert states[-1] == mod.OutputState.RUNNING, states


def test_shutdown_closes_both_streams(threaded):
    mod, eng, host, sink = threaded
    eng.request_start()
    eng.request_open_output(None)
    assert eng._control.wait_idle(_WAIT_S)
    assert eng.input_running and eng.output_running
    eng.shutdown()
    assert host.open_streams() == []
    assert not eng.submit(lambda: None), "no command runs after shutdown"

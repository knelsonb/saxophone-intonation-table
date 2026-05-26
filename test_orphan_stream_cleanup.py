"""v0.5.7.7: when _try_open's worker thread overruns
HOST_API_OPEN_TIMEOUT_S, the eventually-returned stream must be
stopped + closed by the worker itself -- not orphaned.
"""
from __future__ import annotations

import threading
import time
import types
from unittest import mock

import sax_audio_engine as engine_mod


class _SlowStream:
    """Stand-in for sounddevice.InputStream that takes ~2s to construct."""

    instances: list["_SlowStream"] = []

    def __init__(self, **kwargs):
        # Block construction long enough for the main thread to give up
        # at HOST_API_OPEN_TIMEOUT_S (0.8s default).
        time.sleep(2.0)
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        _SlowStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_orphan_stream_disposed_after_timeout() -> None:
    _SlowStream.instances.clear()

    eng = engine_mod.AudioEngine.__new__(engine_mod.AudioEngine)
    eng._lock = threading.RLock()
    eng._make_callback = lambda sr: (lambda *a, **k: None)
    eng._classify_error = lambda exc: engine_mod.AudioEngineError.UNKNOWN

    dev = types.SimpleNamespace(index=0, name='slow-dev', host_api='Test')

    with mock.patch.object(engine_mod.sd, 'InputStream', _SlowStream):
        t0 = time.monotonic()
        ok, kind, msg = eng._try_open(dev, samplerate=48000)
        elapsed = time.monotonic() - t0

    assert not ok, "expected timeout failure"
    assert kind == engine_mod.AudioEngineError.HOSTAPI_FAILURE, kind
    assert 'timed out' in msg, msg
    assert elapsed < 1.5, (
        f"_try_open should have returned near the 0.8s timeout, "
        f"not waited for the 2s open ({elapsed:.2f}s)")

    # Wait for the worker to finish its slow open and run cleanup.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if _SlowStream.instances and _SlowStream.instances[-1].closed:
            break
        time.sleep(0.05)

    assert _SlowStream.instances, "worker never constructed a stream"
    s = _SlowStream.instances[-1]
    assert s.started, "worker should have called start() before checking cancelled"
    assert s.stopped, "orphaned stream was not stopped"
    assert s.closed, "orphaned stream was not closed"

    print("test_orphan_stream_cleanup: OK")


if __name__ == '__main__':
    test_orphan_stream_disposed_after_timeout()

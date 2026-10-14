"""The audio-control thread: one owner for device lifecycle commands.

Opening an audio device can take hundreds of milliseconds, and a wedged driver
can hang until the engine's open timeout gives up on it. Commands that open,
close, retry or re-scan devices therefore run here, one at a time and in the
order they were asked for, on a single thread. The GUI only queues them and
learns the outcome from the engine's signals, so it never waits on a driver,
and two commands can never interleave (a hot-plug recovery cannot race a
device switch the player just made).

``ControlWorker(threaded=False)`` runs each command inline in the caller
instead. The headless test harness uses that mode: a command has finished, and
its signals have been delivered, by the time the call that asked for it
returns, so scenarios stay deterministic. The threaded mode is covered by
tests/contracts/test_audio_control.py.

Qt-free and engine-agnostic: a command is any callable.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

__all__ = ["ControlWorker"]

_STOP = object()


def _report(exc: BaseException) -> None:
    """Hand a failed command to ``threading.excepthook`` (the crash log), as an
    uncaught exception on this thread would be, without ending the thread."""
    args = threading.ExceptHookArgs(
        (type(exc), exc, exc.__traceback__, threading.current_thread()))
    threading.excepthook(args)


class ControlWorker:
    """Runs submitted commands in order on one thread (or inline)."""

    def __init__(self, threaded: bool = True, name: str = "audio-control") -> None:
        self._threaded = bool(threaded)
        self._name = name
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False
        # Commands that raised (on the thread; inline errors reach the caller).
        self.errors = 0

    @property
    def threaded(self) -> bool:
        return self._threaded

    def on_control_thread(self) -> bool:
        """True when called from inside a command (or always, inline)."""
        return (not self._threaded
                or threading.current_thread() is self._thread)

    def submit(self, fn: Callable[..., object], *args, **kwargs) -> bool:
        """Run ``fn(*args, **kwargs)`` after every command submitted before it.

        Threaded: returns at once. Inline: runs it now, and an exception
        propagates to the caller like any other call's would. Returns False
        (and runs nothing) once the worker has been shut down.
        """
        with self._lock:
            if self._closed:
                return False
            if self._threaded:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name=self._name, daemon=True)
                    self._thread.start()
                self._queue.put((fn, args, kwargs))
                return True
        fn(*args, **kwargs)
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                fn, args, kwargs = item
                try:
                    fn(*args, **kwargs)
                except Exception as exc:
                    # One bad command must not stop the ones queued after it.
                    self.errors += 1
                    _report(exc)
            finally:
                self._queue.task_done()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait until every command submitted so far has run. True if it did
        within ``timeout`` seconds. Never call it from inside a command."""
        if not self._threaded or self._thread is None:
            return True
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Let the commands already queued run, then stop the thread; later
        submits are refused. Returns False if the thread is still busy (in a
        hung driver call) after ``timeout`` seconds; it is a daemon thread, so
        it cannot keep the process alive."""
        with self._lock:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                thread = self._thread
                if thread is not None:
                    self._queue.put(_STOP)
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

"""What the player has switched on that needs the audio output.

Every sound source the player turns on (test tone, metronome, drone, a pitch
pad, deck playback) asks for the output through ``want``: it starts at once
while the output runs; otherwise it waits, and one open is requested for all
the waiting users together. Device opens run on the audio-control thread, so
the answer comes later, as an ``OutputStatus`` from the engine
(``on_status``):

* RUNNING: every waiting user starts.
* FAILED: every user that asked is told why, so its toggle can go back off
  and say what to do. Parked users are told too, unless the output failed
  because the device is missing: they keep waiting for it to come back.
* DISCONNECTED: the window stops whatever was sounding and parks it here with
  ``park``. Parked users start again when the output runs again, which the
  engine brings about itself once the device is plugged back in.

Qt-free: users are plain callables, keyed so a second request for the same
thing replaces the first and the player can withdraw one (``drop``).
"""
from __future__ import annotations

from typing import Callable, Optional

__all__ = ["OutputUsers"]

Start = Callable[[], None]
Fail = Callable[[Optional[object]], None]


class OutputUsers:
    def __init__(self, is_running: Callable[[], bool],
                 request_open: Callable[[], None]) -> None:
        self._is_running = is_running
        self._request_open = request_open
        # key -> (start, fail, parked), in the order they asked.
        self._waiting: dict[str, tuple[Start, Fail, bool]] = {}
        # An open was requested and its answer has not arrived yet.
        self._open_pending = False

    def want(self, key: str, start: Start, fail: Fail) -> None:
        """Start ``key`` now if the output runs, else as soon as it does."""
        if self._is_running():
            self._waiting.pop(key, None)
            start()
            return
        self._waiting[key] = (start, fail, False)
        if not self._open_pending:
            self._open_pending = True
            self._request_open()

    def park(self, key: str, start: Start, fail: Fail) -> None:
        """Wait for the output without asking for it to be opened (it was
        unplugged; it reopens when it is back)."""
        self._waiting[key] = (start, fail, True)

    def drop(self, key: str) -> None:
        """The player switched ``key`` off before the output came."""
        self._waiting.pop(key, None)

    def drop_prefix(self, prefix: str) -> None:
        for key in [k for k in self._waiting if k.startswith(prefix)]:
            del self._waiting[key]

    def waiting(self, key: str) -> bool:
        return key in self._waiting

    def any_parked(self) -> bool:
        return any(parked for _s, _f, parked in self._waiting.values())

    def on_status(self, status) -> None:
        """Act on an OutputStatus (duck-typed: ``status.state.value``)."""
        state = getattr(getattr(status, "state", None), "value", "")
        self._open_pending = False
        if state == "running":
            users, self._waiting = list(self._waiting.values()), {}
            for start, _fail, _parked in users:
                start()
        elif state == "failed":
            error = getattr(getattr(status, "error", None), "value", "")
            missing = error in ("no_device", "device_disconnected")
            failed = []
            for key, (_start, fail, parked) in list(self._waiting.items()):
                if parked and missing:
                    continue        # still waiting for the device to return
                del self._waiting[key]
                failed.append(fail)
            for fail in failed:
                fail(status)

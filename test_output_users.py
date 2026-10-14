"""OutputUsers: sound sources waiting for, parked until, or failed by the output."""
from __future__ import annotations

from types import SimpleNamespace

from intonation.audio.engine import AudioEngineError, OutputState, OutputStatus
from intonation.audio.output_users import OutputUsers


class _Rig:
    def __init__(self, running=False):
        self.running = running
        self.opens = 0
        self.log = []
        self.users = OutputUsers(lambda: self.running, self._open)

    def _open(self):
        self.opens += 1

    def user(self, key):
        return (lambda: self.log.append(("start", key)),
                lambda st: self.log.append(("fail", key, st.error if st else None)))


def test_starts_at_once_while_the_output_runs():
    rig = _Rig(running=True)
    rig.users.want("tone", *rig.user("tone"))
    assert rig.log == [("start", "tone")] and rig.opens == 0


def test_waiting_users_share_one_open_and_start_together():
    rig = _Rig()
    rig.users.want("tone", *rig.user("tone"))
    rig.users.want("metro", *rig.user("metro"))
    assert rig.opens == 1 and rig.log == []
    rig.users.on_status(OutputStatus(OutputState.RUNNING))
    assert rig.log == [("start", "tone"), ("start", "metro")]


def test_a_failed_open_tells_every_user_that_asked():
    rig = _Rig()
    rig.users.want("tone", *rig.user("tone"))
    status = OutputStatus(OutputState.FAILED, AudioEngineError.UNSUPPORTED_RATE, "Spk")
    rig.users.on_status(status)
    assert rig.log == [("fail", "tone", AudioEngineError.UNSUPPORTED_RATE)]
    rig.users.want("tone", *rig.user("tone"))
    assert rig.opens == 2, "the next request asks again"


def test_parked_users_wait_out_a_missing_device_but_not_other_failures():
    rig = _Rig()
    rig.users.park("drone", *rig.user("drone"))
    assert rig.opens == 0, "parking does not ask for an open"
    rig.users.on_status(OutputStatus(OutputState.FAILED, AudioEngineError.NO_DEVICE))
    assert rig.log == [] and rig.users.any_parked()
    rig.users.on_status(OutputStatus(OutputState.FAILED, AudioEngineError.DEVICE_BUSY))
    assert rig.log == [("fail", "drone", AudioEngineError.DEVICE_BUSY)]
    assert not rig.users.any_parked()


def test_dropped_users_never_start():
    rig = _Rig()
    rig.users.want("pipe:60", *rig.user("pipe:60"))
    rig.users.want("pipe:62", *rig.user("pipe:62"))
    rig.users.drop_prefix("pipe:6")
    rig.users.on_status(SimpleNamespace(state=OutputState.RUNNING))
    assert rig.log == []

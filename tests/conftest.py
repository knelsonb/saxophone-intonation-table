"""Fixtures for the headless acceptance suite.

``app``        a launched app: English UI, welcome already answered, the list
               table layout, one microphone (44.1 kHz) and one speaker (48 kHz).
``fresh_app``  the same app NOT yet launched and with no saved settings, so the
               first-launch welcome dialog appears. Script it, then ``launch()``::

                   fresh_app.welcome.decide_later()
                   fresh_app.launch()

``make_app``   factory for apps with their own seed files, language or devices.
               Apps from it are not launched; call ``launch()``.
``clock`` / ``host``  the deterministic clock and the fake audio system.
"""
from __future__ import annotations

import pytest

from tests.harness import HeadlessApp, ManualClock, default_host


@pytest.fixture()
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture()
def host(clock):
    h = default_host(clock)
    yield h
    h.close()


@pytest.fixture()
def make_app(tmp_path, clock, host):
    """Factory: ``make_app(lang='en', config=..., history=..., first_launch=...)``.
    Every app shares one home directory, so a second call is a second launch of
    the same user's installation; close the first before launching the second."""
    made: list[HeadlessApp] = []

    def factory(**kw) -> HeadlessApp:
        kw.setdefault("host", host)
        kw.setdefault("clock", clock)
        a = HeadlessApp(tmp_path / "home", **kw)
        made.append(a)
        return a

    yield factory
    errors = []
    for a in reversed(made):
        try:
            a.close()
        except AssertionError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


@pytest.fixture()
def app(make_app) -> HeadlessApp:
    a = make_app()
    a.launch()
    return a


@pytest.fixture()
def fresh_app(make_app) -> HeadlessApp:
    return make_app(first_launch=True)

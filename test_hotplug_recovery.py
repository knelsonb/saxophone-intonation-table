"""
Deterministic tests for the v0.5.7 FAILED -> NEW_DEVICE -> RECOVERY path.

Each test function is independent in what it asserts, but they share a
single fake-sounddevice installation and a single engine-module import
that both happen at module load time.  This preserves the original
requirement: the fake must be in sys.modules['sounddevice'] *before*
sax_audio_engine is imported so that AUDIO_OK is True.

Safe to run under pytest; no GUI or real audio hardware required.
"""
from __future__ import annotations

import importlib
import re
import sys
import types

# ---------------------------------------------------------------------------
# Install the fake sounddevice BEFORE importing sax_audio_engine.
# All test functions below share this single fake and its mutable state.
# ---------------------------------------------------------------------------

_fake_sd = types.ModuleType('sounddevice')

# Mutable state shared by all tests; individual tests reset the fields they
# care about at the start of each phase so they remain independent.
_state: dict = {
    'devices': [],          # list of dicts mimicking sd.query_devices()
    'hostapis': [{'name': 'Windows WASAPI'}],
    'default_input_idx': None,
    'opens': [],            # captured (index, samplerate) on InputStream
    'fail_open': False,
}


def _query_devices(idx=None):
    if idx is None:
        return list(_state['devices'])
    return _state['devices'][idx]


def _query_hostapis():
    return list(_state['hostapis'])


class _DefaultNs:
    device = (None, None)


_fake_sd.default = _DefaultNs()


def _check_input_settings(device=None, channels=None,
                           dtype=None, samplerate=None):
    # Accept all rates so negotiation always returns something.
    return None


class _FakeStream:
    def __init__(self, samplerate, blocksize, channels, dtype,
                 callback, device):
        _state['opens'].append((device, int(samplerate)))
        if _state['fail_open']:
            raise RuntimeError('simulated open failure')
        self._samplerate = samplerate

    def start(self):
        return None

    def stop(self):
        return None

    def close(self):
        return None


_fake_sd.query_devices = _query_devices
_fake_sd.query_hostapis = _query_hostapis
_fake_sd.check_input_settings = _check_input_settings
_fake_sd.InputStream = _FakeStream

sys.modules['sounddevice'] = _fake_sd

# Now import (or re-import) the engine with the fake in place.
if 'sax_audio_engine' in sys.modules:
    del sys.modules['sax_audio_engine']
_engine_mod = importlib.import_module('sax_audio_engine')

AudioEngine = _engine_mod.AudioEngine
AudioEngineState = _engine_mod.AudioEngineState
AudioEngineError = _engine_mod.AudioEngineError
DeviceSelection = _engine_mod.DeviceSelection

# ---------------------------------------------------------------------------
# Helper: a fresh FIIO device descriptor used by several tests.
# ---------------------------------------------------------------------------
_FIIO_DEVICE = {
    'name': 'FIIO DSP Audio',
    'hostapi': 0,
    'max_input_channels': 1,
    'default_samplerate': 48000.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_module_loaded_with_fake_sounddevice():
    """The fake sounddevice must make AUDIO_OK True at import time."""
    assert _engine_mod.AUDIO_OK, 'fake sounddevice failed to load'


def test_no_devices_on_boot_reaches_failed():
    """Phase 1 — empty device list: engine must start in FAILED/NO_DEVICE."""
    _state['devices'] = []
    _state['opens'] = []
    _state['fail_open'] = False

    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.start()

    assert eng.state == AudioEngineState.FAILED, \
        f'expected FAILED on empty boot, got {eng.state}'
    assert eng.last_error == AudioEngineError.NO_DEVICE, \
        f'expected NO_DEVICE, got {eng.last_error}'
    assert _state['opens'] == [], 'should not have attempted to open'


def test_refresh_devices_advances_timestamp_with_no_devices():
    """Phase 2 — tick the poller while still empty: timestamp must advance."""
    _state['devices'] = []
    _state['opens'] = []
    _state['fail_open'] = False

    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.start()

    devs = eng.refresh_devices()

    assert devs == [], 'still no devices'
    assert eng.state == AudioEngineState.FAILED, 'state must not change'
    assert eng.last_devices_refresh_at is not None, \
        'last_devices_refresh_at must advance on every tick'


def test_hotplug_new_device_reaches_running():
    """Phase 3 — plug in a matching device mid-run: engine must reach RUNNING."""
    _state['devices'] = []
    _state['opens'] = []
    _state['fail_open'] = False

    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.start()

    # Simulate hot-plug then poll.
    _state['devices'] = [_FIIO_DEVICE]
    eng.refresh_devices()

    assert _state['opens'], 'engine did NOT attempt to open hot-plugged device'
    assert eng.state == AudioEngineState.RUNNING, (
        f'expected RUNNING after hot-plug, got {eng.state} '
        f'(err={eng.last_error}, msg={eng.last_error_message})')
    assert eng.active_device is not None
    assert eng.active_device.name == 'FIIO DSP Audio'


def test_retry_open_reenumerates_and_reaches_running():
    """Phase 4 — retry_open() must re-enumerate and resolve via the hint."""
    _state['devices'] = []
    _state['opens'] = []
    _state['fail_open'] = False

    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.start()

    assert eng.state == AudioEngineState.FAILED

    # Now plug in the device and call retry_open — must re-enumerate
    # (it can't trust the cached snapshot) and resolve via the hint.
    _state['devices'] = [_FIIO_DEVICE]
    eng.retry_open()

    assert eng.state == AudioEngineState.RUNNING, (
        f'retry_open did not reach RUNNING, got {eng.state} '
        f'(err={eng.last_error}, msg={eng.last_error_message})')


def test_retry_open_with_absent_pinned_device_stays_failed():
    """Phase 5 — retry_open with absent pinned device must report NO_DEVICE."""
    _state['fail_open'] = False
    _state['opens'] = []
    _state['devices'] = [{
        'name': 'Some Other Mic',
        'hostapi': 0,
        'max_input_channels': 1,
        'default_samplerate': 44100.0,
    }]

    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.retry_open()

    assert eng.state == AudioEngineState.FAILED
    assert eng.last_error == AudioEngineError.NO_DEVICE
    assert 'Pinned device not present' in eng.last_error_message, (
        f'expected pinned-device error message, got '
        f'{eng.last_error_message!r}')


def test_vendor_regex_covers_expected_brands():
    """Phase 6 — VENDOR_REGEX must match known pro-audio brands."""
    vendor_re = re.compile(_engine_mod.VENDOR_REGEX, re.IGNORECASE)

    should_match = (
        'Headset (FIIO DSP Audio)',
        'Apogee Duet',
        'Universal Audio Apollo',
        'Roland Quad-Capture',
        'Native Instruments Komplete Audio 6',
        'Tascam US-2x2',
        'PreSonus AudioBox',
        'Zoom F8n',
        'Antelope Discrete 4',
        'IK Multimedia AXE I/O',
    )
    for name in should_match:
        assert vendor_re.search(name), \
            f'vendor regex missed expected brand match: {name!r}'


def test_vendor_regex_does_not_match_generic_windows_audio():
    """Phase 6b — VENDOR_REGEX must NOT match generic Windows system audio."""
    vendor_re = re.compile(_engine_mod.VENDOR_REGEX, re.IGNORECASE)

    should_not_match = (
        'Microphone Array (Realtek)',
        'NVIDIA HDMI Audio',
        'Stereo Mix (Realtek)',
    )
    for name in should_not_match:
        assert not vendor_re.search(name), \
            f'vendor regex falsely matched: {name!r}'

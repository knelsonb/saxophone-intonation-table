"""
Deterministic test for the v0.5.7 FAILED -> NEW_DEVICE -> RECOVERY path.

Boots the engine with an empty device list, ticks refresh_devices(),
then injects a new input device and ticks again. Asserts that the
engine attempts to open the new device (and reaches RUNNING when the
fake open succeeds).

Stand-alone — no pytest dependency. Exit code 0 on pass, non-zero on
fail. Safe to add to CI later.
"""
from __future__ import annotations

import sys
import types


def _run() -> None:
    # Fake sounddevice BEFORE importing the engine so AUDIO_OK is True.
    fake_sd = types.ModuleType('sounddevice')
    state = {
        'devices': [],          # list of dicts mimicking sd.query_devices()
        'hostapis': [{'name': 'Windows WASAPI'}],
        'default_input_idx': None,
        'opens': [],            # captured (index, samplerate) on InputStream
        'fail_open': False,
    }

    def query_devices(idx=None):
        if idx is None:
            return list(state['devices'])
        return state['devices'][idx]

    def query_hostapis():
        return list(state['hostapis'])

    class _DefaultNs:
        device = (None, None)

    fake_sd.default = _DefaultNs()

    def check_input_settings(device=None, channels=None,
                             dtype=None, samplerate=None):
        # Accept all rates so negotiation always returns something.
        return None

    class FakeStream:
        def __init__(self, samplerate, blocksize, channels, dtype,
                     callback, device):
            state['opens'].append((device, int(samplerate)))
            if state['fail_open']:
                raise RuntimeError('simulated open failure')
            self._samplerate = samplerate

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    fake_sd.query_devices = query_devices
    fake_sd.query_hostapis = query_hostapis
    fake_sd.check_input_settings = check_input_settings
    fake_sd.InputStream = FakeStream

    sys.modules['sounddevice'] = fake_sd

    # Import after the fake is in place.
    import importlib
    if 'sax_audio_engine' in sys.modules:
        del sys.modules['sax_audio_engine']
    engine_mod = importlib.import_module('sax_audio_engine')
    assert engine_mod.AUDIO_OK, 'fake sounddevice failed to load'

    AudioEngine = engine_mod.AudioEngine
    AudioEngineState = engine_mod.AudioEngineState
    AudioEngineError = engine_mod.AudioEngineError
    DeviceSelection = engine_mod.DeviceSelection

    # ---- Phase 1: no devices on boot --------------------------------
    eng = AudioEngine()
    eng.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng.start()
    assert eng.state == AudioEngineState.FAILED, \
        f'expected FAILED on empty boot, got {eng.state}'
    assert eng.last_error == AudioEngineError.NO_DEVICE, \
        f'expected NO_DEVICE, got {eng.last_error}'
    assert state['opens'] == [], 'should not have attempted to open'

    # ---- Phase 2: tick the poller, still nothing --------------------
    devs = eng.refresh_devices()
    assert devs == [], 'still no devices'
    assert eng.state == AudioEngineState.FAILED, 'state must not change'
    assert eng.last_devices_refresh_at is not None, \
        'last_devices_refresh_at must advance on every tick'

    # ---- Phase 3: hot-plug a matching device, tick again ------------
    state['devices'] = [{
        'name': 'FIIO DSP Audio',
        'hostapi': 0,
        'max_input_channels': 1,
        'default_samplerate': 48000.0,
    }]
    eng.refresh_devices()
    # The engine should have attempted to open the new device and,
    # because our FakeStream succeeds, reached RUNNING.
    assert state['opens'], 'engine did NOT attempt to open hot-plugged device'
    assert eng.state == AudioEngineState.RUNNING, (
        f'expected RUNNING after hot-plug, got {eng.state} '
        f'(err={eng.last_error}, msg={eng.last_error_message})')
    assert eng.active_device is not None
    assert eng.active_device.name == 'FIIO DSP Audio'

    # ---- Phase 4: retry_open() forces re-enumeration ----------------
    eng2 = AudioEngine()
    state['opens'].clear()
    state['devices'] = []
    eng2.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng2.start()
    assert eng2.state == AudioEngineState.FAILED
    # Now plug in the device and call retry_open — must re-enumerate
    # (it can't trust the cached snapshot) and resolve via the hint.
    state['devices'] = [{
        'name': 'FIIO DSP Audio',
        'hostapi': 0,
        'max_input_channels': 1,
        'default_samplerate': 48000.0,
    }]
    eng2.retry_open()
    assert eng2.state == AudioEngineState.RUNNING, (
        f'retry_open did not reach RUNNING, got {eng2.state} '
        f'(err={eng2.last_error}, msg={eng2.last_error_message})')

    # ---- Phase 5: retry_open with absent pinned device -------------
    eng3 = AudioEngine()
    state['devices'] = [{
        'name': 'Some Other Mic',
        'hostapi': 0,
        'max_input_channels': 1,
        'default_samplerate': 44100.0,
    }]
    eng3.set_preferred_hint(
        DeviceSelection(name='FIIO DSP Audio',
                        host_api='Windows WASAPI', samplerate=0))
    eng3.retry_open()
    assert eng3.state == AudioEngineState.FAILED
    assert eng3.last_error == AudioEngineError.NO_DEVICE
    assert 'Pinned device not present' in eng3.last_error_message, (
        f'expected pinned-device error message, got '
        f'{eng3.last_error_message!r}')

    # ---- Phase 6: vendor regex covers the new brands ----------------
    import re
    vendor_re = re.compile(engine_mod.VENDOR_REGEX, re.IGNORECASE)
    for name in ('Headset (FIIO DSP Audio)', 'Apogee Duet',
                 'Universal Audio Apollo', 'Roland Quad-Capture',
                 'Native Instruments Komplete Audio 6',
                 'Tascam US-2x2', 'PreSonus AudioBox',
                 'Zoom F8n', 'Antelope Discrete 4',
                 'IK Multimedia AXE I/O'):
        assert vendor_re.search(name), \
            f'vendor regex missed expected brand match: {name!r}'
    # And does NOT match generic Windows kit.
    for name in ('Microphone Array (Realtek)', 'NVIDIA HDMI Audio',
                 'Stereo Mix (Realtek)'):
        assert not vendor_re.search(name), \
            f'vendor regex falsely matched: {name!r}'

    print('test_hotplug_recovery: OK')


if __name__ == '__main__':
    try:
        _run()
    except AssertionError as e:
        print(f'test_hotplug_recovery: FAIL — {e}')
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'test_hotplug_recovery: ERROR — {e}')
        sys.exit(2)

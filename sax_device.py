# =============================================================================
# sax_device.py — DeviceController
#
# Extracted from sax_intonation_gui.py as one of the MainWindow controllers.
# The controller owns the audio device picker open flow, the retry-on-failure
# path, the 1 Hz hot-plug poll, the engine-state / devices-changed / interface-appeared Qt slot
# bodies, and the per-active-device write-back to AppConfig.
#
# Device commands are only requested here (engine.request_*): they run on the
# engine's audio-control thread and the outcome comes back through the engine
# signals, so the window never waits on a driver.
#
# Design notes:
#   * The QTimer that drives the hot-plug poll and the Qt signal-slot
#     connections from the engine stay wired in MainWindow.__init__ for
#     grep-ability.  MainWindow keeps one-line wrapper methods so the
#     Qt signal/slot bindings (which PyQt type-checks at connect time)
#     are unchanged — only the slot *bodies* move here.
#   * The AudioPickerDialog lives in sax_dialogs.  The controller just
#     opens it.
#   * The two banner widgets the controller talks to (AudioRecoveryBanner
#     and InfoBanner) are passed in at construction so the controller
#     does not have to reach into MainWindow's widget tree by name.
#   * Live values that may change during a session (the current UI
#     language) reach the controller through a getter callable.  An
#     optional `on_active_device_changed` callback is fired after the
#     write-back so MainWindow can do any downstream refresh it needs
#     (window-title update, etc.) without the controller having to
#     know about it.
#   * The engine-state signal names the device the state is about, so a
#     failure banner never depends on which device happens to be active
#     when the (queued) slot runs.  The RUNNING branch still reads
#     engine.get_active_device() once into a local.
#   * The saved input is persisted only when the player picked a device or
#     the engine opened the saved one (or nothing was saved yet).  A
#     fallback to another input says so and leaves the saved choice alone,
#     so the saved interface is used again once it is back.
#   * The controller's `_audio_ok` flag is captured at construction
#     time from sax_audio_engine.AUDIO_OK (sounddevice importable).
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtWidgets import QDialog

import sax_config
from sax_audio_engine import (
    AUDIO_OK,
    AudioEngineError,
    AudioEngineState,
    DeviceInfo,
    DeviceSelection,
)
from sax_dialogs import AudioPickerDialog

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

    from sax_config import AppConfig


class DeviceController:
    """Owns audio device picker, retry-on-failure, hot-plug poller,
    engine-state / devices-changed / interface-appeared Qt slot handlers,
    and the per-active-device persistence write-back.

    Connected to engine signals via constructor wiring done by MainWindow
    (the connect() calls live in MainWindow.__init__ for grep-ability;
    the slot bodies live here)."""

    def __init__(self, window: 'QMainWindow', *,
                 engine,
                 cfg: 'AppConfig',
                 t_func: Callable[..., str],
                 info_banner,
                 audio_banner,
                 get_lang: Callable[[], str],
                 on_active_device_changed: Optional[Callable[[], None]] = None
                 ) -> None:
        self._w = window
        self._engine = engine
        self._cfg = cfg
        self._t = t_func
        self._info_banner = info_banner
        self._audio_banner = audio_banner
        self._get_lang = get_lang
        self._on_active_device_changed = on_active_device_changed
        # AUDIO_OK only says sounddevice is importable; the engine's open()
        # can still fail at runtime and surface FAILED via state_changed.
        self._audio_ok = bool(AUDIO_OK)
        # The device the player picked, until the open it asked for lands.
        self._user_pick: Optional[DeviceSelection] = None

    # -------------------------------------------------------------------------
    # Persistence write-back
    # -------------------------------------------------------------------------
    def persist_active_device(self) -> None:
        """Write the engine's currently-active device back to config.
        Index is deliberately NOT stored — only name + host API + rate,
        because device indices change between sessions."""
        dev = self._engine.get_active_device()
        if dev is None:
            return
        self._cfg.audio_device_name = dev.name
        self._cfg.audio_device_host_api = dev.host_api
        self._cfg.audio_device_samplerate = int(self._engine.samplerate or 0)
        sax_config.save_config(self._cfg)
        # v0.5.7: keep the engine's hot-plug auto-recovery hint in sync
        # with whatever's actually active. Otherwise the user picks a
        # new device, unplugs it, plugs it back in, and the engine
        # still resolves the stale launch-time hint.
        self._engine.set_preferred_hint(DeviceSelection(
            name=dev.name, host_api=dev.host_api,
            samplerate=int(self._engine.samplerate or 0)))
        if self._on_active_device_changed is not None:
            try:
                self._on_active_device_changed()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Picker open + retry + 1 Hz hot-plug poll
    # -------------------------------------------------------------------------
    def open_audio_picker(self) -> None:
        if not self._audio_ok:
            return
        dlg = AudioPickerDialog(self._w, self._t, self._engine, self._cfg,
                                self._engine.get_active_device())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.chosen()
            if chosen is not None:
                pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                    'auto') or 'auto')
                self._open_picked(DeviceSelection(name=chosen.name,
                                                  host_api=chosen.host_api,
                                                  samplerate=0), pref)

    def _open_picked(self, sel: DeviceSelection,
                     samplerate_pref: Optional[str] = None) -> None:
        """Open a device the player chose; it becomes the saved input once
        it runs."""
        self._user_pick = sel
        self._engine.request_open_device(sel, samplerate_pref=samplerate_pref)

    def retry_audio(self) -> None:
        if not self._audio_ok:
            return
        # v0.5.7: force a fresh PortAudio enumeration before re-opening
        # so a device plugged in after launch can be picked up. The old
        # ``engine.retry()`` reused the stale snapshot and never saw
        # the hot-plugged device.
        self._engine.request_retry_open()

    def poll_devices(self) -> None:
        if not self._audio_ok:
            return
        # One hot-plug tick for inputs and outputs; the engine emits signals
        # on any change.
        self._engine.request_poll()

    # -------------------------------------------------------------------------
    # Engine Qt slot bodies
    # -------------------------------------------------------------------------
    def on_engine_state(self, state, err, msg, device: str = '') -> None:
        """React to engine state transitions: update chip, banner, and
        diagnostics panel device label. ``device`` names the device the
        state is about."""
        dev = self._engine.get_active_device()
        name = dev.name if dev else ''
        host = dev.host_api if dev else ''
        sr = int(self._engine.samplerate or 0)
        w = self._w
        w._audio_chip.update_from_state(state, name, host, sr)
        if self._audio_banner is not None:
            if state == AudioEngineState.FAILED:
                # A device-less NO_DEVICE (nothing plugged in at all) must not
                # borrow the name of a device used earlier.
                failed = device or (
                    name if err != AudioEngineError.NO_DEVICE else '')
                self._audio_banner.show_for(err, failed, msg)
            else:
                self._audio_banner.hide()
        if state in (AudioEngineState.RUNNING, AudioEngineState.FAILED):
            picked, self._user_pick = self._user_pick, None
        else:
            picked = self._user_pick
        if state == AudioEngineState.RUNNING and dev is not None:
            self._settle_device(dev, picked)
            if sr:
                w._tuner.sample_rate = sr
            # First-time-only notice if we wound up at a non-44.1k rate.
            if (sr and sr != 44100
                    and not bool(getattr(self._cfg,
                                          'audio_sr_notice_shown', False))):
                self._cfg.audio_sr_notice_shown = True
                sax_config.save_config(self._cfg)
                # Surface as a passive status-bar line, not a modal, so a
                # sample-rate disclosure never blocks the player.
                w._status_lbl.setText(self._t('audio_sr_notice', sr=sr))

    def _settle_device(self, dev: DeviceInfo,
                       picked: Optional[DeviceSelection]) -> None:
        """An input runs. Remember it if the player chose it, if it is the
        saved one, or if none was saved yet; otherwise it is a fallback for
        a saved device that is missing: say so, and keep the saved choice
        (and the engine's recovery hint) pointing at the missing device."""
        saved = str(getattr(self._cfg, 'audio_device_name', '') or '')
        saved_host = str(getattr(self._cfg, 'audio_device_host_api', '') or '')
        chosen = picked is not None and picked.name == dev.name
        is_saved = (dev.name == saved
                    and (not saved_host or dev.host_api == saved_host))
        if chosen or is_saved or not saved:
            self.persist_active_device()
        elif self._info_banner is not None:
            self._info_banner.show_message(self._t(
                'audio_using_fallback', name=dev.name, saved=saved))

    def on_devices_changed(self, _devices) -> None:
        # Nothing to do at the MainWindow level — the picker reads
        # devices lazily on each open, and the chip is driven by state
        # changes, not the device list. Hook kept so unit tests / future
        # toasts can subscribe without rewiring the engine.
        pass

    def on_interface_appeared(self, device: 'DeviceInfo') -> None:
        """Hot-plug banner for vendor-class interfaces. Polite, non-modal,
        Only fires for matched vendor names so the user isn't trained to
        dismiss it on every wake-from-sleep.

        The banner is non-modal: a QMessageBox.exec() would block the Qt
        event loop and freeze pitch detection on the held note while the
        dialog was up. The user can leave the InfoBanner on screen, or
        click Switch / Dismiss, without interrupting playback.
        """
        # v0.5.7.6: single-snapshot read. Reading active_device twice
        # opens a TOCTOU window — between the not-None check and the
        # .name access the audio worker can tear the stream down and
        # null the device out, raising AttributeError on .name.
        dev = self._engine.get_active_device()
        if dev is not None and dev.name == device.name:
            return
        if self._info_banner is None:
            return

        def _do_switch() -> None:
            self._open_picked(DeviceSelection(
                name=device.name, host_api=device.host_api, samplerate=0))

        self._info_banner.show_message(
            self._t('audio_banner_interface_appeared', name=device.name),
            action_label=self._t('audio_toast_switch'),
            action_callback=_do_switch,
        )

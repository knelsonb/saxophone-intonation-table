# =============================================================================
# sax_device.py — DeviceController
#
# Extracted from sax_intonation_gui.py in Phase 5 of the refactor — the
# fifth and final controller extraction.  The controller owns the audio
# device picker open flow, the retry-on-failure path, the 1 Hz hot-plug
# poll, the engine-state / devices-changed / interface-appeared Qt slot
# bodies, and the per-active-device write-back to AppConfig.
#
# Design notes:
#   * The QTimer that drives the hot-plug poll and the Qt signal-slot
#     connections from the engine stay wired in MainWindow.__init__ for
#     grep-ability.  MainWindow keeps one-line wrapper methods so the
#     Qt signal/slot bindings (which PyQt type-checks at connect time)
#     are unchanged — only the slot *bodies* move here.
#   * The AudioPickerDialog Qt widget class stays in sax_intonation_gui
#     because it is a Qt widget.  The controller just opens it.
#   * The two banner widgets the controller talks to (AudioRecoveryBanner
#     and InfoBanner) are passed in at construction so the controller
#     does not have to reach into MainWindow's widget tree by name.
#   * Live values that may change during a session (the current UI
#     language) reach the controller through a getter callable.  An
#     optional `on_active_device_changed` callback is fired after the
#     write-back so MainWindow can do any downstream refresh it needs
#     (window-title update, etc.) without the controller having to
#     know about it.
#   * The war council's TOCTOU finding lives here: the engine-state /
#     devices-changed / interface-appeared slot bodies read
#     engine.get_active_device() from inside Qt slots that themselves
#     fire on engine state changes.  The structural fix for that
#     (immutable DeviceSnapshot in the signal payload) is a separate
#     engine-surface change deliberately scoped OUT of this refactor.
#     The current pattern is preserved here byte-for-byte.
#   * The controller's `_audio_ok` flag is captured at construction
#     time from the GUI module's AUDIO_OK constant so the controller
#     does not import sax_intonation_gui (no circular import).
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtWidgets import QDialog

import sax_config
from sax_audio_engine import (
    AudioEngineState,
    DeviceInfo,
    DeviceSelection,
)

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
        # AUDIO_OK is captured from the GUI module at construction time
        # so the controller does not import sax_intonation_gui (would
        # cycle).  If sax_intonation_gui imported cleanly to build the
        # window, sounddevice was importable — but the engine's open()
        # can still fail at runtime and surface FAILED via state_changed.
        try:
            from sax_intonation_gui import AUDIO_OK as _audio_ok
        except Exception:
            _audio_ok = False
        self._audio_ok = bool(_audio_ok)

    # -------------------------------------------------------------------------
    # Persistence write-back
    # -------------------------------------------------------------------------
    def persist_active_device(self) -> None:
        """Write the engine's currently-active device back to config.
        Index is deliberately NOT stored — only name + host API + rate,
        per Gandalf's persistence design."""
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
        # AudioPickerDialog is a Qt widget class — it lives in the GUI
        # module.  Imported here (not at module top) so this controller
        # does not pull the entire GUI module at import time.
        from sax_intonation_gui import AudioPickerDialog
        dlg = AudioPickerDialog(self._w, self._t, self._engine, self._cfg,
                                self._engine.get_active_device())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.chosen()
            if chosen is not None:
                pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                    'auto') or 'auto')
                self._engine.open_device(
                    DeviceSelection(name=chosen.name,
                                    host_api=chosen.host_api,
                                    samplerate=0),
                    samplerate_pref=pref)

    def retry_audio(self) -> None:
        if not self._audio_ok:
            return
        # v0.5.7: force a fresh PortAudio enumeration before re-opening
        # so a device plugged in after launch can be picked up. The old
        # ``engine.retry()`` reused the stale snapshot and never saw
        # the hot-plugged device.
        self._engine.retry_open()

    def poll_devices(self) -> None:
        if not self._audio_ok:
            return
        # refresh_devices() emits signals on diff — we just kick it.
        try:
            self._engine.refresh_devices()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Engine Qt slot bodies
    # -------------------------------------------------------------------------
    def on_engine_state(self, state, err, msg) -> None:
        """React to engine state transitions: update chip, banner, and
        diagnostics panel device label."""
        dev = self._engine.get_active_device()
        name = dev.name if dev else ''
        host = dev.host_api if dev else ''
        sr = int(getattr(self._engine, 'samplerate', 0) or 0)
        w = self._w
        if hasattr(w, '_audio_chip'):
            w._audio_chip.update_from_state(state, name, host, sr)
        if self._audio_banner is not None:
            if state == AudioEngineState.FAILED:
                self._audio_banner.show_for(err, name, msg)
            else:
                self._audio_banner.hide()
        if state == AudioEngineState.RUNNING:
            self.persist_active_device()
            if hasattr(w, '_tuner') and sr:
                w._tuner.sample_rate = sr
            # First-time-only notice if we wound up at a non-44.1k rate.
            if (sr and sr != 44100
                    and not bool(getattr(self._cfg,
                                          'audio_sr_notice_shown', False))):
                self._cfg.audio_sr_notice_shown = True
                sax_config.save_config(self._cfg)
                # Surface as a passive status-bar line, not a modal —
                # Frodo-UX memo: never block the user with a dialog
                # over sample-rate disclosure.
                if hasattr(w, '_status_lbl'):
                    w._status_lbl.setText(
                        self._t('audio_sr_notice', sr=sr))

    def on_devices_changed(self, _devices) -> None:
        # Nothing to do at the MainWindow level — the picker reads
        # devices lazily on each open, and the chip is driven by state
        # changes, not the device list. Hook kept so unit tests / future
        # toasts can subscribe without rewiring the engine.
        pass

    def on_interface_appeared(self, device: 'DeviceInfo') -> None:
        """Hot-plug banner for vendor-class interfaces. Polite, non-modal,
        Frodo-UX memo policy: only fires for matched vendor names so the
        user isn't trained to dismiss it on every wake-from-sleep.

        v0.6 Phase-4 (Item 1): the previous QMessageBox.exec() was modal
        and froze pitch detection on the note Frodo was blowing while the
        dialog was up. Now we surface an InfoBanner that he can leave on
        screen — or click Switch / Dismiss without interrupting playback.
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
            self._engine.open_device(DeviceSelection(
                name=device.name, host_api=device.host_api, samplerate=0))

        self._info_banner.show_message(
            self._t('audio_banner_interface_appeared', name=device.name),
            action_label=self._t('audio_toast_switch'),
            action_callback=_do_switch,
        )

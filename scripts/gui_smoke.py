#!/usr/bin/env python3
"""Headless GUI smoke for the saxophone intonation analyzer.

Builds the real ``MainWindow`` under the offscreen Qt platform and asserts the
Sprint-1 navigation shell is wired correctly. This is a *construction* smoke —
it proves the window assembles, the tabs exist, and the status-dot defaults are
right; it does not exercise live audio (there is no device in CI).

Two modal seams would otherwise hang a headless build, so this script defuses
both via the real-path approach rather than patching window internals:

  * sax_intonation_gui.py:2322 — first-launch welcome dialog, fired when
    ``cfg.welcome_shown`` is False. Defused by pointing HOME at a throwaway
    dir containing a config.json with ``welcome_shown: true`` BEFORE import,
    so the welcome branch is never entered and true construction runs.
  * sax_intonation_gui.py:2316 — the "audio disabled" QMessageBox.information,
    fired when ``AUDIO_OK`` is False (always true headless: no PortAudio
    device). AUDIO_OK is import-time hardware state we can't force, so this
    one modal is stubbed to a no-op.

Run:  QT_QPA_PLATFORM=offscreen python3 scripts/gui_smoke.py
Exit: 0 = all assertions passed; 1 = a smoke assertion failed.

Owned jointly by Frodo (assertions / GUI lane) and Treebeard (suite
integration). Treebeard folds it into the pytest suite so CI runs it on the
PyQt6-equipped runner.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile


def _isolate_config() -> str:
    """Point HOME at a fresh dir holding a welcome_shown=True config, so
    MainWindow never enters the first-launch dialog branch. Must run BEFORE
    importing sax_config / sax_intonation_gui — CONFIG_DIR is resolved from
    Path.home() at module import."""
    home = tempfile.mkdtemp(prefix="sax-gui-smoke-")
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home  # Windows runners
    cfg_dir = os.path.join(home, ".intonation_analyzer")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"welcome_shown": True}, f)
    return home


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The app modules live in the repo root; this script sits in scripts/.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    _isolate_config()

    import sax_intonation_gui as G
    from PyQt6.QtWidgets import QApplication

    # The only modal we can't defuse via config: the headless no-audio box.
    G.QMessageBox.information = staticmethod(lambda *a, **k: None)

    app = QApplication.instance() or QApplication([])
    win = G.MainWindow()
    win.show()
    app.processEvents()

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # Navigation shell (D4).
    check(win._tabs.count() == 4,
          f"expected 4 tabs, got {win._tabs.count()}")
    check(win._tab_keys == ["tuner", "metro", "deck", "setup"],
          f"unexpected tab keys: {win._tab_keys}")
    check(win._tabs.widget(0) is win._splitter,
          "TUNER tab is not the existing splitter object (centerpiece churn)")

    # Status dots must be HIDDEN at launch — nothing is running/recording.
    check(win._metro_dot.isHidden(),
          "METRO status dot visible at launch (should be hidden)")
    check(win._deck_dot.isHidden(),
          "DECK status dot visible at launch (should be hidden)")

    # Hooks flip the dots and the deck pulse timer.
    win.set_metro_running(True)
    app.processEvents()
    check(win._metro_dot.isVisible(),
          "set_metro_running(True) did not show the METRO dot")
    win.set_deck_recording(True)
    app.processEvents()
    check(win._deck_pulse_timer.isActive(),
          "set_deck_recording(True) did not start the deck pulse timer")
    win.set_metro_running(False)
    win.set_deck_recording(False)
    app.processEvents()
    check(win._metro_dot.isHidden() and win._deck_dot.isHidden(),
          "status dots not hidden after stopping")
    check(not win._deck_pulse_timer.isActive(),
          "deck pulse timer still active after set_deck_recording(False)")

    # SETUP controls present.
    check(win._out_device_combo.count() >= 1,
          "output device combo has no entries (expected at least System default)")
    check(hasattr(win, "_cb_prefer_duplex"), "duplex preference checkbox missing")
    check(hasattr(win, "_btn_test_tone"), "test-tone button missing")

    # Switch to the SETUP tab first — the real path a user takes, and it makes
    # the page's widgets actually visible (a QTabWidget only shows the current
    # page, so isVisible() on a background page would read False).
    win._tabs.setCurrentIndex(win._tab_keys.index("setup"))
    app.processEvents()
    check(win._cfg.last_active_tab == "setup",
          f"tab switch did not persist last_active_tab: {win._cfg.last_active_tab}")

    # SETUP parity: the instrument range-editor link is reachable from SETUP
    # (also on the toolbar gear) and enabled/wired to the range editor.
    check(hasattr(win, "_btn_setup_range") and win._btn_setup_range.isEnabled(),
          "SETUP instrument range-editor button missing or disabled")

    # SETUP parity: the Response (Fast/Normal/Slow) combo is reachable from SETUP
    # and stays in TWO-WAY sync with the toolbar Response combo through the
    # shared _apply_filter_mode — signals blocked on the mirrored side so neither
    # change double-routes into the engine. Drive the real handler path (not a
    # stub) in both directions and confirm cfg persists.
    check(hasattr(win, "_setup_filter_combo")
          and win._setup_filter_combo.isEnabled(),
          "SETUP Response (filter-mode) combo missing or disabled")
    if hasattr(win, "_setup_filter_combo"):
        _saved = win._filter_combo.currentData()
        _t1 = "slow" if _saved != "slow" else "fast"
        _t2 = "fast" if _t1 != "fast" else "normal"
        # SETUP combo drives → toolbar combo + cfg follow.
        win._setup_filter_combo.setCurrentIndex(
            win._setup_filter_combo.findData(_t1))
        app.processEvents()
        check(win._filter_combo.currentData() == _t1,
              "SETUP Response change did not mirror into the toolbar combo")
        check(win._cfg.filter_mode == _t1,
              "SETUP Response change did not persist cfg.filter_mode")
        # Toolbar combo drives → SETUP combo follows (other direction).
        win._filter_combo.setCurrentIndex(win._filter_combo.findData(_t2))
        app.processEvents()
        check(win._setup_filter_combo.currentData() == _t2,
              "toolbar Response change did not mirror into the SETUP combo")
        # Restore so later assertions see the original state.
        _ri = win._filter_combo.findData(_saved)
        if _ri >= 0:
            win._filter_combo.setCurrentIndex(_ri)
        app.processEvents()

    # SETUP parity: horn nickname editor mirrors the toolbar nickname field
    # (both edit cfg.last_nickname; the toolbar field stays canonical). The
    # editingFinished signal can't be driven headless, so call the REAL
    # handlers after setText and confirm the mirror both directions.
    check(hasattr(win, "_setup_nick_edit"), "SETUP nickname editor missing")
    if hasattr(win, "_setup_nick_edit"):
        win._setup_nick_edit.setText("Tenor SmokeTest")
        win._on_setup_nickname_changed()
        app.processEvents()
        check(win._nick_edit.text() == "Tenor SmokeTest",
              "SETUP nickname did not mirror into the toolbar field")
        win._nick_edit.setText("Alto SmokeTest")
        win._on_nickname_changed()
        app.processEvents()
        check(win._setup_nick_edit.text() == "Alto SmokeTest",
              "toolbar nickname did not mirror into the SETUP field")
        # Blank both so save-on-close can't persist a smoke value.
        win._setup_nick_edit.setText("")
        win._on_setup_nickname_changed()
        app.processEvents()

    # SETUP parity: concert-pitch (A4) combo mirrors the toolbar Kammerton combo
    # through the shared _apply_a4 — the heavy measurement recalc runs once,
    # never double-fired by the sibling (blockSignals). Drive both directions
    # and confirm the engine A4 follows.
    check(hasattr(win, "_setup_a4_combo")
          and win._setup_a4_combo.count() == 21,
          "SETUP A4 combo missing or wrong item count (expected 430..450)")
    if hasattr(win, "_setup_a4_combo"):
        _a4_saved = win._a4_combo.currentIndex()
        # SETUP drives → toolbar + engine follow.
        win._setup_a4_combo.setCurrentIndex(win._setup_a4_combo.findData(442))
        app.processEvents()
        check(win._a4_combo.currentData() == 442,
              "SETUP A4 change did not mirror into the toolbar combo")
        check(int(win._engine.a4) == 442,
              "SETUP A4 change did not reach the engine")
        # Toolbar drives → SETUP follows (other direction).
        win._a4_combo.setCurrentIndex(win._a4_combo.findData(437))
        app.processEvents()
        check(win._setup_a4_combo.currentData() == 437,
              "toolbar A4 change did not mirror into the SETUP combo")
        check(int(win._engine.a4) == 437,
              "toolbar A4 change did not reach the engine")
        # Restore original A4.
        win._a4_combo.setCurrentIndex(_a4_saved)
        app.processEvents()

    # Test tone with no engine mirror (headless): must NOT claim to be playing.
    # The button reverts to unchecked and surfaces a failure status (N2).
    win._btn_test_tone.setChecked(True)
    app.processEvents()
    check(not win._btn_test_tone.isChecked(),
          "test-tone button stayed checked with no output path (button lies)")
    check(win._test_tone_status.isVisible(),
          "test-tone failure status not shown when start failed")

    # METRO panel (Sprint 2) — built inert (no controller in headless). Verify
    # the pure-UI controls respond and the audio actions don't lie.
    win._tabs.setCurrentIndex(win._tab_keys.index("metro"))
    app.processEvents()
    start_bpm = win._metro_bpm
    win._metro_nudge(5)
    check(win._metro_bpm == start_bpm + 5 and win._metro_slider.value() == win._metro_bpm,
          f"BPM nudge/slider out of sync: {win._metro_bpm} / {win._metro_slider.value()}")
    win._metro_nudge(-100000)
    check(win._metro_bpm == win.METRO_BPM_MIN,
          f"BPM did not clamp to min: {win._metro_bpm}")
    win._metro_nudge(100000)
    check(win._metro_bpm == win.METRO_BPM_MAX,
          f"BPM did not clamp to max: {win._metro_bpm}")
    win._on_metro_timesig("6/8")
    checked = [t for t, b in win._metro_ts_btns.items() if b.isChecked()]
    check(checked == ["6/8"],
          f"time-signature selection not exclusive: {checked}")
    # Start with no audible output (headless has no output stream): must NOT
    # claim running — button reverts, inline status shown, green dot stays
    # hidden (the honesty discipline holds whether the controller is absent or
    # simply can't open an output device).
    win._btn_metro_start.setChecked(True)
    app.processEvents()
    check(not win._btn_metro_start.isChecked(),
          "metro start button stayed checked with no audible output (button lies)")
    check(win._metro_status.isVisible(),
          "metro 'unavailable' status not shown when start could not sound")
    check(win._metro_dot.isHidden(),
          "metro green dot shown despite no running metronome")

    # Option-A device-switch regression lock: a RUNNING metronome must be
    # bracketed (stop before / start after) across an output-device reopen, so
    # the mixer.reset_clock() inside the reopen can't silently kill its beat
    # chain. Headless has no real output stream, so drive the GUI orchestration
    # with a tiny fake controller + a stubbed reopen that keeps output "up".
    class _FakeMetro:
        def __init__(self):
            self.calls = []
            self._run = True

        def is_running(self):
            return self._run

        def stop(self):
            self.calls.append("stop")
            self._run = False

        def start(self):
            self.calls.append("start")
            self._run = True

        def set_samplerate(self, sr):
            self.calls.append(("sr", sr))

    fake = _FakeMetro()
    win._metro_ctrl = fake
    win._engine.output_running = True
    win._engine.output_samplerate = 48000
    win._engine.open_output_device = lambda *a, **k: None  # stream stays up
    win._on_output_device_changed(0)
    check("stop" in fake.calls and "start" in fake.calls
          and fake.calls.index("stop") < fake.calls.index("start"),
          f"metronome not bracketed stop-before-start across device switch: {fake.calls}")
    check(fake.is_running(),
          "metronome did not resume after the output device switch")

    # Drone bar + pitch pipes (Sprint 3). The controllers may or may not be
    # present (sax_drone / sax_pitch_pipes land in CI's venv). Either way,
    # headless has NO real output device — so force output "down" and assert the
    # honesty path: an audio action that can't sound must revert + show status,
    # never claim to be running. (The prior device-switch block left output
    # "up"; reset it here.)
    win._engine.output_running = False
    win._engine.open_output_device = lambda *a, **k: None  # never comes up
    win._tabs.setCurrentIndex(win._tab_keys.index("tuner"))
    app.processEvents()
    check(len(win._drone_preset_btns) == 5,
          f"expected 5 drone preset buttons, got {len(win._drone_preset_btns)}")
    check(win._drone_voice_combo.count() >= 5,
          "SETUP drone voice combo has no entries")
    win._btn_drone_on.setChecked(True)
    app.processEvents()
    check(not win._btn_drone_on.isChecked(),
          "drone on-button stayed checked with no audible output (button lies)")
    check(win._drone_status.isVisible(),
          "drone 'unavailable' status not shown when it could not sound")
    win._on_drone_preset("cello")
    check(win._drone_voice_id == "cello" and win._cfg.drone_voice_id == "cello",
          f"drone preset did not update voice id: {win._drone_voice_id}")
    win._drone_semitone_nudge(100)
    check(win._drone_semitones == 12,
          f"drone semitone did not clamp high: {win._drone_semitones}")
    win._drone_semitone_nudge(-100)
    check(win._drone_semitones == -12,
          f"drone semitone did not clamp low: {win._drone_semitones}")
    win._open_pitch_pipes()
    app.processEvents()
    check(len(win._pipe_btns) == 12,
          f"expected 12 pitch-pipe pads, got {len(win._pipe_btns)}")
    win._on_pipe_tapped(60)
    check(win._pipes_status.isVisible(),
          "pitch-pipe 'unavailable' status not shown when it could not sound")
    if win._pipes_dlg is not None:
        win._pipes_dlg.close()
    app.processEvents()

    # D3 duck-attach (GUI-wiring lock): enabling the drone with output "up" must
    # attach its source to the engine's duck consumer, or the drone sounds but
    # never ducks on mic bleed in the live app. The unit/e2e tests attach the
    # consumer manually (engine.coordination_step), so ONLY a GUI-path check
    # catches a controller built without engine=. Skip gracefully if the drone
    # controller didn't construct (no sax_drone in this env).
    if win._drone_ctrl is not None and hasattr(win._engine, "attach_duck_consumer"):
        attached = []
        _real_attach = win._engine.attach_duck_consumer
        win._engine.attach_duck_consumer = (
            lambda c: (attached.append(c), _real_attach(c))[1])
        win._engine.output_running = True
        win._engine.output_samplerate = 44100
        win._engine.open_output_device = lambda *a, **k: None
        win._btn_drone_on.setChecked(True)
        app.processEvents()
        check(bool(attached),
              "enabling the drone did NOT attach a duck consumer "
              "(DroneController missing engine= → no duck in the live app)")
        win._btn_drone_on.setChecked(False)
        app.processEvents()

    # DECK tab (Sprint 4) — built inert here (sax_deck lands in CI's venv; and
    # headless has no input device anyway). This is the HONESTY LOCK Treebeard
    # pins: a record attempt that can't arm the mic must NEVER show the pulsing
    # red dot or claim 'recording' — the dot is driven off controller.state, and
    # with no armable input the state stays idle, so set_deck_recording(True)
    # never fires. Mirror of the drone/metro revert lock, input-side.
    win._tabs.setCurrentIndex(win._tab_keys.index("deck"))
    app.processEvents()
    for _name in ("_btn_deck_record", "_btn_deck_stop", "_btn_deck_play",
                  "_btn_deck_export"):
        check(hasattr(win, _name), f"deck transport button {_name} missing")
    # Nothing recorded/playing and no input: Stop/Play/Export disabled, dot hidden.
    check(not win._btn_deck_stop.isEnabled(),
          "deck Stop enabled with nothing recording/playing")
    check(not win._btn_deck_play.isEnabled(),
          "deck Play enabled with no take")
    check(not win._btn_deck_export.isEnabled(),
          "deck Export enabled with no take")
    check(win._deck_dot.isHidden(),
          "deck recording dot visible at rest (should be hidden)")
    # The honesty seam: attempt to record with no armable input → button reverts
    # unchecked, inline status shown, dot stays HIDDEN + pulse timer INACTIVE.
    win._btn_deck_record.setChecked(True)
    app.processEvents()
    check(not win._btn_deck_record.isChecked(),
          "deck Record stayed checked with no input (button lies)")
    check(win._deck_status.isVisible(),
          "deck 'unavailable' status not shown when record could not arm")
    check(win._deck_dot.isHidden(),
          "deck recording dot shown despite no armed take (false recording)")
    check(not win._deck_pulse_timer.isActive(),
          "deck pulse timer running despite no armed take (false recording)")

    # DECK live-wiring lock (Sprint 4): with sax_deck present, prove the GUI
    # transport actually DRIVES the controller and the pulsing-red dot follows
    # controller state — the deck analog of the drone duck-attach lock (a wiring
    # bug a pure unit test can't see, because the unit tests call the controller
    # directly and bypass the button → handler → controller path). We build a
    # controller on a FAKE input engine + the REAL mixer, swap it in, and drive
    # the real buttons. Faking the engine's input methods keeps this independent
    # of whether the engine input-tap has landed yet. Skips if sax_deck absent.
    if win._deck_ctrl is not None:
        import numpy as _np
        import sax_deck as _sd

        class _FakeInputEngine:
            """Minimal engine the DeckController needs: an armable mic input and
            an output samplerate. Mirrors Gandalf's engine input-tap surface."""
            def __init__(self, mixer):
                self.mixer = mixer
                self.input_running = True
                self.output_running = True
                self.output_samplerate = 44100
                self.samplerate = 44100
                self._armed = False

            def start_input_recording(self, _max_s):
                self._armed = True
                return True

            def is_input_recording(self):
                return self._armed

            def stop_input_recording(self):
                self._armed = False
                return (_np.zeros(2048, dtype=_np.float32), 44100, False)

        win._deck_ctrl = _sd.DeckController(
            win._engine.mixer, 44100, engine=_FakeInputEngine(win._engine.mixer),
            on_state_changed=win._on_deck_state_changed)
        win._engine.output_running = True  # so _ensure_output_open() is True
        win._tabs.setCurrentIndex(win._tab_keys.index("deck"))
        app.processEvents()
        win._sync_deck_transport()
        check(win._btn_deck_record.isEnabled(),
              "deck Record not enabled when the input is armable")

        # Record → recording: the dot must light + pulse, button checked.
        win._btn_deck_record.setChecked(True)
        app.processEvents()
        check(str(win._deck_ctrl.state) == "recording",
              f"deck did not enter recording on Record: {win._deck_ctrl.state}")
        check(win._deck_dot.isVisible() and win._deck_pulse_timer.isActive(),
              "recording dot/pulse not active while recording (GUI-wiring gap)")
        check(win._btn_deck_record.isChecked(),
              "deck Record button not checked while recording")

        # Stop → have-take: dot off, Play + Export enabled.
        win._on_deck_stop()
        app.processEvents()
        check(str(win._deck_ctrl.state) == "have-take",
              f"deck did not reach have-take on Stop: {win._deck_ctrl.state}")
        check(win._deck_dot.isHidden() and not win._deck_pulse_timer.isActive(),
              "deck recording dot still showing after recording stopped")
        check(win._btn_deck_play.isEnabled() and win._btn_deck_export.isEnabled(),
              "deck Play/Export not enabled with a take present")

        # Play → playing: registers the source; the recording dot stays HIDDEN
        # (playback is not recording — no false recording indicator).
        win._on_deck_play()
        app.processEvents()
        check(str(win._deck_ctrl.state) == "playing",
              f"deck did not enter playing on Play: {win._deck_ctrl.state}")
        check(win._deck_dot.isHidden(),
              "deck recording dot lit during playback (false recording)")

        # Clean up: drop the playback source from the real mixer, then shut down.
        win._on_deck_stop()
        app.processEvents()
        win._deck_ctrl.shutdown()

    # ── Theme switching (Sprint 5 live-apply lock) ───────────────────────
    # The SETUP picker exists and _apply_theme switches the palette + re-renders
    # the global stylesheet live without raising. Drives the real handler path
    # (mirrors the drone duck-attach lock), not a stub.
    import sax_theme as _TH
    check(hasattr(win, "_theme_combo"),
          "SETUP theme picker (_theme_combo) missing")
    check(win._theme_combo.count() == len(_TH.THEME_ORDER),
          f"theme combo has {win._theme_combo.count()} items, "
          f"expected {len(_TH.THEME_ORDER)}")
    _start_theme = _TH.active_name()
    for _name in ("night", "light", "dark"):
        win._apply_theme(_name)
        app.processEvents()
        check(_TH.active_name() == _name,
              f"_apply_theme({_name!r}) did not set the active theme")
        check(win._cfg.theme == _name,
              f"_apply_theme({_name!r}) did not persist cfg.theme")
        check(_TH.get_theme(_name).window_bg in win.styleSheet(),
              f"theme {_name!r} window_bg not in the applied stylesheet")
    win._apply_theme(_start_theme)

    win.close()

    if failures:
        print("GUI SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GUI SMOKE OK — nav shell, status dots, SETUP + METRO + drone/pipes "
          "+ deck panels, device-switch bracket, deck honesty lock, theme "
          "switching all sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

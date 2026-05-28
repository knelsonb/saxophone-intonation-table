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

    win.close()

    if failures:
        print("GUI SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GUI SMOKE OK — nav shell, status dots, SETUP + METRO panels, "
          "device-switch bracket all sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

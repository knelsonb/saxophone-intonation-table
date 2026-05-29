#!/usr/bin/env python3
"""Offscreen theme screenshot harness — the de-inline VERIFIER.

Renders the real MainWindow (every nav tab) in each theme to PNGs so the
THEME-INLINE de-inline can be checked VISUALLY without a display:

  * DARK byte-identical — diff dark-*.png before vs after a de-inline batch
    (a clean batch leaves them pixel-identical: the shipped default can't
    silently regress).
  * light / night re-theme — eyeball light-*.png / night-*.png to confirm a
    de-inlined widget actually picks up the palette instead of staying
    dark-slate.

QWidget.grab() paints a widget into a QPixmap via its own paint path, so it
works under QT_QPA_PLATFORM=offscreen (unlike QScreen.grabWindow, which needs a
real screen). Optional extra targets (e.g. the range-editor dialog with its
ok/warn validation styling) are grabbed too.

Run:  QT_QPA_PLATFORM=offscreen python3 scripts/theme_shots.py [outdir]
Out:  {outdir}/{theme}-{tab}.png  +  {outdir}/{theme}-rangedlg.png
"""
from __future__ import annotations

import json
import os
import sys
import tempfile


def _isolate_config() -> str:
    """Throwaway HOME with welcome_shown=True so MainWindow builds without the
    first-launch dialog (same trick as scripts/gui_smoke.py)."""
    home = tempfile.mkdtemp(prefix="sax-shots-")
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    cfg_dir = os.path.join(home, ".intonation_analyzer")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"welcome_shown": True}, f)
    return home


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    _isolate_config()
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sax_shots"
    os.makedirs(outdir, exist_ok=True)

    import sax_intonation_gui as G
    import sax_theme as TH
    from PyQt6.QtWidgets import QApplication

    # The one modal we can't defuse via config (headless no-audio box).
    G.QMessageBox.information = staticmethod(lambda *a, **k: None)

    app = QApplication.instance() or QApplication([])
    win = G.MainWindow()
    win.resize(1280, 840)
    win.show()
    for _ in range(4):
        app.processEvents()

    saved: list[str] = []

    def _grab(widget, name: str) -> None:
        for _ in range(2):
            app.processEvents()
        path = os.path.join(outdir, name + ".png")
        widget.grab().save(path)
        saved.append(path)

    for theme in TH.THEME_ORDER:                 # dark, night, light
        win._apply_theme(theme)
        for _ in range(3):
            app.processEvents()
        for i, key in enumerate(win._tab_keys):  # tuner, metro, deck, setup
            win._tabs.setCurrentIndex(i)
            _grab(win, f"{theme}-{key}")
        # Range-editor dialog: built directly (not exec'd, which would block)
        # so we capture its inputs + ok/warn validation styling per theme.
        try:
            dlg = G.RangeEditorDialog(parent=win, instrument=win.instrument) \
                if hasattr(G, "RangeEditorDialog") else None
            if dlg is not None:
                dlg.resize(520, 360)
                _grab(dlg, f"{theme}-rangedlg")
                dlg.close()
        except Exception as exc:                 # never let one target sink the run
            print(f"  (rangedlg skipped for {theme}: {exc})")

    win.close()
    print(f"wrote {len(saved)} screenshots to {outdir}")
    for p in saved:
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

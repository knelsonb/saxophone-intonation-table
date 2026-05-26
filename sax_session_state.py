# =============================================================================
# sax_session_state.py — SessionStateController
#
# Extracted from sax_intonation_gui.py in Phase 5 of the refactor. The
# controller owns the round-trip of MainWindow's persistable session
# state — window geometry, splitter widths, the currently-selected
# instrument / nickname / display mode / A4 / language — through the
# AppConfig dataclass.
#
# Design notes:
#   * The controller does NOT call sax_config.save_config(). Persistence
#     timing belongs to the caller (closeEvent in MainWindow's case).
#     Keeping save_config out of here makes the controller pure and
#     testable: a test can hand it a fake window + a bare AppConfig and
#     assert on the post-save() field values without touching disk.
#   * The window and every widget the controller pokes are passed in
#     explicitly through the constructor. Verbose but grep-friendly:
#     the dependency surface is the constructor signature, full stop.
#   * No behavioural change vs. the inlined implementation that used
#     to live on MainWindow — the order of operations, the defensive
#     try/except blocks and the off-screen-rescue logic are preserved
#     verbatim.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QByteArray
from PyQt6.QtGui import QGuiApplication

if TYPE_CHECKING:
    from PyQt6.QtWidgets import (
        QMainWindow, QSplitter, QComboBox, QLineEdit,
    )
    from sax_config import AppConfig


class SessionStateController:
    """Save and restore the MainWindow's persistable session state
    (window geometry, splitter sizes, current instrument/nickname,
    display mode, A4, language) through sax_config's AppConfig."""

    def __init__(self, window: 'QMainWindow', cfg: 'AppConfig',
                 *,
                 engine,
                 splitter: 'QSplitter | None' = None,
                 instr_combo: 'QComboBox | None' = None,
                 nick_edit: 'QLineEdit | None' = None,
                 disp_combo: 'QComboBox | None' = None,
                 a4_combo: 'QComboBox | None' = None,
                 lang_combo: 'QComboBox | None' = None) -> None:
        self._w = window
        self._cfg = cfg
        self._engine = engine
        self._splitter = splitter
        self._instr_combo = instr_combo
        self._nick_edit = nick_edit
        self._disp_combo = disp_combo
        self._a4_combo = a4_combo
        self._lang_combo = lang_combo

    # -------------------------------------------------------------------------
    def restore(self) -> None:
        """Apply persisted window geometry, splitter sizes, nickname,
        language/instrument/display/A4 to the freshly-built widgets.

        Called from MainWindow.__init__ right after _build_ui. Lang /
        instrument / display were already applied earlier (they affect
        _build_ui's first paint); here we sync the combos to those
        values plus the widget-level state. First launch (empty cfg)
        is a no-op."""
        w = self._w
        cfg = self._cfg

        # Window geometry — only restore if the saved rect still fits on
        # at least one connected screen. A user who unplugged a monitor
        # should not get a window stranded off-screen.
        geom_b64 = getattr(cfg, 'window_geometry', "")
        if geom_b64:
            try:
                ba = QByteArray.fromBase64(geom_b64.encode('ascii'))
                if not ba.isEmpty() and w.restoreGeometry(ba):
                    screens = QGuiApplication.screens()
                    if screens:
                        visible = False
                        win_geom = w.frameGeometry()
                        for scr in screens:
                            if scr.availableGeometry().intersects(win_geom):
                                visible = True
                                break
                        if not visible:
                            # Off-screen — let Qt's default placement
                            # take over by clearing the geometry.
                            w.resize(1100, 700)
                            primary = QGuiApplication.primaryScreen()
                            if primary is not None:
                                center = primary.availableGeometry().center()
                                w.move(center.x() - w.width() // 2,
                                       center.y() - w.height() // 2)
            except Exception:
                pass
        state_b64 = getattr(cfg, 'window_state', "")
        if state_b64:
            try:
                ba = QByteArray.fromBase64(state_b64.encode('ascii'))
                if not ba.isEmpty():
                    w.restoreState(ba)
            except Exception:
                pass

        # Splitter widths.
        sizes = list(getattr(cfg, 'splitter_sizes', []) or [])
        if (self._splitter is not None and len(sizes) == 2
                and all(s >= 0 for s in sizes) and sum(sizes) > 0):
            self._splitter.setSizes(sizes)

        # Nickname text.
        nick = getattr(cfg, 'last_nickname', "")
        if nick and self._nick_edit is not None:
            self._nick_edit.setText(nick)

        # Language combo — match w.lang set earlier in __init__.
        if self._lang_combo is not None:
            for i in range(self._lang_combo.count()):
                if self._lang_combo.itemData(i) == w.lang:
                    self._lang_combo.blockSignals(True)
                    self._lang_combo.setCurrentIndex(i)
                    self._lang_combo.blockSignals(False)
                    break
            # _build_ui used the lang in effect at construction; if we
            # overrode it post-load, retranslate now to repaint labels.
            w._retranslate()

        # Display combo — match w.display.
        if self._disp_combo is not None:
            target_idx = 0 if w.display == 'griff' else 1
            if self._disp_combo.currentIndex() != target_idx:
                self._disp_combo.blockSignals(True)
                self._disp_combo.setCurrentIndex(target_idx)
                self._disp_combo.blockSignals(False)

        # A4 — apply to combo and engine.
        a4 = int(getattr(cfg, 'last_a4_hz', 440))
        if 430 <= a4 <= 450:
            if self._a4_combo is not None:
                self._a4_combo.blockSignals(True)
                self._a4_combo.setCurrentIndex(a4 - 430)
                self._a4_combo.blockSignals(False)
            try:
                self._engine.a4 = float(a4)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    def save(self) -> None:
        """Capture window state into self._cfg. Does NOT call
        sax_config.save_config — the caller (MainWindow.closeEvent)
        decides when to persist."""
        w = self._w
        cfg = self._cfg
        # Window geometry + state — QByteArray base64 round-trip.
        geom = w.saveGeometry()
        state = w.saveState()
        try:
            cfg.window_geometry = bytes(geom.toBase64()).decode('ascii')
        except Exception:
            cfg.window_geometry = ""
        try:
            cfg.window_state = bytes(state.toBase64()).decode('ascii')
        except Exception:
            cfg.window_state = ""
        # Splitter sizes.
        if self._splitter is not None:
            cfg.splitter_sizes = list(self._splitter.sizes())
        # Per-widget UI state.
        cfg.last_instrument_key = w.instrument
        if self._nick_edit is not None:
            cfg.last_nickname = self._nick_edit.text().strip()
        cfg.last_display_mode = w.display
        try:
            cfg.last_a4_hz = int(self._engine.a4)
        except Exception:
            pass
        cfg.last_lang = w.lang

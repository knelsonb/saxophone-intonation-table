# =============================================================================
# sax_session_state.py — SessionStateController
#
# Extracted from sax_intonation_gui.py as a MainWindow controller. The
# controller owns the round-trip of MainWindow's persistable session
# state — window geometry, splitter widths, the session's instrument /
# nickname / display mode / A4 and the UI language — through the
# AppConfig dataclass.
#
# Design notes:
#   * The controller does NOT save the settings. Persistence timing
#     belongs to the caller (closeEvent in MainWindow's case).
#     Keeping the store out of here makes the controller pure and
#     testable: a test can hand it a fake window + a bare AppConfig and
#     assert on the post-save() field values without touching disk.
#   * The window, the splitter, the SessionModel and the Translator are
#     passed in explicitly through the constructor: the dependency surface
#     is the constructor signature, full stop. The combos and editors that
#     show those values follow the session, so restoring the session is
#     enough to put them right.
#   * No behavioural change vs. the inlined implementation that used
#     to live on MainWindow — the order of operations, the defensive
#     try/except blocks and the off-screen-rescue logic are preserved
#     verbatim.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import QByteArray
from PyQt6.QtGui import QGuiApplication

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow, QSplitter

    from sax_config import AppConfig
    from sax_session import SessionModel
    from sax_translator import Translator


class SessionStateController:
    """Save and restore the MainWindow's persistable session state
    (window geometry, splitter sizes, current instrument/nickname,
    display mode, A4, language) through sax_config's AppConfig."""

    def __init__(self, window: 'QMainWindow', cfg: 'AppConfig',
                 *,
                 session: 'SessionModel',
                 translator: 'Translator',
                 splitter: 'QSplitter | None' = None,
                 apply_a4: 'Callable[[float], None]',
                 keep_saved_instrument: bool = False) -> None:
        self._w = window
        self._cfg = cfg
        self._session = session
        self._tr = translator
        self._splitter = splitter
        # The window's A4 apply path: it re-derives the table at the saved
        # A4, and the session hands the value to the combos, engine, drone
        # and pitch pipes.
        self._apply_a4 = apply_a4
        # Set when the saved instrument could not be resolved (its custom
        # instruments file was unreadable) and the window fell back to its
        # default. While the window still shows that fallback, save() keeps
        # the saved choice instead of overwriting it.
        self._fallback_instrument = (session.instrument if keep_saved_instrument
                                     else None)

    # -------------------------------------------------------------------------
    def restore(self) -> None:
        """Apply persisted window geometry, splitter sizes, nickname and A4
        to the freshly-built window.

        Called from MainWindow.__init__ right after _build_ui. Language,
        instrument and display were applied earlier (they affect _build_ui's
        first paint). First launch (empty cfg) is a no-op."""
        w = self._w
        cfg = self._cfg

        # Window geometry — only restore if the saved rect still fits on
        # at least one connected screen. A user who unplugged a monitor
        # should not get a window stranded off-screen.
        geom_b64 = cfg.window_geometry
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
        state_b64 = cfg.window_state
        if state_b64:
            try:
                ba = QByteArray.fromBase64(state_b64.encode('ascii'))
                if not ba.isEmpty():
                    w.restoreState(ba)
            except Exception:
                pass

        # Splitter widths.
        sizes = list(cfg.splitter_sizes)
        if (self._splitter is not None and len(sizes) == 2
                and all(s >= 0 for s in sizes) and sum(sizes) > 0):
            self._splitter.setSizes(sizes)

        # Nickname text (both editors follow the session).
        if cfg.last_nickname:
            self._session.set_nickname(cfg.last_nickname)

        # A4 — one path to the combos, engine, drone and pitch pipes.
        a4 = cfg.last_a4_hz
        if 430 <= a4 <= 450:
            self._apply_a4(float(a4))

    # -------------------------------------------------------------------------
    def save(self) -> None:
        """Capture window state into self._cfg. Does NOT call
        the settings — the caller (MainWindow.closeEvent)
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
        # The session.
        session = self._session
        if session.instrument != self._fallback_instrument:
            cfg.last_instrument_key = session.instrument
        cfg.last_nickname = session.label
        cfg.last_display_mode = session.display
        cfg.last_a4_hz = int(session.a4)
        cfg.last_lang = self._tr.lang

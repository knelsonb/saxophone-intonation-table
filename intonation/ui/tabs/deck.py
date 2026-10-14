# =============================================================================
# intonation/ui/tabs/deck.py — the DECK tab (tape deck)
#
# As in the Android useDeck hook: record the MIC to a single take, play it
# back through the mixer, export it to a WAV. DeckController (intonation/audio/sources/deck.py)
# owns the state machine, the engine's input-recording tap and the WAV I/O;
# this panel is its transport.
#
# Honesty discipline: the pulsing-red dot (``recording_changed``) and the
# 'recording' state appear ONLY when the controller actually armed the mic
# input — never on a bare button press. Playback asks for the output through
# OutputUsers and says why when none can open.
#
# deck.pump() runs on a GUI-side timer. Playback-end and cap-hit transitions
# are detected there (the audio thread only flips a lock-free .finished
# flag); pump() performs the transition and fires on_state_changed ON THE GUI
# THREAD, so buttons are always relabelled on-thread.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from intonation.audio.sources.deck import DeckController
from intonation.ui.output_status import show_with_reason

if TYPE_CHECKING:
    from intonation.audio.engine import AudioEngine
    from intonation.audio.output_users import OutputUsers
    from intonation.i18n.translator import Translator
    from intonation.persistence.config import AppConfig

DECK_STATES = ('idle', 'recording', 'have_take', 'playing')

# How often the transport asks the controller for finished playback, a
# recording that hit its length cap, and whether a take can be armed.
PUMP_INTERVAL_MS = 100


class DeckTab(QWidget):
    """The DECK tab; see the module comment."""

    # True while a take is really recording (drives the tab's red dot).
    recording_changed = pyqtSignal(bool)

    def __init__(self, *, tr: 'Translator', cfg: 'AppConfig',
                 engine: 'AudioEngine', output_users: 'OutputUsers',
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._cfg = cfg
        self._output_users = output_users
        self._played_once = False
        self._last_state: Optional[str] = None
        self._last_can_record: Optional[bool] = None
        # max_seconds comes from the clamped settings field.
        self._ctrl = DeckController(
            engine.mixer, int(engine.mixer.samplerate),
            engine=engine,           # arms the input-recording tap
            max_seconds=float(cfg.deck_max_seconds),
            scratch_dir=cfg.deck_scratch_dir or None,
            on_state_changed=self._on_state_changed)
        self._build()
        # The GUI-thread pump tick (see the module comment).
        self._pump_timer = QTimer(self)
        self._pump_timer.setInterval(PUMP_INTERVAL_MS)
        self._pump_timer.timeout.connect(self._pump)
        self._pump_timer.start()
        self._sync()

    @property
    def controller(self) -> DeckController:
        return self._ctrl

    def _build(self) -> None:
        tr = self._tr
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel()
        tr.bind(title.setText, 'deck_group')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size:18px;font-weight:bold;letter-spacing:3px;')
        outer.addWidget(title)

        # Big state readout: empty / recording / take ready / playing.
        self._state_lbl = QLabel('')
        self._state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_lbl.setObjectName('deckState')   # themed accent readout
        outer.addWidget(self._state_lbl)

        # Transport: Record (toggle, green→red) · Stop · Play/Replay · Export.
        row = QHBoxLayout()
        self._btn_record = QPushButton()
        tr.bind(self._btn_record.setText, 'deck_record')
        self._btn_record.setCheckable(True)
        tr.bind(self._btn_record.setToolTip, 'deck_record_tip')
        self._btn_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_record.toggled.connect(self._on_record_toggle)
        self._btn_record.setObjectName('transportRec')   # green/red + themed disabled
        self._btn_stop = QPushButton()
        tr.bind(self._btn_stop.setText, 'deck_stop')
        self._btn_play = QPushButton()
        tr.bind(self._btn_play.setText, 'deck_play')
        tr.bind(self._btn_play.setToolTip, 'deck_play_tip')
        self._btn_export = QPushButton()
        tr.bind(self._btn_export.setText, 'deck_export')
        tr.bind(self._btn_export.setToolTip, 'deck_export_tip')
        self._btn_stop.clicked.connect(self.stop)
        self._btn_play.clicked.connect(self.play)
        self._btn_export.clicked.connect(self._on_export)
        for b in (self._btn_stop, self._btn_play, self._btn_export):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName('transport')   # themed via QPushButton#transport
        for b in (self._btn_record, self._btn_stop, self._btn_play, self._btn_export):
            row.addWidget(b)
        outer.addLayout(row)

        # Records the mic only, a single take (as in the Android app).
        hint = QLabel()
        tr.bind(hint.setText, 'deck_hint')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setObjectName('dimLabel')
        outer.addWidget(hint)

        # Inline status — shown only when an action can't run / fails or to
        # confirm an export, so the transport never silently lies.
        self._status = QLabel('')
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        outer.addWidget(self._status)
        outer.addStretch()

    # ── State ────────────────────────────────────────────────────────────────
    def state(self) -> str:
        """The controller's state as one of DECK_STATES. intonation.audio.sources.deck spells it
        'have-take' (a str-enum value); hyphen and underscore are both
        accepted, so the panel can't drift from that spelling."""
        s = self._ctrl.state.value.replace('-', '_')
        return s if s in DECK_STATES else 'idle'

    def _can_record(self) -> bool:
        """Honest pre-click probe: can a take be armed right now? Backed by
        the controller's can_record() (engine.input_running);
        start_record()'s False-return is the authoritative check after a
        click either way."""
        try:
            return bool(self._ctrl.can_record())
        except Exception:
            return False

    def _sync(self) -> None:
        """Paint every transport control, the recording dot and the state
        readout from the controller's state. Called at build, on every
        on_state_changed, and on the pump tick (for can_record changes)."""
        state = self.state()
        can_record = self._can_record()
        recording = state == 'recording'
        playing = state == 'playing'
        have_take = state in ('have_take', 'playing')

        # Record toggle: checked iff recording; enabled while recording (so the
        # untoggle can stop it) or when a fresh arm is honestly possible.
        blocked = self._btn_record.blockSignals(True)
        self._btn_record.setChecked(recording)
        self._btn_record.blockSignals(blocked)
        self._btn_record.setEnabled(
            recording or (state in ('idle', 'have_take') and can_record))
        # Stop: only while something is sounding/capturing.
        self._btn_stop.setEnabled(recording or playing)
        # Play/Replay: a take exists (have-take → play; playing → restart@0).
        self._btn_play.setEnabled(have_take)
        self._tr.bind(self._btn_play.setText,
                      'deck_replay' if (self._played_once or playing) else 'deck_play')
        # Export: whenever a take exists.
        self._btn_export.setEnabled(have_take)

        # The pulsing-red dot follows ONLY a real 'recording' state — never a
        # bare button press.
        self.recording_changed.emit(recording)

        key = {'idle': 'deck_status_empty', 'recording': 'deck_status_recording',
               'have_take': 'deck_status_have_take',
               'playing': 'deck_status_playing'}[state]
        self._tr.bind(self._state_lbl.setText, key)

        self._last_state = state
        self._last_can_record = can_record

    def _show_status(self, key: str, *, error: bool = True, output_status=None) -> None:
        """Inline transport status. Red for can't/failed, green for confirmation
        (export done) — never leave a stale message of the wrong colour.
        ``output_status`` adds why the output could not open."""
        self._status.setStyleSheet('color:%s;font-size:12px;padding-top:4px;'
                                   % ('#c0392b' if error else '#2ecc71'))
        show_with_reason(self._tr, self._status, key, output_status)
        self._status.setVisible(True)

    def _on_state_changed(self) -> None:
        """Zero-arg slot for DeckController.on_state_changed — always on the
        GUI thread. Clears a stale status and repaints from the new state."""
        self._status.setVisible(False)
        self._sync()

    def _pump(self) -> None:
        try:
            self._ctrl.pump()   # may fire on_state_changed → _sync
        except Exception:
            pass
        if (self.state() != self._last_state
                or self._can_record() != self._last_can_record):
            self._sync()

    # ── Transport ────────────────────────────────────────────────────────────
    def _on_record_toggle(self, on: bool) -> None:
        if not on:
            # Un-toggling Record while recording = stop the take.
            self.stop()
            return
        self._status.setVisible(False)
        # Recording arms the MIC INPUT (not the output). The controller opens it
        # and returns False if it can't — then revert + status, so there is no
        # false 'recording' state when the input isn't open.
        try:
            ok = bool(self._ctrl.start_record())
        except Exception:
            ok = False
        if ok:
            self._played_once = False   # a fresh take replaces the old one
            # success → controller fires on_state_changed → recording paint.
            return
        blocked = self._btn_record.blockSignals(True)
        self._btn_record.setChecked(False)
        self._btn_record.blockSignals(blocked)
        self.recording_changed.emit(False)
        self._show_status('deck_unavailable')
        self._sync()

    def stop(self) -> None:
        try:
            self._ctrl.stop()           # fires on_state_changed
        except Exception:
            self._sync()

    def play(self) -> None:
        self._status.setVisible(False)
        # Playback sounds through the output mixer: it starts once an output
        # runs (never 'playing' with nothing to hear). Mirrors drone/metro.
        self._output_users.want('deck', self._play_now, self._play_failed)

    def _play_now(self) -> None:
        try:
            ok = bool(self._ctrl.play())   # fires on_state_changed
        except Exception:
            ok = False
        if ok:
            self._played_once = True
        else:
            self._play_failed(None)

    def _play_failed(self, status) -> None:
        self._show_status('deck_play_unavailable', output_status=status)
        self._sync()

    def _on_export(self) -> None:
        t = self._tr.t
        path, _sel = QFileDialog.getSaveFileName(
            self, t('deck_export_title'), self._cfg.last_take_path,
            t('deck_export_filter'))
        if not path:
            return
        if not path.lower().endswith('.wav'):
            path += '.wav'
        try:
            ok = bool(self._ctrl.export(path))
        except Exception:
            ok = False
        if ok:
            self._cfg.last_take_path = path
            self._show_status('deck_export_done', error=False)
        else:
            self._show_status('deck_export_fail')

    # ── Output lifecycle ─────────────────────────────────────────────────────
    def park_output(self) -> None:
        """The output is gone: stop playback (the take stays)."""
        if self.state() == 'playing':
            self._ctrl.stop()

    def shut_down(self) -> None:
        """Stop the pump, disarm the input tap and unregister playback;
        close() is idempotent, never raises, never orphans."""
        self._pump_timer.stop()
        self._ctrl.close()

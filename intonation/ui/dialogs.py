"""Modal dialogs of the main window: the audio-input picker, the
per-instrument range editor, the first-launch welcome, the custom-instrument
prompt, and the DE | EN toggle they share.

The dialogs own no application state beyond what they edit: the window passes
in its translation function, the engine and the config, and acts on the
result after ``exec()`` returns.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from intonation.audio.engine import (
    SAMPLERATE_PREF_VALUES,
    VENDOR_REGEX,
    AudioEngineError,
    AudioEngineState,
    DeviceInfo,
    DeviceSelection,
    query_input_devices,
)
from intonation.catalog import instruments
from intonation.core.pitch import midi_note_name, note_name_to_midi
from intonation.persistence import config

if TYPE_CHECKING:
    from intonation.audio.engine import AudioEngine

# t_func(key, **fmt) -> translated text in the current UI language. Each
# dialog binds it to the name ``_t``, as the window does, so the i18n
# contract scan (tests/contracts/test_i18n_contracts.py) checks every key and
# its format arguments here too.
Translate = Callable[..., str]
# add_lang_toggle bound to the window: (dialog, layout, on_change) -> None.
LangToggle = Callable[..., None]


# =============================================================================
# DE | EN toggle shared by the app's own dialogs
# =============================================================================
def add_lang_toggle(dlg: QDialog, layout: QLayout, *,
                    get_lang: Callable[[], str],
                    set_lang: Callable[[str], None],
                    t_func: Translate,
                    on_change: Callable[[], None] | None = None) -> None:
    """Drop a small DE | EN toggle at the top of a modal dialog so the user
    can switch language without cancelling out to find the toolbar combo.

    ``set_lang`` switches the whole UI (the window routes it through its
    language combo, so every subscriber updates). ``on_change``, if given,
    runs after a switch so the dialog can relabel its own widgets; the
    window title is relabelled here from the dialog's ``_lang_title_key``
    property."""
    _t = t_func
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addStretch()

    def make_btn(label, code):
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(get_lang() == code)
        btn.setFixedWidth(46)
        btn.setStyleSheet("""
            QPushButton{background:#2a2a3e;color:#bbb;border:1px solid #444;
                         border-radius:4px;padding:3px 8px;font-size:11px;}
            QPushButton:checked{background:#2c5282;color:white;border:1px solid #6699cc;}
            QPushButton:hover:!checked{background:#34344a;}
        """)
        return btn

    btn_de = make_btn('DE', 'de')
    btn_en = make_btn('EN', 'en')

    def switch(target):
        # A checkable button unchecks itself when clicked while active,
        # so re-sync both buttons on every click, including the no-op
        # click on the language already in use.
        if get_lang() == target:
            btn_de.setChecked(get_lang() == 'de')
            btn_en.setChecked(get_lang() == 'en')
            return
        set_lang(target)
        btn_de.setChecked(get_lang() == 'de')
        btn_en.setChecked(get_lang() == 'en')
        dlg.setWindowTitle(_t(dlg.property('_lang_title_key') or 'window_title'))
        if on_change is not None:
            on_change()

    btn_de.clicked.connect(lambda: switch('de'))
    btn_en.clicked.connect(lambda: switch('en'))
    row.addWidget(btn_de)
    row.addWidget(btn_en)
    layout.addLayout(row)


# =============================================================================
# First-launch welcome
# =============================================================================
class WelcomeDialog(QDialog):
    """First-launch dialog. Offers opt-in for the persistent measurement
    history (log.jsonl); ``keep_history()`` reports the choice after
    ``exec()``. "Decide later" (reject) leaves the question open, so the
    window shows it again on the next launch."""

    def __init__(self, parent: QWidget, t_func: Translate,
                 default_log_path: Path, lang_toggle: LangToggle) -> None:
        super().__init__(parent)
        self._t = t_func
        self.setProperty('_lang_title_key', 'welcome_title')
        self.setWindowTitle(self._t('welcome_title'))
        self.setMinimumWidth(480)
        self.setObjectName('welcomeDlg')   # themed via QDialog#welcomeDlg
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 14, 24, 18)

        info = QLabel(self._t('welcome_info'))
        info.setWordWrap(True)
        self._cb_persist = QCheckBox(self._t('welcome_persist'))
        self._cb_persist.setChecked(False)
        path_lbl = QLabel(self._t('welcome_path', path=str(default_log_path)))
        path_lbl.setObjectName('dimLabel')   # themed via QLabel#dimLabel
        path_lbl.setWordWrap(True)
        btns = QDialogButtonBox()
        btn_go = btns.addButton(self._t('welcome_continue'),
                                QDialogButtonBox.ButtonRole.AcceptRole)
        btn_later = btns.addButton(self._t('welcome_skip'),
                                   QDialogButtonBox.ButtonRole.RejectRole)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        def relabel():
            info.setText(self._t('welcome_info'))
            self._cb_persist.setText(self._t('welcome_persist'))
            path_lbl.setText(self._t('welcome_path', path=str(default_log_path)))
            btn_go.setText(self._t('welcome_continue'))
            btn_later.setText(self._t('welcome_skip'))

        # The language toggle sits at the top: this dialog runs on first
        # launch, before the user has touched the toolbar language combo.
        lang_toggle(self, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addWidget(self._cb_persist)
        layout.addWidget(path_lbl)
        layout.addWidget(btns)

    def keep_history(self) -> bool:
        return self._cb_persist.isChecked()


# =============================================================================
# Custom instrument prompt
# =============================================================================
def ask_custom_instrument(parent: QWidget, t_func: Translate,
                          lang_toggle: LangToggle) -> tuple[str, int] | None:
    """Ask for a custom instrument's display name and transposition.

    Returns ``(name, transp)``, or None when cancelled or when the name is
    empty or has no letter or digit (the instrument key is a slug of those;
    the user is told why)."""
    _t = t_func
    dlg = QDialog(parent)
    dlg.setProperty('_lang_title_key', 'custom_dlg_title')
    dlg.setWindowTitle(_t('custom_dlg_title'))
    dlg.setMinimumWidth(380)
    dlg.setStyleSheet("""
        QDialog { background: #1a1a2e; color: #ddd; }
        QLabel  { color: #bbb; font-size: 13px; }
        QLineEdit, QSpinBox {
            background: #252540; border: 1px solid #444; border-radius: 5px;
            color: #eee; padding: 6px 10px; font-size: 13px;
        }
        QLineEdit:focus, QSpinBox:focus { border: 1px solid #6699cc; }
        QDialogButtonBox QPushButton {
            background: #2c5282; color: white; border: none;
            border-radius: 5px; padding: 6px 18px; font-size: 13px;
        }
    """)
    layout = QVBoxLayout(dlg)
    layout.setSpacing(10)
    layout.setContentsMargins(20, 14, 20, 16)
    info = QLabel(_t('custom_dlg_info'))
    info.setStyleSheet('color: #888; font-size: 12px;')
    info.setWordWrap(True)
    form = QFormLayout()
    form.setSpacing(8)
    edit_name = QLineEdit()
    edit_name.setPlaceholderText('e.g. Eb Mezzo-Soprano')
    name_lbl = QLabel(_t('custom_lbl_name'))
    form.addRow(name_lbl, edit_name)
    spin_transp = QSpinBox()
    spin_transp.setRange(-36, 36)
    spin_transp.setValue(0)
    transp_lbl = QLabel(_t('custom_lbl_transp'))
    form.addRow(transp_lbl, spin_transp)
    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)

    def relabel():
        info.setText(_t('custom_dlg_info'))
        name_lbl.setText(_t('custom_lbl_name'))
        transp_lbl.setText(_t('custom_lbl_transp'))

    lang_toggle(dlg, layout, on_change=relabel)
    layout.addWidget(info)
    layout.addLayout(form)
    layout.addWidget(btns)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    name = edit_name.text().strip()
    if not name:
        QMessageBox.warning(parent, _t('err_title'), _t('custom_err_name'))
        return None
    # The instrument key is a slug of letters and digits; a name with
    # neither would produce an empty key.
    if not any(c.isalnum() for c in name):
        QMessageBox.warning(parent, _t('err_title'), _t('custom_err_chars'))
        return None
    return (name, int(spin_transp.value()))


# =============================================================================
# Audio-input picker
# =============================================================================
def _promote_vendor_prefix(name: str) -> str:
    """If ``name`` contains a known vendor brand mid-string (typical of
    Windows naming like "Headset (FIIO DSP Audio)" or "Microphone
    (2- Scarlett Solo)"), return a ``"{VENDOR} · {rest}"`` form with
    the matched vendor lifted to the front and the parenthesised vendor
    fragment stripped from the body.

    Returns the original ``name`` unchanged if no vendor regex matches
    or if the match already sits at position 0 (no promotion needed)."""
    if not name:
        return name
    m = re.search(VENDOR_REGEX, name, re.IGNORECASE)
    if m is None:
        return name
    vendor = m.group(0).upper()
    # Step 1: for each parenthesised fragment that contains a vendor
    # token, strip only the vendor occurrence from the paren interior and
    # keep the rest.  Examples:
    #   "Microphone (2- Scarlett Solo)" -> "Microphone (2- Solo)"
    #   "Line In (FIIO) - ASUS"         -> "Line In  - ASUS"  (parens empty, dropped)
    # Non-vendor parens are preserved verbatim.  v0.6 change: previously
    # the WHOLE paren was dropped when it contained a vendor, losing
    # meaningful non-vendor content like "(2- Solo)".  Empty parens
    # produced by full-vendor stripping collapse to a single space.
    paren_re = re.compile(r'\([^)]*\)')
    cleaned_parts: list[str] = []
    pos = 0
    for pm in paren_re.finditer(name):
        cleaned_parts.append(name[pos:pm.start()])
        inner = pm.group(0)
        if re.search(VENDOR_REGEX, inner, re.IGNORECASE) is None:
            cleaned_parts.append(inner)
        else:
            stripped = re.sub(VENDOR_REGEX, '', inner,
                              count=1, flags=re.IGNORECASE)
            # Tidy paren whitespace produced by the strip.
            stripped = re.sub(r'\(\s+', '(', stripped)
            stripped = re.sub(r'\s+\)', ')', stripped)
            stripped = re.sub(r'\s+', ' ', stripped)
            # If only the vendor was inside, collapse the now-empty
            # parens to a single space (Step 3 collapses runs of space).
            if stripped in ('()', '( )'):
                cleaned_parts.append(' ')
            else:
                cleaned_parts.append(stripped)
        pos = pm.end()
    cleaned_parts.append(name[pos:])
    body = ''.join(cleaned_parts)
    # Step 2 (v0.5.7.4): strip bare occurrences of the matched vendor
    # token still left in the body. Without this, "FIIO Q3" produced
    # "FIIO · FIIO Q3" because step 1 only touches paren-wrapped vendor
    # text. We only strip the SAME vendor that was hoisted to the prefix
    # — other vendor-list tokens in the body are part of the device
    # name proper (e.g. "Universal Audio Apollo Twin X" should keep
    # "Apollo" in the body even though Apollo is a known brand).
    body = re.sub(
        rf'\b{re.escape(m.group(0))}\b', ' ', body, count=1, flags=re.IGNORECASE
    )
    # Step 3: clean up whitespace and orphan separators left by removal.
    # v0.6 dropped a `re.sub(r'\s*-\s*', ' - ', body)` step that used to
    # normalise dash spacing; with the new paren-content-preservation
    # logic, dashes inside parens (e.g. "(2- Solo)") must stay intact.
    body = re.sub(r'\s*\(\s*\)\s*', ' ', body)  # empty parens
    body = re.sub(r'\s+', ' ', body).strip(' -·:|')
    if not body:
        # Input was effectively just the vendor token (possibly wrapped
        # in parens) — return it bare rather than echoing back "VENDOR · ".
        return vendor
    return f'{vendor} · {body}'


class AudioPickerDialog(QDialog):
    """Modal device picker. Two-line rows, dedup-by-name with an
    expandable host-API sublist. Vendor regex ranks external interfaces
    to the top. The dialog never opens a stream itself — accepting the
    selection just calls ``engine.open_device(spec)`` and lets the
    state machine do the rest."""

    # The sample-rate re-open runs on the engine's audio-control thread
    # (open_device walks candidates with an open timeout per attempt, which
    # would freeze the UI on a slow device). The command emits this signal
    # carrying (prev_pref, new_pref) and a QueuedConnection on the receiving
    # slot marshals back to the GUI thread for the UI updates.
    _sr_reopen_done = pyqtSignal(str, str)

    def __init__(self, parent, t_func, engine: 'AudioEngine',
                 cfg: config.AppConfig, current: 'DeviceInfo | None',
                 store: config.ConfigStore):
        super().__init__(parent)
        self._t = t_func
        self._engine = engine
        self._cfg = cfg
        self._store = store
        self._current = current
        self._chosen: 'DeviceInfo | None' = None
        # v0.5.7.2: debounce flag — ignore further combo changes while
        # an open is in flight. The combo is also disabled, but the
        # flag guards against programmatic edits / queued signals that
        # could slip in around the worker boundary.
        self._sr_switch_in_flight: bool = False
        self._sr_prev_pref: str = 'auto'
        # v0.5.7.3: close-during-worker UAF guard. If the user closes the
        # dialog while an sr-reopen worker is still running, the worker's
        # finally block used to emit `_sr_reopen_done` on a QObject that
        # Qt had already destroyed (segfault / undefined behaviour). The
        # threading.Event lets the worker bail out of the emit, and we
        # also disconnect the signal in closeEvent as a second line of
        # defence.
        self._closing_event: threading.Event = threading.Event()
        # True while this dialog holds a WaitCursor override. Released in
        # _restore_cursor, which every exit path calls: a dialog closed
        # before the sr-reopen worker reports back must not leave the
        # application stuck on the busy cursor.
        self._cursor_overridden = False
        self._sr_reopen_done.connect(
            self._on_sr_reopen_done, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle(self._t('audio_picker_title'))
        self.setModal(True)
        self.setMinimumWidth(560)
        # v0.5.7.1: dialog used to open at Qt's minimum-content height,
        # which fit ~3 rows and forced the user to scroll past the most
        # interesting devices (vendor interfaces below the system mics).
        # Open at a size where the full typical list fits unscrolled.
        self.setMinimumSize(480, 480)
        self.setObjectName('pickerDlg')   # themed via QDialog#pickerDlg
        self._build()
        self._refill()
        # v0.5.7.1: explicit default size so the picker opens at a
        # comfortable height for the typical Windows device count.
        self.resize(560, 600)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # v0.5.6: sample-rate preference row. Sits ABOVE the device list
        # so the user reads the policy first and the chosen device second.
        sr_row = QHBoxLayout()
        sr_lbl = QLabel(self._t('samplerate_label') + ':')
        sr_lbl.setObjectName('dimLabel')
        sr_row.addWidget(sr_lbl)
        self._sr_combo = QComboBox()
        self._sr_combo.addItem(self._t('samplerate_auto'), 'auto')
        for hz in (192000, 96000, 88200, 48000, 44100):
            # Number-with-thin-space formatting reads as "192 000 Hz" in
            # both DE and EN locales without dragging in locale plumbing.
            label = f'{hz:,}'.replace(',', ' ') + ' Hz'
            self._sr_combo.addItem(label, str(hz))
        cur_pref = self._cfg.audio_samplerate_pref
        idx = self._sr_combo.findData(cur_pref)
        if idx < 0:
            idx = 0
        self._sr_combo.setCurrentIndex(idx)
        # v0.5.7.2: snapshot for revert-on-failure path.
        self._sr_prev_pref = cur_pref
        self._sr_combo.currentIndexChanged.connect(self._on_sr_changed)
        # themed by the global QComboBox rule in build_app_qss (no inline).
        sr_row.addWidget(self._sr_combo, 1)
        root.addLayout(sr_row)

        self._sr_error_lbl = QLabel('')
        self._sr_error_lbl.setObjectName('errorLabel')   # themed via QLabel#errorLabel
        self._sr_error_lbl.setVisible(False)
        self._sr_error_lbl.setWordWrap(True)
        root.addWidget(self._sr_error_lbl)

        top = QHBoxLayout()
        self._btn_rescan = QPushButton(self._t('audio_picker_rescan'))
        self._btn_rescan.clicked.connect(self._refill)
        top.addWidget(self._btn_rescan)
        top.addStretch()
        root.addLayout(top)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _i: self._accept())
        root.addWidget(self._list, 1)

        opts = QHBoxLayout()
        self._cb_show_all = QCheckBox(self._t('audio_picker_show_all'))
        self._cb_show_all.setChecked(self._cfg.show_all_host_apis)
        self._cb_show_all.toggled.connect(self._on_toggle_show_all)
        opts.addWidget(self._cb_show_all)
        self._cb_ks = QCheckBox(self._t('audio_picker_prefer_ks'))
        self._cb_ks.setChecked(self._cfg.prefer_wdmks)
        self._cb_ks.toggled.connect(self._on_toggle_ks)
        opts.addWidget(self._cb_ks)
        opts.addStretch()
        root.addLayout(opts)

        btns = QHBoxLayout()
        self._btn_use = QPushButton(self._t('audio_picker_use'))
        self._btn_use.clicked.connect(self._accept)
        self._btn_cancel = QPushButton(self._t('audio_picker_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self._btn_use)
        btns.addWidget(self._btn_cancel)
        root.addLayout(btns)

    def _on_sr_changed(self, _idx: int) -> None:
        """User picked a new sample-rate policy. Persist immediately,
        then reopen the active device with the new pref on a worker
        thread.

        open_device walks candidates with an open timeout per attempt,
        which on Qt's main thread froze the dialog for seconds. The
        re-open is queued on the engine's audio-control thread (in order
        with every other device command), the combo is disabled while it
        runs, and the UI updates are marshalled back through
        ``_sr_reopen_done`` (QueuedConnection) so all widget mutation
        happens on the GUI thread again.
        """
        if self._sr_switch_in_flight:
            # Debounce: ignore further changes until the current worker
            # finishes. The combo is disabled below, so this only fires
            # for queued signals that slipped past the disable.
            return
        new_pref = str(self._sr_combo.currentData() or 'auto')
        if new_pref not in SAMPLERATE_PREF_VALUES:
            new_pref = 'auto'
        prev_pref = self._sr_prev_pref
        if new_pref == prev_pref:
            return
        self._cfg.audio_samplerate_pref = new_pref
        self._store.save(self._cfg)
        self._sr_error_lbl.setVisible(False)
        dev = self._engine.get_active_device()
        if dev is None:
            self._sr_prev_pref = new_pref
            return
        sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                              samplerate=0)

        # Block the GUI from issuing more re-opens until this one lands.
        self._sr_switch_in_flight = True
        self._sr_combo.setEnabled(False)
        if not self._cursor_overridden:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._cursor_overridden = True

        # v0.5.7.3: capture the event by closure (not self.) so the
        # worker can check it even if the dialog is mid-teardown. The
        # event lives independently of the QObject's lifetime.
        closing_event = self._closing_event

        def _worker() -> None:
            try:
                self._engine.stop()
                self._engine.open_device(sel, samplerate_pref=new_pref)
            finally:
                # Even if the engine raises, hand control back to the
                # GUI thread so the combo gets re-enabled — UNLESS the
                # dialog is being closed, in which case the QObject
                # may already be destroyed and emitting on a dead
                # receiver is undefined behaviour.
                if not closing_event.is_set():
                    try:
                        self._sr_reopen_done.emit(prev_pref, new_pref)
                    except RuntimeError:
                        # "wrapped C/C++ object has been deleted" —
                        # raced past our event check. Swallow; there's
                        # no receiver to hand off to anyway.
                        pass

        self._engine.submit(_worker)

    def _on_sr_reopen_done(self, prev_pref: str, new_pref: str) -> None:
        """Runs on the GUI thread (QueuedConnection). Re-enables the
        combo, surfaces the inline error on UNSUPPORTED_RATE, and
        reverts the selection if the new pref was refused."""
        self._restore_cursor()
        failed_rate = (
            self._engine.state == AudioEngineState.FAILED
            and self._engine.last_error == AudioEngineError.UNSUPPORTED_RATE
        )
        if failed_rate:
            shown_rate = new_pref if new_pref != 'auto' else '–'
            self._sr_error_lbl.setText(
                self._t('samplerate_unsupported', rate=shown_rate))
            self._sr_error_lbl.setVisible(True)
            # Revert to previous pref in combo + cfg, then retry the
            # open with the previous pref so the user is left in the
            # same audio state they started with.
            self._cfg.audio_samplerate_pref = prev_pref
            self._store.save(self._cfg)
            self._sr_combo.blockSignals(True)
            revert_idx = self._sr_combo.findData(prev_pref)
            if revert_idx < 0:
                revert_idx = 0
            self._sr_combo.setCurrentIndex(revert_idx)
            self._sr_combo.blockSignals(False)
            dev = self._engine.get_active_device()
            if dev is not None:
                sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                                      samplerate=0)
                # Queued like any device command, so we don't re-introduce
                # the main-thread block we just removed. Fire and forget —
                # failure here just leaves the engine in FAILED, which the
                # banner handles separately.
                self._engine.request_open_device(sel,
                                                 samplerate_pref=prev_pref)
            self._sr_prev_pref = prev_pref
        else:
            self._sr_prev_pref = new_pref
        self._sr_combo.setEnabled(True)
        self._sr_switch_in_flight = False

    def closeEvent(self, event) -> None:
        """v0.5.7.3: signal any in-flight sr-reopen worker that we're
        going away so it skips its emit (which would otherwise land on a
        deleted QObject). Also disconnect the signal as belt-and-
        suspenders — if a queued emission is already sitting in the
        event loop, the disconnect makes it a no-op."""
        self._closing_event.set()
        try:
            self._sr_reopen_done.disconnect(self._on_sr_reopen_done)
        except (TypeError, RuntimeError):
            # Already disconnected or signal/slot already torn down.
            pass
        self._restore_cursor()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """``accept()`` / ``reject()`` route through here, so set the
        closing flag here too. closeEvent isn't always called on
        programmatic accept/reject paths. The worker's completion signal is
        skipped once closing, so the cursor is released here."""
        self._closing_event.set()
        self._restore_cursor()
        super().done(result)

    def _restore_cursor(self) -> None:
        """Pop this dialog's WaitCursor override, once."""
        if not self._cursor_overridden:
            return
        self._cursor_overridden = False
        try:
            QGuiApplication.restoreOverrideCursor()
        except Exception:
            pass

    def _on_toggle_show_all(self, checked: bool) -> None:
        self._cfg.show_all_host_apis = bool(checked)
        self._store.save(self._cfg)
        self._refill()

    def _on_toggle_ks(self, checked: bool) -> None:
        self._cfg.prefer_wdmks = bool(checked)
        self._store.save(self._cfg)
        self._engine.set_prefer_wdmks(bool(checked))

    def _rank(self, d: DeviceInfo) -> int:
        """Higher = sorts earlier. Vendor regex dominates."""
        score = 0
        if re.search(VENDOR_REGEX, d.name, re.IGNORECASE):
            score += 100
        api = d.host_api.lower()
        if 'wasapi' in api:
            score += 20
        elif 'wdm-ks' in api:
            score += 10
        if d.default_samplerate >= 48000:
            score += 5
        low = d.name.lower()
        if any(x in low for x in ('webcam', 'hdmi', 'nvidia', 'amd')):
            score -= 50
        if any(x in low for x in ('microphone array', 'stereo mix')):
            score -= 20
        return score

    def _refill(self) -> None:
        self._list.clear()
        # Only lists devices; plugging in and out is handled by the engine's
        # hot-plug poll.
        devices = query_input_devices()
        if not devices:
            it = QListWidgetItem(self._t('audio_picker_no_devices'))
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(it)
            return
        show_all = self._cb_show_all.isChecked()
        # Group by canonical name unless show-all is on.
        groups: dict[str, list[DeviceInfo]] = {}
        order: list[str] = []
        for d in devices:
            key = d.name if not show_all else f'{d.name}\0{d.host_api}'
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(d)
        # Pick a representative for each group (WASAPI > WDM-KS > MME > rest).
        def api_pref(d: DeviceInfo) -> int:
            api = d.host_api.lower()
            if 'wasapi' in api:
                return 0
            if 'wdm-ks' in api:
                return 1
            if api == 'mme':
                return 2
            if 'directsound' in api:
                return 3
            return 9
        rows: list[tuple[DeviceInfo, list[DeviceInfo]]] = []
        for key in order:
            members = sorted(groups[key], key=api_pref)
            rows.append((members[0], members))
        # Sort rows by rank, current device pinned to the top.
        def sort_key(row):
            primary = row[0]
            is_current = (
                self._current is not None
                and primary.name == self._current.name
                and primary.host_api == self._current.host_api)
            return (0 if is_current else 1, -self._rank(primary),
                    primary.name.lower())
        rows.sort(key=sort_key)
        for primary, members in rows:
            self._add_row(primary, members)

    def _add_row(self, primary: DeviceInfo,
                 members: list[DeviceInfo]) -> None:
        is_current = (
            self._current is not None
            and primary.name == self._current.name
            and primary.host_api == self._current.host_api)
        meta = self._t('audio_picker_row_meta',
                       api=primary.host_api or '—',
                       ch=primary.max_input_channels,
                       sr=primary.default_samplerate)
        badge = ''
        if is_current:
            badge = f'  ◀ {self._t("audio_picker_current")}'
        elif len(members) > 1:
            badge = f'  ▸ {self._t("audio_picker_apis_more", n=len(members))}'
        # v0.5.7: if a vendor brand appears mid-string in the Windows
        # device name (e.g. "Headset (FIIO DSP Audio)"), promote the
        # brand to the front of the row label so saxophone players can
        # scan by brand instead of by Windows device-naming convention.
        # The full original name remains in the row's tooltip.
        display_name = _promote_vendor_prefix(primary.name)
        text = f'{display_name}{badge}\n    {meta}'
        it = QListWidgetItem(text)
        it.setData(Qt.ItemDataRole.UserRole, primary)
        it.setToolTip(primary.name)
        self._list.addItem(it)
        if is_current:
            self._list.setCurrentItem(it)

    def _accept(self) -> None:
        it = self._list.currentItem()
        if it is None:
            self.reject()
            return
        dev = it.data(Qt.ItemDataRole.UserRole)
        if not isinstance(dev, DeviceInfo):
            self.reject()
            return
        self._chosen = dev
        self.accept()

    def chosen(self) -> 'DeviceInfo | None':
        return self._chosen


# =============================================================================
# Per-instrument range editor (v0.5.5)
# =============================================================================
class RangeEditorDialog(QDialog):
    """Modal editor for the (lo, hi) fingered-MIDI range of one
    instrument. Persists to ~/.intonation_analyzer/instrument_ranges.json
    via instruments.save_range_override / clear_range_override.

    The dialog owns no policy: the caller (MainWindow) refreshes its
    table after accept() so the new range takes effect immediately."""

    def __init__(self, parent, t, instrument_key: str,
                 display_name: str, current_lo: int, current_hi: int,
                 baked_lo: int, baked_hi: int, has_baked: bool,
                 display: str = 'griff', transp: int = 0):
        """``current_lo`` / ``current_hi`` / ``baked_lo`` / ``baked_hi``
        are ALWAYS fingered MIDI (canonical disk format). The dialog
        renders them as sounding MIDI when ``display == 'klingend'`` by
        adding ``transp`` semitones, and reverses the offset on save —
        the JSON on disk never changes format regardless of which user
        opens the editor.
        """
        super().__init__(parent)
        self._t = t
        self._key = instrument_key
        self._baked = (baked_lo, baked_hi)  # canonical fingered MIDI
        self._has_baked = has_baked
        self._display = 'klingend' if display == 'klingend' else 'griff'
        self._transp = int(transp) if self._display == 'klingend' else 0
        self.setWindowTitle(self._t('range_editor_title', name=display_name))
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)
        self._lo_spin = QSpinBox()
        self._lo_spin.setRange(0, 127)
        self._lo_spin.setValue(int(current_lo) + self._transp)
        self._hi_spin = QSpinBox()
        self._hi_spin.setRange(0, 127)
        self._hi_spin.setValue(int(current_hi) + self._transp)
        # v0.6.2: explicit sub-control geometry for the up/down buttons.
        # Setting bg + border + padding on QSpinBox without styling
        # ::up-button / ::down-button leaves the buttons with effectively
        # zero clickable width on Windows — the up arrow visibly draws
        # but the hit-rect collapses. Reserving 18px of right padding +
        # positioning the two sub-controls top/bottom right restores
        # clickability and keeps the dark theme.
        self._lo_spin.setObjectName('rangeSpin')   # themed via QSpinBox#rangeSpin
        self._hi_spin.setObjectName('rangeSpin')
        # Primary input is the note-name QLineEdit;
        # the MIDI spinbox is kept visible for power users. Wire-up:
        # the name edit drives the spin (parses on editingFinished), and
        # the spin drives the edit (formats on valueChanged). Cycle guard
        # via _suppress_sync so we don't bounce between the two.
        self._suppress_sync = False
        self._lo_name = QLineEdit()
        self._hi_name = QLineEdit()
        self._lo_name.setObjectName('rangeName')   # themed via QLineEdit#rangeName
        self._hi_name.setObjectName('rangeName')
        self._lo_name.setText(midi_note_name(int(current_lo) + self._transp))
        self._hi_name.setText(midi_note_name(int(current_hi) + self._transp))
        self._lo_name.setPlaceholderText(self._t('range_note_name_hint'))
        self._hi_name.setPlaceholderText(self._t('range_note_name_hint'))
        self._lo_name.setToolTip(self._t('range_note_name_hint'))
        self._hi_name.setToolTip(self._t('range_note_name_hint'))
        if self._display == 'klingend':
            lo_label = self._t('range_lo_label_sounding')
            hi_label = self._t('range_hi_label_sounding')
        else:
            lo_label = self._t('range_lo_label')
            hi_label = self._t('range_hi_label')
        # Each row hosts name-edit + spinbox so power users still see the
        # MIDI integer. Name-edit is first because it's the primary input.
        def _make_row(name_edit, spin) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(name_edit, 1)
            h.addWidget(spin, 0)
            return w
        form.addRow(lo_label, _make_row(self._lo_name, self._lo_spin))
        form.addRow(hi_label, _make_row(self._hi_name, self._hi_spin))
        layout.addLayout(form)

        # Mode hint so the user knows what the spinboxes mean and that
        # the file format is stable across users with different display
        # preferences.
        mode_note_key = ('range_mode_note_sound'
                        if self._display == 'klingend'
                        else 'range_mode_note_griff')
        self._mode_lbl = QLabel(self._t(mode_note_key))
        # colour inherits QDialog#rangeDlg QLabel (text_dim); keep font only.
        self._mode_lbl.setStyleSheet('font-size:11px;font-style:italic;')
        self._mode_lbl.setWordWrap(True)
        layout.addWidget(self._mode_lbl)

        self._preview_lbl = QLabel('')
        self._preview_lbl.setStyleSheet('font-size:12px;')
        self._preview_lbl.setWordWrap(True)
        layout.addWidget(self._preview_lbl)

        self._error_lbl = QLabel('')
        self._error_lbl.setObjectName('errorLabel')   # red, overrides dialog rule
        self._error_lbl.setWordWrap(True)
        self._error_lbl.setVisible(False)
        layout.addWidget(self._error_lbl)

        # Buttons row.
        btn_row = QHBoxLayout()
        if has_baked:
            restore_label = self._t('range_restore_default')
        else:
            # Show baked numbers in the current display mode so they
            # match what the spinboxes display.
            disp_lo = baked_lo + self._transp
            disp_hi = baked_hi + self._transp
            restore_label = self._t('range_restore_fallback',
                                    lo=disp_lo, hi=disp_hi)
        self._btn_restore = QPushButton(restore_label)
        self._btn_restore.clicked.connect(self._on_restore)
        self._btn_cancel = QPushButton(self._t('range_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_save = QPushButton(self._t('range_save'))
        self._btn_save.clicked.connect(self._on_save)
        for _b in (self._btn_restore, self._btn_cancel, self._btn_save):
            _b.setObjectName('chrome')   # themed via QPushButton#chrome (+disabled)
        # Enter after typing a note must save. Without this Qt gives Enter to
        # the first button created, Restore, which silently undid the typing.
        self._btn_restore.setAutoDefault(False)
        self._btn_cancel.setAutoDefault(False)
        self._btn_save.setDefault(True)
        btn_row.addWidget(self._btn_restore)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_save)
        layout.addLayout(btn_row)

        # Live preview wiring.
        self._lo_spin.valueChanged.connect(self._on_spin_changed)
        self._hi_spin.valueChanged.connect(self._on_spin_changed)
        self._lo_name.editingFinished.connect(
            lambda: self._on_name_edited(self._lo_name, self._lo_spin))
        self._hi_name.editingFinished.connect(
            lambda: self._on_name_edited(self._hi_name, self._hi_spin))
        self._update_preview()

        # Themed via QDialog#rangeDlg in build_app_qss (re-themes per palette;
        # was a hardcoded dark modal).
        self.setObjectName('rangeDlg')

    def _set_invalid(self, w, bad: bool) -> None:
        """Toggle the [invalid] QSS property + repolish so the themed warn style
        (QSS [invalid="true"]) applies — replaces the old runtime setStyleSheet
        ok/bad swap so validation styling re-themes with the palette."""
        w.setProperty('invalid', bool(bad))
        w.style().unpolish(w)
        w.style().polish(w)

    def _on_spin_changed(self) -> None:
        # Spin changed → sync the matching note-name edit, then refresh
        # the preview. Guarded so we don't recurse when the name-edit
        # slot is the one mutating the spin.
        if not self._suppress_sync:
            self._suppress_sync = True
            try:
                self._lo_name.setText(midi_note_name(self._lo_spin.value()))
                self._hi_name.setText(midi_note_name(self._hi_spin.value()))
                # Clear any invalid mark if the user previously typed a bad name.
                self._set_invalid(self._lo_name, False)
                self._set_invalid(self._hi_name, False)
            finally:
                self._suppress_sync = False
        self._update_preview()

    def _on_name_edited(self, edit, spin) -> None:
        # Parse the user's text. On success, push the MIDI into the spin
        # (and let _on_spin_changed reformat). On failure, paint the edit
        # red and leave the spin alone — the brief is strict about not
        # silently mutating the underlying number when parsing fails.
        if self._suppress_sync:
            return
        text = edit.text()
        midi = note_name_to_midi(text)
        if midi is None:
            self._set_invalid(edit, True)
            self._error_lbl.setText(self._t('range_note_name_invalid'))
            self._error_lbl.setVisible(True)
            return
        # Valid name → clear the invalid mark, push to spin (which re-formats
        # the edit canonically via _on_spin_changed).
        self._set_invalid(edit, False)
        spin.setValue(midi)

    def _update_preview(self) -> None:
        lo = self._lo_spin.value()
        hi = self._hi_spin.value()
        invalid = lo > hi
        # Visual cue on the offending field — themed warn via the [invalid]
        # property (QSpinBox#rangeSpin[invalid="true"] in build_app_qss; the
        # spin sub-controls are themed there too, keeping them clickable).
        self._set_invalid(self._lo_spin, invalid)
        self._set_invalid(self._hi_spin, invalid)
        if invalid:
            self._error_lbl.setText(self._t('range_invalid'))
            self._error_lbl.setVisible(True)
            self._preview_lbl.setText('')
            self._btn_save.setEnabled(False)
            return
        self._error_lbl.setVisible(False)
        self._btn_save.setEnabled(True)
        semis = hi - lo
        octs = semis / 12.0
        self._preview_lbl.setText(self._t(
            'range_preview_fmt',
            lo_name=midi_note_name(lo), hi_name=midi_note_name(hi),
            semis=semis, octs=octs))

    def _on_restore(self) -> None:
        # Baked is canonical fingered; render in current display mode.
        lo, hi = self._baked
        self._lo_spin.setValue(lo + self._transp)
        self._hi_spin.setValue(hi + self._transp)

    def _on_save(self) -> None:
        # A typed name that does not parse leaves its spin box on the old
        # value; saving then would store a range the player did not ask for.
        # Keep the dialog open with the error shown instead.
        for edit in (self._lo_name, self._hi_name):
            if note_name_to_midi(edit.text()) is None:
                self._set_invalid(edit, True)
                self._show_error(self._t('range_note_name_invalid'))
                return
        # Spinbox values are in the current display mode. Convert back
        # to fingered (canonical) before persisting so the file format
        # is stable across users with different display preferences.
        disp_lo = self._lo_spin.value()
        disp_hi = self._hi_spin.value()
        if disp_lo > disp_hi:
            return
        lo = disp_lo - self._transp
        hi = disp_hi - self._transp
        # In sounding display the spin boxes span 0..127 sounding, which
        # can land outside 0..127 fingered (an Eb baritone's sounding 127
        # is fingered 148). Say so and keep the dialog open.
        if not (0 <= lo <= 127 and 0 <= hi <= 127):
            self._show_error(self._t('range_out_of_midi', lo=lo, hi=hi))
            return
        # If the user dialed back to the baked default AND a baked entry
        # exists, clear the override instead of writing a redundant
        # entry. Keeps the overrides file lean and lets future baked
        # changes flow through.
        try:
            if self._has_baked and (lo, hi) == self._baked:
                instruments.clear_range_override(self._key)
            else:
                instruments.save_range_override(self._key, lo, hi)
        except (OSError, ValueError,
                instruments.RangeOverridesUnreadable) as exc:
            # Nothing was saved, so the dialog must not close as if it had.
            self._show_error(self._t(
                'range_save_failed',
                path=str(instruments._RANGES_FILE), err=str(exc)))
            return
        self.accept()

    def _show_error(self, text: str) -> None:
        self._error_lbl.setText(text)
        self._error_lbl.setVisible(True)

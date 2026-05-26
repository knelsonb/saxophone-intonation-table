# =============================================================================
# sax_instrument_controller.py — InstrumentController
#
# Extracted from sax_intonation_gui.py in Phase 5 of the refactor. The
# controller owns instrument-family selection, instrument selection,
# nickname editing, per-instrument range editor, custom-instrument
# registration, and the autotune-from-history flow.
#
# Module name is `sax_instrument_controller.py` because `sax_instruments.py`
# already exists for the catalog/transposition data.
#
# Design notes:
#   * Live values that change during a session (the active instrument
#     key, the per-note stats dict, the display mode, the UI language)
#     reach the controller through getter callables, not via fixed
#     references captured at construction time. That way the controller
#     reads the *current* value at action time rather than freezing
#     whatever was set on the window when __init__ ran.
#   * The controller invokes `set_instrument(key)` and `set_a4(hz)`
#     setter callbacks when it changes those values — MainWindow owns
#     the source-of-truth attributes (`self.instrument`, the A4 dataset)
#     so the controller never reaches across the boundary to mutate
#     window state directly.
#   * `on_instrument_changed` and `on_a4_changed` are optional
#     notifications fired AFTER the set callback so MainWindow can
#     refresh the table / range banner / OOR counter without the
#     controller having to know about those concerns.
#   * The controller does NOT call sax_config.save_config(). The
#     MainWindow owns persistence timing (consistent with the
#     SessionStateController / ExportController patterns).
#   * The controller does NOT touch self._table — that's the
#     TableController's territory. It only notifies via callbacks.
#   * No behavioural change vs. the inlined implementation — dialog
#     text, autotune math, custom-instrument key generation, and the
#     RangeEditorDialog round-trip are preserved verbatim.
# =============================================================================
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QSpinBox,
    QDialogButtonBox, QMessageBox,
)

import sax_instruments
import sax_config
from sax_instruments import (
    families as instrument_families,
    instruments_in,
    transp_map as build_transp_map,
    display_name as instrument_display_name,
    family_of as instrument_family_of,
    register_custom,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow, QComboBox, QLineEdit as _QLineEdit
    from sax_config import AppConfig


class InstrumentController:
    """Owns instrument-family selection, instrument selection, nickname
    editing, per-instrument range editor, custom-instrument
    registration, and the autotune-from-history flow.

    Reads MainWindow widgets via constructor injection; reads live
    runtime values (current instrument key, A4, stats dict) via
    getter callables so changes during a session are observed
    correctly."""

    def __init__(self, window, *,
                 cfg, engine, log, t_func,
                 family_combo, instr_combo, nick_edit, a4_combo,
                 get_stats, get_display_mode, get_lang,
                 set_instrument, set_a4,
                 on_instrument_changed=None,
                 on_range_changed=None,
                 on_a4_changed=None) -> None:
        self._w = window
        self._cfg = cfg
        self._engine = engine
        self._log = log
        self._t = t_func
        self._family_combo = family_combo
        self._instr_combo = instr_combo
        self._nick_edit = nick_edit
        self._a4_combo = a4_combo
        # Callables, not values — the active values change at runtime;
        # we want to read them at action time, not at controller init.
        self._get_stats = get_stats
        self._get_display_mode = get_display_mode
        self._get_lang = get_lang
        # Setter callbacks — MainWindow owns the source-of-truth fields.
        self._set_instrument = set_instrument
        self._set_a4 = set_a4
        # Optional post-change notifications.  `on_instrument_changed`
        # fires after the active instrument *key* changed (combo select);
        # `on_range_changed` fires after the per-instrument fingered
        # range was edited (key unchanged, so MainWindow does NOT reset
        # the wrong-instrument detector).
        self._on_instrument_changed = on_instrument_changed
        self._on_range_changed = on_range_changed
        self._on_a4_changed = on_a4_changed

    # ── Family / sub-instrument combos ───────────────────────────────────────
    def on_family_changed(self, _idx) -> None:
        """Family combo changed → repopulate the sub-instrument combo with
        the new family's entries and select the first one."""
        self.populate_instrument_combo(select_key=None)

    def populate_instrument_combo(self, select_key: str | None) -> None:
        family_key = self._family_combo.currentData()
        if family_key is None:
            return
        lang = self._get_lang()
        self._instr_combo.blockSignals(True)
        self._instr_combo.clear()
        target_idx = 0
        for i, (key, name_de, name_en) in enumerate(instruments_in(family_key)):
            label = name_de if lang == 'de' else name_en
            self._instr_combo.addItem(label, key)
            if key == select_key:
                target_idx = i
        if self._instr_combo.count():
            self._instr_combo.setCurrentIndex(target_idx)
        self._instr_combo.blockSignals(False)
        if self._instr_combo.count():
            self.on_instr_changed(self._instr_combo.currentIndex())

    def select_family_for_instrument(self, instrument_key: str) -> None:
        """Move the family combo to the family that contains the given key,
        without triggering a sub-instrument repopulate."""
        family_key = instrument_family_of(instrument_key)
        if family_key is None:
            return
        for i in range(self._family_combo.count()):
            if self._family_combo.itemData(i) == family_key:
                self._family_combo.blockSignals(True)
                self._family_combo.setCurrentIndex(i)
                self._family_combo.blockSignals(False)
                return

    def on_instr_changed(self, idx) -> None:
        # Local imports keep the module decoupled from the GUI file while
        # still using the same shared module-level state. AUDIO_OK is a
        # module-level flag; importing it at call time picks up the
        # value frozen at GUI import.
        from sax_intonation_gui import AUDIO_OK
        key = self._instr_combo.itemData(idx)
        if key is None:
            return
        # Family-combo scrubs re-populate the sub-instrument combo, which
        # auto-fires this handler even when the resolved key matches the
        # already-active instrument. Bail out so we don't spawn a fresh run
        # for every flicker through the family list.
        w = self._w
        if key == w.instrument:
            return
        self._set_instrument(key)
        self._engine.instr_key = key
        # Instrument switch ⇒ new run, so per-run aggregates stay coherent.
        # Empty predecessor runs are coalesced inside `start_run`.
        if AUDIO_OK and getattr(w, '_recording', False):
            self._log.start_run(instrument=key,
                                a4_hz=self._engine.a4,
                                label=self._nick_edit.text().strip())
        if self._on_instrument_changed is not None:
            self._on_instrument_changed()

    # ── Nickname ─────────────────────────────────────────────────────────────
    def on_nickname_changed(self) -> None:
        """User finished editing the nickname. Stamp the new label onto the
        current run so the next CSV export and table summary pick it up."""
        from sax_intonation_gui import AUDIO_OK
        nickname = self._nick_edit.text().strip()
        if AUDIO_OK and getattr(self._w, '_recording', False):
            self._log.set_current_run_metadata(label=nickname)

    # ── Range editor ─────────────────────────────────────────────────────────
    def open_range_editor(self) -> None:
        """Open the per-instrument range editor for the active instrument.
        Accepts → persist via sax_instruments override DB → notify caller
        so the new bounds take effect immediately."""
        from sax_intonation_gui import RangeEditorDialog, TRANSP_MAP
        w = self._w
        key = w.instrument
        cur_lo, cur_hi = sax_instruments.fingered_range(key)
        baked_lo, baked_hi = sax_instruments.baked_fingered_range(key)
        has_baked = sax_instruments.has_baked_range(key)
        lang = self._get_lang()
        name = instrument_display_name(key, lang)
        # v0.5.7.1: pass display mode + instrument transposition so the
        # dialog can render values in whichever notation the user is
        # already reading on the matrix. File format stays fingered.
        display = self._get_display_mode()
        transp = TRANSP_MAP.get(key, 0)
        dlg = RangeEditorDialog(
            w, self._t, key, name,
            cur_lo, cur_hi, baked_lo, baked_hi, has_baked,
            display=display, transp=transp)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Range bounds changed but the *instrument key* did NOT,
            # so MainWindow should re-seed expected notes + refresh
            # without zeroing the wrong-instrument counter. Distinct
            # callback from on_instrument_changed for exactly that
            # reason.
            if self._on_range_changed is not None:
                self._on_range_changed()

    # ── Custom-instrument registration ───────────────────────────────────────
    def on_add_custom(self) -> None:
        """Prompt the user for a custom instrument and register it."""
        from sax_intonation_gui import _rebuild_transp_map
        result = self._ask_custom_instrument()
        if result is None:
            return
        name, transp = result
        # Build a stable key: 'custom_' + slugified name. If the slug collides
        # with an existing custom (different display name but same slug, e.g.
        # 'Mezzo' vs 'mezzo'), append a numeric suffix instead of silently
        # overwriting the older entry — old CSV exports may still reference
        # the original transposition for that key.
        base = 'custom_' + ''.join(
            c.lower() if c.isalnum() else '_' for c in name).strip('_')[:32]
        if not base or base == 'custom_':
            return
        existing_keys = set(build_transp_map().keys())
        key = base
        suffix = 2
        while key in existing_keys:
            # If the user re-typed the EXACT same display name, treat it as
            # an intentional re-add (let register_custom replace the row).
            existing = [c for c in sax_config.load_customs() if c.key == key]
            if existing and existing[0].name_en == name:
                break
            key = f"{base}_{suffix}"
            suffix += 1
        register_custom(key, transp, name, name)
        _rebuild_transp_map()
        # Persist for next session.
        customs = sax_config.load_customs()
        customs = [c for c in customs if c.key != key]
        customs.append(sax_config.CustomInstrument(
            key=key, transp=transp, name_de=name, name_en=name))
        sax_config.save_customs(customs)
        # Refresh combos and select the new instrument.
        lang = self._get_lang()
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for fk, de, en in instrument_families():
            self._family_combo.addItem(de if lang == 'de' else en, fk)
        self._family_combo.blockSignals(False)
        self.select_family_for_instrument(key)
        self.populate_instrument_combo(select_key=key)

    def _ask_custom_instrument(self) -> tuple[str, int] | None:
        w = self._w
        dlg = QDialog(w)
        dlg.setProperty('_lang_title_key', 'custom_dlg_title')
        dlg.setWindowTitle(self._t('custom_dlg_title'))
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
        info = QLabel(self._t('custom_dlg_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)
        form = QFormLayout()
        form.setSpacing(8)
        edit_name = QLineEdit()
        edit_name.setPlaceholderText('e.g. Eb Mezzo-Soprano')
        name_lbl = QLabel(self._t('custom_lbl_name'))
        form.addRow(name_lbl, edit_name)
        spin_transp = QSpinBox()
        spin_transp.setRange(-36, 36)
        spin_transp.setValue(0)
        transp_lbl = QLabel(self._t('custom_lbl_transp'))
        form.addRow(transp_lbl, spin_transp)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('custom_dlg_info'))
            name_lbl.setText(self._t('custom_lbl_name'))
            transp_lbl.setText(self._t('custom_lbl_transp'))

        # The DE | EN toggle lives on MainWindow because it pokes the
        # main lang combo so all subscribers update too. Reach back via
        # the documented helper rather than duplicating it here.
        w._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        name = edit_name.text().strip()
        if not name:
            QMessageBox.warning(w, self._t('err_title'),
                                self._t('custom_err_name'))
            return None
        return (name, int(spin_transp.value()))

    # ── Autotune ─────────────────────────────────────────────────────────────
    def on_autotune(self) -> None:
        # The autotune flow reads the live stats dict via the getter; the
        # lock that guards it lives on MainWindow because the audio
        # callback's _on_note path holds it too. Reach back for the lock
        # via the documented attribute — the controller does not own a
        # parallel lock.
        w = self._w
        stats = self._get_stats()
        with w._lock:
            items = [(st.mean, st.n) for st in stats.values() if st.n >= 5]
            # v0.6 Phase-4 (Item 6): count progress so the blocker is
            # actionable. ``qualified`` = notes meeting the ≥5 threshold;
            # ``touched`` = distinct notes that have ANY samples. Both are
            # shown to the user when there isn't enough data yet.
            qualified = len(items)
            touched = sum(1 for st in stats.values() if st.n > 0)

        if len(items) < 3:
            msg = (self._t('autotune_nodata') + '\n\n'
                   + self._t('autotune_nodata_progress',
                              qualified=qualified, touched=touched))
            QMessageBox.warning(w, self._t('autotune_title'), msg)
            return

        means   = np.array([m for m, _ in items])
        weights = np.array([math.sqrt(n) for _, n in items])
        order   = np.argsort(means)
        cumw    = np.cumsum(weights[order])
        offset_ct     = float(means[order][np.searchsorted(cumw, cumw[-1] / 2.0)])
        mean_weighted = float(np.sum(means * weights) / np.sum(weights))

        a4_current    = self._engine.a4
        a4_optimal    = a4_current * (2.0 ** (offset_ct / 1200.0))
        a4_rounded    = round(a4_optimal)
        a4_clamped    = max(430, min(450, a4_rounded))

        sign  = '+' if offset_ct    >= 0 else ''
        sign2 = '+' if mean_weighted >= 0 else ''
        msg   = self._t('autotune_result',
                         notes=len(items), sign=sign, offset=offset_ct,
                         sign2=sign2, mean=mean_weighted,
                         current=a4_current, optimal=a4_optimal, rounded=a4_rounded)
        if a4_clamped != a4_rounded:
            msg += self._t('autotune_clamp', rounded=a4_rounded, clamped=a4_clamped)
        msg += self._t('autotune_confirm', clamped=a4_clamped)

        if QMessageBox.question(
            w, self._t('autotune_title'), msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            # set_a4 callback owns the combo+engine update. We block
            # signals at the combo level so _on_a4_changed does NOT fire
            # (the autotune flow does its own stats clear + refresh
            # via on_a4_changed). This matches the v0.5.x behaviour
            # before the extraction.
            self._set_a4(float(a4_clamped))
            with w._lock:
                stats.clear()
            if self._on_a4_changed is not None:
                self._on_a4_changed()

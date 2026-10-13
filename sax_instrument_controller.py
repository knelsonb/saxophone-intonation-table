# =============================================================================
# sax_instrument_controller.py — InstrumentController
#
# Extracted from sax_intonation_gui.py as a MainWindow controller. The
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
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
)

import sax_config
import sax_instruments
from sax_audio_engine import AUDIO_OK
from sax_dialogs import RangeEditorDialog, ask_custom_instrument
from sax_instruments import (
    display_name as instrument_display_name,
    families as instrument_families,
    family_of as instrument_family_of,
    instruments_in,
    register_custom,
    transp_map,
)

# Autotune: a note counts toward the suggestion once it has this many samples,
# and the suggestion needs at least AUTOTUNE_MIN_NOTES such notes.
AUTOTUNE_MIN_SAMPLES = 5
AUTOTUNE_MIN_NOTES = 3
# The A4 combo accepts whole Hz in this range; a suggestion is clamped to it.
A4_MIN_HZ = 430
A4_MAX_HZ = 450


@dataclass(frozen=True)
class A4Suggestion:
    """Outcome of the autotune maths. ``offset_ct`` is the weighted median of
    the per-note mean deviations; ``optimal`` the A4 that would zero it;
    ``rounded`` that value to a whole Hz (half to even); ``clamped`` the result
    limited to the A4 combo range."""
    notes: int
    offset_ct: float
    mean_weighted_ct: float
    optimal: float
    rounded: int
    clamped: int


def suggest_a4(note_stats: Iterable[tuple[float, int]],
               a4_current: float) -> Optional[A4Suggestion]:
    """Suggest an A4 from per-note ``(mean_cents, sample_count)`` pairs, or
    None when fewer than AUTOTUNE_MIN_NOTES notes have AUTOTUNE_MIN_SAMPLES.

    The mean deviations are measured against ``a4_current``, so a player who
    sits sharp of it (positive cents) gets a higher A4 back. The offset is a
    median weighted by sqrt(n), which keeps a few long-held notes from
    outvoting the rest of the scale and resists a single stray note."""
    items = [(m, n) for m, n in note_stats if n >= AUTOTUNE_MIN_SAMPLES]
    if len(items) < AUTOTUNE_MIN_NOTES:
        return None
    means = np.array([m for m, _ in items])
    weights = np.array([math.sqrt(n) for _, n in items])
    order = np.argsort(means)
    cumw = np.cumsum(weights[order])
    offset_ct = float(means[order][np.searchsorted(cumw, cumw[-1] / 2.0)])
    mean_weighted = float(np.sum(means * weights) / np.sum(weights))
    optimal = a4_current * (2.0 ** (offset_ct / 1200.0))
    rounded = round(optimal)
    clamped = max(A4_MIN_HZ, min(A4_MAX_HZ, rounded))
    return A4Suggestion(notes=len(items), offset_ct=offset_ct,
                        mean_weighted_ct=mean_weighted, optimal=optimal,
                        rounded=rounded, clamped=clamped)


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
        if AUDIO_OK and w._recording:
            self._log.start_run(instrument=key,
                                a4_hz=self._engine.a4,
                                label=self._nick_edit.text().strip())
        if self._on_instrument_changed is not None:
            self._on_instrument_changed()

    # ── Nickname ─────────────────────────────────────────────────────────────
    def on_nickname_changed(self) -> None:
        """User finished editing the nickname. A nickname names a horn, so a
        new nickname starts a new run, as an instrument or A4 change does:
        relabelling the current run would move readings already taken on the
        previous horn under the new name. An empty current run is coalesced
        by start_run, so typing a name before playing just relabels it.
        Clearing the nickname opens a run with an empty label."""
        nickname = self._nick_edit.text().strip()
        if not (AUDIO_OK and self._w._recording):
            return
        current = self._log.current_run()
        # editingFinished also fires on a plain focus change; only an actual
        # change of name opens a run.
        if current is not None and current.label == nickname:
            return
        self._log.start_run(instrument=self._w.instrument,
                            a4_hz=self._engine.a4,
                            label=nickname)

    # ── Range editor ─────────────────────────────────────────────────────────
    def open_range_editor(self) -> None:
        """Open the per-instrument range editor for the active instrument.
        Accepts → persist via sax_instruments override DB → notify caller
        so the new bounds take effect immediately."""
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
        transp = transp_map().get(key, 0)
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
        existing_keys = set(transp_map().keys())
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
        # Persist for next session. Strict load: if the existing file can't be
        # read right now, keep the instrument for this session only rather
        # than rewriting the file from an empty list.
        try:
            customs = sax_config.load_customs(strict=True)
        except OSError:
            customs = None
        if customs is not None:
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
        # The DE | EN toggle switches the whole window, so the dialog
        # borrows the window's toggle rather than building its own.
        return ask_custom_instrument(self._w, self._t,
                                     self._w._add_dialog_lang_toggle)

    # ── Autotune ─────────────────────────────────────────────────────────────
    def on_autotune(self) -> None:
        # The live stats dict comes through the getter; like every reader
        # of it, this runs on the Qt thread.
        w = self._w
        stats = self._get_stats()
        note_stats = [(st.mean, st.n) for st in stats.values()]
        # Count progress so the blocker is actionable. ``qualified`` = notes
        # meeting the sample threshold; ``touched`` = distinct notes that have
        # ANY samples. Both are shown to the user when there isn't enough data.
        qualified = sum(1 for _m, n in note_stats if n >= AUTOTUNE_MIN_SAMPLES)
        touched = sum(1 for _m, n in note_stats if n > 0)

        a4_current = self._engine.a4
        suggestion = suggest_a4(note_stats, a4_current)
        if suggestion is None:
            msg = (self._t('autotune_nodata') + '\n\n'
                   + self._t('autotune_nodata_progress',
                              qualified=qualified, touched=touched))
            QMessageBox.warning(w, self._t('autotune_title'), msg)
            return

        offset_ct = suggestion.offset_ct
        mean_weighted = suggestion.mean_weighted_ct
        a4_optimal = suggestion.optimal
        a4_rounded = suggestion.rounded
        a4_clamped = suggestion.clamped

        sign  = '+' if offset_ct    >= 0 else ''
        sign2 = '+' if mean_weighted >= 0 else ''
        msg   = self._t('autotune_result',
                         notes=suggestion.notes, sign=sign, offset=offset_ct,
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
            # Pause detection across the change, as the manual A4 path does,
            # so no reading lands between the stats clear and the new run.
            self._engine.pause_emissions()
            try:
                self._set_a4(float(a4_clamped))
                stats.clear()
                if self._on_a4_changed is not None:
                    self._on_a4_changed()
            finally:
                self._engine.resume_emissions()

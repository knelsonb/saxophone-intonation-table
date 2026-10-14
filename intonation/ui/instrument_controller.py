# =============================================================================
# intonation/ui/instrument_controller.py — InstrumentController
#
# Extracted from intonation/ui/main_window.py as a MainWindow controller. The
# controller owns instrument-family selection, instrument selection,
# the per-instrument range editor, custom-instrument registration, and the
# autotune-from-history flow.
#
# Named instrument_controller to keep it apart from catalog/instruments.py,
# which holds the catalog/transposition data.
#
# Design notes:
#   * The instrument, display mode, A4 and the per-note statistics live on
#     the SessionModel (intonation.core.session); the controller reads and sets them
#     there, so whatever else follows the session (the engine, the drone,
#     the history log's runs) follows a change made here too.
#   * `on_instrument_changed`, `on_range_changed` and `on_a4_changed` are
#     optional notifications fired AFTER the change so the window can
#     refresh the table / range banner / OOR counter without the
#     controller having to know about those concerns.
#   * The controller does NOT save the settings. The MainWindow owns
#     persistence timing (consistent with the SessionStateController /
#     ExportController patterns). It saves custom instruments through the
#     ConfigStore it is given.
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

from intonation.catalog import instruments
from intonation.catalog.instruments import (
    display_name as instrument_display_name,
    families as instrument_families,
    family_of as instrument_family_of,
    instruments_in,
    register_custom,
    transp_map,
)
from intonation.i18n.strings import STRINGS
from intonation.persistence import config
from intonation.ui.dialogs import RangeEditorDialog, ask_custom_instrument

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
    """Owns instrument-family selection, instrument selection, the
    per-instrument range editor, custom-instrument registration, and the
    autotune-from-history flow.

    Fills and follows the two instrument combos it is given (again in the
    new language after a switch) and works on the SessionModel; dialogs
    open over ``parent``."""

    def __init__(self, parent, *,
                 session, store, engine, tr,
                 family_combo, instr_combo, add_lang_toggle,
                 on_instrument_changed=None,
                 on_range_changed=None,
                 on_a4_changed=None) -> None:
        self._parent = parent
        self._session = session
        self._store = store
        self._engine = engine
        self._tr = tr
        self._t = tr.t
        self._family_combo = family_combo
        self._instr_combo = instr_combo
        # The window's DE | EN toggle for its own dialogs.
        self._add_lang_toggle = add_lang_toggle
        # Optional post-change notifications.  `on_instrument_changed`
        # fires after the active instrument *key* changed (combo select);
        # `on_range_changed` fires after the per-instrument fingered
        # range was edited (key unchanged, so MainWindow does NOT reset
        # the wrong-instrument detector).
        self._on_instrument_changed = on_instrument_changed
        self._on_range_changed = on_range_changed
        self._on_a4_changed = on_a4_changed
        family_combo.currentIndexChanged.connect(self.on_family_changed)
        instr_combo.currentIndexChanged.connect(self.on_instr_changed)
        # Instrument names come from the catalog in each language, not from
        # translation keys, so the combos are filled again after a switch.
        tr.language_changed.connect(lambda _lang: self.refill())

    def instrument_label(self, key: str) -> str:
        """The instrument's name as reports and banners show it: the long
        translated name when there is one, else the catalog name."""
        long_key = f'instr_long_{key}'
        if long_key in STRINGS[self._tr.lang]:
            return self._t(long_key)
        return instrument_display_name(key, self._tr.lang)

    # ── Family / sub-instrument combos ───────────────────────────────────────
    def refill(self, select_key: str | None = None) -> None:
        """Fill both combos in the current language and show ``select_key``
        (default: the session's instrument), selecting it when it changed."""
        key = select_key if select_key is not None else self._session.instrument
        combo = self._family_combo
        blocked = combo.blockSignals(True)
        combo.clear()
        lang = self._tr.lang
        for family_key, name_de, name_en in instrument_families():
            combo.addItem(name_de if lang == 'de' else name_en, family_key)
        combo.blockSignals(blocked)
        self.select_family_for_instrument(key)
        self.populate_instrument_combo(select_key=key)

    def on_family_changed(self, _idx) -> None:
        """Family combo changed → repopulate the sub-instrument combo with
        the new family's entries and select the first one."""
        self.populate_instrument_combo(select_key=None)

    def populate_instrument_combo(self, select_key: str | None) -> None:
        family_key = self._family_combo.currentData()
        if family_key is None:
            return
        lang = self._tr.lang
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
        if key == self._session.instrument:
            return
        # The session's followers open a new run, so per-run aggregates stay
        # coherent (empty predecessor runs are coalesced by the log).
        self._session.set_instrument(key)
        if self._on_instrument_changed is not None:
            self._on_instrument_changed()

    # ── Range editor ─────────────────────────────────────────────────────────
    def open_range_editor(self) -> None:
        """Open the per-instrument range editor for the active instrument.
        Accepts → persist via intonation.catalog.instruments override DB → notify caller
        so the new bounds take effect immediately."""
        key = self._session.instrument
        # The editor edits the keyed tier; an override widens the extended
        # tier but never narrows it (see instruments.extended_range).
        cur_lo, cur_hi = instruments.keyed_range(key)
        baked_lo, baked_hi = instruments.baked_keyed_range(key)
        has_baked = instruments.has_baked_range(key)
        name = instrument_display_name(key, self._tr.lang)
        # v0.5.7.1: pass display mode + instrument transposition so the
        # dialog can render values in whichever notation the user is
        # already reading on the matrix. File format stays fingered.
        display = self._session.display
        transp = transp_map().get(key, 0)
        dlg = RangeEditorDialog(
            self._parent, self._t, key, name,
            cur_lo, cur_hi, baked_lo, baked_hi, has_baked,
            display=display, transp=transp,
            baked_extended=(instruments.baked_extended_range(key)
                            if has_baked else None))
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
            existing = [c for c in self._store.load_customs() if c.key == key]
            if existing and existing[0].name_en == name:
                break
            key = f"{base}_{suffix}"
            suffix += 1
        register_custom(key, transp, name, name)
        # Persist for next session. Strict load: if the existing file can't be
        # read right now, keep the instrument for this session only rather
        # than rewriting the file from an empty list.
        try:
            customs = self._store.load_customs(strict=True)
        except OSError:
            customs = None
        if customs is not None:
            customs = [c for c in customs if c.key != key]
            customs.append(config.CustomInstrument(
                key=key, transp=transp, name_de=name, name_en=name))
            self._store.save_customs(customs)
        # Refresh the combos (the new family entry) and select the instrument.
        self.refill(select_key=key)

    def _ask_custom_instrument(self) -> tuple[str, int] | None:
        # The DE | EN toggle switches the whole window, so the dialog
        # borrows the window's toggle rather than building its own.
        return ask_custom_instrument(self._parent, self._t, self._add_lang_toggle)

    # ── Autotune ─────────────────────────────────────────────────────────────
    def on_autotune(self) -> None:
        # Like every reader of the session's stats, this runs on the Qt thread.
        w = self._parent
        session = self._session
        stats = session.stats
        note_stats = [(st.mean, st.n) for st in stats.values()]
        # Count progress so the blocker is actionable. ``qualified`` = notes
        # meeting the sample threshold; ``touched`` = distinct notes that have
        # ANY samples. Both are shown to the user when there isn't enough data.
        qualified = sum(1 for _m, n in note_stats if n >= AUTOTUNE_MIN_SAMPLES)
        touched = sum(1 for _m, n in note_stats if n > 0)

        a4_current = session.a4
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
            # The session's followers move the combos, the engine, the drone
            # and the pitch pipes and open the new run. Unlike a manual A4
            # change, accepting a suggestion starts the table afresh (the
            # stats clear below), refreshed through on_a4_changed.
            # Pause detection across the change, as the manual A4 path does,
            # so no reading lands between the stats clear and the new run.
            self._engine.pause_emissions()
            try:
                session.set_a4(float(a4_clamped))
                session.clear_stats()
                if self._on_a4_changed is not None:
                    self._on_a4_changed()
            finally:
                self._engine.resume_emissions()

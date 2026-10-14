# =============================================================================
# intonation/ui/controls.py — small controls the toolbar and the SETUP tab share
#
# Concert pitch and response appear twice (toolbar and SETUP), so they are
# built here once and the two copies cannot differ in their entries. Both
# follow the value they show rather than being read: see ``show_data``.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QComboBox

from intonation.audio.engine import FILTER_MODE_DEFAULT, FILTER_PRESETS

if TYPE_CHECKING:
    from intonation.i18n.translator import Translator

# The concert pitches the A4 selectors offer (whole Hz).
A4_CHOICES = range(430, 451)

# Mic-gain SETUP control: the offered dB steps.
MIC_GAIN_DB_STEPS = (-12, -6, 0, 6, 12, 18, 24)


def db_to_linear(db: float) -> float:
    """dB gain → linear multiplier (0 dB → 1.0)."""
    return 10.0 ** (float(db) / 20.0)


def show_data(combo: QComboBox, data: Any) -> None:
    """Select the entry holding ``data`` without emitting a change: the combo
    is following a value set elsewhere, not being changed by the player. An
    entry the combo does not offer leaves it as it is."""
    idx = combo.findData(data)
    if idx >= 0 and idx != combo.currentIndex():
        blocked = combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(blocked)


def a4_combo(a4: float) -> QComboBox:
    """An A4 selector ("NNN Hz", language-independent) showing ``a4``."""
    combo = QComboBox()
    for hz in A4_CHOICES:
        combo.addItem(f'{hz} Hz', hz)
    show_data(combo, int(a4))
    return combo


def response_combo(tr: 'Translator', mode: str) -> QComboBox:
    """A Response selector: Fast / Normal / Slow in the current language
    (and every later one), showing ``mode``. The engine ignores an unknown
    mode, so callers pass the mode the engine actually uses."""
    combo = QComboBox()
    tr.bind_items(combo, [('filter_fast', 'fast'),
                          ('filter_normal', 'normal'),
                          ('filter_slow', 'slow')])
    tr.bind(combo.setToolTip, 'filter_tip')
    show_data(combo, mode if mode in FILTER_PRESETS else FILTER_MODE_DEFAULT)
    return combo

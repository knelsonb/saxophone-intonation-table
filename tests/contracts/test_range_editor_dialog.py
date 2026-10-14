"""The range editor's read-only line about the extended range.

The editor edits the guide rows (the keyed tier). Whether a played note counts
as out of range is judged on the extended tier, which an edit can only widen;
the editor states that range so a player who opens it from the
wrong-instrument hint can see the limit that raised it.
"""
from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from intonation.catalog import instruments as si
from intonation.i18n.translator import translate
from intonation.ui.dialogs import RangeEditorDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def editor(key: str, lang: str = "en", display: str = "griff") -> RangeEditorDialog:
    lo, hi = si.baked_keyed_range(key)
    return RangeEditorDialog(
        None, lambda k, **kw: translate(lang, k, **kw), key, key,
        lo, hi, lo, hi, True, display=display,
        transp=si.transp_map()[key],
        baked_extended=si.baked_extended_range(key))


def test_the_alto_editor_names_its_altissimo_limit(qapp):
    dlg = editor("eb_alto")
    assert dlg._extended_lbl.text() == translate(
        "en", "range_extended_note", lo_name="A#/Bb3", hi_name="F7")


def test_an_edit_past_the_extended_range_widens_the_line(qapp):
    dlg = editor("eb_alto")
    dlg._hi_spin.setValue(103)                      # fingered G7
    dlg._lo_spin.setValue(57)                       # fingered A3
    assert dlg._extended_lbl.text() == translate(
        "en", "range_extended_note", lo_name="A3", hi_name="G7")


def test_the_line_follows_the_sounding_display(qapp):
    dlg = editor("eb_alto", lang="de", display="klingend")
    # Fingered Bb3..F7 on an Eb alto sounds Db3..Ab6.
    assert dlg._extended_lbl.text() == translate(
        "de", "range_extended_note", lo_name="C#/Db3", hi_name="G#/Ab6")


def test_without_a_baked_range_the_line_is_hidden(qapp):
    dlg = RangeEditorDialog(None, lambda k, **kw: translate("en", k, **kw),
                            "custom_x", "X", 48, 79, 48, 79, False)
    assert dlg._extended_lbl.isHidden()

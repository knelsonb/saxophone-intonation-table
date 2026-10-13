"""i18n key-coverage / de-en parity net for sax_i18n.STRINGS.

Every translation key must exist in BOTH languages with a non-empty value, so
no control ships English-only. This locks that invariant: any later string
addition that forgets one language, or leaves it blank, fails loudly instead of
silently shipping a half-translated control. Pure Python; runs under system python.
"""
from __future__ import annotations

import sax_i18n as i18n


def test_languages_present():
    langs = set(i18n.available_languages())
    assert {"de", "en"} <= langs, f"expected at least de+en, got {sorted(langs)}"


def test_every_key_present_in_every_language():
    """No key may exist in one language but not another — the i18n sweep
    invariant (no half-translated string)."""
    all_keys = i18n.keys()
    for lang in i18n.available_languages():
        missing = sorted(all_keys - set(i18n.STRINGS[lang]))
        assert not missing, f"language {lang!r} is missing keys: {missing}"


def test_no_empty_translations():
    """Every translation value is a non-empty, non-whitespace string — a blank
    entry is as much a gap as a missing key."""
    for lang, table in i18n.STRINGS.items():
        blanks = sorted(k for k, v in table.items()
                        if not (isinstance(v, str) and v.strip()))
        assert not blanks, f"language {lang!r} has empty/blank values: {blanks}"

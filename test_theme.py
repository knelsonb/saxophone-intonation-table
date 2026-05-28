"""Tests for sax_theme — the Sprint-5 theme palette module.

Pure (no Qt): runs under system python like the other logic nets. Locks the
palette contract (every theme complete), the coerce/active behaviour, the
QSS-builder placeholder resolution, and the DARK == legacy-colours guarantee
that makes routing the app through build_app_qss(DARK) a visual no-op.
"""
from __future__ import annotations

import re
from dataclasses import fields

import pytest

import sax_theme as th

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# Colour roles that the global stylesheet consumes (ok/warn/bad are for status
# dots + cents bars in the painted widgets, not the QSS).
_QSS_ROLES = (
    "window_bg", "base_bg", "alt_bg", "tab_bg", "tab_hover_bg",
    "panel_border", "input_border", "text", "text_dim", "text_bright",
    "accent", "button_bg",
)


@pytest.fixture(autouse=True)
def _restore_active():
    """Several tests flip the module-level active theme; restore it so the
    global never leaks across tests."""
    before = th.active_name()
    yield
    th.set_active(before)


def _colour_roles():
    return [f.name for f in fields(th.ThemePalette) if f.name != "name"]


def test_every_theme_is_complete_and_hex():
    roles = _colour_roles()
    for name, pal in th.THEMES.items():
        assert pal.name == name
        for role in roles:
            val = getattr(pal, role)
            assert isinstance(val, str) and _HEX.match(val), \
                f"{name}.{role}={val!r} is not a #rrggbb hex"


def test_theme_order_covers_all_themes():
    assert set(th.THEME_ORDER) == set(th.THEMES)
    assert len(th.THEME_ORDER) == len(th.THEMES) == 3
    assert th.THEME_ORDER[0] == "dark"  # default anchor first


def test_default_theme_present_and_dark():
    assert th.DEFAULT_THEME == "dark"
    assert th.DEFAULT_THEME in th.THEMES


def test_coerce_theme_name():
    assert th.coerce_theme_name("light") == "light"
    assert th.coerce_theme_name("DARK") == "dark"
    assert th.coerce_theme_name("  Night ") == "night"
    assert th.coerce_theme_name("bogus") == "dark"
    assert th.coerce_theme_name(None) == "dark"
    assert th.coerce_theme_name(123) == "dark"
    assert th.coerce_theme_name("") == "dark"


def test_get_theme():
    assert th.get_theme("night") is th.NIGHT
    assert th.get_theme("light") is th.LIGHT
    assert th.get_theme("nope") is th.DARK


def test_active_roundtrip():
    th.set_active("light")
    assert th.active() is th.LIGHT
    assert th.active_name() == "light"
    th.set_active("bogus")  # coerces to dark
    assert th.active() is th.DARK
    assert th.active_name() == "dark"


def test_dark_matches_legacy_inline_colours():
    # No-op-wiring guarantee: DARK reproduces the pre-Sprint-5 hexes the global
    # stylesheet used, so build_app_qss(DARK) is visually identical to the old
    # hand-written sheet. If someone "improves" a DARK colour, this fails loudly.
    assert th.DARK.window_bg == "#12121a"
    assert th.DARK.base_bg == "#1e1e2e"
    assert th.DARK.alt_bg == "#252535"
    assert th.DARK.tab_bg == "#161620"
    assert th.DARK.tab_hover_bg == "#1f1f2e"
    assert th.DARK.accent == "#6699cc"
    assert th.DARK.button_bg == "#34495e"
    assert th.DARK.button_hover == "#3d566e"
    assert th.DARK.button_pressed == "#2c3e50"
    assert th.DARK.ok == "#2ecc71"
    assert th.DARK.warn == "#c8a020"
    assert th.DARK.bad == "#c0392b"
    assert th.DARK.grid == "#282c3c"
    assert th.DARK.accent_muted == "#2d4a7a"


def test_build_app_qss_resolves_all_placeholders():
    for name, pal in th.THEMES.items():
        qss = th.build_app_qss(pal)
        assert "QMainWindow" in qss
        assert "QComboBox" in qss
        assert "QTabBar::tab" in qss
        for role in _QSS_ROLES:
            assert getattr(pal, role) in qss, f"{name}: {role} missing from qss"
        # no leftover python-format artefacts (f-string fully rendered)
        assert "{p." not in qss
        assert "{self." not in qss


def test_themes_are_visually_distinct():
    # Each theme must have its own window background (sanity: not all the same).
    assert len({p.window_bg for p in th.THEMES.values()}) == 3

    def _lum(hexstr: str) -> int:
        h = hexstr.lstrip("#")
        return int(h[0:2], 16) + int(h[2:4], 16) + int(h[4:6], 16)

    # LIGHT genuinely light; DARK/NIGHT genuinely dark (0..765 summed-RGB).
    assert _lum(th.LIGHT.window_bg) > 600
    assert _lum(th.DARK.window_bg) < 150
    assert _lum(th.NIGHT.window_bg) < 150

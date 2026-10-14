"""Theme palette contract: every theme is complete, the QSS builder resolves
every placeholder, and an unknown name degrades to the default.

Pure Python, no Qt. Name coercion through the config file is covered by
test_config_roundtrip.py; this file covers the palette module itself.
"""
from __future__ import annotations

import re
from dataclasses import fields

import pytest

from intonation.ui import theme as th

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# Colour roles the global stylesheet reads (ok/warn/bad paint status dots and
# cents bars outside the QSS).
_QSS_ROLES = (
    "window_bg", "base_bg", "alt_bg", "tab_bg", "tab_hover_bg",
    "panel_border", "input_border", "text", "text_dim", "text_bright",
    "accent", "button_bg",
)


@pytest.fixture(autouse=True)
def _restore_active_theme():
    before = th.active_name()
    yield
    th.set_active(before)


def _colour_roles() -> list[str]:
    return [f.name for f in fields(th.ThemePalette) if f.name != "name"]


@pytest.mark.parametrize("name", th.THEME_ORDER)
def test_every_theme_defines_every_role_as_hex(name: str) -> None:
    palette = th.THEMES[name]
    assert palette.name == name
    for role in _colour_roles():
        value = getattr(palette, role)
        assert isinstance(value, str) and _HEX.match(value), (
            f"{name}.{role}={value!r} is not a #rrggbb hex")


def test_theme_order_lists_every_theme_once_with_dark_first() -> None:
    assert sorted(th.THEME_ORDER) == sorted(th.THEMES)
    assert len(set(th.THEME_ORDER)) == len(th.THEME_ORDER) == 3
    assert th.THEME_ORDER[0] == th.DEFAULT_THEME == "dark"


@pytest.mark.parametrize("name", th.THEME_ORDER)
def test_build_app_qss_resolves_every_placeholder(name: str) -> None:
    palette = th.THEMES[name]
    qss = th.build_app_qss(palette)
    for selector in ("QMainWindow", "QComboBox", "QTabBar::tab"):
        assert selector in qss
    for role in _QSS_ROLES:
        assert getattr(palette, role) in qss, f"{name}: {role} missing from the QSS"
    assert "{p." not in qss and "{self." not in qss, "an f-string field was left unrendered"


@pytest.mark.parametrize("raw,expected", [
    ("light", "light"), ("DARK", "dark"), ("  Night ", "night"),
    ("bogus", "dark"), ("", "dark"), (None, "dark"), (123, "dark"),
])
def test_coerce_theme_name(raw, expected: str) -> None:
    assert th.coerce_theme_name(raw) == expected


def test_get_theme_falls_back_to_dark() -> None:
    assert th.get_theme("night") is th.NIGHT
    assert th.get_theme("light") is th.LIGHT
    assert th.get_theme("nope") is th.DARK


def test_active_theme_follows_set_active_and_coerces() -> None:
    th.set_active("light")
    assert (th.active() is th.LIGHT, th.active_name()) == (True, "light")
    th.set_active("bogus")
    assert (th.active() is th.DARK, th.active_name()) == (True, "dark")


def test_themes_differ_and_light_is_light() -> None:
    assert len({p.window_bg for p in th.THEMES.values()}) == 3

    def brightness(hexstr: str) -> int:
        h = hexstr.lstrip("#")
        return int(h[0:2], 16) + int(h[2:4], 16) + int(h[4:6], 16)   # 0..765

    assert brightness(th.LIGHT.window_bg) > 600
    assert brightness(th.DARK.window_bg) < 150
    assert brightness(th.NIGHT.window_bg) < 150

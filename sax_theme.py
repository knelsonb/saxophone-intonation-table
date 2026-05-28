"""Theme palettes for the intonation analyzer (Sprint-5 parity feature).

Android (v1.1) ships three themes — ``dark`` / ``night`` / ``light`` — plus a
night-vision tuning emphasis; the desktop was dark-only. This module is the
single source of truth for every themeable colour, so the GUI stops hard-coding
hex literals inline.

Design:

* **Pure data + string building — NO Qt import.** The palette is a frozen
  dataclass of named colour *roles*; the GUI turns a palette into a global
  Qt stylesheet via :func:`build_app_qss`, and the custom-painted widgets
  (tuner needle, cents bars, table, chart) read role colours via
  :func:`active` and wrap them in ``QColor`` themselves. Keeping this module
  Qt-free means it unit-tests in system python like the other logic modules.
* **DARK reproduces the pre-Sprint-5 inline colours** for the roles the global
  stylesheet uses, so routing the app through ``build_app_qss(DARK)`` is a
  visual no-op — the refactor is provably safe before NIGHT/LIGHT are judged
  on appearance.
* **NIGHT** is red-shifted and low-blue (preserves dark-adapted vision while
  tuning in a pit/dark hall); **LIGHT** is dark-on-light with semantic colours
  darkened for contrast on a pale background.

Themes are read/written on the GUI thread only (never the audio callback), so
the module-level "active" pointer needs no lock.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = [
    "ThemePalette",
    "DARK", "NIGHT", "LIGHT",
    "THEMES", "THEME_ORDER", "DEFAULT_THEME",
    "coerce_theme_name", "get_theme",
    "active", "active_name", "set_active",
    "build_app_qss",
]


@dataclass(frozen=True)
class ThemePalette:
    """A complete set of themeable colour roles. Every field is a CSS/Qt hex
    string. Every theme must define every role (enforced by the dataclass +
    test_theme.test_every_theme_is_complete)."""

    name: str

    # -- surfaces (back to front) ------------------------------------------
    window_bg: str        # the main window / default widget background
    base_bg: str          # raised inputs: combos, line edits, list views
    alt_bg: str           # hover / alternate surface
    tab_bg: str           # an unselected nav tab
    tab_hover_bg: str     # a hovered nav tab

    # -- lines --------------------------------------------------------------
    panel_border: str     # group-box / splitter / pane separators
    input_border: str     # combo / edit borders

    # -- text ---------------------------------------------------------------
    text: str             # primary body text
    text_dim: str         # secondary / inactive labels
    text_bright: str      # selected / emphasised text

    # -- accent -------------------------------------------------------------
    accent: str           # selected tab, focus ring, needle, links

    # -- buttons ------------------------------------------------------------
    button_bg: str
    button_hover: str
    button_pressed: str

    # -- semantic (status dots, cents bars) --------------------------------
    ok: str               # in tune / running / success green
    warn: str             # near / opening / caution amber
    bad: str              # out / failed / error red

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# DARK — reproduces the pre-Sprint-5 inline palette for the global-stylesheet
# roles, so build_app_qss(DARK) renders identically to the old hand-written QSS.
# ---------------------------------------------------------------------------
DARK = ThemePalette(
    name="dark",
    window_bg="#12121a",
    base_bg="#1e1e2e",
    alt_bg="#252535",
    tab_bg="#161620",
    tab_hover_bg="#1f1f2e",
    panel_border="#333333",
    input_border="#444444",
    text="#dddddd",
    text_dim="#999999",
    text_bright="#ffffff",
    accent="#6699cc",
    button_bg="#34495e",
    button_hover="#3d566e",
    button_pressed="#2c3e50",
    ok="#2ecc71",
    warn="#c8a020",
    bad="#c0392b",
)

# ---------------------------------------------------------------------------
# NIGHT — warm, red-shifted, minimal blue. For tuning in a darkened hall/pit
# without wrecking dark-adapted vision. Semantic colours stay distinguishable
# but desaturated (bright pure green/blue would defeat the point).
# ---------------------------------------------------------------------------
NIGHT = ThemePalette(
    name="night",
    window_bg="#100808",
    base_bg="#1d1010",
    alt_bg="#281616",
    tab_bg="#160c0c",
    tab_hover_bg="#241414",
    panel_border="#3a2020",
    input_border="#4a2828",
    text="#d6a0a0",
    text_dim="#a06868",
    text_bright="#f0c8c8",
    accent="#d07058",
    button_bg="#3a2222",
    button_hover="#4a2c2c",
    button_pressed="#2c1818",
    ok="#8fa05a",
    warn="#c89030",
    bad="#d04030",
)

# ---------------------------------------------------------------------------
# LIGHT — dark-on-light. Semantic colours darkened so they read on a pale
# background (the bright dark-theme green/amber wash out on white).
# ---------------------------------------------------------------------------
LIGHT = ThemePalette(
    name="light",
    window_bg="#eef0f4",
    base_bg="#ffffff",
    alt_bg="#e0e4ec",
    tab_bg="#dde0e8",
    tab_hover_bg="#e8ebf2",
    panel_border="#c0c4cc",
    input_border="#a8acb6",
    text="#1a1c24",
    text_dim="#5a5e68",
    text_bright="#000000",
    accent="#2c5fa0",
    button_bg="#d4d8e2",
    button_hover="#c4c9d6",
    button_pressed="#b4bac8",
    ok="#1a8a3a",
    warn="#9a6800",
    bad="#c0291b",
)

THEMES: dict[str, ThemePalette] = {p.name: p for p in (DARK, NIGHT, LIGHT)}
# Display/cycle order; DARK first so it stays the default + the cycle anchor.
THEME_ORDER: tuple[str, ...] = ("dark", "night", "light")
DEFAULT_THEME = "dark"


def coerce_theme_name(name) -> str:
    """Normalise an arbitrary stored value to a valid theme name, defaulting to
    DARK. Mirrors the coerce-clamp-default discipline of the sax_config fields."""
    if isinstance(name, str):
        key = name.strip().lower()
        if key in THEMES:
            return key
    return DEFAULT_THEME


def get_theme(name) -> ThemePalette:
    """Return the palette for ``name`` (coerced; DARK on anything unknown)."""
    return THEMES[coerce_theme_name(name)]


# ---------------------------------------------------------------------------
# Active theme — module-level pointer, GUI-thread only (no lock needed).
# ---------------------------------------------------------------------------
_active: ThemePalette = DARK


def active() -> ThemePalette:
    """The currently applied palette. Custom-painted widgets read role colours
    from this each repaint, so a theme switch repaints in the new colours."""
    return _active


def active_name() -> str:
    return _active.name


def set_active(name) -> ThemePalette:
    """Set the active palette by name (coerced). Returns the palette applied."""
    global _active
    _active = get_theme(name)
    return _active


# ---------------------------------------------------------------------------
# Global Qt stylesheet, parameterised by a palette. This is the old hand-written
# MainWindow stylesheet with the hex literals replaced by palette roles, so
# build_app_qss(DARK) == the pre-Sprint-5 stylesheet (modulo whitespace).
# ---------------------------------------------------------------------------
def build_app_qss(p: ThemePalette) -> str:
    """Return the application-wide stylesheet for palette ``p``."""
    return f"""
        QMainWindow,QWidget{{background:{p.window_bg};color:{p.text};}}
        QGroupBox{{border:1px solid {p.panel_border};border-radius:6px;font-size:12px;
                  color:{p.text_dim};margin-top:6px;padding-top:4px;}}
        QGroupBox::title{{subcontrol-origin:margin;left:8px;top:-2px;}}
        QComboBox{{background:{p.base_bg};border:1px solid {p.input_border};border-radius:5px;
                  color:{p.text};padding:4px 8px;font-size:13px;min-height:28px;}}
        QComboBox:hover{{background:{p.alt_bg};border:1px solid {p.accent};}}
        QComboBox::drop-down{{border:none;width:20px;background:{p.base_bg};}}
        QComboBox QAbstractItemView{{background:{p.base_bg};color:{p.text};
                  border:1px solid {p.input_border};outline:0;
                  selection-background-color:{p.button_bg};
                  selection-color:{p.text_bright};}}
        QComboBox QAbstractItemView::item{{background:{p.base_bg};color:{p.text};
                  padding:4px 8px;border:none;}}
        QComboBox QAbstractItemView::item:selected{{background:{p.button_bg};
                  color:{p.text_bright};}}
        QComboBox QAbstractItemView::item:hover{{background:{p.alt_bg};
                  color:{p.text_bright};}}
        QSplitter::handle{{background:{p.panel_border};}}
        QTabWidget::pane{{border:none;border-top:1px solid {p.panel_border};top:-1px;}}
        QTabBar::tab{{background:{p.tab_bg};color:{p.text_dim};font-size:12px;
                     font-weight:bold;letter-spacing:1px;
                     padding:8px 18px;margin-right:2px;
                     border:1px solid {p.panel_border};border-bottom:none;
                     border-top-left-radius:6px;border-top-right-radius:6px;}}
        QTabBar::tab:hover{{background:{p.tab_hover_bg};color:{p.text};}}
        QTabBar::tab:selected{{background:{p.window_bg};color:{p.accent};
                     border-bottom:2px solid {p.accent};}}
    """

"""Frozen-aware bundled-asset path resolution.

The SoundFont (assets/GeneralUser-GS.sf2) is the first asset the app loads at
RUNTIME (the icon is used only at build time / via the .desktop file). In a
PyInstaller onefile build the bundle is unpacked to a temp dir exposed as
``sys._MEIPASS``, so a path relative to ``__file__`` or the CWD — fine in a dev
checkout — does NOT exist in the frozen binary. ``asset_path`` resolves against
``sys._MEIPASS`` when frozen and the module's own directory otherwise, so the
same call works in both contexts.

The PyInstaller spec bundles the whole ``assets/`` directory into an ``assets``
subdir of the bundle, so callers pass the path RELATIVE to the repo root, e.g.
``asset_path('assets', 'GeneralUser-GS.sf2')``.
"""

from __future__ import annotations

import os
import sys


def base_dir() -> str:
    """Root for bundled assets: the PyInstaller unpack dir when frozen, else
    this module's directory (the repo root in a dev checkout)."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def asset_path(*parts: str) -> str:
    """Absolute path to a bundled asset, valid in both dev and frozen builds.

    Example: ``asset_path('assets', 'GeneralUser-GS.sf2')``.
    """
    return os.path.join(base_dir(), *parts)

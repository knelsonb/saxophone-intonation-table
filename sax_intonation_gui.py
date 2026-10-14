"""Deprecated alias of intonation.ui.main_window, removed in the next release.

``python sax_intonation_gui.py`` still starts the app, as ``python -m intonation``.
The alias is explained in legacy_names.py.
"""
import importlib
import sys
import warnings

if __name__ == '__main__':
    from intonation.__main__ import main
    sys.exit(main())

_NEW = "intonation.ui.main_window"
warnings.warn(f"{__name__} is deprecated; import {_NEW}", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = importlib.import_module(_NEW)

"""Deprecated alias of intonation.ui.dialogs, removed in the next release.

The alias is explained in legacy_names.py.
"""
import importlib
import sys
import warnings

_NEW = "intonation.ui.dialogs"
warnings.warn(f"{__name__} is deprecated; import {_NEW}", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = importlib.import_module(_NEW)

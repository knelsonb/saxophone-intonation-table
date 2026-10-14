"""The pre-2.0 ``sax_*`` module names, kept importable for one release.

This is the only test that imports an old name. Every other test imports the
``intonation`` package directly, so this file is what keeps the shims honest
until they are deleted (together with legacy_names.py and this file).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legacy_names import LEGACY_MODULES

_REPO = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _old_names_imported_afresh():
    """Each test imports the shims itself (so their warning fires) and leaves
    sys.modules without the old names, as it found it."""
    saved = {name: sys.modules.pop(name) for name in list(sys.modules)
             if name in LEGACY_MODULES}
    yield
    for name in LEGACY_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_every_old_module_has_a_shim_and_every_shim_is_listed():
    shims = {p.stem for p in _REPO.glob("sax_*.py")}
    assert shims == set(LEGACY_MODULES)


def test_every_new_home_exists_in_the_package():
    for new in LEGACY_MODULES.values():
        assert new.startswith("intonation.")
        assert importlib.util.find_spec(new) is not None, new


@pytest.mark.parametrize("old", sorted(LEGACY_MODULES))
def test_the_old_name_is_the_new_module_object(old):
    """The shim aliases the module itself, so module identity holds: classes,
    enums and module state are shared, and nothing is imported twice."""
    new = importlib.import_module(LEGACY_MODULES[old])
    with pytest.warns(DeprecationWarning, match=LEGACY_MODULES[old]):
        aliased = importlib.import_module(old)
    assert aliased is new
    assert sys.modules[old] is new


def test_a_patch_through_the_old_name_reaches_the_new_one(monkeypatch):
    with pytest.warns(DeprecationWarning):
        import sax_config
    from intonation.persistence import config

    sentinel = object()
    monkeypatch.setattr(sax_config, "TIME_SIG_VALUES", sentinel)
    assert config.TIME_SIG_VALUES is sentinel


@pytest.mark.parametrize("script", ["sax_intonation_gui.py", "main.py"])
def test_the_old_launch_scripts_start_the_package_entry(script):
    """``python sax_intonation_gui.py`` and ``python main.py`` hand over to
    ``intonation.__main__.main``. The entry is replaced by a stub that returns
    an exit code, so no window opens."""
    run = (
        "import runpy, sys\n"
        "import intonation.__main__ as entry\n"
        "entry.main = lambda: 7\n"
        f"runpy.run_path({script!r}, run_name='__main__')\n"
    )
    proc = subprocess.run([sys.executable, "-c", run], cwd=_REPO, capture_output=True,
                          text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(_REPO)})
    assert proc.returncode == 7, proc.stderr

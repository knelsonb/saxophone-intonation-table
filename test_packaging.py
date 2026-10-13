"""Packaging acceptance net — the #1 freeze trap, locked.

The main packaging risk: a PyInstaller onefile must bundle
AND, at runtime, find the 32 MB GeneralUser-GS.sf2 — but a path relative to
__file__/CWD that works in a dev checkout does NOT exist in the frozen binary
(assets unpack to sys._MEIPASS). sax_assets.asset_path closes that. This file:

  * Unit-locks the freeze-trap fix DETERMINISTICALLY (runs on every suite, no
    PyInstaller build): asset_path must resolve under sys._MEIPASS when frozen,
    and under the module dir otherwise.
  * Confirms the SF2 is present in the dev layout asset_path resolves to.
  * Runs the frozen-chain smoke (tools/tsf_pack_smoke.py) as a subprocess
    in dev — the SAME import-tsf -> asset_path -> sfload -> synth -> non-silent
    chain the frozen binary runs — so the runtime chain is locked without a
    multi-minute build. (The actual frozen onefile build stays a CI gate via
    that same script: it checks that the _tinysoundfont extension and the SF2
    are both bundled and that a GM-19 note synthesises non-silent.)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import sax_assets

_REPO = Path(__file__).parent
_SMOKE = _REPO / "tools" / "tsf_pack_smoke.py"
_SF2_PARTS = ("assets", "GeneralUser-GS.sf2")


@pytest.fixture(autouse=True)
def _isolate_frozen_state(monkeypatch):
    """Keep sys.frozen / sys._MEIPASS from leaking either direction across these
    tests.

    A test elsewhere that sets sys.frozen=True and forgets to restore it would
    silently flip asset_path's dev/frozen branch, making the dev-baseline tests
    below (which resolve against the module dir, where the real SF2 lives) fail
    in a confusing, order-dependent way. So force the clean NON-frozen baseline
    these tests assume (the frozen test sets its own values over it) and let
    monkeypatch put back exactly what was there.

    This goes through the same ``monkeypatch`` instance the tests use, so the
    undo order is correct whichever fixture was set up first. A subprocess (the
    pack smoke) is already isolated from parent globals, so this only governs
    the in-process asset_path tests.
    """
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


# ---------------------------------------------------------------------------
# 1. The freeze-trap fix — asset_path resolution (deterministic, no build).
# ---------------------------------------------------------------------------
def test_asset_path_dev_uses_module_dir():
    """In a dev checkout (not frozen) assets resolve relative to the module
    directory (the repo root)."""
    expected = os.path.dirname(os.path.abspath(sax_assets.__file__))
    assert sax_assets.base_dir() == expected
    assert sax_assets.asset_path("assets", "x.sf2") == os.path.join(
        expected, "assets", "x.sf2")


def test_asset_path_frozen_uses_meipass(monkeypatch, tmp_path):
    """THE freeze-trap lock: when sys.frozen + sys._MEIPASS are set (PyInstaller
    onefile at runtime), assets MUST resolve under _MEIPASS — not __file__ /
    CWD, which don't exist in the bundle. This is the exact bug sax_assets
    exists to prevent; assert it directly without needing a real frozen build."""
    meipass = str(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", meipass, raising=False)
    assert sax_assets.base_dir() == meipass, (
        "frozen build must resolve assets under sys._MEIPASS")
    assert sax_assets.asset_path(*_SF2_PARTS) == os.path.join(meipass, *_SF2_PARTS)


def test_asset_path_joins_parts():
    p = sax_assets.asset_path("a", "b", "c.bin")
    assert p.endswith(os.path.join("a", "b", "c.bin"))


# ---------------------------------------------------------------------------
# 2. The bundled SoundFont is actually present (dev layout).
# ---------------------------------------------------------------------------
def test_soundfont_asset_present_and_sized():
    sf2 = sax_assets.asset_path(*_SF2_PARTS)
    assert os.path.exists(sf2), (
        f"GeneralUser-GS.sf2 not found at {sf2} — the drone's SoundFont must "
        f"be in the tree (and bundled via the spec's assets/ datas)")
    size_mb = os.path.getsize(sf2) / 1e6
    assert size_mb > 1.0, f"SF2 looks truncated ({size_mb:.1f} MB)"


# ---------------------------------------------------------------------------
# 3. The runtime SF2-load -> synth chain (the pack smoke, run in dev).
#    Needs tinysoundfont -> venv/CI only.
# ---------------------------------------------------------------------------
def test_pack_smoke_chain_synthesizes_non_silent():
    pytest.importorskip(
        "tinysoundfont",
        reason="tinysoundfont not installed; packaging synth-chain is venv/CI only")
    assert _SMOKE.exists(), f"packaging smoke script missing at {_SMOKE}"
    # The script lives in tools/, so running it directly puts tools/ on sys.path,
    # not the repo root — add the repo root so its `import sax_assets` resolves
    # in dev. (In the frozen binary every module is bundled, so this is dev-only.)
    env = {**os.environ, "PYTHONPATH": str(_REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, str(_SMOKE)],
        capture_output=True, text=True, cwd=str(_REPO), env=env, timeout=120,
    )
    assert proc.returncode == 0, (
        "the SF2-load -> GM-program -> synth -> non-silent chain (the same one "
        f"the frozen binary runs) must pass.\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}")
    assert "PASS" in proc.stdout, f"smoke did not report PASS:\n{proc.stdout}"


# ---------------------------------------------------------------------------
# 4. Every app module ships in the frozen binary (REL-FROZEN-SMOKE).
#    PyInstaller starts from the spec's entry script and follows its imports.
#    An app module it does not reach, and the spec does not list in
#    hiddenimports, is left out of the onefile and the feature that needs it
#    is dead there (the sax_deck-missing-from-spec bug class). The check below
#    follows only imports that run when a module is imported (not those inside
#    a function), so a module the app loads lazily must be declared. No
#    PyInstaller build required.
# ---------------------------------------------------------------------------
def _spec_tree():
    # Parse the spec via AST so a module NAME that only appears in a comment does
    # not count as declared (a regex over the file text would false-pass).
    import ast
    return ast.parse((_REPO / "intonation_analyzer.spec").read_text(encoding="utf-8"))


def _spec_hiddenimports() -> set:
    import ast
    hidden = set()
    for node in ast.walk(_spec_tree()):
        if (isinstance(node, ast.keyword) and node.arg == "hiddenimports"
                and isinstance(node.value, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    hidden.add(elt.value)
    return hidden


def _import_time_imports(module: str) -> set:
    """In-repo modules ``module`` imports at import time: every import
    statement outside a function or class body, including ones under a
    module-level ``if`` or ``try``."""
    import ast
    path = _REPO / f"{module}.py"
    if not path.exists():
        return set()
    found = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Import):
                found.update(a.name for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.add(child.module)
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return {m for m in found if (_REPO / f"{m}.py").exists()}


def _bundled_modules() -> set:
    pending = ["main", *sorted(_spec_hiddenimports())]
    seen: set = set()
    while pending:
        mod = pending.pop()
        if mod in seen:
            continue
        seen.add(mod)
        pending.extend(_import_time_imports(mod) - seen)
    return seen


def test_spec_starts_from_the_launcher():
    import ast
    analysis = next(node for node in ast.walk(_spec_tree())
                    if isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "Analysis")
    scripts = ast.unparse(analysis.args[0])
    assert "'main.py'" in scripts, f"spec entry script is {scripts}, expected main.py"


def test_every_app_module_is_bundled():
    app_modules = {p.stem for p in _REPO.glob("sax_*.py")}
    missing = app_modules - _bundled_modules()
    assert not missing, (
        f"app modules neither imported at import time from main.py nor listed in "
        f"intonation_analyzer.spec hiddenimports: {sorted(missing)} — PyInstaller "
        f"would leave them out of the frozen binary (the sax_deck-missing-from-spec "
        f"bug class)")

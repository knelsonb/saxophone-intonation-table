"""Packaging acceptance net — the #1 freeze trap, locked.

The main packaging risk: a PyInstaller onefile must bundle
AND, at runtime, find the 32 MB GeneralUser-GS.sf2 — but a path relative to
__file__/CWD that works in a dev checkout does NOT exist in the frozen binary
(assets unpack to sys._MEIPASS). intonation.assets.asset_path closes that. This file:

  * Unit-locks the freeze-trap fix DETERMINISTICALLY (runs on every suite, no
    PyInstaller build): asset_path must resolve under sys._MEIPASS when frozen,
    and under the repo root otherwise.
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

from intonation import assets

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
def test_asset_path_dev_uses_the_repo_root():
    """In a dev checkout (not frozen) assets resolve relative to the repo root,
    where assets/ lives — the directory above the intonation package."""
    expected = str(_REPO.resolve())
    assert os.path.dirname(os.path.dirname(os.path.abspath(assets.__file__))) == expected
    assert assets.base_dir() == expected
    assert os.path.isdir(os.path.join(assets.base_dir(), "assets"))
    assert assets.asset_path("assets", "x.sf2") == os.path.join(
        expected, "assets", "x.sf2")


def test_asset_path_frozen_uses_meipass(monkeypatch, tmp_path):
    """THE freeze-trap lock: when sys.frozen + sys._MEIPASS are set (PyInstaller
    onefile at runtime), assets MUST resolve under _MEIPASS — not __file__ /
    CWD, which don't exist in the bundle. This is the exact bug
    intonation.assets exists to prevent; assert it directly without needing a
    real frozen build."""
    meipass = str(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", meipass, raising=False)
    assert assets.base_dir() == meipass, (
        "frozen build must resolve assets under sys._MEIPASS")
    assert assets.asset_path(*_SF2_PARTS) == os.path.join(meipass, *_SF2_PARTS)


def test_asset_path_joins_parts():
    p = assets.asset_path("a", "b", "c.bin")
    assert p.endswith(os.path.join("a", "b", "c.bin"))


# ---------------------------------------------------------------------------
# 2. The bundled SoundFont is actually present (dev layout).
# ---------------------------------------------------------------------------
def test_soundfont_asset_present_and_sized():
    sf2 = assets.asset_path(*_SF2_PARTS)
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
    # not the repo root — add the repo root so its `import intonation.assets` resolves
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
#    is dead there (the deck-missing-from-spec bug class). The check below
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


def _module_path(module: str) -> Path | None:
    """The source file of an in-repo module or package, or None."""
    base = _REPO.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _with_parents(module: str) -> set:
    """Importing ``a.b.c`` runs ``a`` and ``a.b`` first."""
    parts = module.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)}


def _import_time_imports(module: str) -> set:
    """In-repo modules ``module`` imports at import time: every import
    statement outside a function or class body, including ones under a
    module-level ``if`` or ``try``, with the packages each one passes through.
    ``from package import name`` counts ``package.name`` when that is a module."""
    import ast
    path = _module_path(module)
    if path is None:
        return set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Import):
                found.update(a.name for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                if child.level:
                    anchor = package.split(".")[:len(package.split(".")) - child.level + 1]
                    base = ".".join(anchor + ([child.module] if child.module else []))
                found.add(base)
                found.update(f"{base}.{a.name}" for a in child.names)
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return {m for name in found for m in _with_parents(name) if _module_path(m)}


_ENTRY = "intonation.__main__"


def _bundled_modules() -> set:
    pending = [_ENTRY, *_with_parents(_ENTRY), *sorted(_spec_hiddenimports())]
    seen: set = set()
    while pending:
        mod = pending.pop()
        if mod in seen:
            continue
        seen.add(mod)
        pending.extend(_import_time_imports(mod) - seen)
    return seen


def _package_modules() -> set:
    """Every module and package under intonation/, as dotted names."""
    names = set()
    for path in (_REPO / "intonation").rglob("*.py"):
        parts = path.relative_to(_REPO).with_suffix("").parts
        names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def test_spec_starts_from_the_package_entry():
    import ast
    analysis = next(node for node in ast.walk(_spec_tree())
                    if isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "Analysis")
    scripts = ast.unparse(analysis.args[0])
    assert "'intonation'" in scripts and "'__main__.py'" in scripts, (
        f"spec entry script is {scripts}, expected intonation/__main__.py")


def test_the_reachability_walk_sees_the_package():
    """Guards the walk below against going vacuous: from the entry alone it
    must already reach the crash handler through ``from intonation import``,
    and a lazily imported module must not count."""
    reached = _import_time_imports(_ENTRY)
    assert {"intonation", "intonation.crash"} <= reached
    assert "intonation.ui.main_window" not in reached     # imported inside main()
    assert len(_package_modules()) > 40


def test_every_app_module_is_bundled():
    app_modules = _package_modules()
    missing = app_modules - _bundled_modules()
    assert not missing, (
        f"app modules neither imported at import time from intonation/__main__.py "
        f"nor listed in intonation_analyzer.spec hiddenimports: {sorted(missing)} — "
        f"PyInstaller would leave them out of the frozen binary (the "
        f"deck-missing-from-spec bug class)")

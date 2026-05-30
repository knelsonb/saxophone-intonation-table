"""Packaging acceptance net (parity Sprint 3) — the #1 freeze trap, locked.

The flagged risk of the whole parity effort: a PyInstaller onefile must bundle
AND, at runtime, find the 32 MB GeneralUser-GS.sf2 — but a path relative to
__file__/CWD that works in a dev checkout does NOT exist in the frozen binary
(assets unpack to sys._MEIPASS). sax_assets.asset_path closes that. This file:

  * Unit-locks the freeze-trap fix DETERMINISTICALLY (runs on every suite, no
    PyInstaller build): asset_path must resolve under sys._MEIPASS when frozen,
    and under the module dir otherwise.
  * Confirms the SF2 is present in the dev layout asset_path resolves to.
  * Runs Gandalf's frozen-chain smoke (tools/tsf_pack_smoke.py) as a subprocess
    in dev — the SAME import-tsf -> asset_path -> sfload -> synth -> non-silent
    chain the frozen binary runs — so the runtime chain is locked without a
    multi-minute build. (The actual frozen onefile build stays a CI gate via
    that same script, proven 2026-05-28: 108 MB artifact, _tinysoundfont .so +
    SF2 both bundled, GM-19 synth non-silent, exit 0.)
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
def _isolate_frozen_state():
    """Belt-and-suspenders: keep sys.frozen / sys._MEIPASS from leaking either
    direction across these tests.

    No test in the suite mutates these without monkeypatch today (verified by
    grep), so this is defense-in-depth, not a patch for a live leak. But the
    failure mode is nasty: a future test elsewhere that sets sys.frozen=True and
    forgets to restore it would silently flip asset_path's dev/frozen branch,
    making the dev-baseline tests below (which resolve against the module dir,
    where the real SF2 lives) fail in a confusing, order-dependent way. So we:
      1. snapshot whatever is there,
      2. force the clean NON-frozen baseline these tests assume (the frozen test
         monkeypatches over it; monkeypatch restores to this clean baseline),
      3. restore the exact original on teardown so we never leak the other way.
    A subprocess (the pack smoke) is already isolated from parent globals, so
    this only governs the in-process asset_path tests.
    """
    had_frozen, frozen_val = hasattr(sys, "frozen"), getattr(sys, "frozen", None)
    had_meipass, meipass_val = hasattr(sys, "_MEIPASS"), getattr(sys, "_MEIPASS", None)
    if had_frozen:
        del sys.frozen
    if had_meipass:
        del sys._MEIPASS
    try:
        yield
    finally:
        if had_frozen:
            sys.frozen = frozen_val
        elif hasattr(sys, "frozen"):
            del sys.frozen
        if had_meipass:
            sys._MEIPASS = meipass_val
        elif hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS


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
# 3. The runtime SF2-load -> synth chain (Gandalf's smoke, run in dev).
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
# 4. Guard-imported modules must be in the spec hiddenimports (REL-FROZEN-SMOKE).
#    The Sprint 1-4 audio controllers are imported under try/except in the GUI
#    (so the dev app degrades gracefully if one is absent). PyInstaller's static
#    analysis CANNOT see a guarded import, so each must be declared in the spec's
#    hiddenimports or the frozen binary ships that tab INERT — exactly the
#    war-council bug where sax_deck was missing and the whole Deck tab would have
#    been dead (_DeckController=None) in the onefile. This locks it deterministically,
#    no PyInstaller build required.
# ---------------------------------------------------------------------------
def _guard_imported_sax_modules() -> set:
    import ast
    tree = ast.parse((_REPO / "sax_intonation_gui.py").read_text(encoding="utf-8"))
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and (sub.module or "").startswith("sax_"):
                guarded.add(sub.module)
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    if alias.name.startswith("sax_"):
                        guarded.add(alias.name)
    return guarded


def _spec_hiddenimports() -> set:
    # Parse the spec via AST so a module NAME that only appears in a comment does
    # not count as declared (a regex over the file text would false-pass).
    import ast
    tree = ast.parse((_REPO / "intonation_analyzer.spec").read_text(encoding="utf-8"))
    hidden = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.keyword) and node.arg == "hiddenimports"
                and isinstance(node.value, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    hidden.add(elt.value)
    return hidden


def test_guard_imported_modules_are_in_spec_hiddenimports():
    guarded = _guard_imported_sax_modules()
    assert guarded, "expected to find try/except-guarded sax_ imports in the GUI"
    missing = guarded - _spec_hiddenimports()
    assert not missing, (
        f"guard-imported modules NOT in intonation_analyzer.spec hiddenimports: "
        f"{sorted(missing)} — PyInstaller would ship them INERT in the frozen "
        f"binary (the war-council sax_deck-missing bug class)")

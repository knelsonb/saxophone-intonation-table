"""Frozen-packaging smoke for the tinysoundfont drone synth (Sprint 3).

The packaging RISK is whether a PyInstaller onefile bundles, and at runtime
loads, BOTH the compiled tinysoundfont extension AND the 32 MB SoundFont, and
finds the SF2 via the sys._MEIPASS-aware path. This script exercises exactly
that chain — import tinysoundfont, resolve the SF2 with sax_assets.asset_path,
load it, select a GM program, synthesize a note, assert non-silent — and exits
0 on success / 1 on failure. Built as a tiny console onefile with the SAME
bundling the real spec uses (collect tinysoundfont binaries + SF2 datas), it is
the runnable day-one proof that the frozen binary can sound the drone, without
needing the GUI artifact (which can't synth a note non-interactively).
"""
import sys


def main() -> int:
    import numpy as np
    from sax_assets import asset_path, base_dir
    frozen = getattr(sys, 'frozen', False)
    print(f"frozen={frozen} base_dir={base_dir()}")
    try:
        import tinysoundfont as tsf
    except Exception as exc:  # the .so failed to bundle/load
        print(f"FAIL: tinysoundfont import failed in frozen binary: {exc}")
        return 1
    sf2 = asset_path('assets', 'GeneralUser-GS.sf2')
    import os
    if not os.path.exists(sf2):
        print(f"FAIL: SF2 not found at {sf2} (asset_path/_MEIPASS bundling broken)")
        return 1
    print(f"SF2 resolved: {sf2} ({os.path.getsize(sf2)/1e6:.1f} MB)")
    try:
        sy = tsf.Synth()
        sfid = sy.sfload(sf2)
        sy.program_select(0, sfid, 0, 19)   # GM 19 church organ (a drone preset)
        sy.noteon(0, 60, 100)
        mv = sy.generate(4800)              # stereo interleaved float32
        arr = np.frombuffer(mv, dtype=np.float32)
        peak = float(np.max(np.abs(arr)))
    except Exception as exc:
        print(f"FAIL: synth chain raised: {exc}")
        return 1
    print(f"synth: {arr.size} stereo samples, peak={peak:.4f}")
    if peak <= 0.01:
        print("FAIL: synthesized audio is silent")
        return 1
    print("PASS: frozen binary loads the SF2 + synthesizes a GM note (non-silent)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

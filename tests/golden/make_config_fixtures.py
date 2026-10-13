"""Regenerate the config_v*.json fixtures from the real AppConfig of each release.

    python tests/golden/make_config_fixtures.py

For every release tag below this runs that tag's own ``sax_config.py`` (read with
``git show``), builds an ``AppConfig`` holding a plausible user's choices, and
writes ``asdict(config)`` the way the release's ``save_config`` did (2-space
JSON). The files therefore have exactly the keys, key order and value types that
release wrote, which is what an upgrade has to cope with.

The field set did not change between v1.0.0 and v1.2.0 (37 fields); v0.6.2 is
included because it is the last release with a smaller schema (24 fields: no
mic gain, output device, metronome, deck or theme settings).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# One distinct, plausible user per release.
USERS = {
    "v0.6.2": dict(
        welcome_shown=True, persistence_enabled=True, allow_out_of_range=False,
        filter_mode="slow", min_n_visible=3, last_instrument_key="bb_tenor",
        last_nickname="Selmer Mark VI", last_display_mode="klingend", last_a4_hz=443,
        last_lang="de", audio_device_name="Microphone (FIIO DSP Audio)",
        audio_device_host_api="Windows WASAPI", audio_device_samplerate=48000,
        splitter_sizes=[420, 620],
    ),
    "v1.0.0": dict(
        welcome_shown=True, persistence_enabled=False, filter_mode="fast",
        mic_gain_db=6.0, last_instrument_key="eb_alto", last_a4_hz=442,
        last_nickname="Yamaha YAS-62", last_active_tab="metro", last_bpm=96,
        last_time_sig="3/4", click_volume=0.8, theme="night",
        output_device_name="Speakers (Realtek)", output_device_host_api="Windows WASAPI",
        splitter_sizes=[400, 640],
    ),
    "v1.1.0": dict(
        welcome_shown=True, persistence_enabled=True, last_instrument_key="eb_bari",
        last_lang="de", last_display_mode="klingend", last_a4_hz=440,
        # v1.0/v1.1 saved whatever the user typed here; v1.2 clamps it to 0..4.
        matrix_extra_octaves=9, layout_mode_preference="matrix", theme="light",
        last_active_tab="deck", deck_max_seconds=120,
    ),
    "v1.2.0": dict(
        welcome_shown=True, persistence_enabled=True,
        last_instrument_key="custom_mezzo_soprano", last_a4_hz=441,
        last_nickname="Müller Bariton", last_lang="en", min_n_visible=2,
        last_active_tab="setup", mic_gain_db=-6.0, theme="dark",
    ),
}
FILENAME = {"v0.6.2": "config_v0.6.json", "v1.0.0": "config_v1.0.json",
            "v1.1.0": "config_v1.1.json", "v1.2.0": "config_v1.2.json"}


def load_config_module(tag: str):
    src = subprocess.run(["git", "show", f"{tag}:sax_config.py"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    tmp = Path(tempfile.mkdtemp()) / f"sax_config_{tag.replace('.', '_')}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(tmp.stem, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tmp.stem] = mod        # dataclasses needs the module registered
    sys.path.insert(0, str(REPO))      # sax_atomic / sax_theme are imported by it
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    for tag, values in USERS.items():
        cfg = load_config_module(tag).AppConfig(**values)
        out = HERE / FILENAME[tag]
        out.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
        print(f"{tag}: wrote {out.name} ({len(asdict(cfg))} fields)")


if __name__ == "__main__":
    main()

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Intonation Analyzer (onefile, GUI).

Sounddevice ships the portaudio binary inside the wheel as
``_sounddevice_data/``. PyInstaller's stock hook picks up the Python
extension but historically misses the portaudio DLL/.so/.dylib next to it,
which results in a fully-imported sounddevice that crashes the moment an
input stream is started. ``collect_dynamic_libs`` + ``collect_data_files``
explicitly grab everything sounddevice ships."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_dynamic_libs, collect_data_files, collect_submodules,
)

block_cipher = None
HERE = Path(SPECPATH).resolve()

# Icon selection: Windows wants .ico, macOS auto-converts a PNG to .icns at
# build time (PyInstaller handles it via Pillow), Linux ignores the icon
# parameter entirely (Linux apps surface their icon through the .desktop
# file, not the ELF). Passing the wrong format here is silent on macOS/Linux
# in old PyInstaller versions but produces a bundle with no icon.
if sys.platform.startswith('win'):
    ICON_PATH = str(HERE / 'assets' / 'icon.ico')
elif sys.platform == 'darwin':
    ICON_PATH = str(HERE / 'assets' / 'icon.png')
else:
    ICON_PATH = None

sd_binaries = collect_dynamic_libs('sounddevice')
sd_datas = collect_data_files('sounddevice', include_py_files=False)
sd_hidden = collect_submodules('sounddevice')

# tinysoundfont (Sprint 3 drone synth) ships its synth as a compiled extension
# (_tinysoundfont.*.so/.pyd) inside the wheel. It is imported LAZILY (guarded
# try/except) by sax_drone, so PyInstaller's static analysis can miss it —
# collect the binary + submodules + hiddenimport it explicitly. The SoundFont
# itself (assets/GeneralUser-GS.sf2) rides along in the assets/ datas below; the
# app resolves it at runtime via sax_assets.asset_path (sys._MEIPASS-aware).
tsf_binaries = collect_dynamic_libs('tinysoundfont')
tsf_hidden = collect_submodules('tinysoundfont')

a = Analysis(
    [str(HERE / 'sax_intonation_gui.py')],
    pathex=[str(HERE)],
    binaries=sd_binaries + tsf_binaries,
    datas=[
        (str(HERE / 'assets'), 'assets'),
        *sd_datas,
    ],
    hiddenimports=[
        'sax_intonation_log',
        'sax_intonation_chart',
        'sax_instruments',
        'sax_config',
        'sax_flow_layout',
        'sax_audio_engine',
        # Sprint 1-3 audio-output modules. sax_drone / sax_pitch_pipes are
        # guard-imported (try/except), so list them explicitly to be safe.
        'sax_mixer',
        'sax_coordination',
        'sax_metronome',
        'sax_drone',
        'sax_pitch_pipes',
        'sax_assets',
        'tinysoundfont',
        '_sounddevice',
        *sd_hidden,
        *tsf_hidden,
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='intonation-analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=ICON_PATH,
)

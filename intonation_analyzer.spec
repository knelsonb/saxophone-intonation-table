# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Intonation Analyzer (onefile, GUI).

Sounddevice ships the portaudio binary inside the wheel as
``_sounddevice_data/``. PyInstaller's stock hook picks up the Python
extension but historically misses the portaudio DLL/.so/.dylib next to it,
which results in a fully-imported sounddevice that crashes the moment an
input stream is started. ``collect_dynamic_libs`` + ``collect_data_files``
explicitly grab everything sounddevice ships."""

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_dynamic_libs, collect_data_files, collect_submodules,
)

block_cipher = None
HERE = Path(SPECPATH).resolve()

sd_binaries = collect_dynamic_libs('sounddevice')
sd_datas = collect_data_files('sounddevice', include_py_files=False)
sd_hidden = collect_submodules('sounddevice')

a = Analysis(
    [str(HERE / 'sax_intonation_gui.py')],
    pathex=[str(HERE)],
    binaries=sd_binaries,
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
        '_sounddevice',
        *sd_hidden,
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
    icon=str(HERE / 'assets' / 'icon.ico'),
)

# -*- mode: python ; coding: utf-8 -*-

import moviepy, pathlib
import sys, os, importlib.util, re

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs
)

from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable,
    StringStruct, VarFileInfo, VarStruct
)

spec_dir = pathlib.Path(os.path.dirname(sys.argv[0]))
src = spec_dir / "pmveaver.py"
spec = importlib.util.spec_from_file_location("pmveaver", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
VERSION = getattr(mod, "__version__", "0.0.0")

def _vers_tuple(v: str):
    m = re.search(r'(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?', v)
    if not m:
        return (0, 0, 0, 0)
    major, minor, patch, build = m.groups()
    return (int(major), int(minor), int(patch), int(build or 0))

filevers = prodvers = _vers_tuple(VERSION)

versionBlock = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=filevers, prodvers=prodvers, mask=0x3F, flags=0x0,
        OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)
    ),
    kids=[
        StringFileInfo([
            StringTable('040904B0', [
                StringStruct('FileDescription', 'PMVeaver CLI'),
                StringStruct('FileVersion', VERSION),
                StringStruct('ProductName', 'PMVeaver CLI'),
                StringStruct('ProductVersion', VERSION),
                StringStruct('OriginalFilename', 'pmveaver.exe'),
                StringStruct('CompanyName', 'Dazzog'),
                StringStruct('LegalCopyright', 'Copyright © 2025 Dazzog'),
            ])
        ]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])
    ]
)

moviepy_dir = pathlib.Path(moviepy.__file__).parent

# Hidden Imports, die MoviePy/Librosa oft dynamisch laden
hiddenimports = (
    ['moviepy.editor', 'moviepy.video.fx.all', 'moviepy.audio.fx.all', 'proglog'] +
    collect_submodules('imageio') +
    collect_submodules('imageio_ffmpeg') +
    collect_submodules('numpy') +
    collect_submodules('scipy') +
    collect_submodules('numba') +
    collect_submodules('llvmlite') +
    collect_submodules('resampy') +
    collect_submodules('audioread')
)

datas = (
    # MoviePy KOMPLETT als Daten in die EXE legen -> garantiert importierbar
    [(str(moviepy_dir), 'moviepy')] +
    collect_data_files('imageio') +
    collect_data_files('librosa') +
    collect_data_files('resampy') +
    collect_data_files('audioread') +
    collect_data_files('pooch')
)

# SoundFile bringt die libsndfile-1.dll & Co mit
binaries = collect_dynamic_libs('soundfile')

block_cipher = None

a = Analysis(
    ['pmveaver.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest','numpy.f2py.tests','torch','PyQt5', 'PyQt6', 'PySide2', 'PySide'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='pmveaver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='assets/icon_cli.ico',
    version=versionBlock,
)

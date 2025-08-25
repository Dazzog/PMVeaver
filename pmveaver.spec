# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs
)
import moviepy, pathlib

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
    icon='assets/icon.ico',
)

# -*- mode: python ; coding: utf-8 -*-

import pathlib
import sys, os, importlib.util, re

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs
)
from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable,
    StringStruct, VarFileInfo, VarStruct
)

spec_dir = pathlib.Path(os.path.dirname(sys.argv[0]))
src = spec_dir / "pmveaver_gui.py"
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
                StringStruct('FileDescription', 'PMVeaver GUI'),
                StringStruct('FileVersion', VERSION),
                StringStruct('ProductName', 'PMVeaver GUI'),
                StringStruct('ProductVersion', VERSION),
                StringStruct('OriginalFilename', 'pmveaver_gui.exe'),
                StringStruct('CompanyName', 'Dazzog'),
                StringStruct('LegalCopyright', 'Copyright © 2025 Dazzog'),
            ])
        ]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])
    ]
)

try:
    SPEC_DIR = pathlib.Path(__file__).parent.resolve()
except NameError:
    SPEC_DIR = pathlib.Path(os.getcwd()).resolve()

block_cipher = None

a = Analysis(
    ['pmveaver_gui.py'],
    pathex=[],
    binaries=[],
    datas=[(str(SPEC_DIR / 'assets' / 'icon.ico'), 'assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest','numpy.f2py.tests','torch','PyQt5', 'PyQt6', 'PySide2', 'PySide'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='pmveaver_gui',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/icon.ico',
    version=versionBlock,
)
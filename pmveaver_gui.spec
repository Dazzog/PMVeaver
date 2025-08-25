# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

try:
    SPEC_DIR = Path(__file__).parent.resolve()
except NameError:
    SPEC_DIR = Path(os.getcwd()).resolve()

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
)
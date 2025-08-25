# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['montage_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'numpy.tests', 'scipy.tests', 'sklearn.tests',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,   # alles anhängen
    a.zipfiles,
    a.datas,
    name='montage_gui',
    debug=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)
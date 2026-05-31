# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\mzf-data\\demo_data\\22211880124-王蕊-新\\系统代码\\12-app.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\mzf-data\\demo_data\\22211880124-王蕊-新\\系统代码\\best_resnet18_binary.pth', '.'), ('E:\\mzf-data\\demo_data\\22211880124-王蕊-新\\系统代码\\best_resnet18_multi.pth', '.')],
    hiddenimports=['pydicom', 'scipy.ndimage', 'sklearn.metrics', 'tqdm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='COPD_Diagnosis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

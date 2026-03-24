# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = [('flimkit', 'flimkit'), ('flimkit/icon.png', 'flimkit')]
datas += collect_data_files('TKinterModernThemes')
datas += copy_metadata('readchar')
datas += copy_metadata('inquirer')
datas += copy_metadata('blessed')
datas += copy_metadata('tkinterdnd2')
datas += copy_metadata('TKinterModernThemes')


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tkinter', 'tkinter.ttk', 'PIL', 'inquirer', 'readchar', 'blessed', 'tqdm', 'matplotlib', 'numpy', 'cv2', 'opencv_python', 'phasorpy', 'pandas', 'scipy', 'xarray', 'tifffile', 'ptufile', 'tkinterdnd2', 'TKinterModernThemes'],
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
    name='FLIMKit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['flimkit\\icon.ico'],
)

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build recipe for the MimicAI Flask server.

This produces a one-directory application so Flask's templates/static files and
optional local model directories remain available beside the executable.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)
datas = [
    (str(project_root / "templates"), "templates"),
    (str(project_root / "static"), "static"),
]

# These directories contain locally downloaded models and user profiles. They
# are ignored by Git, so include them for local builds when they exist but do
# not make release builds fail when CI does not have them.
for directory in ("kokoro_models", "piper_models", "voices"):
    source = project_root / directory
    if source.is_dir():
        datas.append((str(source), directory))

# Several audio/ML dependencies discover modules and package data at runtime.
# Collect their package data to keep the frozen application self-contained.
hiddenimports = []
binaries = []
for package in ("flask", "flask_cors", "torch", "torchaudio", "transformers", "onnxruntime", "kokoro_onnx"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "IPython", "jupyter", "notebook"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="MimicAI",
    exclude_binaries=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="MimicAI",
)

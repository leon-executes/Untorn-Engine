"""Build Untorn Engine from this checkout. Windows x64 / CPython 3.12.

Copyright 2026 Parham (Leon) Faraji. PolyForm-Noncommercial-1.0.0.
Third-party components retain their respective licenses.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
PACKAGE = DIST / "Untorn Engine"

PROJECT_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-build.txt",
    "build_portable.py",
    "build_exe.cmd",
    ".gitignore",
    ".gitattributes",
]
PROJECT_DIRS = ["source", "licenses", "docs", "tests", "samples", ".github"]


def source_archive():
    archive = DIST / "Untorn-Engine-Source.zip"
    files = [ROOT / name for name in PROJECT_FILES]
    for name in PROJECT_DIRS:
        files.extend(
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, "Untorn-Engine/" + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("Source archive integrity failed")
    archive.with_suffix(".sha256").write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n",
        encoding="ascii",
    )


def seal(package=PACKAGE):
    files = sorted(p for p in package.rglob("*") if p.is_file() and p.name != "package-sha256.json")
    if any("Untorn-output" in p.relative_to(package).parts for p in files):
        raise RuntimeError(
            "Refusing to package a user workspace. Move it outside the release folder first."
        )
    manifest = {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (package / "package-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = DIST / "Untorn-Engine-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [package / "package-sha256.json"]:
            z.write(p, "Untorn Engine/" + p.relative_to(package).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("Archive integrity failed")
        for path, digest in manifest.items():
            if hashlib.sha256(z.read("Untorn Engine/" + path)).hexdigest() != digest:
                raise RuntimeError("Archive hash mismatch: " + path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".sha256").write_text(digest + "  " + archive.name + "\n", encoding="ascii")
    print(f"Archive verified: {archive} ({archive.stat().st_size:,} bytes)", flush=True)


def main():
    if os.name != "nt" or sys.maxsize <= 2**32:
        raise SystemExit("Build on Windows with 64-bit CPython.")
    if (PACKAGE / "Untorn-output").exists():
        raise SystemExit(
            "Release folder contains a user workspace. Move the complete folder before rebuilding."
        )
    if not PACKAGE.resolve().is_relative_to(DIST.resolve()):
        raise SystemExit("Invalid build target")
    BUILD.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    test = subprocess.run(
        [sys.executable, str(ROOT / "tests/test_engine.py")], capture_output=True, text=True
    )
    (BUILD / "source-tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode:
        raise SystemExit("Source checks failed; see build/source-tests.txt")
    version = BUILD / "version.txt"
    version.write_text(
        """VSVersionInfo(ffi=FixedFileInfo(filevers=(2,3,0,0),prodvers=(2,3,0,0),mask=0x3f,flags=0x0,OS=0x40004,fileType=0x1,subtype=0x0,date=(0,0)),kids=[StringFileInfo([StringTable('040904B0',[StringStruct('CompanyName','Parham (Leon) Faraji'),StringStruct('FileDescription','Untorn Engine — high-definition image upscaling'),StringStruct('FileVersion','2.3.0'),StringStruct('ProductName','Untorn Engine'),StringStruct('ProductVersion','2.3.0'),StringStruct('LegalCopyright','Copyright 2026 Parham (Leon) Faraji (@leon-executes)')])]),VarFileInfo([VarStruct('Translation',[1033,1200])])])""",
        encoding="utf-8",
    )
    src = ROOT / "source"
    subprocess.run([sys.executable, str(src / "create_icon.py")], check=True)
    binaries = [(str(src / "bin" / name), "bin") for name in ["sinterp.exe", "untorn_gpu.exe"]]
    spec = f"""a=Analysis([{str(src / "untorn_engine.py")!r}],pathex=[{str(src)!r}],binaries={binaries!r},datas=[({str(src / "untorn.ico")!r},'.')],hiddenimports=[],hookspath=[],hooksconfig={{}},runtime_hooks=[],excludes=['tkinter','matplotlib','scipy','torch','cv2','wx.html2'],noarchive=False)
pyz=PYZ(a.pure)
gui=EXE(pyz,a.scripts,[],exclude_binaries=True,name='Untorn Engine',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,icon={str(src / "untorn.ico")!r},version={str(version)!r})
cli=EXE(pyz,a.scripts,[],exclude_binaries=True,name='UntornCLI',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=True,icon={str(src / "untorn.ico")!r},version={str(version)!r})
coll=COLLECT(gui,cli,a.binaries,a.datas,strip=False,upx=False,name='Untorn Engine')
"""
    path = BUILD / "engine.spec"
    path.write_text(spec, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(DIST),
            "--workpath",
            str(BUILD / "pyinstaller"),
            str(path),
        ],
        check=True,
    )
    for name in PROJECT_DIRS:
        shutil.copytree(
            ROOT / name,
            PACKAGE / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in PROJECT_FILES:
        shutil.copy2(ROOT / name, PACKAGE / name)
    verification = PACKAGE / "verification"
    verification.mkdir(exist_ok=True)
    shutil.copy2(BUILD / "source-tests.txt", verification / "source-tests.txt")
    version_text = subprocess.run(
        [str(PACKAGE / "UntornCLI.exe"), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    if "Untorn Engine 2.3.0" not in version_text:
        raise RuntimeError("Packaged console identity failed")
    (verification / "build-checks.json").write_text(
        json.dumps(
            {
                "version": version_text,
                "source_tests": True,
                "desktop": "native wxWidgets",
                "web_server": False,
                "browser_runtime": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    seal()
    source_archive()


if __name__ == "__main__":
    main()

"""Build the Windows x64 interpolation and OpenCL helpers with MinGW-w64."""

from pathlib import Path
import os
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "source/native"
BUILD = ROOT / "build/native"
OUTPUT = ROOT / "source/bin"


def main():
    if os.name != "nt":
        raise SystemExit("Build the native helpers on Windows x64.")
    gcc, gxx = shutil.which("gcc"), shutil.which("g++")
    if not gcc or not gxx:
        raise SystemExit("Place MinGW-w64 gcc and g++ on PATH.")
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(NATIVE / "contours.tar.gz") as archive:
        archive.extractall(BUILD, filter="data")
    src = BUILD / "sinterp-src"
    sources = ["sinterpcli.c", "sinterp.c", "sset.c", "invmat.c", "svd2x2.c", "pen.c", "basic.c"]
    subprocess.run(
        [
            gcc,
            "-O2",
            "-std=c99",
            "-I" + str(src),
            str(NATIVE / "imageio-bmp.c"),
            *[str(src / s) for s in sources],
            "-o",
            str(OUTPUT / "sinterp.exe"),
            "-static-libgcc",
            "-lm",
        ],
        check=True,
    )
    subprocess.run(
        [
            gxx,
            "-O2",
            "-std=c++17",
            "-static",
            "-static-libgcc",
            "-static-libstdc++",
            str(NATIVE / "untorn_opencl.cpp"),
            "-o",
            str(OUTPUT / "untorn_gpu.exe"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

"""
build_zips.py
=============

Builds the two downloadable ZIP files of the project, verifies them and writes
them to the repository root so they are easy to find:

    CNN_Results_Package.zip     all reports, tables, figures and notebooks
    GAF_CNN_Complete.zip        the whole project folder (code, data, results,
                                notebooks, run output)

Both archives are verified after writing (every entry is read back and checked),
so a corrupted download cannot happen.

Usage:
    python3 build_zips.py [--repo ..] 
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

ZIP_DATE = (2024, 10, 10, 0, 0, 0)          # fixed timestamp -> reproducible archives


def add_file(z, path, arcname):
    """Write one file with a fixed timestamp (smaller and reproducible)."""
    info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with open(path, "rb") as f:
        z.writestr(info, f.read())


def build_results_zip(repo):
    """Every result of the project in one small archive."""
    src = repo / "results"
    out = repo / "CNN_Results_Package.zip"

    wanted = {".pdf", ".docx", ".txt", ".csv", ".png", ".json", ".html", ".ipynb", ".md"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(src.iterdir()):
            if p.is_file() and p.suffix.lower() in wanted and p.suffix.lower() != ".zip":
                add_file(z, p, p.name)
    return out


def build_bundle_zip(repo):
    """The whole GAF_CNN_Complete folder (the nested zips are left out)."""
    src = repo / "GAF_CNN_Complete"
    out = repo / "GAF_CNN_Complete.zip"

    skip_names = {"GAF_CNN_Complete.zip", "CNN_Results_Package.zip", "GAF_CNN_Run_Outputs.zip"}
    skip_dirs = {"__pycache__", ".ipynb_checkpoints"}

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            if p.name in skip_names or any(part in skip_dirs for part in rel.parts):
                continue
            add_file(z, p, f"GAF_CNN_Complete/{rel.as_posix()}")
    return out


def verify(path):
    """Read every entry back; returns (n_files, uncompressed_bytes, sha256)."""
    n, total = 0, 0
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"{path}: corrupt entry {bad}")
        for info in z.infolist():
            if not info.is_dir():
                z.read(info.filename)     # decompress fully -> catches CRC errors
                n += 1
                total += info.file_size
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return n, total, sha


def report(path):
    n, total, sha = verify(path)
    print(f"\n{path.name}")
    print(f"  location    : {path}")
    print(f"  files       : {n}")
    print(f"  size on disk: {path.stat().st_size:,} bytes ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  uncompressed: {total:,} bytes")
    print(f"  sha256      : {sha}")
    print(f"  verified    : OK (every entry read back)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    results_zip = build_results_zip(repo)
    bundle_zip = build_bundle_zip(repo)

    for p in (results_zip, bundle_zip):
        report(p)


if __name__ == "__main__":
    main()

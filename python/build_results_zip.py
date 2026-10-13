"""
build_results_zip.py
====================

Packs the results folder into results/CNN_Results_Package.zip:
every report, table, figure and notebook, so the whole result set can be
downloaded in one file.

Usage:
    python3 build_results_zip.py [--results_dir ../results]
"""

import argparse
import os
import zipfile
from pathlib import Path


def build(results_dir):
    results_dir = Path(results_dir).resolve()
    zip_path = results_dir / "CNN_Results_Package.zip"
    extensions = {".pdf", ".docx", ".txt", ".csv", ".png", ".json", ".html", ".ipynb"}

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(results_dir.iterdir()):
            if p.is_file() and p != zip_path and p.suffix.lower() in extensions:
                z.write(p, arcname=p.name)

    print(f"{zip_path}  ({zip_path.stat().st_size:,} bytes, "
          f"{len(zipfile.ZipFile(zip_path).namelist())} files)")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../results")
    a = ap.parse_args()
    build(a.results_dir)

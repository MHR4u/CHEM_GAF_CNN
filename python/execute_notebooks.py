"""
execute_notebooks.py
====================

Runs every notebook of a folder in place, so the .ipynb files are delivered with
their outputs already inside (tables, plots, download links) and can be read
without running anything.

Usage:
    python3 execute_notebooks.py <folder>            # all notebooks in the folder
    python3 execute_notebooks.py <folder> a.ipynb    # only the named notebooks

Requires the libraries of requirements.txt plus nbclient / ipykernel.
"""

import os
import sys
import glob

import nbformat
from nbclient import NotebookClient


def execute(folder, only=None, timeout=1800):
    paths = sorted(glob.glob(os.path.join(folder, "*.ipynb")))
    if only:
        paths = [p for p in paths if os.path.basename(p) in set(only)]

    done = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": folder}},   # notebooks run inside their own folder
        )
        client.execute()
        nbformat.write(nb, path)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        errors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
        images = [o for c in code_cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        print(f"executed {os.path.basename(path)}: {len(code_cells)} code cells, "
              f"{len(errors)} errors, {len(images)} images")
        done.append(path)
    return done


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = sys.argv[1]
    execute(folder, only=sys.argv[2:] or None)

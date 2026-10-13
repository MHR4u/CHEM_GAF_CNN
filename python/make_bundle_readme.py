"""
make_bundle_readme.py
=====================

Writes GAF_CNN_Complete/README.md, the index of the single folder that holds the
whole project, filled with the real numbers of the gaf_cnn.sh run.

Usage:
    python3 make_bundle_readme.py [--bundle ../GAF_CNN_Complete]
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default or {}


def nb_summary(path):
    """Short description of an executed notebook: cells, images, links."""
    try:
        nb = json.loads(Path(path).read_text())
    except Exception:
        return ""
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    pngs = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    links = sum(1 for c in code for o in c.get("outputs", [])
                if "href" in str(o.get("data", {}).get("text/html", "")))
    return f"{len(code)} code cells, {pngs} figures, {links} download links"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="../GAF_CNN_Complete")
    a = ap.parse_args()
    B = Path(a.bundle).resolve()
    RUN = B / "run_output" / "gaf_cnn_sh"

    hist = read_json(RUN / "training_history.json")
    epochs_run = len(hist.get("accuracy", []))
    best_val_loss = min(hist["val_loss"]) if hist.get("val_loss") else None
    best_epoch = (hist["val_loss"].index(best_val_loss) + 1) if best_val_loss is not None else None
    best_val_acc = (hist["val_accuracy"][hist["val_loss"].index(best_val_loss)] * 100) if best_val_loss is not None else None
    final_train_acc = (hist["accuracy"][-1] * 100) if hist.get("accuracy") else None

    test = read_json(RUN / "test_eval" / "shuffled_summary.json")
    compare = read_json(RUN / "compare_tmp" / "compare_summary.json")

    model_files = sorted(RUN.glob("model/*.keras"))
    model_name = model_files[0].name if model_files else "not found"
    model_size = f"{model_files[0].stat().st_size:,} bytes" if model_files else "-"

    def n_files(p):
        p = Path(p)
        return sum(1 for _ in p.rglob("*") if _.is_file()) if p.exists() else 0

    notebooks = sorted((B / "notebooks").glob("*.ipynb"))
    nb_lines = "\n".join(
        f"| `{nb.name}` | {nb_summary(nb)} |" for nb in notebooks
    )

    def fnum(v, suffix=""):
        return "-" if v is None else f"{v:.2f}{suffix}"

    L = f"""# GAF-CNN — Complete folder

**Everything of this project in one folder**: the code, the data, the results, the notebooks and
the outputs of the `gaf_cnn.sh` run.

Created: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

## What is inside

| Folder | Content |
|---|---|
| `code/` | all sources: `python/` (CNN scripts, report builders, notebook builders), `java/` (GAF image generator) and `run_support/` (the capture layer used during the run) |
| `data/` | the input data: voltammetric CSVs (`data/voltammetry/`) and {n_files(B / 'data' / 'gafimages')} GAF images (`data/gafimages/`) plus the {n_files(B / 'data' / 'voltammetry')} files of raw generator output and source CSVs (`data/voltammetry/`) |
| `results/` | every produced result: final report (PDF/Word/text), complete output file, HTML report, per image predictions, confusion matrices, manifests and the ZIP package |
| `notebooks/` | all notebooks, **already executed**, so every table, figure and link is visible without running anything |
| `run_output/gaf_cnn_sh/` | every output of the `gaf_cnn.sh` run: console log, training history, plots, the trained model and the structured test evaluation |
| `original_model/` | the model that was in the repository before the run (kept, so the earlier reports stay reproducible) |
| `GAF_CNN_Complete.zip` | the whole folder as one archive |

---

## The `gaf_cnn.sh` run

Command that was executed:

```bash
cd python
bash gaf_cnn.sh
# -> python3 ./gaf_cnn.py --base_dir "../data/gafimages/shuffled/" --plot \\
#        --nb_epoch 4000 --batch_size 16 --patience 1000
```

### Training output

| | |
|---|---|
| Epochs run | **{epochs_run}** (of 4000 requested; early stopping on `val_loss` with patience 1000) |
| Best epoch | {best_epoch} (lowest `val_loss`) |
| Best `val_loss` | {best_val_loss:.6f} |
| Validation accuracy at best epoch | {fnum(best_val_acc, ' %')} |
| Final training accuracy | {fnum(final_train_acc, ' %')} |
| Model saved by the run | `{model_name}` ({model_size}) |

### Test output

| | |
|---|---|
| Test split | `data/gafimages/shuffled/Test` ({test.get('n_test_images', '-')} images) |
| Correct / wrong | {test.get('n_correct', '-')} / {test.get('n_wrong', '-')} |
| Accuracy | **{fnum(test.get('accuracy_percent'), ' %')}** |
| Structure of the result | `run_output/gaf_cnn_sh/test_eval/` (predictions, confusion matrix, report, figures) |

---

## Notebooks — one notebook per output of the run

| Notebook | Content |
|---|---|
{nb_lines}

Each notebook is independent (it loads what it needs itself) and every file it mentions can be
downloaded from its last cells.

---

## Windows / Linux note

The notebooks find this folder automatically, so they work wherever the folder is placed.
To run them again: `python3 code/execute_notebooks.py notebooks` (needs the libraries listed in
`code/requirements.txt` plus `nbclient` and `ipykernel`).
"""
    out = B / "README.md"
    out.write_text(L)
    print("written:", out, f"({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

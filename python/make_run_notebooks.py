"""
make_run_notebooks.py
=====================

Turns the outputs of a `gaf_cnn.sh` run into Jupyter notebooks, one notebook per
output, so every result of the run can be read and downloaded from a notebook.

Inputs  (folder produced by the run, default ../run_output/gaf_cnn_sh):
    training_console.txt    full console output of gaf_cnn.sh (stdout + stderr)
    training_history.json   accuracy / loss per epoch (captured by run_support/sitecustomize.py)
    training_curve_*.png    the plots the script made with plt.show() (captured)
    model/*.keras           the model saved by the ModelCheckpoint callback
    test_eval/              structured re-evaluation of the saved model (csv/json)

Outputs (default ../GAF_CNN_Complete/notebooks):
    GAF_CNN_Run_01_Training_Output.ipynb   -> training phase: log, curves, epoch table, model
    GAF_CNN_Run_02_Test_Output.ipynb       -> test phase: predictions, confusion matrix, report
    GAF_CNN_Run_03_All_Outputs.ipynb       -> index of every output + download links

Usage:
    python3 make_run_notebooks.py [--run_dir ../run_output/gaf_cnn_sh] [--out_dir ../GAF_CNN_Complete/notebooks]
"""

import argparse
import json
import os
import re
from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# --------------------------------------------------------------------------- #
#  helpers used while building
# --------------------------------------------------------------------------- #

def clean_log(text):
    """Remove ANSI colours and the carriage-return progress spam."""
    text = ANSI.sub("", text)
    out = []
    for raw_line in text.splitlines():
        line = raw_line
        if "\r" in line:
            line = line.split("\r")[-1]          # keep only the final state of the line
        line = line.replace("\b", "")
        out.append(line.rstrip())
    # drop empty lines that only came from the progress bar
    return "\n".join(l for l in out if l.strip() != "" or True)


def parse_config(log):
    cfg = {}
    for key in ["base_dir", "nb_epochs", "batch_size", "patience"]:
        m = re.search(rf"^{key}:\s*(.+)$", log, re.MULTILINE)
        if m:
            cfg[key] = m.group(1).strip()
    m = re.search(r"Found (\d+) images belonging to (\d+) classes", log)
    if m:
        cfg["train_images"], cfg["classes"] = int(m.group(1)), int(m.group(2))
    found = re.findall(r"Found (\d+) images belonging to (\d+) classes", log)
    if len(found) > 1:
        cfg["val_images"] = int(found[1][0])
    m = re.search(r"LEARNING - (\S+)", log)
    if m:
        cfg["model_file"] = m.group(1)
    m = re.search(r"Number of rows[^\n]*", log)
    m2 = re.search(r"Prediction of (\d+) rows", log)
    if m2:
        cfg["test_images"] = int(m2.group(1))
    return cfg


def test_section(log):
    """The part of the console log written by gaf_cnn.py's test() function."""
    idx = log.find("TESTING WITH -")
    if idx < 0:
        return ""
    start = log.rfind("*****", 0, idx)
    start = log.rfind("*****", 0, start) if start > 0 else idx
    return log[max(start, 0):].strip()


def header_cells(notebook_title, subtitle, n_output="1 of 3"):
    return [
        md(f"# {notebook_title}\n\n"
           f"**{subtitle}**\n\n"
           f"<small>Notebook {n_output} &middot; generated automatically from the output files of the "
           f"`gaf_cnn.sh` run, so it is reproducible: every number below is read from the files "
           f"the run produced.</small>"),
        md("## 0. Setup — imports and paths\n\n"
           "The cell below finds the `GAF_CNN_Complete` folder automatically, so the notebook works "
           "wherever the folder is placed."),
        code('''import json, os, re, sys, glob, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, FileLink, Image as IPImage

%matplotlib inline


def find_bundle_root(start=None):
    """Walk up until we find the folder that contains run_output/gaf_cnn_sh."""
    p = Path(start or os.getcwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "run_output" / "gaf_cnn_sh" / "training_console.txt").exists():
            return cand
        if (cand / "training_console.txt").exists():      # notebook inside run_output
            return cand.parent.parent
    raise FileNotFoundError("Could not find run_output/gaf_cnn_sh - run the notebook from inside "
                            "the GAF_CNN_Complete folder")


BUNDLE = find_bundle_root()
RUN    = BUNDLE / "run_output" / "gaf_cnn_sh"
CODE   = BUNDLE / "code"
NB_DIR = BUNDLE / "notebooks"
DATA   = BUNDLE / "data"
sys.path.insert(0, str(CODE))

# keep line length sane in the tables
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("bundle folder :", BUNDLE)
print("run folder    :", RUN)
print("files in run  :", sorted(p.name for p in RUN.iterdir()))'''),
    ]


# --------------------------------------------------------------------------- #
#  notebook 1 - training output
# --------------------------------------------------------------------------- #

def notebook_training():
    cells = header_cells(
        "GAF-CNN — Training Output (gaf_cnn.sh)",
        "Everything the training phase of `gaf_cnn.sh` produced",
        "1 of 3",
    )

    cells.append(md("""## 1. What was run

The original command line script of the repository:

```bash
cd python
bash gaf_cnn.sh
# -> python3 ./gaf_cnn.py --base_dir "../data/gafimages/shuffled/" --plot \\
#        --nb_epoch 4000 --batch_size 16 --patience 1000
```

The training phase trains the small CNN (`simple_cnn`) on the `shuffled/Train` split and validates on
`shuffled/Val`. The next cells show the configuration it printed and the results it produced."""))

    cells.append(code('''# the console output of the run (colours and progress-bar spam removed)
log = (RUN / "training_console.txt").read_text(errors="replace")
log = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", log)

def clean(text):
    out = []
    for line in text.splitlines():
        if "\\r" in line:
            line = line.split("\\r")[-1]
        out.append(line.replace("\\b", "").rstrip())
    return "\\n".join(out)

log = clean(log)
print("console log:", len(log.splitlines()), "lines,", len(log), "characters")

cfg = {}
for key in ["base_dir", "nb_epochs", "batch_size", "patience"]:
    m = re.search(rf"^{key}:\\s*(.+)$", log, re.MULTILINE)
    if m:
        cfg[key] = m.group(1).strip()
found = re.findall(r"Found (\\d+) images belonging to (\\d+) classes", log)
if found:
    cfg["train_images"], cfg["classes"] = int(found[0][0]), int(found[0][1])
if len(found) > 1:
    cfg["validation_images"] = int(found[1][0])
m = re.search(r"LEARNING - (\\S+)", log)
if m:
    cfg["model_file"] = m.group(1)
m = re.search(r"Prediction of (\\d+) rows", log)
if m:
    cfg["test_images"] = int(m.group(1))

display(pd.DataFrame(cfg.items(), columns=["setting", "value"]).set_index("setting"))'''))

    cells.append(md("## 2. How the training went\n\n"
                    "The run's own history (captured while `model.fit()` ran) gives the exact "
                    "accuracy / loss of every epoch."))

    cells.append(code('''history = json.loads((RUN / "training_history.json").read_text())
epochs_run = len(history["accuracy"])

epochs = pd.DataFrame({
    "epoch": range(1, epochs_run + 1),
    "train_accuracy": history["accuracy"],
    "train_loss": history["loss"],
    "val_accuracy": history["val_accuracy"],
    "val_loss": history["val_loss"],
})
epochs.to_csv(RUN / "training_epochs.csv", index=False)   # full table for download

best_i = int(epochs["val_loss"].idxmin())
best = epochs.loc[best_i]

summary = pd.DataFrame({
    "value": [
        cfg.get("nb_epochs"), epochs_run,
        int(epochs["epoch"].max()) if int(epochs["epoch"].max()) < int(cfg.get("nb_epochs", 0)) else "not needed - all epochs completed",
        best["epoch"], f"{best['val_loss']:.6f}", f"{best['val_accuracy'] * 100:.2f} %",
        f"{epochs.iloc[-1]['train_accuracy'] * 100:.2f} %", f"{epochs.iloc[-1]['train_loss']:.6f}",
        f"{epochs['val_accuracy'].max() * 100:.2f} %",
    ]},
    index=["epochs requested", "epochs actually run", "early stopping",
           "best epoch (lowest val_loss)", "best val_loss", "val_accuracy at best epoch",
           "final train accuracy", "final train loss", "best val_accuracy seen"],
)
display(Markdown("### Training summary"))
display(summary)'''))

    cells.append(code('''# the two figures the script itself plotted (captured from plt.show())
display(Markdown("### Training curves produced by the script (`--plot`)"))
for png in sorted(RUN.glob("training_curve_*.png")):
    display(Markdown(f"`{png.name}`"))
    display(IPImage(filename=str(png), width=560))'''))

    cells.append(code('''# clean redraw from the exact history, plus the table of epochs
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs["epoch"], epochs["train_accuracy"] * 100, label="train accuracy", lw=1)
axes[0].plot(epochs["epoch"], epochs["val_accuracy"] * 100, label="validation accuracy", lw=1)
axes[0].scatter([best["epoch"]], [best["val_accuracy"] * 100], color="red", zorder=5, label="best epoch")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy %"); axes[0].set_title("Accuracy")
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

axes[1].plot(epochs["epoch"], epochs["train_loss"], label="train loss", lw=1)
axes[1].plot(epochs["epoch"], epochs["val_loss"], label="validation loss", lw=1)
axes[1].scatter([best["epoch"]], [best["val_loss"]], color="red", zorder=5, label="best epoch")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss"); axes[1].set_title("Loss")
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
fig.suptitle(f"gaf_cnn.sh training - {epochs_run} epochs", fontsize=12)
fig.tight_layout()
plt.show()

display(Markdown(f"### Epoch table ({epochs_run} epochs)"))
display(Markdown("First 5 epochs:")); display(epochs.head())
display(Markdown("Best epoch (lowest validation loss):")); display(epochs.iloc[max(0, best_i - 2): best_i + 3])
display(Markdown("Last 5 epochs:")); display(epochs.tail())
display(Markdown(f"Complete table also saved as `run_output/gaf_cnn_sh/training_epochs.csv` "
                 f"({epochs_run} rows)."))'''))

    cells.append(md("## 3. The model this training produced\n\n"
                    "`ModelCheckpoint` saved the best model (lowest `val_loss`) to disk."))

    cells.append(code('''import tensorflow as tf
import zipfile
from gaf_cnn import simple_cnn
%matplotlib inline

CLASSES = ["Benzochinone", "Catecolo", "Ferro-Ferri", "Idrochinone", "Idrochinone-Benzochinone"]

model_files = sorted(RUN.glob("model/*.keras")) or sorted(CODE.glob("CNN_*classes_*.keras"))
model_path = model_files[0]
print("model file :", model_path.name)
print("size       :", f"{model_path.stat().st_size:,} bytes")
print("sha256     :", hashlib.sha256(model_path.read_bytes()).hexdigest()[:32], "...")

with zipfile.ZipFile(model_path) as z:
    meta = json.loads(z.read("metadata.json").decode())
print("saved by   :", "keras", meta.get("keras_version"), "on", meta.get("date_saved"))

# rebuild the architecture with the repository's own function, then load the trained weights
trained = simple_cnn(n_classes=len(CLASSES))
trained(tf.zeros((1, 224, 224, 3)))
trained.load_weights(model_path)
print("\\nmodel architecture (rebuild of the repository's simple_cnn()):")
trained.summary()'''))

    cells.append(md("## 4. Files produced by the training phase"))

    cells.append(code('''files = [
    ("Full console output of gaf_cnn.sh", RUN / "training_console.txt"),
    ("Training history (accuracy/loss per epoch)", RUN / "training_history.json"),
    ("Epoch table (CSV)", RUN / "training_epochs.csv"),
    ("Trained model (best val_loss)", model_path),
]
files += [("Training curve (as plotted by the script)", p) for p in sorted(RUN.glob("training_curve_*.png"))]

display(Markdown("### Download"))
for label, p in files:
    if Path(p).exists():
        display(FileLink(str(p), result_html_prefix=f"{label}: "))
    else:
        print(f"{label}: not found ({p})")'''))

    return cells


# --------------------------------------------------------------------------- #
#  notebook 2 - test output
# --------------------------------------------------------------------------- #

def notebook_test():
    cells = header_cells(
        "GAF-CNN — Test Output (gaf_cnn.sh)",
        "Everything the test phase of `gaf_cnn.sh` produced: predictions, confusion matrix, accuracy, report",
        "2 of 3",
    )

    cells.append(md("""## 1. The test output as the script printed it

After training, `gaf_cnn.py` loads the best saved model and runs it on the test split
(`shuffled/Test`, 25 images, no shuffling). Below is that part of the console output, verbatim."""))

    cells.append(code('''raw = (RUN / "training_console.txt").read_text(errors="replace")
raw = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", raw).replace("\\b", "")

idx = raw.find("TESTING WITH -")
start = raw.rfind("*****", 0, idx)
start = raw.rfind("*****", 0, start) if start > 0 else idx
test_raw = raw[max(start, 0):].strip()

print(f"captured test output: {len(test_raw.splitlines())} lines")
print(test_raw)'''))

    cells.append(md("""## 2. The same test, structured

The numbers above are the script's own output. To make them readable (and comparable with the report
files) the saved model is evaluated once more with the repository's evaluation helper
(`gaf_cnn_full_output.run_split`), which uses the same test set, the same generator settings
(`rescale 1/255`, `shuffle=False`, `batch_size=1`) and the same class order. The numbers are identical;
they are just written as tables instead of a printed array."""))

    cells.append(code('''import importlib
import tensorflow as tf
import gaf_cnn_full_output as full_output
from gaf_cnn import simple_cnn
%matplotlib inline

# load the model this run produced (notebooks are independent of each other)
model_path = (sorted(RUN.glob("model/*.keras")) or sorted(CODE.glob("CNN_*classes_*.keras")))[0]
trained = simple_cnn(n_classes=5)
trained(tf.zeros((1, 224, 224, 3)))
trained.load_weights(model_path)
print("model evaluated :", model_path.name, f"({model_path.stat().st_size:,} bytes)")

TEST_DIR = DATA / "gafimages" / "shuffled" / "Test"
EVAL_DIR = RUN / "test_eval"
EVAL_DIR.mkdir(exist_ok=True)

import contextlib, io as _io
with contextlib.redirect_stdout(_io.StringIO()):
    summary = full_output.run_split(trained, str(TEST_DIR), "shuffled", str(EVAL_DIR), str(model_path))

print("test images      :", summary["n_test_images"], "in", len(summary["class_names"]), "classes")
print("correct / wrong  :", summary["n_correct"], "/", summary["n_wrong"])
print("accuracy         :", f"{summary['accuracy_percent']:.2f} %")
print("result files     :", sorted(p.name for p in EVAL_DIR.iterdir()))'''))

    cells.append(md("## 3. Overall result and accuracy per class"))

    cells.append(code('''display(Markdown(f"### Accuracy: **{summary['accuracy_percent']:.2f} %** "
                 f"({summary['n_correct']} of {summary['n_test_images']} test images correct)"))

per_class = pd.DataFrame({
    "class": summary["class_names"],
    "accuracy %": [summary["accuracy_by_class_percent"][c] for c in summary["class_names"]],
}).set_index("class")
display(per_class.round(2))

display(Markdown("### Confusion matrix (rows = true class, columns = predicted class)"))
cm = pd.DataFrame(summary["confusion_matrix"], index=summary["class_names"], columns=summary["class_names"])
display(cm)'''))

    cells.append(md("## 4. Confusion matrix - figure"))

    cells.append(code("""from IPython.display import Image as _Img
png = EVAL_DIR / 'shuffled_confusion_matrix.png'
if png.exists():
    display(_Img(filename=str(png), width=520))
    print('figure saved as:', png.relative_to(BUNDLE).as_posix())
"""))

    cells.append(md("## 5. Classification report (precision / recall / F1)"))

    cells.append(code('''print((EVAL_DIR / "shuffled_classification_report.txt").read_text())'''))

    cells.append(md("## 6. Per image predictions\n\nEvery test image, the class the model predicted and the "
                    "confidence (probability of the winning class). Wrong predictions are shown in red."))

    cells.append(code('''pred = pd.read_csv(EVAL_DIR / "shuffled_predictions.csv")
show = pd.DataFrame({
    "#": range(len(pred)),
    "file": pred["file"],
    "true class": pred["true_class"],
    "predicted class": pred["predicted_class"],
    "confidence": pred["max_confidence"].round(4),
    "result": pred["correct"].astype(str).map({"True": "OK", "False": "WRONG"}),
})

try:
    display(show.style.apply(lambda r: ["background-color: #ffe3e3" if r["result"] == "WRONG" else "" for _ in r], axis=1))
except Exception:
    display(show)'''))

    cells.append(md("## 7. Misclassified images"))

    cells.append(code('''wrong = pred[pred["correct"].astype(str) == "False"]
if len(wrong) == 0:
    display(Markdown("### No misclassified images — every test image was classified correctly."))
else:
    display(Markdown(f"### {len(wrong)} misclassified image(s) out of {len(pred)}"))
    for _, r in wrong.iterrows():
        display(Markdown(f"**{r['file']}** — true `{r['true_class']}`, predicted "
                         f"`{r['predicted_class']}` (confidence {r['max_confidence']:.4f})"))
        hits = glob.glob(str(DATA / "gafimages" / "shuffled" / "Test" / r["true_class"] / r["file"]))
        if hits:
            display(IPImage(filename=hits[0], width=190))
        probs = pd.DataFrame({c: [round(float(r[f"prob_{c}"]), 4)] for c in summary["class_names"]},
                             index=["probability"]).T
        display(probs)'''))

    cells.append(md("## 8. Full probability matrix and downloads"))

    cells.append(code('''display(Markdown("### Probability of each class for every test image"))
probs = pred[["file", "true_class"] + [f"prob_{c}" for c in summary["class_names"]]].copy()
probs.columns = ["file", "true"] + summary["class_names"]
display(probs.round(4))'''))

    cells.append(code('''display(Markdown("### Download"))
for label, p in [
    ("Per image predictions (CSV)", EVAL_DIR / "shuffled_predictions.csv"),
    ("Confusion matrix (CSV)", EVAL_DIR / "shuffled_confusion_matrix.csv"),
    ("Confusion matrix (PNG)", EVAL_DIR / "shuffled_confusion_matrix.png"),
    ("Classification report", EVAL_DIR / "shuffled_classification_report.txt"),
    ("Summary (JSON)", EVAL_DIR / "shuffled_summary.json"),
    ("Full text output", EVAL_DIR / "shuffled_full_output.txt"),
    ("Raw console output of the run", RUN / "training_console.txt"),
]:
    if Path(p).exists():
        display(FileLink(str(p), result_html_prefix=f"{label}: "))
    else:
        print(f"{label}: not found ({p})")'''))

    return cells


# --------------------------------------------------------------------------- #
#  notebook 3 - index of every output
# --------------------------------------------------------------------------- #

def notebook_all():
    cells = header_cells(
        "GAF-CNN — All Outputs of the Run",
        "Index of every file produced by `gaf_cnn.sh`, with download links",
        "3 of 3",
    )

    cells.append(md("""## 1. The run

`gaf_cnn.sh` = training + test of the repository's CNN. It produced the outputs listed below,
each one readable in its own notebook:

| # | Notebook | Covers |
|---|---|---|
| 01 | `GAF_CNN_Run_01_Training_Output.ipynb` | training log, curves, epoch table, trained model |
| 02 | `GAF_CNN_Run_02_Test_Output.ipynb` | predictions, confusion matrix, accuracy, classification report |
| 03 | this notebook | index of all output files + download links |"""))

    cells.append(md("## 2. Every output file"))

    cells.append(code('''rows = []
for p in sorted(RUN.rglob("*")):
    if p.is_file():
        rows.append({
            "output": p.relative_to(RUN).as_posix(),
            "what it is": {
                "training_console.txt": "full console output of gaf_cnn.sh",
                "training_history.json": "accuracy / loss of every epoch",
                "training_epochs.csv": "the same history as a table",
                "training_curve_": "figure plotted by the script (--plot)",
            }.get(p.name, "test / evaluation output" if "test_eval" in p.parts else ""),
            "size": f"{p.stat().st_size:,} B",
        })
outputs = pd.DataFrame(rows)
display(outputs)

display(Markdown(f"**{len(outputs)} output files**, "
                 f"{sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()):,} bytes in total."))'''))

    cells.append(md("## 3. Accuracy of this run, next to the repository's original model\n\n"
                    "This run trained a fresh model. The table compares its test accuracy with the "
                    "model that was already in the repository, on the same test images."))

    cells.append(code('''import importlib
import tensorflow as tf
import gaf_cnn_full_output as lib
from gaf_cnn import simple_cnn
%matplotlib inline

model_path = (sorted(RUN.glob("model/*.keras")) or sorted(CODE.glob("CNN_*classes_*.keras")))[0]


def accuracy_of(path, test_dir):
    """Same evaluation as in notebook 02, only the headline numbers."""
    import contextlib, io as _io
    model = simple_cnn(n_classes=5)
    model(tf.zeros((1, 224, 224, 3)))
    model.load_weights(path)
    with contextlib.redirect_stdout(_io.StringIO()):
        s = lib.run_split(model, str(test_dir), "compare", str(RUN / "compare_tmp"), str(path))
    return s["accuracy_percent"], s["n_correct"], s["n_test_images"]


orig = BUNDLE / "original_model" / "CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras"
test_dir = DATA / "gafimages" / "shuffled" / "Test"
rows = []
new_acc, new_ok, n = accuracy_of(model_path, test_dir)
rows.append({"model": f"this run ({model_path.name})", "test split": "shuffled",
             "correct": f"{new_ok}/{n}", "accuracy %": round(new_acc, 2)})
if orig.exists():
    o_acc, o_ok, _ = accuracy_of(orig, test_dir)
    rows.append({"model": "original model of the repository", "test split": "shuffled",
                 "correct": f"{o_ok}/{n}", "accuracy %": round(o_acc, 2)})
display(pd.DataFrame(rows))'''))

    cells.append(md("## 4. Download everything"))

    cells.append(code('''import zipfile
ZIP = BUNDLE / "GAF_CNN_Run_Outputs.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and p.name != "compare_tmp":
            z.write(p, arcname=p.relative_to(BUNDLE).as_posix())
    for nb in sorted(NB_DIR.glob("*.ipynb")):
        z.write(nb, arcname=f"notebooks/{nb.name}")
print(f"{ZIP.name}: {ZIP.stat().st_size:,} bytes, "
      f"{len(zipfile.ZipFile(ZIP).namelist())} files")

display(Markdown("### Download links"))
for label, p in [
    ("All run outputs (ZIP)", ZIP),
    ("Full console output of gaf_cnn.sh", RUN / "training_console.txt"),
    ("Trained model produced by this run", model_path),
    ("Training notebook", NB_DIR / "GAF_CNN_Run_01_Training_Output.ipynb"),
    ("Test notebook", NB_DIR / "GAF_CNN_Run_02_Test_Output.ipynb"),
    ("Raw results (CSV) of the test", RUN / "test_eval" / "shuffled_predictions.csv"),
]:
    if Path(p).exists():
        display(FileLink(str(p), result_html_prefix=f"{label}: "))
    else:
        print(f"{label}: not found")'''))

    cells.append(md("""---
### Summary

`gaf_cnn.sh` was run end to end on the repository's own data (`data/gafimages/shuffled`).
Training and test outputs are stored in `run_output/gaf_cnn_sh/` and each of them is readable in
these three notebooks. All numbers come from the files the run produced, so the notebooks are
reproducible without retraining.
"""))

    return cells


# --------------------------------------------------------------------------- #

def write_clean_log(run_dir):
    """The raw console log contains ANSI colours and one progress bar per step.
    Write a readable version next to it (same content, one line per epoch)."""
    raw = (run_dir / "training_console.txt").read_text(errors="replace")
    lines = []
    for line in raw.splitlines():
        if "\r" in line:
            line = line.split("\r")[-1]
        line = ANSI.sub("", line).replace("\b", "").rstrip()
        if line.strip():
            lines.append(line)
    out = run_dir / "training_console_clean.txt"
    out.write_text("\n".join(lines) + "\n")
    return out, len(lines)


def build(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path, n_lines = write_clean_log(run_dir)
    print(f"cleaned console log: {clean_path} ({n_lines} lines, "
          f"{clean_path.stat().st_size:,} bytes vs "
          f"{(run_dir / 'training_console.txt').stat().st_size:,} bytes raw)")

    specs = [
        ("GAF_CNN_Run_01_Training_Output.ipynb", notebook_training),
        ("GAF_CNN_Run_02_Test_Output.ipynb", notebook_test),
        ("GAF_CNN_Run_03_All_Outputs.ipynb", notebook_all),
    ]
    for name, fn in specs:
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": name.replace(".ipynb", ""),
        }
        path = out_dir / name
        nbf.write(nb, str(path))
        print("written:", path, f"({len(nb.cells)} cells)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="../run_output/gaf_cnn_sh")
    ap.add_argument("--out_dir", default="../GAF_CNN_Complete/notebooks")
    a = ap.parse_args()
    build(a.run_dir, a.out_dir)

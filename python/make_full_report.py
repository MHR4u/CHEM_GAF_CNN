"""
make_full_report.py
===================

Combines every output produced by gaf_cnn_full_output.py into:

  results/CNN_COMPLETE_OUTPUT.txt        -> ONE single complete text file with
                                            everything the CNN produced on all
                                            splits (report + probabilities +
                                            confusion matrices + metrics + report)
  results/CNN_FULL_REPORT.html           -> self contained HTML report (open in
                                            any browser): metrics, tables,
                                            confusion matrices, sample images
  results/all_generated_images_manifest.csv -> inventory of every image file used
                                            by the CNN (path, split, class, size)
  results/README.md                      -> what each file contains

Usage:
    python3 make_full_report.py --results_dir ../results --data_dir ../data/gafimages
"""

import base64
import csv
import glob
import json
import os
import argparse
from datetime import datetime, timezone

CLASSES = [
    "Benzochinone",
    "Catecolo",
    "Ferro-Ferri",
    "Idrochinone",
    "Idrochinone-Benzochinone",
]
SPLITS = ["shuffled", "concentrations_based"]


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_manifest(data_dir, out_csv):
    """Inventory of every generated image in data/ (CNN input sets + raw GAF output)."""
    rows = []
    data_dir = os.path.abspath(data_dir)
    root = os.path.dirname(data_dir)  # .../data
    for path in sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True)):
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        if parts[0] == "gafimages" and len(parts) >= 4:
            # gafimages/<layout>/<Train|Val|Test>/<class>/<file>
            kind = "CNN input image (gafimages)"
            layout, split, cls = parts[1], parts[2], parts[3]
        elif parts[0] == "voltammetry" and len(parts) >= 5:
            # voltammetry/<class>/GAFData/<sensor>/<file>
            kind = "raw GAF generator output (voltammetry)"
            layout, split, cls = "voltammetry/" + parts[2], "-", parts[1]
        else:
            kind, layout, split, cls = "other", "", "", parts[-2] if len(parts) > 1 else ""
        rows.append(
            {
                "file": parts[-1],
                "class": cls,
                "split": split,
                "layout": layout,
                "kind": kind,
                "size_bytes": os.path.getsize(path),
                "path": os.path.relpath(path, os.getcwd()),
            }
        )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "class", "split", "layout", "kind", "size_bytes", "path"]
        )
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--data_dir", default="../data/gafimages")
    args = ap.parse_args()

    rd = args.results_dir
    summaries = json.load(open(os.path.join(rd, "all_splits_summary.json")))
    manifest_csv = os.path.join(rd, "all_generated_images_manifest.csv")
    rows = build_manifest(args.data_dir, manifest_csv)

    # ---------------------------------------------------------- master txt file
    master_path = os.path.join(rd, "CNN_COMPLETE_OUTPUT.txt")
    with open(master_path, "w") as out:
        out.write("#" * 100 + "\n")
        out.write("#  CNN - COMPLETE GENERATED OUTPUT (all splits, everything in one file)\n")
        out.write("#  Model : CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras\n")
        out.write("#  Built : " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + "\n")
        out.write("#" * 100 + "\n\n")

        out.write("TABLE OF CONTENTS\n")
        out.write("  1. OVERALL RESULTS (all splits)\n")
        for i, s in enumerate(summaries, start=2):
            out.write(f"  {i}. FULL OUTPUT - {s['split'].upper()} split\n")
        out.write("\n" + "#" * 100 + "\n\n")

        out.write("=" * 100 + "\n1. OVERALL RESULTS (all splits)\n" + "=" * 100 + "\n\n")
        out.write(f"{'split':<24}{'images':>8}{'correct':>9}{'wrong':>7}{'accuracy %':>12}\n")
        out.write("-" * 60 + "\n")
        for s in summaries:
            out.write(
                f"{s['split']:<24}{s['n_test_images']:>8}{s['n_correct']:>9}"
                f"{s['n_wrong']:>7}{s['accuracy_percent']:>12.2f}\n"
            )
        out.write("\nAccuracy per class (%):\n")
        out.write(f"{'class':<28}" + "".join(f"{s['split']:>22}" for s in summaries) + "\n")
        out.write("-" * 72 + "\n")
        for c in CLASSES:
            out.write(f"{c:<28}" + "".join(f"{s['accuracy_by_class_percent'][c]:>22.2f}" for s in summaries) + "\n")
        out.write("\n\n")

        for i, s in enumerate(summaries, start=2):
            full_txt = os.path.join(rd, f"{s['split']}_full_output.txt")
            out.write("#" * 100 + "\n")
            out.write(f"# {i}. FULL OUTPUT - {s['split'].upper()} split  ({s['test_dir']})\n")
            out.write("#" * 100 + "\n\n")
            with open(full_txt) as f:
                out.write(f.read())
            out.write("\n\n")

        # ------------------------------------------- inventory of all images
        out.write("#" * 100 + "\n")
        out.write(f"# {len(summaries) + 2}. INVENTORY OF ALL GENERATED IMAGES ({len(rows)} PNG files in data/)\n")
        out.write("#" * 100 + "\n\n")
        counts = {}
        for r in rows:
            counts.setdefault((r["kind"], r["layout"], r["split"], r["class"]), 0)
            counts[(r["kind"], r["layout"], r["split"], r["class"])] += 1
        out.write(f"{'kind':<38}{'layout':<24}{'split':<8}{'class':<28}{'files':>6}\n")
        out.write("-" * 104 + "\n")
        for (kind, layout, split, cls), n in sorted(counts.items()):
            out.write(f"{kind:<38}{layout:<24}{split:<8}{cls:<28}{n:>6}\n")
        out.write(f"\nTOTAL: {len(rows)} images\n\n")
        out.write("Complete list (also available as all_generated_images_manifest.csv):\n\n")
        for r in rows:
            out.write(f"  [{r['kind'][:4]}] {r['path']}\n")
        out.write("\nEnd of complete CNN output\n")

    # ------------------------------------------------------------- HTML report
    html_path = os.path.join(rd, "CNN_FULL_REPORT.html")
    sample_imgs = []
    for cls in CLASSES:
        found = sorted(glob.glob(os.path.join(args.data_dir, "shuffled", "Test", cls, "*.png")))
        if found:
            sample_imgs.append((cls, found[0]))

    with open(html_path, "w") as h:
        h.write("""<!doctype html><html><head><meta charset="utf-8">
<title>GAF CNN - complete output</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}
 header{background:#12304d;color:#fff;padding:22px 28px}
 header h1{margin:0 0 6px;font-size:22px}
 header p{margin:0;opacity:.85;font-size:13px}
 main{max-width:1150px;margin:0 auto;padding:24px 18px 60px}
 section{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:18px 20px;margin:0 0 18px}
 h2{font-size:17px;margin:0 0 12px;border-bottom:1px solid #eef1f4;padding-bottom:8px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border:1px solid #e4e8ec;padding:6px 9px;text-align:left}
 th{background:#f1f4f7}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 .ok{color:#0a7a3d;font-weight:600}.bad{color:#b3261e;font-weight:600}
 .kpis{display:flex;gap:14px;flex-wrap:wrap}
 .kpi{flex:1 1 170px;background:#f8fafc;border:1px solid #e4e8ec;border-radius:8px;padding:12px 14px}
 .kpi b{display:block;font-size:24px;margin-top:4px}
 .imgs{display:flex;flex-wrap:wrap;gap:12px}
 .imgs figure{margin:0;text-align:center;font-size:12px}
 .imgs img{border:1px solid #dfe3e7;border-radius:6px;background:#fff}
 pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:12px;max-height:420px}
 code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:12.5px}
 .muted{color:#5b6570;font-size:12.5px}
</style></head><body>
<header>
  <h1>GAF CNN &ndash; complete generated output</h1>
  <p>Model: CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras &nbsp;|&nbsp; 5 classes &nbsp;|&nbsp; inputs 224x224x3 from GAF images (100x100 RGB on disk)</p>
</header><main>
""")
        # KPIs
        h.write('<section><h2>1. Overall results</h2><div class="kpis">')
        for s in summaries:
            h.write(
                f'<div class="kpi"><span class="muted">{s["split"]} split</span>'
                f'<b>{s["accuracy_percent"]:.1f}%</b>'
                f'<span class="muted">{s["n_correct"]}/{s["n_test_images"]} images correct</span></div>'
            )
        h.write(
            f'<div class="kpi"><span class="muted">model parameters</span><b>{summaries[0]["model_params"]:,}</b>'
            f'<span class="muted">trained weights loaded</span></div>'
        )
        h.write("</div></section>")

        # per class accuracy
        h.write('<section><h2>2. Accuracy per class (%)</h2><table><tr><th>class</th>')
        for s in summaries:
            h.write(f'<th>{s["split"]}</th>')
        h.write("</tr>")
        for c in CLASSES:
            h.write(f"<tr><td>{c}</td>")
            for s in summaries:
                v = s["accuracy_by_class_percent"][c]
                cls = "ok" if v == 100 else ("bad" if v < 90 else "")
                h.write(f'<td class="num {cls}">{v:.2f}</td>')
            h.write("</tr>")
        h.write("</table></section>")

        # confusion matrices
        h.write('<section><h2>3. Confusion matrices</h2>')
        for s in summaries:
            png = os.path.join(rd, f"{s['split']}_confusion_matrix.png")
            h.write(
                f'<p class="muted">{s["split"]} split &ndash; rows = true class, columns = predicted class</p>'
                f'<img src="data:image/png;base64,{b64_image(png)}" style="max-width:640px;width:100%">'
            )
        h.write("</section>")

        # per image predictions
        for s in summaries:
            h.write(
                f'<section><h2>4. Per image predictions &ndash; {s["split"]} split</h2>'
                f'<table><tr><th>#</th><th>file</th><th>true class</th><th>predicted class</th>'
                f'<th>confidence</th><th>result</th></tr>'
            )
            with open(os.path.join(rd, f"{s['split']}_predictions.csv")) as f:
                for i, row in enumerate(csv.DictReader(f)):
                    good = row["correct"] == "True"
                    h.write(
                        f'<tr><td class="num">{i}</td><td>{row["file"]}</td><td>{row["true_class"]}</td>'
                        f'<td>{row["predicted_class"]}</td><td class="num">{float(row["max_confidence"]):.4f}</td>'
                        f'<td class="{"ok" if good else "bad"}">{"OK" if good else "WRONG"}</td></tr>'
                    )
            h.write("</table></section>")

        # sample images
        h.write('<section><h2>5. Sample images fed to the CNN (one per class)</h2><div class="imgs">')
        for cls, path in sample_imgs:
            h.write(
                f'<figure><img src="data:image/png;base64,{b64_image(path)}" width="150">'
                f'<figcaption>{cls}<br><span class="muted">{os.path.basename(path)}</span></figcaption></figure>'
            )
        h.write("</div></section>")

        # architecture
        with open(os.path.join(rd, f"{summaries[0]['split']}_full_output.txt")) as f:
            txt = f.read()
        start = txt.find("--- MODEL ARCHITECTURE")
        end = txt.find("--- IMAGES FED TO THE CNN")
        arch = txt[start:end].strip() if start >= 0 else ""
        h.write(f"<section><h2>6. Model architecture</h2><pre>{arch}</pre></section>")
        h.write(
            '<section><h2>7. Files</h2><p class="muted">Complete text output: '
            '<code>results/CNN_COMPLETE_OUTPUT.txt</code> &middot; per split raw output: '
            '<code>&lt;split&gt;_full_output.txt</code> &middot; probabilities: '
            '<code>&lt;split&gt;_predictions.csv</code> &middot; machine readable: '
            '<code>&lt;split&gt;_summary.json</code> and <code>all_splits_summary.json</code></p></section>'
        )
        h.write("</main></body></html>")

    # ------------------------------------------------------------------ README
    with open(os.path.join(rd, "README.md"), "w") as f:
        f.write("# CNN - complete generated output\n\n")
        f.write(f"Model: `CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras` "
                f"({summaries[0]['model_params']:,} parameters, Keras 3.3.3, saved 2024-10-10)\n\n")
        f.write("| File | What it contains |\n|---|---|\n")
        f.write("| `CNN_COMPLETE_OUTPUT.txt` | **Everything in one file**: overall results + full output of every split (architecture, all image probabilities, per image results, confusion matrices, accuracy, classification report, misclassified images) |\n")
        for s in summaries:
            n = s["split"]
            acc = s["accuracy_percent"]
            f.write(f"| `{n}_full_output.txt` | Full text output for the {n} split (accuracy {acc:.2f}%) |\n")
            f.write(f"| `{n}_predictions.csv` | One row per test image: true class, predicted class, probability of each of the 5 classes, confidence, correct/wrong |\n")
            f.write(f"| `{n}_confusion_matrix.csv` | Confusion matrix ({n}) |\n")
            f.write(f"| `{n}_confusion_matrix.png` | Confusion matrix image ({n}) |\n")
            f.write(f"| `{n}_classification_report.txt` | sklearn precision/recall/f1 report ({n}) |\n")
            f.write(f"| `{n}_summary.json` | Machine readable summary ({n}) |\n")
        f.write("| `all_splits_summary.json` | Summaries of all splits in one JSON |\n")
        f.write("| `CNN_FULL_REPORT.html` | Self contained HTML report (open in browser) |\n")
        f.write(
            f"| `all_generated_images_manifest.csv` | Inventory of all {len(rows)} generated PNG images in `data/` "
            "(kind, file, class, split, layout, size, path): the CNN input sets and the raw GAF generator output |\n\n"
        )
        f.write("## How it was produced\n\n```bash\n")
        f.write("# 1) run the trained CNN and capture every output into files\n")
        f.write("cd python\npython3 gaf_cnn_full_output.py \\\n")
        f.write("  --model CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras \\\n")
        f.write("  --splits ../data/gafimages/shuffled/Test ../data/gafimages/concentrations_based/Test \\\n")
        f.write("  --out_dir ../results\n\n")
        f.write("# 2) build the single complete file + html report\n")
        f.write("python3 make_full_report.py --results_dir ../results --data_dir ../data/gafimages\n```\n")
        f.write("\nThe original `gaf_cnn.py` only printed these values to the screen; "
                "`gaf_cnn_full_output.py` writes every one of them to disk, unchanged, "
                "using the same architecture (`simple_cnn`) and the same test procedure.\n")

    print("Wrote:")
    for p in [master_path, html_path, manifest_csv, os.path.join(rd, "README.md")]:
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()

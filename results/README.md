# CNN - complete generated output

Model: `CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras` (20,641 parameters, Keras 3.3.3, saved 2024-10-10)

| File | What it contains |
|---|---|
| `CNN_COMPLETE_OUTPUT.txt` | **Everything in one file**: overall results + full output of every split (architecture, all image probabilities, per image results, confusion matrices, accuracy, classification report, misclassified images) |
| `shuffled_full_output.txt` | Full text output for the shuffled split (accuracy 96.00%) |
| `shuffled_predictions.csv` | One row per test image: true class, predicted class, probability of each of the 5 classes, confidence, correct/wrong |
| `shuffled_confusion_matrix.csv` | Confusion matrix (shuffled) |
| `shuffled_confusion_matrix.png` | Confusion matrix image (shuffled) |
| `shuffled_classification_report.txt` | sklearn precision/recall/f1 report (shuffled) |
| `shuffled_summary.json` | Machine readable summary (shuffled) |
| `concentrations_based_full_output.txt` | Full text output for the concentrations_based split (accuracy 100.00%) |
| `concentrations_based_predictions.csv` | One row per test image: true class, predicted class, probability of each of the 5 classes, confidence, correct/wrong |
| `concentrations_based_confusion_matrix.csv` | Confusion matrix (concentrations_based) |
| `concentrations_based_confusion_matrix.png` | Confusion matrix image (concentrations_based) |
| `concentrations_based_classification_report.txt` | sklearn precision/recall/f1 report (concentrations_based) |
| `concentrations_based_summary.json` | Machine readable summary (concentrations_based) |
| `all_splits_summary.json` | Summaries of all splits in one JSON |
| `CNN_FULL_REPORT.html` | Self contained HTML report (open in browser) |
| `all_generated_images_manifest.csv` | Inventory of all 264 generated PNG images in `data/` (kind, file, class, split, layout, size, path): the CNN input sets and the raw GAF generator output |

## How it was produced

```bash
# 1) run the trained CNN and capture every output into files
cd python
python3 gaf_cnn_full_output.py \
  --model CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras \
  --splits ../data/gafimages/shuffled/Test ../data/gafimages/concentrations_based/Test \
  --out_dir ../results

# 2) build the single complete file + html report
python3 make_full_report.py --results_dir ../results --data_dir ../data/gafimages
```

The original `gaf_cnn.py` only printed these values to the screen; `gaf_cnn_full_output.py` writes every one of them to disk, unchanged, using the same architecture (`simple_cnn`) and the same test procedure.

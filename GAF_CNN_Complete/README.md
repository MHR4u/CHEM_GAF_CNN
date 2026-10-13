# GAF-CNN — Complete folder

**Everything of this project in one folder**: the code, the data, the results, the notebooks and
the outputs of the `gaf_cnn.sh` run.

Created: 2026-10-13 15:05:00 UTC

---

## What is inside

| Folder | Content |
|---|---|
| `code/` | all sources: `python/` (CNN scripts, report builders, notebook builders), `java/` (GAF image generator) and `run_support/` (the capture layer used during the run) |
| `data/` | the input data: voltammetric CSVs (`data/voltammetry/`) and 189 GAF images (`data/gafimages/`) plus the 80 files of raw generator output and source CSVs (`data/voltammetry/`) |
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
# -> python3 ./gaf_cnn.py --base_dir "../data/gafimages/shuffled/" --plot \
#        --nb_epoch 4000 --batch_size 16 --patience 1000
```

### Training output

| | |
|---|---|
| Epochs run | **2418** (of 4000 requested; early stopping on `val_loss` with patience 1000) |
| Best epoch | 1418 (lowest `val_loss`) |
| Best `val_loss` | 0.024688 |
| Validation accuracy at best epoch | 100.00 % |
| Final training accuracy | 89.86 % |
| Model saved by the run | `CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras` (230,098 bytes) |

### Test output

| | |
|---|---|
| Test split | `data/gafimages/shuffled/Test` (25 images) |
| Correct / wrong | 23 / 2 |
| Accuracy | **92.00 %** |
| Structure of the result | `run_output/gaf_cnn_sh/test_eval/` (predictions, confusion matrix, report, figures) |

---

## Notebooks — one notebook per output of the run

| Notebook | Content |
|---|---|
| `GAF_CNN_Final_Report.ipynb` | 12 code cells, 3 figures, 9 download links |
| `GAF_CNN_Run_01_Training_Output.ipynb` | 7 code cells, 3 figures, 6 download links |
| `GAF_CNN_Run_02_Test_Output.ipynb` | 10 code cells, 3 figures, 7 download links |
| `GAF_CNN_Run_03_All_Outputs.ipynb` | 4 code cells, 0 figures, 6 download links |

Each notebook is independent (it loads what it needs itself) and every file it mentions can be
downloaded from its last cells.

---

## Windows / Linux note

The notebooks find this folder automatically, so they work wherever the folder is placed.
To run them again: `python3 code/execute_notebooks.py notebooks` (needs the libraries listed in
`code/requirements.txt` plus `nbclient` and `ipykernel`).

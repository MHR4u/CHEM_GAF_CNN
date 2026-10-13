#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# finalize_run.sh - runs after gaf_cnn.sh has finished.
# It collects every output of the run, builds the notebooks and packages the
# single GAF_CNN_Complete folder.
# ---------------------------------------------------------------------------
set -e
REPO=/home/user/CHEM_GAF_CNN
BUNDLE=$REPO/GAF_CNN_Complete
RUN_SRC=$REPO/run_output/gaf_cnn_sh
RUN_DST=$BUNDLE/run_output/gaf_cnn_sh
PY=/home/user/.venv/bin/python
MODEL=CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras

echo "== 1. collect the run output =="
mkdir -p "$RUN_DST/model"
cp -f "$RUN_SRC"/training_console.txt "$RUN_SRC"/training_history.json "$RUN_DST"/ 2>/dev/null || true
cp -f "$RUN_SRC"/training_curve_*.png "$RUN_DST"/ 2>/dev/null || true
# the model the run trained (written by the ModelCheckpoint callback into python/)
cp -f "$REPO/python/$MODEL" "$RUN_DST/model/$MODEL"
echo "   model produced by the run: $(stat -c%s "$RUN_DST/model/$MODEL") bytes"
ls -la "$RUN_DST"

echo "== 2. keep the repository consistent =="
# the reports and notebooks delivered earlier were made with the model that was
# in the repository before this run -> put that one back in python/
cp -f "$BUNDLE/original_model/$MODEL" "$REPO/python/$MODEL"
echo "   python/$MODEL restored to the original (pre-run) model"

echo "== 3. build the notebooks from the run output =="
cd "$REPO/python"
$PY make_run_notebooks.py --run_dir "$RUN_DST" --out_dir "$BUNDLE/notebooks"

echo "== 4. notebooks done (source). Execution happens in the next step."

echo "== 5. execute the notebooks so they contain their outputs =="
cd "$BUNDLE/notebooks"
PYTHONPATH=/home/user/.venv/lib/python3.11/site-packages TF_CPP_MIN_LOG_LEVEL=3 \
  $PY "$REPO/python/execute_notebooks.py" "$BUNDLE/notebooks"

echo "== 6. copy the notebooks next to the other results =="
cp -f "$BUNDLE"/notebooks/*.ipynb "$REPO/results/"
cp -f "$BUNDLE"/notebooks/*.ipynb "$BUNDLE/results/"
cp -f "$REPO"/GAF_CNN_Final_Report.ipynb "$BUNDLE/results/" 2>/dev/null || true

echo "== 7. rebuild the results package =="
$PY "$REPO/python/build_results_zip.py"

echo "== 8. refresh the copy of results inside the bundle =="
rm -rf "$BUNDLE/results"; mkdir -p "$BUNDLE/results"
cp -rf "$REPO"/results/* "$BUNDLE/results/"


echo "== 9. bundle zip =="
cd "$BUNDLE"
rm -f GAF_CNN_Complete.zip
zip -qr GAF_CNN_Complete.zip . -x "GAF_CNN_Complete.zip"
echo "   $(stat -c%s GAF_CNN_Complete.zip) bytes"

echo "ALL DONE"

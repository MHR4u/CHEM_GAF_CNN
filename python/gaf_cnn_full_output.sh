#!/usr/bin/env bash
# Same CNN as gaf_cnn.sh, but every result it generates is written to files
# inside ../results/ instead of only being printed on the screen.
set -e
cd "$(dirname "$0")"

# Use $PYTHON if it is set (e.g. a virtualenv), otherwise python3
PYTHON="${PYTHON:-python3}"

"$PYTHON" gaf_cnn_full_output.py \
  --model CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras \
  --splits ../data/gafimages/shuffled/Test ../data/gafimages/concentrations_based/Test \
  --out_dir ../results

"$PYTHON" make_full_report.py --results_dir ../results --data_dir ../data/gafimages

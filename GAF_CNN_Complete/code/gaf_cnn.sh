#!/usr/bin/env bash
python3 ./gaf_cnn.py  --base_dir "../data/gafimages/shuffled/" --plot --nb_epoch 4000 --batch_size 16 --patience 1000

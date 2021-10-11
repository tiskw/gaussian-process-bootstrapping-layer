#!/bin/sh


COMMON_AEGS="--device cuda:0"

# Without data augmentation.
python3 train.py --gpb_layer_pos ""                  > results/baseline.tsv
python3 train.py --gpb_layer_pos "top"               > results/gpb_txx.tsv
python3 train.py --gpb_layer_pos "middle"            > results/gpb_xmx.tsv
python3 train.py --gpb_layer_pos "bottom"            > results/gpb_xxb.tsv
python3 train.py --gpb_layer_pos "top,middle"        > results/gpb_tmx.tsv
python3 train.py --gpb_layer_pos "top,bottom"        > results/gpb_txb.tsv
python3 train.py --gpb_layer_pos "middle,bottom"     > results/gpb_xmb.tsv
python3 train.py --gpb_layer_pos "top,middle,bottom" > results/gpb_tmb.tsv

# With data augmentation.
python3 train.py --data_aug --gpb_layer_pos ""                  > results/baseline_da.tsv
python3 train.py --data_aug --gpb_layer_pos "top"               > results/gpb_txx_da.tsv
python3 train.py --data_aug --gpb_layer_pos "middle"            > results/gpb_xmx_da.tsv
python3 train.py --data_aug --gpb_layer_pos "bottom"            > results/gpb_xxb_da.tsv
python3 train.py --data_aug --gpb_layer_pos "top,middle"        > results/gpb_tmx_da.tsv
python3 train.py --data_aug --gpb_layer_pos "top,bottom"        > results/gpb_txb_da.tsv
python3 train.py --data_aug --gpb_layer_pos "middle,bottom"     > results/gpb_xmb_da.tsv
python3 train.py --data_aug --gpb_layer_pos "top,middle,bottom" > results/gpb_tmb_da.tsv

# Save trained model.
python3 train.py --data_aug --save results/model_plain_da.pth --gpb_layer_pos ""
python3 train.py --data_aug --save results/model_gpb_b_da.pth --gpb_layer_pos "bottom"

# Compute variance values.
python3 print_variance.py --model results/model_plain_da.pth > results/variance_plain_da.txt
python3 print_variance.py --model results/model_gpb_b_da.pth > results/variance_gpb_b_da.txt


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker

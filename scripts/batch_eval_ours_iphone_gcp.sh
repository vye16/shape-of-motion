#!/bin/bash

EXPNAME=$1

seq_names=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "spin" "sriracha-tree" "teddy")
out_dir="/mnt/out/$EXPNAME"
for seq_name in "${seq_names[@]}"; do
    seq_dir="$out_dir/$seq_name"
    mkdir -p $seq_dir
    gsutil -mq cp -r "gs://xcloud-shared/qianqianwang/flow3d/ours/iphone/$EXPNAME/${seq_name}/results" $seq_dir
    done

python scripts/evaluate_iphone.py --data_dir /home/qianqianwang_google_com/datasets/iphone/dycheck  --result_dir /mnt/out/$EXPNAME
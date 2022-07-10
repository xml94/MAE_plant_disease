#!/bin/bash
dataset=$1
num_label=$2
batch=$3

export gpu=0,1,2,3
export epoch=50
export eval_epoch=5
export test_epoch='best'

for dataset_split in "train1shot" "train5shot" "train10shot" "train20shot"
do
  for mode in "ViT" "ViT_IN" "MAE_IN" "MAE_CLEF"
  do
    export name="${dataset}_${dataset_split}_${mode}"
    export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
    CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
    --eval \
    --resume "checkpoint/${name}/checkpoint-${test_epoch}.pth" \
    --batch_size ${batch} \
    --data_path ${IMAGENET_DIR} \
    --visualize_epoch 0 \
    --max_num 1 \
    --test_mode 'test' \
    --nb_classes ${num_label} \
    --mode ${mode}
  done
done
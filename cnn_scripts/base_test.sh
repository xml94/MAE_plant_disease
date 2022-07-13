#!/bin/bash
dataset=$1
num_label=$2
batch=$3

#export gpu=0,1,2,3
export gpu=0,1
export save_model_epoch=500
export epoch=50
export eval_epoch=5
export test_epoch='best'

for dataset_split in "train20" "train40" "train60" "train80"
do
  for mode in "CNN" "CNN_super" "MOCO"
  do
    export name="${dataset}_${dataset_split}_${mode}"
    export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
#    -m torch.distributed.launch --nproc_per_node=4
    CUDA_VISIBLE_DEVICES=${gpu} python3 cnn_finetune.py \
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
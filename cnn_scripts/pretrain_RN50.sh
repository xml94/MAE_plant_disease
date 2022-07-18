#!/bin/bash
dataset=$1
num_label=$2
batch=$3

export epoch=50
export eval_epoch=5
export mode="CNN_super"
export export save_model_epoch=500
export name="PlantCLEF_${mode}"
export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"

CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=4 cnn_finetune.py \
    --accum_iter 2 \
    --batch_size ${batch} \
    --epochs ${epoch} \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --nb_classes ${num_label} \
    --output_dir checkpoint/${name} \
    --log_dir checkpoint/${name}/"log" \
    --eval_epoch ${eval_epoch} \
    --save_model_epoch ${save_model_epoch} \
    --mode ${mode}
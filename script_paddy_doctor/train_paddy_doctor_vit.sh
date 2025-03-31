#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export gpu=0
export batch=64
export num_label=10
export epoch=50
export PRETRAIN_CHKPT='None'
export save_model_epoch=10
export mode="ViT"
export eval_epoch=10

export name="train_paddy_doctor"
export IMAGENET_DIR="/home/oem/Mingle/MAE_plant_disease/datasets/PaddyDoctor10407/raw/"
CUDA_VISIBLE_DEVICES=${gpu} torchrun main_finetune.py \
    --accum_iter 1 \
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
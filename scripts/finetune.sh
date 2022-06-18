#!/bin/bash
##########################
#  PVD
##########################
export epoch=400
export name="base_PVD_split_118_epoch400"
export batch=128
export IMAGENET_DIR="./../../datasets/PlantVillage_split_118"
export PRETRAIN_CHKPT='./ckpt/checkpoint-38.pth'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
    --accum_iter 8 \
    --batch_size ${batch} \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${epoch} \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --nb_classes 38 \
    --output_dir checkpoint/${name} \
    --log_dir checkpoint/${name}/"log" \
    --eval_epoch 50

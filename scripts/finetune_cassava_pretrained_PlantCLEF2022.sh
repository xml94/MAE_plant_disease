#!/bin/bash
##########################
#  cassava
##########################
export epoch=50
export dataset="train90_val5_test5"
export name="cassava_PlantCLEF2022_${dataset}"
export batch=32
export IMAGENET_DIR="/home/oem/Mingle/datasets/cassava/${dataset}"
export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 8 \
    --batch_size ${batch} \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${epoch} \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --nb_classes 5 \
    --output_dir checkpoint/${name} \
    --log_dir checkpoint/${name}/"log" \
    --eval_epoch 1 \
    --save_model_epoch 10
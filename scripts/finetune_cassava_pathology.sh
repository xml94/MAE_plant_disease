#!/bin/bash
##########################
#  cassava
##########################
export epoch=100
export name="base_clef2022_epoch21_cassava_epoch100"
export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/_cassava"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
export PRETRAIN_CHKPT='./output_dir/checkpoint-21.pth'
export batch=8
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 4 \
    --batch_size ${batch} \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${epoch} \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --nb_classes 5 \
    --output_dir checkpoint/${name} \
    --log_dir checkpoint/${name}/"log" \
    --eval_epoch 100

#export epoch=400
#export name="base_clef2022_epoch15_cassava_shot100_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/_cassava_split_shot100"
##export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export PRETRAIN_CHKPT='./output_dir/checkpoint-15.pth'
#export batch=8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_large_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 5 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


#export epoch=400
#export name="base_clef2022_epoch15_cassava_shot50_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/_cassava_split_shot50"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export batch=8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_base_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 5 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50

##########################
#  pathology 2020
##########################
#export epoch=300
#export name="base_pathology_shot100_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/_pathology2020_split_shot100"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export batch=256
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_base_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 38 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log"
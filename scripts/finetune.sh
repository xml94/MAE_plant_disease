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

#export epoch=400
#export name="base_PVD_split_118_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_118"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export batch=128
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


#export epoch=400
#export name="base_PVD_split_424_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_424"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export batch=128
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
#
#
#export epoch=400
#export name="base_PVD_split_811_epoch400"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_811"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#export batch=128
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
#
#
#export epoch=400
#export name="base_PVD_shot1_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot1"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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
#
#
#export epoch=400
#export name="base_PVD_shot10_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot10"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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


#export epoch=400
#export name="base_PVD_shot20_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot20"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


#export epoch=400
#export name="base_PVD_shot30_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot30"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


#export epoch=400
#export name="base_PVD_shot50_epoch400"
#export batch=64
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot50"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


#export epoch=400
#export name="base_PVD_shot100_epoch400"
#export batch=64
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot100"
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
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
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


#export epoch=400
#export name="base_PVD_shot20_clef_epoch38_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot20"
#export PRETRAIN_CHKPT='./output_dir/checkpoint-38.pth'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_large_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 38 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50
#
#export epoch=400
#export name="base_PVD_shot10_clef_epoch38_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot10"
#export PRETRAIN_CHKPT='./output_dir/checkpoint-38.pth'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_large_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 38 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50

#export epoch=400
#export name="base_PVD_shot1_clef_epoch38_epoch400"
#export batch=2
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot1"
#export PRETRAIN_CHKPT='./output_dir/checkpoint-38.pth'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size ${batch} \
#    --model vit_large_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 38 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 50


##########################
#  Clef_plant
##########################
#export IMAGENET_DIR='/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/'
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_large.pth'
#export name="IN1k_Clef2022"
#export epoch=100
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size 32 \
#    --model vit_large_patch16  \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir checkpoint/${name}/"log" \
#    --nb_classes 80000 \
#    --resume "./checkpoint/${name}/checkpoint-49.pth" --start_epoch 49 \
#    --eval_epoch 1 \
#    --save_model_epoch 1

#--finetune ${PRETRAIN_CHKPT} \

##########################
#  Clef-Fungi
##########################
#export IMAGENET_DIR='/home/oem/Mingle/datasets/FungiCLEF2022/'
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_large.pth'
#export name="ClefFungi2022_epoch100"
#export epoch=100
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size 32 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --model vit_large_patch16  \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir checkpoint/${name}/"log" \
#    --nb_classes 1604 \
#    --eval_epoch 10 \
#    --save_model_epoch 10 \
#    --output_dir checkpoint/${name}

##########################
#  ImageNet-1k
##########################
#export epoch=100
#export clef_epoch=40
#export name="large_IN_clef${clef_epoch}"
#export batch=32
#export IMAGENET_DIR="/home/oem/Mingle/datasets/CLS-LOC/"
#export PRETRAIN_CHKPT="./output_dir/checkpoint-${clef_epoch}.pth"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 8 \
#    --batch_size ${batch} \
#    --model vit_large_patch16 \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --nb_classes 1000 \
#    --output_dir checkpoint/${name} \
#    --log_dir checkpoint/${name}/"log" \
#    --eval_epoch 1 \
#    --save_model_epoch 10


##########################
#  Rice_leaf_2022
##########################
#export clef_epoch=24
#export epoch=100
#export IMAGENET_DIR='/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_ratio82/'
##export PRETRAIN_CHKPT="./output_dir/checkpoint-${clef_epoch}.pth"
##export name="rice_leaf_2022_ratio82_mae_base_clef_${clef_epoch}"
#export PRETRAIN_CHKPT="./checkpoint/mae_pretrain_vit_base.pth"
#export name="rice_leaf_2022_ratio82_mae_base_epoch${epoch}"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size 32 \
#    --model vit_base_patch16  \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir checkpoint/${name}/"log" \
#    --nb_classes 4 \
#    --eval_epoch 50 \
#    --save_model_epoch 50 \
#    --output_dir checkpoint/${name}

#export shot=1
#export epoch=400
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_shot${shot}/"
#export PRETRAIN_CHKPT="./checkpoint/mae_pretrain_vit_base.pth"
#export name="rice_leaf_2022_shot${shot}_mae_base_epoch${epoch}"
#CUDA_VISIBLE_DEVICES=0 python  main_finetune.py \
#    --accum_iter 2 \
#    --batch_size 2 \
#    --model vit_base_patch16  \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir checkpoint/${name}/"log" \
#    --nb_classes 4 \
#    --eval_epoch 50 \
#    --save_model_epoch 50 \
#    --output_dir checkpoint/${name}

#for shot in 10 30 50 100
#do
#  export epoch=100
#  export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_shot${shot}/"
#  export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_base.pth'
#  export name="rice_leaf_2022_shot${shot}_mae_base_epoch${epoch}"
#  CUDA_VISIBLE_DEVICES=0 python  main_finetune.py \
#      --accum_iter 4 \
#      --batch_size 4 \
#      --model vit_base_patch16  \
#      --finetune ${PRETRAIN_CHKPT} \
#      --epochs ${epoch} \
#      --blr 5e-4 --layer_decay 0.65 \
#      --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#      --dist_eval --data_path ${IMAGENET_DIR} \
#      --log_dir checkpoint/${name}/"log" \
#      --nb_classes 4 \
#      --eval_epoch 50 \
#      --save_model_epoch 50 \
#      --output_dir checkpoint/${name}
#done




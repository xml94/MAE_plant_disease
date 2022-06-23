#!/bin/bash

export dataset='PlantVillage'
export num_label=38
export batch=32
export epoch=90
export accum_iter=1
export save_model_epoch=1000
export train_type='linprobe'
##########################
#  PlantCLEF2022
##########################
#"train1shot_val20_test20"
for dataset_split in  "train5shot_val20_test20" "train10shot_val20_test20" "train20shot_val20_test20"
do
  export name="${dataset}_${dataset_split}_${train_type}_PlantCLEF2022"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
  CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
      --accum_iter ${accum_iter} \
      --batch_size ${batch} \
      --model vit_large_patch16 \
      --finetune ${PRETRAIN_CHKPT} \
      --epochs ${epoch} \
      --blr 0.1 \
      --weight_decay 0.0 \
      --dist_eval --data_path ${IMAGENET_DIR} \
      --nb_classes ${num_label} \
      --output_dir checkpoint/${name} \
      --log_dir checkpoint/${name}/"log" \
      --eval_epoch 10 \
      --save_model_epoch ${save_model_epoch}
done


##########################
#  MAE
##########################
for dataset_split in "train1shot_val20_test20" "train5shot_val20_test20" "train10shot_val20_test20" "train20shot_val20_test20"
do
  export name="${dataset}_${dataset_split}_${train_type}_MAE"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
  CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
      --accum_iter ${accum_iter} \
      --batch_size ${batch} \
      --model vit_large_patch16 \
      --finetune ${PRETRAIN_CHKPT} \
      --epochs ${epoch} \
      --blr 1e-3 --layer_decay 0.75 \
      --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
      --dist_eval --data_path ${IMAGENET_DIR} \
      --nb_classes ${num_label} \
      --output_dir checkpoint/${name} \
      --log_dir checkpoint/${name}/"log" \
      --eval_epoch 10 \
      --save_model_epoch ${save_model_epoch}
done
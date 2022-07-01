#!/bin/bash
dataset=$1
num_label=$2
batch=$3
eval_epoch=$4

export gpu=0,1,2,3
export save_model_epoch=500
export epoch=50


for mode in "ViT" "ViT_IN" "MAE_IN" "MAE_CLEF"
do
  if [ $mode = "ViT" ]
  then
    PRETRAIN_CHKPT='None'
    export epoch=200
  fi
  if [ $mode = "ViT_IN" ]
  then
    export PRETRAIN_CHKPT='./ckpt/L_16_imagenet1k.pth'
  fi
  if [ $mode = "MAE_IN" ]
  then
    export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
  fi
  if [ $mode = "MAE_CLEF" ]
  then
    export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
  fi

  for dataset_split in "train20" "train40" "train60" "train80"
  do
    export name="${dataset}_${dataset_split}_${mode}"
    export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
    CUDA_VISIBLE_DEVICES=${gpu} python main_finetune.py \
        --batch_size ${batch} \
        --finetune ${PRETRAIN_CHKPT} \
        --epochs ${epoch} \
        --blr 1e-3 --layer_decay 0.75 \
        --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --dist_eval --data_path ${IMAGENET_DIR} \
        --nb_classes ${num_label} \
        --output_dir checkpoint/${name} \
        --log_dir checkpoint/${name}/"log" \
        --eval_epoch ${eval_epoch} \
        --save_model_epoch ${save_model_epoch}
  done
done
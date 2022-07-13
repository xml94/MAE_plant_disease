#!/bin/bash
dataset=$1
num_label=$2
batch=$3

export gpu=0,1,2,3
export save_model_epoch=500

#"MAE_IN" "MAE_CLEF"
for mode in "ViT" "ViT_IN"
do
  if [ $mode = "ViT" ]
  then
    PRETRAIN_CHKPT='None'
    export epoch=50
    export eval_epoch=5
  fi
  if [ $mode = "ViT_IN" ]
  then
    export PRETRAIN_CHKPT='./ckpt/L_16_imagenet1k.pth'
    export epoch=50
    export eval_epoch=5
  fi
  if [ $mode = "MAE_IN" ]
  then
    export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
    export epoch=50
    export eval_epoch=5
  fi
  if [ $mode = "MAE_CLEF" ]
  then
    export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
    export epoch=50
    export eval_epoch=5
  fi

  for dataset_split in "train20" "train40" "train60" "train80"
  do
    export name="${dataset}_${dataset_split}_${mode}"
    export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
        --accum_iter 1 \
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
        --save_model_epoch ${save_model_epoch} \
        --mode ${mode}
  done
done
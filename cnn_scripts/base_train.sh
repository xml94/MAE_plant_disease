#!/bin/bash
dataset=$1
num_label=$2
batch=$3

#export gpu=0,1,2,3
export gpu=0,1
export save_model_epoch=500

for mode in "MOCO" "CNN" "CNN_super"
do
  if [ $mode = "CNN" ]
  then
    export epoch=200
    export eval_epoch=10
  fi
  if [ $mode = "CNN_super" ]
  then
    export epoch=200
    export eval_epoch=10
  fi
  if [ $mode = "MOCO" ]
  then
    export epoch=50
    export eval_epoch=5
  fi

  for dataset_split in "train20" "train40" "train60" "train80"
  do
    export name="${dataset}_${dataset_split}_${mode}"
    export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=2 cnn_finetune.py \
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
  done
done
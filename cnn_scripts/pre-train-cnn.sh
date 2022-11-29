#!/bin/bash
'''
pretrain the models in PlantCLEF2022 dataset

Model version
* ResNet50
* DenseNet

Supervised way
* cnn from scratch
* supervised cnn in ImageNet
* self-supervised cnn
'''

#export mode="CNN_super"
#export mode="CNN"
#export mode="MoCo"


for mode in "CNN_super" "CNN" "MoCo"
do
  export name="${mode}_PlantCLEF2022"

  export IMAGENET_DIR='/home/oem/Mingle/datasets/ClefPlant2022_train/'
  export all_epoch=100
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 cnn_finetune.py \
      --accum_iter 1 \
      --batch_size 128 \
      --epochs ${all_epoch} \
      --blr 5e-4 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
      --dist_eval --data_path ${IMAGENET_DIR} \
      --log_dir "checkpoint/${name}/log" \
      --nb_classes 80000 \
      --eval_epoch 1 \
      --save_model_epoch 20 \
      --output_dir checkpoint/${name} \
      --mode ${mode}
done

#CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=4 cnn_finetune.py \
#        --accum_iter 2 \
#        --batch_size ${batch} \
#        --epochs ${epoch} \
#        --blr 1e-3 --layer_decay 0.75 \
#        --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#        --dist_eval --data_path ${IMAGENET_DIR} \
#        --nb_classes ${num_label} \
#        --output_dir checkpoint/${name} \
#        --log_dir checkpoint/${name}/"log" \
#        --eval_epoch ${eval_epoch} \
#        --save_model_epoch ${save_model_epoch} \
#        --mode ${mode}
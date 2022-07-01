#!/bin/bash

##########################
#  shared
##########################
export epoch=90
export accum_iter=1
export save_model_epoch=1000
export train_type='finetune'

##########################
#  PlantVillage
##########################
#export dataset='PlantVillage'
#export num_label=38
#export batch=38
##########################
#  cassava
##########################
#export dataset='cassava'
#export num_label=5
#export batch=4
##########################
#  Apple2020
##########################
#export dataset='Apple2020'
#export num_label=4
#export batch=4
##########################
#  Apple2021
##########################
#export dataset='Apple2021'
#export num_label=6
#export batch=6
##########################
#  Rice2020
##########################
#export dataset='Rice2020'
#export num_label=4
#export batch=4
##########################
#  IVADL_tomato
##########################
#export dataset='IVADL_tomato'
#export num_label=9
#export batch=8
##########################
#  IVADL_rose
##########################
#export dataset='IVADL_rose'
#export num_label=6
#export batch=6
##########################
#  CitrusLeaf
##########################
#export dataset='CitrusLeaf'
#export num_label=4
#export batch=4
##########################
#  ChineseStrawberry
##########################
#export dataset='ChineseStrawberry'
#export num_label=4
#export batch=4
##########################
#  CGIAR_wheat
##########################
export dataset='CGIAR_wheat'
export num_label=3
export batch=2
##########################
#  Rice1462
##########################
#export dataset='Rice1462'
#export num_label=9
#export batch=8
##########################
#  Final code
##########################
##########################
#  PlantCLEF2022
##########################
export gpu=2
for dataset_split in "train1shot_val20_test20" "train5shot_val20_test20" "train10shot_val20_test20" "train20shot_val20_test20"
do
  export name="${dataset}_${dataset_split}_${train_type}_PlantCLEF2022"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
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


##########################
#  MAE
##########################
for dataset_split in "train1shot_val20_test20" "train5shot_val20_test20" "train10shot_val20_test20" "train20shot_val20_test20"
do
  export name="${dataset}_${dataset_split}_${train_type}_MAE"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
      --accum_iter ${accum_iter} \
      --batch_size ${batch} \
      --model vit_large_patch16 \
      --finetune ${PRETRAIN_CHKPT} \
      --epochs ${epoch} \
      --blr 1e-3 \
      --weight_decay 0.00 \
      --dist_eval --data_path ${IMAGENET_DIR} \
      --nb_classes ${num_label} \
      --output_dir checkpoint/${name} \
      --log_dir checkpoint/${name}/"log" \
      --eval_epoch 10 \
      --save_model_epoch ${save_model_epoch}
done



##########################
#  TaiwanTomato
##########################
#export dataset='TaiwanTomato'
#export num_label=6
#export batch=6
##########################
#  PlantDoc_cls
##########################
#export dataset='PlantDoc_cls'
#export num_label=27
#export batch=26
##########################
#  PlantCLEF2022
##########################
#export gpu=1
#for dataset_split in "train1shot_val20" "train5shot_val20" "train10shot_val20" "train20shot_val20"
#do
#  export name="${dataset}_${dataset_split}_${train_type}_PlantCLEF2022"
#  export batch=${batch}
#  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
#  export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
#  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
#      --accum_iter ${accum_iter} \
#      --batch_size ${batch} \
#      --model vit_large_patch16 \
#      --finetune ${PRETRAIN_CHKPT} \
#      --epochs ${epoch} \
#      --blr 1e-3 --layer_decay 0.75 \
#      --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#      --dist_eval --data_path ${IMAGENET_DIR} \
#      --nb_classes ${num_label} \
#      --output_dir checkpoint/${name} \
#      --log_dir checkpoint/${name}/"log" \
#      --eval_epoch 10 \
#      --save_model_epoch ${save_model_epoch}
#done


##########################
#  MAE
##########################
#for dataset_split in "train1shot_val20" "train5shot_val20" "train10shot_val20" "train20shot_val20"
#do
#  export name="${dataset}_${dataset_split}_${train_type}_MAE"
#  export batch=${batch}
#  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
#  export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
#  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
#      --accum_iter ${accum_iter} \
#      --batch_size ${batch} \
#      --model vit_large_patch16 \
#      --finetune ${PRETRAIN_CHKPT} \
#      --epochs ${epoch} \
#      --blr 1e-3 \
#      --weight_decay 0.00 \
#      --dist_eval --data_path ${IMAGENET_DIR} \
#      --nb_classes ${num_label} \
#      --output_dir checkpoint/${name} \
#      --log_dir checkpoint/${name}/"log" \
#      --eval_epoch 10 \
#      --save_model_epoch ${save_model_epoch}
#done







##########################
#  PDD271_Sample
##########################
#export dataset='PDD271_Sample'
#export num_label=271
#export batch=32

#export gpu=1
#for dataset_split in "train1shot_val1shot_test4shot" "train5shot_val1shot_test4shot"
#do
#  export name="${dataset}_${dataset_split}_${train_type}_PlantCLEF2022"
#  export batch=${batch}
#  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
#  export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'
#  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
#      --accum_iter ${accum_iter} \
#      --batch_size ${batch} \
#      --model vit_large_patch16 \
#      --finetune ${PRETRAIN_CHKPT} \
#      --epochs ${epoch} \
#      --blr 1e-3 --layer_decay 0.75 \
#      --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#      --dist_eval --data_path ${IMAGENET_DIR} \
#      --nb_classes ${num_label} \
#      --output_dir checkpoint/${name} \
#      --log_dir checkpoint/${name}/"log" \
#      --eval_epoch 10 \
#      --save_model_epoch ${save_model_epoch}
#done
#for dataset_split in "train1shot_val1shot_test4shot" "train5shot_val1shot_test4shot"
#do
#  export name="${dataset}_${dataset_split}_${train_type}_MAE"
#  export batch=${batch}
#  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
#  export PRETRAIN_CHKPT='./ckpt/mae_pretrain_vit_large.pth'
#  CUDA_VISIBLE_DEVICES=${gpu} python "main_${train_type}.py" \
#      --accum_iter ${accum_iter} \
#      --batch_size ${batch} \
#      --model vit_large_patch16 \
#      --finetune ${PRETRAIN_CHKPT} \
#      --epochs ${epoch} \
#      --blr 1e-3 \
#      --weight_decay 0.00 \
#      --dist_eval --data_path ${IMAGENET_DIR} \
#      --nb_classes ${num_label} \
#      --output_dir checkpoint/${name} \
#      --log_dir checkpoint/${name}/"log" \
#      --eval_epoch 10 \
#      --save_model_epoch ${save_model_epoch}
#done
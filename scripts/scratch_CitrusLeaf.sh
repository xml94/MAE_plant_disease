export dataset='CitrusLeaf'
export num_label=4
export batch=16

export epoch=200
export accum_iter=1
export save_model_epoch=5000
##########################
#  PlantCLEF2022
##########################
for dataset_split in "train20_val20_test20" "train40_val20_test20" "train60_val20_test20" "train80_val10_test10"
do
  export name="${dataset}_${dataset_split}_scratch_PlantCLEF2022"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
      --accum_iter ${accum_iter} \
      --batch_size ${batch} \
      --model vit_large_patch16 \
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
for dataset_split in "train20_val20_test20" "train40_val20_test20" "train60_val20_test20" "train80_val10_test10"
do
  export name="${dataset}_${dataset_split}_scratch_MAE"
  export batch=${batch}
  export IMAGENET_DIR="./../datasets/${dataset}/${dataset_split}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
      --accum_iter ${accum_iter} \
      --batch_size ${batch} \
      --model vit_large_patch16 \
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
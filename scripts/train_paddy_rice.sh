export gpu=0,1,2,3
export save_model_epoch=500
export batch=32
export epoch=200
export eval_epoch=5
export num_label=10
export mode=MAE_CLEF
export PRETRAIN_CHKPT='./ckpt/PlantCLEF2022_MAE_vit_large_patch16.pth'

export IMAGENET_DIR="/home/oem/Mingle/datasets/paddy-disease-classification/train80"
export name="paddy_rice_PlantCLEF_ViT_train80"

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

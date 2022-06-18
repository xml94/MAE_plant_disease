#CUDA_VISIBLE_DEVICES=0,1,2,3 python submitit_pretrain.py \
#    --job_dir checkpoint/pretain_clef_2022 \
#    --nodes 1 \
#    --use_volta32 \
#    --batch_size 64 \
#    --model mae_vit_large_patch16 \
#    --norm_pix_loss \
#    --mask_ratio 0.75 \
#    --epochs 800 \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    --data_path /home/oem/Mingle/leaf_dataset/ClefPlant2022


export name="clef_pretrain_epoch100"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
--data_path /home/oem/Mingle/datasets/leaf_disease/ClefPlant2022 \
--model mae_vit_large_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 100 \
--warmup_epochs 5 \
--blr 1.5e-4 --weight_decay 0.05 \
--batch_size 32 \
--accum_iter 4 \
--output_dir "checkpoint/${name}" \
--log_dir "checkpoint/${name}/logs"
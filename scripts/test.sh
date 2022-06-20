##########################
#  Apple2020
##########################
export dataset_mode="train20_val20_test20"
export name="Apple2020_${dataset_mode}_finetune_MAE"
export num_label=4
export IMAGENET_DIR="/home/oem/Mingle/datasets/Apple2020/${dataset_mode}"
export test_epoch=best
CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
--eval \
--resume "checkpoint/${name}/checkpoint-${test_epoch}.pth" \
--model vit_large_patch16 \
--batch_size 1 \
--data_path ${IMAGENET_DIR} \
--nb_classes ${num_label} \
--visualize_epoch 0 \
--max_num 1 \
--test_mode 'test'

##########################
#  paddy rice
##########################
export IMAGENET_DIR="/home/oem/Mingle/datasets/paddy-disease-classification/test"
export name="paddy_rice_PlantCLEF_ViT"
export epoch=best
CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py \
--eval \
--resume "checkpoint/${name}/checkpoint-${epoch}.pth" \
--model vit_large_patch16 \
--batch_size 1 \
--data_path ${IMAGENET_DIR} \
--nb_classes 10 \
--visualize_epoch 1 \
--max_num 1
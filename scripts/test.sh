##########################
#  PVD
##########################
#export shot=10
#export clef_epoch=38
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/PlantVillage_split_shot${shot}"
#export name="base_PVD_shot${shot}_clef_epoch${clef_epoch}_epoch400"
#export epoch=400
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#--eval \
#--resume checkpoint/${name}/checkpoint-${epoch}.pth \
#--model vit_large_patch16 \
#--batch_size 128 \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 38

#export epoch=400
#CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py \
#--eval \
#--resume checkpoint/${name}/checkpoint-${epoch}.pth \
#--model vit_base_patch16 \
#--batch_size 1 \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 38


##########################
#  cassava
##########################
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/_cassava_split_shot50/"
#export name="base_cassava_shot50_epoch400"
#export epoch=400
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#--eval \
#--resume checkpoint/${name}/checkpoint-${epoch}.pth \
#--model vit_base_patch16 \
#--batch_size 128 \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 5


##########################
#  Clef_plant
##########################
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022_test/"
#export name="IN1k_Clef2022"
#export epoch=77
#CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py \
#--eval \
#--resume "output_dir/checkpoint-${epoch}.pth" \
#--model vit_large_patch16 \
#--batch_size 1 \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 80000 \
#--visualize_epoch "${epoch}" \
#--max_num 30


##########################
#  Clef_fungi
##########################
export test_epoch=100
export IMAGENET_DIR="/home/oem/Mingle/datasets/FungiCLEF2022/real_test"
export name="ClefFungi2022_epoch100_ClefPlant2022_epoch67"
CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
--eval \
--resume "checkpoint/${name}/checkpoint-${test_epoch}.pth" \
--model vit_large_patch16 \
--batch_size 1 \
--data_path ${IMAGENET_DIR} \
--nb_classes 1604 \
--visualize_epoch "${test_epoch}" \
--max_num 1


##########################
#  ImageNet
##########################
#export shot=10
#export clef_epoch=67
#export name="large_IN_clef${clef_epoch}"
#export batch=32
#export IMAGENET_DIR="/home/oem/Mingle/datasets/CLS-LOC/"
#export epoch=12
#CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py \
#--eval \
#--resume "checkpoint/${name}/checkpoint-${epoch}.pth" \
#--model vit_large_patch16 \
#--batch_size ${batch} \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 1000


##########################
#  Rice_leaf_2022
##########################
#export shot=100
#export epoch=100
##export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_ratio82"
#export IMAGENET_DIR="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_shot${shot}/"
##export name="rice_leaf_2022_ratio82_mae_base_epoch${epoch}"
#export name="rice_leaf_2022_shot${shot}_mae_base_epoch${epoch}"
#CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
#--eval \
#--resume "checkpoint/${name}/checkpoint-${epoch}.pth" \
#--model vit_base_patch16 \
#--batch_size 32 \
#--data_path ${IMAGENET_DIR} \
#--nb_classes 4 \
#--visualize_epoch 0
##########################
#  PlantVillage
##########################
#export dataset="PlantVillage"
#export num_label=38
#export batch=32
##########################
#  cassava
##########################
#export dataset='cassava'
#export num_label=5
#export batch=32
##########################
#  Apple2020
##########################
#export dataset='Apple2020'
#export num_label=4
#export batch=32
##########################
#  Apple2021
##########################
#export dataset='Apple2021'
#export num_label=6
#export batch=32
##########################
#  Rice2020
##########################
#export dataset='Rice2020'
#export num_label=4
#export batch=32
##########################
#  IVADL_tomato
##########################
#export dataset='IVADL_tomato'
#export num_label=9
#export batch=32
##########################
#  IVADL_rose
##########################
#export dataset='IVADL_rose'
#export num_label=6
#export batch=32
##########################
#  CitrusLeaf
##########################
#export dataset='CitrusLeaf'
#export num_label=4
#export batch=16
##########################
#  ChineseStrawberry
##########################
export dataset='ChineseStrawberry'
export num_label=4
export batch=16
##########################
#  Final code
##########################
for pretrain in "PlantCLEF2022" "MAE"
do
  for dataset_mode in "train20_val20_test20" "train40_val20_test20" "train60_val20_test20" "train80_val10_test10"
  do
    export name="${dataset}_${dataset_mode}_finetune_${pretrain}"
    export IMAGENET_DIR="/home/oem/Mingle/datasets/${dataset}/${dataset_mode}"
    export test_epoch=best
    CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
    --eval \
    --resume "checkpoint/${name}/checkpoint-${test_epoch}.pth" \
    --model vit_large_patch16 \
    --batch_size ${batch} \
    --data_path ${IMAGENET_DIR} \
    --visualize_epoch 0 \
    --max_num 1 \
    --test_mode 'test' \
    --nb_classes ${num_label}
  done
done



##########################
#  PlantDoc_cls
##########################
#export dataset='PlantDoc_cls'
#export num_label=27
#export batch=32
##########################
#  TaiwanTomato
##########################
#export dataset='TaiwanTomato'
#export num_label=6
#export batch=16

##########################
#  Final code
##########################
#for pretrain in "PlantCLEF2022" "MAE"
#do
#  for dataset_mode in "train20_val20" "train40_val20" "train60_val20" "train80_val20"
#  do
#    export name="${dataset}_${dataset_mode}_finetune_${pretrain}"
#    export IMAGENET_DIR="/home/oem/Mingle/datasets/${dataset}/${dataset_mode}"
#    export test_epoch=best
#    CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
#    --eval \
#    --resume "checkpoint/${name}/checkpoint-${test_epoch}.pth" \
#    --model vit_large_patch16 \
#    --batch_size ${batch} \
#    --data_path ${IMAGENET_DIR} \
#    --visualize_epoch 0 \
#    --max_num 1 \
#    --test_mode 'test' \
#    --nb_classes ${num_label}
#  done
#done
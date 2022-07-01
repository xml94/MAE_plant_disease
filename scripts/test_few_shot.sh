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
#export batch=32
##########################
#  ChineseStrawberry
##########################
#export dataset='ChineseStrawberry'
#export num_label=4
#export batch=32
##########################
#  CGIAR_wheat
##########################
export dataset='CGIAR_wheat'
export num_label=3
export batch=16
##########################
#  Rice1462
##########################
#export dataset='Rice1462'
#export num_label=9
#export batch=16

for pretrain in "PlantCLEF2022" "MAE"
do
  for dataset_mode in "train1shot_val20_test20" "train5shot_val20_test20" "train10shot_val20_test20" "train20shot_val20_test20"
  do
    export name="${dataset}_${dataset_mode}_finetune_${pretrain}"
    export IMAGENET_DIR="/home/oem/Mingle/datasets/${dataset}/${dataset_mode}"
    export test_epoch=best
    CUDA_VISIBLE_DEVICES=3 python3 main_finetune.py \
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
#export batch=32
#
#for pretrain in "PlantCLEF2022" "MAE"
#do
#  for dataset_mode in "train1shot_val20" "train5shot_val20" "train10shot_val20" "train20shot_val20"
#  do
#    export name="${dataset}_${dataset_mode}_finetune_${pretrain}"
#    export IMAGENET_DIR="/home/oem/Mingle/datasets/${dataset}/${dataset_mode}"
#    export test_epoch=best
#    CUDA_VISIBLE_DEVICES=3 python3 main_finetune.py \
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




##########################
#  PDD271_Sample
##########################
#export dataset='PDD271_Sample'
#export num_label=271
#export batch=32
#for pretrain in "PlantCLEF2022" "MAE"
#do
#  for dataset_mode in "train1shot_val1shot_test4shot" "train5shot_val1shot_test4shot"
#  do
#    export name="${dataset}_${dataset_mode}_finetune_${pretrain}"
#    export IMAGENET_DIR="/home/oem/Mingle/datasets/${dataset}/${dataset_mode}"
#    export test_epoch=best
#    CUDA_VISIBLE_DEVICES=3 python3 main_finetune.py \
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
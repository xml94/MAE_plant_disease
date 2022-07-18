export code="./cnn_scripts/base_few_shot.sh"

#dataset=$1
#num_label=$2
#batch=$3
#gpu=$4

export gpu=0
#sh ${code} "Apple2020" 4 4 $gpu
sh ${code} "Apple2021" 6 4 $gpu
#sh ${code} "cassava" 5 4 $gpu
#sh ${code} "CGIAR_wheat" 3 2 $gpu
#sh ${code} "ChineseStrawberry" 4 4 $gpu
#sh ${code} "CitrusLeaf" 4 4 $gpu
#sh ${code} "IVADL_rose" 6 4 $gpu
#sh ${code} "IVADL_tomato" 9 8 $gpu
#sh ${code} "PlantDoc_cls" 27 16 $gpu
#sh ${code} "PlantVillage" 38 32 $gpu
#sh ${code} "Rice1462" 9 8 $gpu
#sh ${code} "Rice2020" 4 4 $gpu
#sh ${code} "TaiwanTomato" 6 4 $gpu
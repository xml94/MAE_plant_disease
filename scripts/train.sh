export code="./scripts/base_train.sh"

#dataset=$1
#num_label=$2
#batch=$3
#eval_epoch=$4


sh ${code} "Apple2020" 4 32 100
sh ${code} "Apple2021" 6 32 100
sh ${code} "cassava" 5 32 100
sh ${code} "CGIAR_wheat" 3 16 100
sh ${code} "ChineseStrawberry" 4 16 100
sh ${code} "CitrusLeaf" 4 16 100
sh ${code} "IVADL_rose" 6 32 100
sh ${code} "IVADL_tomato" 9 32 100
sh ${code} "PlantDoc_cls" 27 32 100
sh ${code} "PlantVillage" 38 32 100
sh ${code} "Rice1462" 9 32 100
sh ${code} "Rice2020" 4 32 100
sh ${code} "TaiwanTomato" 6 16 100
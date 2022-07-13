export base_dir="/home/oem/Mingle/plant_disease_dataset/"
export code='visualize_dataset/visualize_dataset.py'

#########################################################
# original dataset without test
#########################################################
python ${code} --base_dir ${base_dir} --src_dir "Apple2020"
python ${code} --base_dir ${base_dir} --src_dir "Apple2021"
python ${code} --base_dir ${base_dir} --src_dir "cassava"
python ${code} --base_dir ${base_dir} --src_dir "CGIAR_wheat"
python ${code} --base_dir ${base_dir} --src_dir "ChineseStrawberry"
python ${code} --base_dir ${base_dir} --src_dir "CitrusLeaf"
python ${code} --base_dir ${base_dir} --src_dir "IVADL_rose"
python ${code} --base_dir ${base_dir} --src_dir "IVADL_tomato"
python ${code} --base_dir ${base_dir} --src_dir "PlantVillage"
python ${code} --base_dir ${base_dir} --src_dir "Rice1462"
python ${code} --base_dir ${base_dir} --src_dir "Rice2020"
python ${code} --base_dir ${base_dir} --src_dir "PlantDoc_cls" --test 1
python ${code} --base_dir ${base_dir} --src_dir "TaiwanTomato" --test 1

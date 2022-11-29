export code='data/make_train_val_test.py'
export base_dir="/data/Mingle/DATASETS"


#########################################################
# original dataset without test
#########################################################
python ${code} --base_dir ${base_dir} --src_dir "Apple2020_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "Apple2021_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "cassava_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "CGIAR_wheat_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "ChineseStrawberry_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "CitrusLeaf_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "IVADL_rose_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "IVADL_tomato_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "PlantVillage_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "Rice1462_back" --test 0
#python ${code} --base_dir ${base_dir} --src_dir "Rice2020_back" --test 0

#python ${code} --base_dir ${base_dir} --src_dir "CottonWeedID15" --test 0


#########################################################
# original dataset with test
#########################################################
#python ${code} --base_dir ${base_dir} --src_dir "PlantDoc_cls_back" --test 1
#python ${code} --base_dir ${base_dir} --src_dir "taiwanTomato_back" --test 1

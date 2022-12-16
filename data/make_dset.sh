export code='data/make_train_val_test.py'
export base_dir="/home/oem/Mingle/datasets/"

#########################################################
# original dataset without test
#########################################################
#python ${code} --dset_name "${base_dir}/Apple2020_back"
#python ${code} --dset_name "${base_dir}/Apple2021_back"
#python ${code} --dset_name "${base_dir}/cassava_back"
#python ${code} --dset_name "${base_dir}/CGIAR_wheat_back"
#python ${code} --dset_name "${base_dir}/ChineseStrawberry_back"
#python ${code} --dset_name "${base_dir}/CitrusLeaf_back"
#python ${code} --dset_name "${base_dir}/IVADL_rose_back"
#python ${code} --dset_name "${base_dir}/IVADL_tomato_back"
#python ${code} --dset_name "${base_dir}/PlantVillage_back"
#python ${code} --dset_name "${base_dir}/Rice1462_back"
#python ${code} --dset_name "${base_dir}/Rice2020_back"
#python ${code} --dset_name "${base_dir}/CottonWeedID15"


#########################################################
# original dataset with test
#########################################################
#python ${code} --dset_name "${base_dir}/PlantDoc_cls_back" --with_test
#python ${code} --dset_name "${base_dir}/taiwanTomato_back" --with_test
python ${code} --dset_name "${base_dir}/paddy-disease-classification" --with_test

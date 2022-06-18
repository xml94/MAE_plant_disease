# split dataset
##########################
#  plant village
##########################
export ratio_train=0.2
export number_train=20
export src_dir='./../../datasets/PlantVillage'
export tgt_src="./../../datasets/PlantVillage_ratio${ratio_train}"
rm -rf ${tgt_src}
#
python split_train_val_test.py --src_dir ${src_dir} --tgt_dir ${tgt_src} \
--mode 'number' \
--number_train ${number_train} --ratio_train ${ratio_train}

#--mode 'ratio' --ratio_train ${ratio_train} --ratio_val ${ratio_val}
#--mode 'number' --number_train 1



##########################
#  cassava
##########################
#export ratio_train=0.8
#export number_train=1
#export src_dir='/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022/'
#export tgt_src="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_shot${number_train}"
##export tgt_src="/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022_ratio82"
#rm -rf ${tgt_src}
#
#python split_train_val_test.py --src_dir ${src_dir} --tgt_dir ${tgt_src} \
#--mode 'number' \
#--number_train ${number_train} --ratio_train ${ratio_train}



##########################
#  pathology 2020
##########################
#export ratio_train=0.8
#export number_train=100
#export src_dir='/home/oem/Mingle/datasets/leaf_disease/_pathology2020/train'
#export tgt_src="/home/oem/Mingle/datasets/leaf_disease/_pathology2020_split_shot${number_train}"
#rm -rf ${tgt_src}
#
#python split_train_val_test.py --src_dir ${src_dir} --tgt_dir ${tgt_src} \
#--mode 'number' \
#--number_train ${number_train} --ratio_train ${ratio_train}
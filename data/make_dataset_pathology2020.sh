#################
## pathology 2020
#################
## train
#export input_dir="/home/oem/Mingle/datasets/leaf_disease/plant-pathology-2020-fgvc7"
#export output_dir='/home/oem/Mingle/datasets/leaf_disease/pathology2020'
#export input_file='train.csv'
#python make_dataset_from_file.py \
#--input_dir ${input_dir} \
#--output_dir ${output_dir} \
#--input_file ${input_file}
#
## test
#export input_dir="/home/oem/Mingle/datasets/leaf_disease/plant-pathology-2020-fgvc7"
#export output_dir='/home/oem/Mingle/datasets/leaf_disease/pathology2020'
#export input_file='test.csv'
#mkdir -p "${output_dir}/test"
#cp ${input_dir}/images/Test* ${output_dir}/test/


#################
## cassava
#################
# train
export input_dir="/home/oem/Mingle/datasets/leaf_disease/cassava"
export output_dir='/home/oem/Mingle/datasets/leaf_disease/cassava2022'
export input_file='train.csv'
python make_dataset_from_file.py \
--input_dir ${input_dir} \
--output_dir ${output_dir} \
--input_file ${input_file} \
--prefix 'train_images' \
--subfix ''
"""
download dataset from https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data
step 1: make every class from train.csv
step 2: make train, val, and test dataset
    setting 1: 1 shot
    setting 2: 5 shot
    setting 3: 10 shot
    setting 4: 20%, 20%, 20% (remain images are not used)
    setting 5: 40%, 20%, 20%
    setting 6: 60%, 20%, 20%
"""

import numpy as np
import pandas as pd
import shutil
import argparse
import os
import os.path as osp

# use seed to reproduce the data split
np.random.seed(15)



# osize = (300, 300)
target_dir = "/data/Mingle/DATASETS_after/cassava/all"
source_dir = "/data/Mingle/DATASETS/Cassava/"
file_name = "/data/Mingle/DATASETS/Cassava/train.csv"


meta_data = pd.read_csv(file_name, header=0)
values = meta_data.values
labels = np.unique(values.T[1])


####################################################################
# step 1: make all datasets
####################################################################
# for label in labels:
#     abs_dir = osp.join(target_dir, str(label))
#     os.makedirs(abs_dir, exist_ok=True)
#
# for i in range(values.shape[0]):
#     img_name = values[i][0]
#     label = values[i][1]
#
#     abs_src_name = osp.join(source_dir, "train_images", img_name)
#     abs_tgt_name = osp.join(target_dir, str(label), img_name)
#
#     shutil.copyfile(abs_src_name, abs_tgt_name)


source_dir = target_dir # "/data/Mingle/DATASETS_after/cassava/all"
####################################################################
# step 2: make train, val dataset for different ratio
####################################################################
train_ratio = [20, 40, 60]
val_ratio = [20, 20, 20]
test_ratio = [20, 20, 20]

for i in range(len(train_ratio)):
    train_ratio_ = train_ratio[i]
    val_ratio_ = val_ratio[i]
    test_ratio_ = test_ratio[i]
    print(f"\nStarting: train: {train_ratio_}, val: {val_ratio_}, test: {test_ratio_}")
    all_ratio = train_ratio_ + val_ratio_ + test_ratio_
    file_dir = f"train{train_ratio_}_val{val_ratio_}_test{test_ratio_}"
    target_dir = osp.join(source_dir.replace('all', ''), file_dir)
    os.makedirs(target_dir, exist_ok=True)

    # make label directory for train, val, test dataset
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        for label in labels:
            dataset_label_dir = osp.join(target_dir, dataset, str(label))
            os.makedirs(dataset_label_dir, exist_ok=True)

    # copy images for each label
    for label in labels:
        src_label_dir = osp.join(source_dir, str(label))
        img_list = sorted(os.listdir(src_label_dir))
        num_img = int(len(img_list) * all_ratio / 100.0)
        random_perm = np.random.permutation(num_img)
        print(f"checking: useful images: {num_img}")

        for j, num in enumerate(random_perm):
            if j <= num_img * train_ratio_ / all_ratio:
                dataset_ = 'train'
            elif j <= num_img * (train_ratio_ + val_ratio_) / all_ratio:
                dataset_ = 'val'
            else:
                dataset_ = 'test'
            abs_src_img_name = osp.join(src_label_dir, img_list[num])
            abs_tgt_img_name = osp.join(target_dir, dataset_, str(label), img_list[num])
            shutil.copyfile(abs_src_img_name, abs_tgt_img_name)
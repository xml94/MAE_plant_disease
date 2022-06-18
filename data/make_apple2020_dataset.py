"""
download dataset from https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data
step 1: make every class from train.csv
step 2: make train, val, and test dataset
    setting 1: 20%, 20%, 20% (remain images are not used)
    setting 2: 40%, 20%, 20%
    setting 3: 60%, 20%, 20%
    setting 4: 80%, 10%, 10%
step 3: make train, val, and test dataset
    setting 1: 1 shot, 20%, 20%
    setting 2: 5 shot, 20%, 20%
    setting 3: 10 shot, 20%, 20%
    setting 4: 20 shot, 20%, 20%
"""

import numpy as np
import pandas as pd
import shutil
import argparse
import os
import os.path as osp
from tqdm import tqdm


###### please rewrite the following directory
abs_source_dir = "/data/Mingle/DATASETS/Apple2020/"
abs_target_dir = "/data/Mingle/DATASETS_after/Apple2020/all"

###### please do NOT change the following codes
###### But you can block some parts
file_name = osp.join(abs_source_dir, "train.csv")
meta_data = pd.read_csv(file_name, header=0)
values = meta_data.values
label_names = meta_data.columns.to_list()[1:]
labels = np.argmax(values[:, 1:], axis=1)

# use seed to reproduce the data split
np.random.seed(15)
# because 15th is my wife's and my son's birthday. Yes, same day!
####################################################################
# step 1: make all datasets for each class
####################################################################
for label in label_names:
    abs_dir = osp.join(abs_target_dir, str(label))
    os.makedirs(abs_dir, exist_ok=True)

for i in tqdm(range(values.shape[0])):
    img_name = values[i][0]
    label = label_names[labels[i]]

    abs_src_name = osp.join(abs_source_dir, "images", img_name + '.jpg')
    abs_tgt_name = osp.join(abs_target_dir, str(label), img_name + '.jpg')

    shutil.copyfile(abs_src_name, abs_tgt_name)



####################################################################
# step 2: make train, val dataset for different ratio
####################################################################
source_dir = abs_target_dir
train_ratio = [20, 40, 60, 80]
val_ratio = [20, 20, 20, 10]
test_ratio = [20, 20, 20, 10]
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
        for label in label_names:
            dataset_label_dir = osp.join(target_dir, dataset, str(label))
            os.makedirs(dataset_label_dir, exist_ok=True)

    # copy images for each label
    for label in label_names:
        src_label_dir = osp.join(source_dir, str(label))
        img_list = sorted(os.listdir(src_label_dir))
        num_img = int(len(img_list) * all_ratio / 100.0)
        random_perm = np.random.permutation(num_img)
        print(f"checking: {label} useful images: {num_img}")

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

####################################################################
# step 3: make train, val dataset for few shot
####################################################################
source_dir = abs_target_dir
few_shot = [1, 5, 10, 20]
val_ratio = [20, 20, 20, 20]
test_ratio = [20, 20, 20, 20]
for i in range(len(few_shot)):
    shots = few_shot[i]
    val_ratio_ = val_ratio[i]
    test_ratio_ = test_ratio[i]
    print(f"\nStarting: train: {shots} shot, val: {val_ratio_}%, test: {test_ratio_}%")
    all_ratio = val_ratio_ + test_ratio_
    file_dir = f"train{shots}shot_val{val_ratio_}_test{test_ratio_}"
    target_dir = osp.join(source_dir.replace('all', ''), file_dir)
    os.makedirs(target_dir, exist_ok=True)

    # make label directory for train, val, test dataset
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        for label in label_names:
            dataset_label_dir = osp.join(target_dir, dataset, str(label))
            os.makedirs(dataset_label_dir, exist_ok=True)

    # copy images for each label
    for label in label_names:
        src_label_dir = osp.join(source_dir, str(label))
        print(src_label_dir)
        img_list = sorted(os.listdir(src_label_dir))
        num_img = int(len(img_list) * all_ratio / 100.0)
        random_perm = np.random.permutation(num_img + shots)
        print(f"checking: {label} useful images: {num_img}")

        for j, num in enumerate(random_perm):
            if j < shots:
                dataset_ = 'train'
            elif j < num_img * val_ratio_ / all_ratio + shots:
                dataset_ = 'val'
            else:
                dataset_ = 'test'
            abs_src_img_name = osp.join(src_label_dir, img_list[num])
            abs_tgt_img_name = osp.join(target_dir, dataset_, str(label), img_list[num])
            shutil.copyfile(abs_src_img_name, abs_tgt_img_name)
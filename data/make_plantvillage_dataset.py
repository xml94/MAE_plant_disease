"""
download dataset from https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color
step 1:
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
import subprocess


###### please rewrite the following directory
abs_source_dir = "/data/Mingle/DATASETS/PlantVillage"
abs_target_dir = "/data/Mingle/DATASETS_after/PlantVillage/raw"

###### please do NOT change the following codes
###### But you can block some parts

# use seed to reproduce the data split
np.random.seed(15)
# because 15th is my wife's and my son's birthday. Yes, same day!
####################################################################
# step 1: copy all datasets
####################################################################
temp_dir = abs_target_dir.replace('raw', '')
if osp.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)
list_files = subprocess.run(["cp", "-r", abs_source_dir, abs_target_dir])
label_names = os.listdir(abs_source_dir)


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
    file_dir = f"train{train_ratio_}"
    target_dir = osp.join(source_dir.replace('raw', ''), file_dir)
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
    file_dir = f"train{shots}shot"
    target_dir = osp.join(source_dir.replace('raw', ''), file_dir)
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
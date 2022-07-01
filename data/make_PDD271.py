"""
download dataset from https://github.com/liuxindazz/PDD271
step 1: change the directory to PDD271_Sample
step 2: make train, val, test
    setting 1: 1 shot, 1 shot, 4 shot
    setting 2: 5 shot, 1 shot, 4 shot
"""

import numpy as np
import pandas as pd
import shutil
import argparse
import os
import os.path as osp
from tqdm import tqdm
import subprocess
from PIL import Image


###### please rewrite the following directory
abs_source_dir = "/data/Mingle/DATASETS/PDD271_Sample"
abs_target_dir = "/data/Mingle/DATASETS_after/PDD271_Sample/all"
label_names = os.listdir(abs_source_dir)

###### please do NOT change the following codes
###### But you can block some parts

# use seed to reproduce the data split
np.random.seed(15)
# because 15th is my wife's and my son's birthday. Yes, same day!
####################################################################
# step 1: copy datasets, resize and limit the maximum number of images for each label
####################################################################
####################################################################
# step 1: copy all datasets
####################################################################
temp_dir = abs_target_dir.replace('all', '')
if osp.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)
src_dir = abs_source_dir
list_files = subprocess.run(["cp", "-r", src_dir, abs_target_dir])
# remove label_rename.txt
rm_dir = osp.join(abs_target_dir, 'label_rename.txt')
os.remove(rm_dir)
label_names = os.listdir(abs_target_dir)


####################################################################
# step 3: make train, val dataset for few shot
####################################################################
source_dir = abs_target_dir
few_shot = [1, 5]
val_shot = [1, 1]
test_shot = [4, 4]
for i in range(len(few_shot)):
    shots = few_shot[i]
    val_shot_ = val_shot[i]
    test_shot_ = test_shot[i]
    all_shot_ = shots + val_shot_ + test_shot_
    print(f"\nStarting: train: {shots} shot, val: {val_shot_}shot, test: {test_shot_}shot")
    file_dir = f"train{shots}shot_val{val_shot_}shot_test{test_shot_}shot"
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
        random_perm = np.random.permutation(all_shot_)
        print(f"checking: {label} useful images: {all_shot_}")

        for j, num in enumerate(random_perm):
            if j < shots:
                dataset_ = 'train'
            elif j < shots + val_shot_:
                dataset_ = 'val'
            else:
                dataset_ = 'test'
            abs_src_img_name = osp.join(src_label_dir, img_list[num])
            abs_tgt_img_name = osp.join(target_dir, dataset_, str(label), img_list[num])
            shutil.copyfile(abs_src_img_name, abs_tgt_img_name)
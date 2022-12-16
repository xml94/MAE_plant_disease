"""
Make train, val, and test dataset according to raw dataset
input: raw dataset directory
    the directory to save the original images
    two types:
        split into train and test
        no split into train and test
output: images directory as train, val, and test.
    two types:
        ratio: 4 settings
            train: 20%, 40%, 60%, 80%
            val: 20%, 20%, 20%, 10%
            test: 20%, 20%, 20%, 10%
        few shot: 4 settings:
            train: 1 shot, 5 shot, 10 shot, 20 shot
            val: 20%, 20%, 20%, 20%
            test: 20%, 20%, 20%, 20%
    if there is test dataset in the original dataset,
    then we do not make test.
 output examples:
    ├── train10shot
        ├── test
        ├── train
        └── val
    ├── train1shot
        ├── test
        ├── train
        └── val
    ├── train20
        ├── test
        ├── train
        └── val
    ├── train40
        ├── test
        ├── train
        └── val
"""
import numpy as np
import shutil
import os.path as osp
import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dset_name', type=str, required=True, help='a dataset with raw dataset')
parser.add_argument('--with_test', action='store_true', help='testing dataset exist in the original dataset')
parser = parser.parse_args()
src_dir = parser.dset_name
with_test = parser.with_test
tgt_path = src_dir
abs_src_dir = osp.join(src_dir, 'raw')

####################################################################
# use seed to reproduce the data split
####################################################################
random_seed = 15
# because 15th is my wife's and my son's birthday. Yes, same day!

if not with_test:
    label_names = os.listdir(osp.join(abs_src_dir, 'train'))
    np.random.seed(15)
    ####################################################################
    # step 1: make train, val dataset for different ratio
    ####################################################################
    train_ratio = [20, 40, 60, 80]
    val_ratio = [20, 20, 20, 10]
    test_ratio = [20, 20, 20, 10]
    for i in range(len(train_ratio)):
        train_ratio_ = train_ratio[i]
        val_ratio_ = val_ratio[i]
        test_ratio_ = test_ratio[i]
        all_ratio = train_ratio_ + val_ratio_ + test_ratio_
        file_dir = f"train{train_ratio_}"
        tgt_plant_split_dir = osp.join(tgt_path, file_dir)
        if osp.exists(tgt_plant_split_dir):
            shutil.rmtree(tgt_plant_split_dir)
        os.makedirs(tgt_plant_split_dir)

        # make label directory for train, val, test dataset
        dset_modes = ['train', 'val', 'test']
        for dset_mode in dset_modes:
            for label in label_names:
                dataset_label_dir = osp.join(tgt_plant_split_dir, dset_mode, str(label))
                os.makedirs(dataset_label_dir)

        # copy images for each label
        for label in label_names:
            src_label_dir = osp.join(abs_src_dir, str(label))
            img_list = sorted(os.listdir(src_label_dir))
            num_img = int(len(img_list) * all_ratio / 100.0)
            np.random.seed(15)
            random_perm = np.random.permutation(num_img)
            for j, num in enumerate(random_perm):
                if j <= num_img * train_ratio_ / all_ratio:
                    dset_mode = 'train'
                elif j <= num_img * (train_ratio_ + val_ratio_) / all_ratio:
                    dset_mode = 'val'
                else:
                    dset_mode = 'test'
                abs_src_img_name = osp.join(src_label_dir, img_list[num])
                abs_tgt_img_name = osp.join(tgt_plant_split_dir, dset_mode, str(label), img_list[num])
                shutil.copyfile(abs_src_img_name, abs_tgt_img_name)

    ####################################################################
    # step 2: make train, val dataset for few shot
    ####################################################################
    few_shot = [1, 5, 10, 20]
    val_ratio = [20, 20, 20, 20]
    test_ratio = [20, 20, 20, 20]
    for i in range(len(few_shot)):
        shots = few_shot[i]
        val_ratio_ = val_ratio[i]
        test_ratio_ = test_ratio[i]
        all_ratio = val_ratio_ + test_ratio_
        file_dir = f"train{shots}shot"
        tgt_plant_split_dir = osp.join(tgt_path, file_dir)
        if osp.exists(tgt_plant_split_dir):
            shutil.rmtree(tgt_plant_split_dir)
        os.makedirs(tgt_plant_split_dir)

        # make label directory for train, val, test dataset
        dset_modes = ['train', 'val', 'test']
        for dset_mode in dset_modes:
            for label in label_names:
                dataset_label_dir = osp.join(tgt_plant_split_dir, dset_mode, str(label))
                os.makedirs(dataset_label_dir, exist_ok=True)

        # copy images for each label
        for label in label_names:
            src_label_dir = osp.join(abs_src_dir, str(label))
            img_list = sorted(os.listdir(src_label_dir))
            num_img = int(len(img_list) * all_ratio / 100.0)
            np.random.seed(random_seed)
            random_perm = np.random.permutation(num_img + shots)
            for j, num in enumerate(random_perm):
                if j < shots:
                    dset_mode = 'train'
                elif j < num_img * val_ratio_ / all_ratio + shots:
                    dset_mode = 'val'
                else:
                    dset_mode = 'test'
                abs_src_img_name = osp.join(src_label_dir, img_list[num])
                abs_tgt_img_name = osp.join(tgt_plant_split_dir, dset_mode, str(label), img_list[num])
                shutil.copyfile(abs_src_img_name, abs_tgt_img_name)
else:
    abs_src_dir = osp.join(abs_src_dir, 'train')
    label_names = os.listdir(abs_src_dir)
    np.random.seed(15)

    ####################################################################
    # step 1: make train, val dataset for different ratio
    ####################################################################
    np.random.seed(15)
    train_ratio = [20, 40, 60, 80]
    val_ratio = [20, 20, 20, 20]
    for i in range(len(train_ratio)):
        train_ratio_ = train_ratio[i]
        val_ratio_ = val_ratio[i]
        all_ratio = train_ratio_ + val_ratio_
        file_dir = f"train{train_ratio_}"
        tgt_plant_split_dir = osp.join(tgt_path, file_dir)
        if osp.exists(tgt_plant_split_dir):
            shutil.rmtree(tgt_plant_split_dir)
        os.makedirs(tgt_plant_split_dir)

        # make label directory for train, val, test dataset
        dset_modes = ['train', 'val']
        for dset_mode in dset_modes:
            for label in label_names:
                dataset_label_dir = osp.join(tgt_plant_split_dir, dset_mode, str(label))
                os.makedirs(dataset_label_dir)

        # copy images for each label
        for label in label_names:
            src_label_dir = osp.join(abs_src_dir, str(label))
            img_list = sorted(os.listdir(src_label_dir))
            num_img = int(len(img_list) * all_ratio / 100.0)
            np.random.seed(random_seed)
            random_perm = np.random.permutation(num_img)
            for j, num in enumerate(random_perm):
                if j <= num_img * train_ratio_ / all_ratio:
                    dset_mode = 'train'
                elif j <= num_img * (train_ratio_ + val_ratio_) / all_ratio:
                    dset_mode = 'val'
                else:
                    continue
                abs_src_img_name = osp.join(src_label_dir, img_list[num])
                abs_tgt_img_name = osp.join(tgt_plant_split_dir, dset_mode, str(label), img_list[num])
                shutil.copyfile(abs_src_img_name, abs_tgt_img_name)

    ####################################################################
    # step 2: make train, val dataset for few shot
    ####################################################################
    few_shot = [1, 5, 10, 20]
    val_ratio = [20, 20, 20, 20]
    for i in range(len(few_shot)):
        shots = few_shot[i]
        val_ratio_ = val_ratio[i]
        all_ratio = val_ratio_
        file_dir = f"train{shots}shot"
        tgt_plant_split_dir = osp.join(tgt_path, file_dir)
        if osp.exists(tgt_plant_split_dir):
            shutil.rmtree(tgt_plant_split_dir)
        os.makedirs(tgt_plant_split_dir)

        # make label directory for train, val, test dataset
        dset_modes = ['train', 'val']
        for dset_mode in dset_modes:
            for label in label_names:
                dataset_label_dir = osp.join(tgt_plant_split_dir, dset_mode, str(label))
                os.makedirs(dataset_label_dir, exist_ok=True)

        # copy images for each label
        for label in label_names:
            src_label_dir = osp.join(abs_src_dir, str(label))
            img_list = sorted(os.listdir(src_label_dir))
            num_img = int(len(img_list) * all_ratio / 100.0)
            np.random.seed(random_seed)
            random_perm = np.random.permutation(num_img + shots)
            for j, num in enumerate(random_perm):
                if j < shots:
                    dset_mode = 'train'
                elif j < num_img * val_ratio_ / all_ratio + shots:
                    dset_mode = 'val'
                else:
                    continue
                abs_src_img_name = osp.join(src_label_dir, img_list[num])
                abs_tgt_img_name = osp.join(tgt_plant_split_dir, dset_mode, str(label), img_list[num])
                shutil.copyfile(abs_src_img_name, abs_tgt_img_name)

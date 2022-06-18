"""

Aim to split a dataset into train, val, and test, for image classification.
Input:
    src_dir: <str>, the source directory including many images under the class
    tgt_dir: <str>, the target directory you want to save the train, val dataset
    mode: <str>
        number: put how many images for train for each class, for example, 1, 5, 10 (few-shot)
        ratio: put the percentage of images for each class, for example, 1%, 10%, 50%
    number: <list of <int>>, for example (10, 10), the rest will be test
    ratio: <list of <float>>, for example (0.05, 0.05), 1- 0.05 - 0.05 will be test
    random: <bool>, how to split the images, random or from sorted
Example:
    src_dir: dir_a
        --class_1
            -img_1
            -img_2
        --class_2
            -img_3
            -img_4
    target_dir: dir_b
        --train
            --class_1
                --img_1
                --img_2
            --class_2
        --val
"""
import os
import shutil
import os.path as osp
import argparse
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cal_num_for_class(dir, number, ratio, mode):
    n_images = len(os.listdir(dir))
    print(f"checking: {n_images} images in total.")

    number_ = []
    if mode.lower() == 'number':
        number_.append(int(number[0]))
        number_.append(int(n_images - number_[0]))
        assert number_[1] >= 0, "number of train and val should be less than the number of total images."
    else:

        assert mode.lower() == 'ratio', "please check mode, should be in [number, ratio]!"
        r_train = ratio[0]
        r_val = 1 - r_train
        assert r_val >= 0, "Please make sure that the sum of ratio should be less than one!"
        number_.append(int(np.floor(r_train * n_images)))
        number_.append(int(n_images - number_[0]))
    return number_


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, required=True, help='the dir includes class directory')
parser.add_argument('--tgt_dir', type=str, required=True, help='the dir you want to save the split dataset')
parser.add_argument('--mode', type=str, choices=['number', 'ratio'], default='ratio', help='the mode to split')
parser.add_argument('--number_train', type=int, default=10, help='how many images for train')
parser.add_argument('--ratio_train', type=float, default=0.60, help='the percentage of images for train')
parser.add_argument('--random', type=str2bool, default=False, help='how to split the image, random or sorted')
opt = parser.parse_args()


src_dir, tgt_dir = opt.src_dir, opt.tgt_dir
class_names = sorted(os.listdir(src_dir))
number = [opt.number_train, 0]
ratio = [opt.ratio_train, 1 - opt.ratio_train]
mode = opt.mode

for class_name in class_names:
    os.makedirs(osp.join(tgt_dir, 'train', class_name), exist_ok=True)
    os.makedirs(osp.join(tgt_dir, 'val', class_name), exist_ok=True)

    abs_src_dir = osp.join(src_dir, class_name)
    split_number = cal_num_for_class(abs_src_dir, number, ratio, mode)
    n_images = len(os.listdir(abs_src_dir))

    # how to split the data, random or not
    if opt.random:
        permute = np.random.permutation(n_images)
    else:
        permute = list(range(n_images))
    permute[0:split_number[0]] = np.zeros(split_number[0])
    permute[split_number[0]:split_number[1]] = np.ones(split_number[1])

    for num, file in enumerate(os.listdir(abs_src_dir)):
        abs_src_file = osp.join(abs_src_dir, file)
        if permute[num] == 0:
            abs_tgt_file = osp.join(tgt_dir, 'train', class_name, file)
        else:
            assert permute[num] == 1
            abs_tgt_file = osp.join(tgt_dir, 'val', class_name, file)
        shutil.copyfile(abs_src_file, abs_tgt_file)

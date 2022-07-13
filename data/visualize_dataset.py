"""
Visualize a dataset for classification
input: a directory
    class_1: img_1, img_2, ...
    class_2: img_1, img_2, ...
    ...
output: a directory
    class_1_img, class_2_img, ...
"""
import os
import os.path as osp
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='',
                    help='the directory to save all plant disease dataset')
parser.add_argument('--src_dir', type=str, required=True,
                    help='name of plant disease dataset')
parser.add_argument('--num', type=str, default=3,
                    help='how many images to show for each class')
parser.add_argument('--test', type=int, default=0, help='if test dataset exist in the original')
parser = parser.parse_args()

num = parser.num
test = parser.test
src_dir = osp.join(parser.base_dir, parser.src_dir, 'all')
tgt_dir = osp.join(parser.base_dir, 'vis', parser.src_dir)
os.makedirs(tgt_dir, exist_ok=True)

np.random.seed(15)

if os.path.exists(tgt_dir):
    shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)
else:
    os.makedirs(tgt_dir)

if not test:
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            abs_dir = os.path.join(root, dir_name)
            file_list = os.listdir(abs_dir)
            if len(file_list) > num:
                rands = np.random.randint(len(file_list), size=num)
                for i in range(num):
                    src_file_name = os.path.join(abs_dir, file_list[rands[i]])
                    tgt_file_name = os.path.join(tgt_dir, dir_name + f"_{i}.jpg")
                    shutil.copyfile(src_file_name, tgt_file_name)
else:
    for root, dirs, files in os.walk(osp.join(src_dir, 'train')):
        for dir_name in dirs:
            abs_dir = os.path.join(root, dir_name)
            file_list = os.listdir(abs_dir)
            if len(file_list) > num:
                rands = np.random.randint(len(file_list), size=num)
                for i in range(num):
                    src_file_name = os.path.join(abs_dir, file_list[rands[i]])
                    tgt_file_name = os.path.join(tgt_dir, dir_name + f"_{i}.jpg")
                    shutil.copyfile(src_file_name, tgt_file_name)
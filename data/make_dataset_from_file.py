"""
Make dataset from file
input:
    file_name: (train.csv)
        image_id
        label
    mode: one-hot label or convert to one-hot label
    labels: 0-n
output:
    0: <class_name>
        --img1
        --img2
    1: <class_name>
    ...
"""
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='the source directory')
parser.add_argument('--input_file', type=str, required=True, help='the text file')
parser.add_argument('--output_dir', type=str, required=True, help='the dir to save the images')
parser.add_argument('--prefix', type=str, default='images', help='the folder with images')
parser.add_argument('--subfix', type=str, default='.jpg', help='the source image format')
parser.add_argument('--stage', type=str, default='train', help='train or test')
opt = parser.parse_args()

input_dir = opt.input_dir
output_dir = opt.output_dir
file_name = osp.join(input_dir, opt.input_file)
stage = opt.stage

file = pd.read_csv(file_name, )
keys = file.columns.tolist()
values = file.values
img_names = values[:, 0]
img_labels = values[:, 1:]

if len(keys) > 2: #one-hot label
    class_names = keys[1:]
    img_labels = np.argmax(img_labels, axis=1)
    assert np.max(img_labels) == len(class_names) - 1
else: # not one-hot label
    class_names = list(range(np.max(img_labels + 1)))

# make dataset class directory
for name in class_names:
    abs_dir = osp.join(output_dir, stage, str(name))
    os.makedirs(abs_dir, exist_ok=True)

if opt.prefix:
    input_dir = osp.join(input_dir, opt.prefix)

for idx, img_name in enumerate(img_names):
    abs_src_img_name = osp.join(input_dir, img_name + opt.subfix)
    label_idx = img_labels[idx]
    # print(type(label_idx))
    # print(label_idx)
    # print(class_names[label_idx])
    abs_tgt_img_name = osp.join(output_dir, stage, str(class_names[int(label_idx)]), img_name + opt.subfix)
    shutil.copyfile(abs_src_img_name, abs_tgt_img_name)
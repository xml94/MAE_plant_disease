"""
Visualize a dataset for classification
input: a directory
    class_1: img_1, img_2, ...
    class_2: img_1, img_2, ...
    ...
output: a directory
    image and their labels
"""
import os
import os.path as osp
import shutil
import numpy as np
import argparse

from PIL import Image
from torchvision.utils import make_grid, save_image
import torch
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as F
plt.rcParams["savefig.bbox"] = 'tight'
matplotlib.rcParams['savefig.dpi'] = 1200


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='',
                    help='the directory to save all plant disease dataset')
parser.add_argument('--src_dir', type=str, required=True,
                    help='name of plant disease dataset')
parser.add_argument('--num', type=str, default=3,
                    help='how many images to show for each class')
parser.add_argument('--test', type=int, default=0, help='if test dataset exist in the original')
parser.add_argument('--osize', type=list, default=(512, 512),
                    help="the size of every image after resizing")
parser.add_argument('--copy', type=int, default=1,
                    help="if copy the original images into the new visualization directory")
parser = parser.parse_args()

num = parser.num
test = parser.test
osize = parser.osize
copy = parser.copy
if not test:
    src_dir = osp.join(parser.base_dir, parser.src_dir, 'raw')
else:
    src_dir = osp.join(parser.base_dir, parser.src_dir, 'raw', 'train')
if not osp.exists(src_dir):
    print('Please check if the directory exist: {src_dir}')
tgt_dir = osp.join(parser.base_dir, 'vis', parser.src_dir)
os.makedirs(tgt_dir, exist_ok=True)

np.random.seed(15)

if os.path.exists(tgt_dir):
    shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)
else:
    os.makedirs(tgt_dir)


img_list = []
label = "Label name from the second row to the last"
for root, dirs, files in os.walk(src_dir):
    for dir_name in dirs:
        abs_dir = os.path.join(root, dir_name)
        file_list = os.listdir(abs_dir)
        if len(file_list) > num:
            label = label + '\n' + str(dir_name)
            rands = np.random.randint(len(file_list), size=num)
            for i in range(num):
                src_file_name = os.path.join(abs_dir, file_list[rands[i]])
                tgt_file_name = os.path.join(tgt_dir, dir_name + f"_{i}.jpg")
                if copy:
                    shutil.copyfile(src_file_name, tgt_file_name)
                img = Image.open(src_file_name).convert('RGB')
                img = img.resize(osize)
                img_list.append(torch.tensor(np.array(img).transpose(2, 0, 1)))
img = make_grid(img_list, nrow=num)
img_name = osp.join(osp.join(parser.base_dir, 'vis'), parser.src_dir + '_all_img.png')
show(img)
plt.savefig(img_name)
label_name = osp.join(osp.join(parser.base_dir, 'vis'), parser.src_dir + 'all_label.txt')
with open(label_name, 'w') as file:
    file.writelines(label)

print(f"Done for {parser.src_dir}")
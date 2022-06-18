"""
Split image dataset based on meta-data.
Input:
    meta-data.csv
Output:
    train <dir>
        cls1: img_1, ...
        cls2, img_n, ...
        ...
    val <dir>:
        ...
"""
import pandas as pd
import os.path as osp
import os
import shutil


prefix = "train"
base = "/home/oem/Mingle/datasets/FungiCLEF2022"
data_dir = f"{base}/DF20-{prefix}_metadata.csv"
src_base = f"/home/oem/Mingle/datasets/FungiCLEF2022/train_val_300/DF20_300"
tgt_base = f"/home/oem/Mingle/datasets/FungiCLEF2022/{prefix}"
os.makedirs(tgt_base, exist_ok=True)

# image_path: 33
# class_id: 34
img_path_idx = 33 - 1
class_idx = 34 - 1
meta_data = pd.read_csv(data_dir, header=0)

values = meta_data.values
labels = meta_data.columns.to_list()
image_paths = values.T[img_path_idx]
class_ids = values.T[class_idx]
assert labels[img_path_idx] == 'image_path'
assert labels[class_idx] == 'class_id'

# print(len(image_paths))
# print(image_paths[0])

for idx, img_path in enumerate(image_paths):
    img_path = img_path.replace('JPG', 'jpg')
    class_id = class_ids[idx]
    abs_tgt_class_path = osp.join(tgt_base, str(class_id))
    if not osp.exists(abs_tgt_class_path):
        os.makedirs(abs_tgt_class_path)

    abs_src_img_path = osp.join(src_base, img_path)
    abs_tgt_img_path = osp.join(abs_tgt_class_path, img_path)
    shutil.copyfile(abs_src_img_path, abs_tgt_img_path)

# /home/oem/Mingle/datasets/FungiCLEF2022/train_val_300/DF20_300/2238546328-30620.JPG
# /home/oem/Mingle/datasets/FungiCLEF2022/train_val_300/
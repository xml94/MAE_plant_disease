import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path as osp
import argparse
plt.rcParams["savefig.bbox"] = 'tight'
matplotlib.rcParams['savefig.dpi'] = 1200


def read_acc(file_name):
    with open(file_name) as file:
        meta_data = file.readlines()
        val_acc  = []
        for data in meta_data:
            val_acc.append(np.array(data.split(" ")[7].replace(',', ''), dtype=float))
        val_acc = np.array(val_acc)
    return val_acc


parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, required=True,
                    help='the dataset')
parser.add_argument('--vis_dir', type=str, default='analysis_results', help='the director to save images')
parser.add_argument('--base', type=str, default='/data/Mingle/checkpoints/', help='the base directory to save the files')
parser = parser.parse_args()

base_folder = parser.base
dset = parser.dset

dset_name ={
'Apple2020': 'Apple2020',
'Apple2021': 'Apple2021',
'cassava': "Cassava",
'CGIAR_wheat': "CGIARWheat",
'ChineseStrawberry': "Strawberry2021",
'CitrusLeaf': "CitrusLeaf",
'IVADL_rose': "IVADLRose",
'IVADL_tomato': "IVADLTomato",
'PlantDoc_cls': "PlantDocCls",
'PlantVillage': "PlantVillage",
'Rice1462': "Rice1426",
'Rice2020': "Rice5932",
'TaiwanTomato': "TaiwanTomato"
}



methods = ["CNN", "CNN_super", "MOCO", "ViT", "ViT_IN", "MAE_IN", 'MAE_CLEF']
dset_modes = ['train1shot', 'train5shot', 'train10shot', 'train20shot',
              "train20", "train40", 'train60', 'train80']
titles = ['1-shot', '5-shot', '10-shot', '20-shot', 'Ratio20', 'Ratio40', 'Ratio60', 'Ratio80']
legend = ["RN50" ,"RN50-IN", "MoCo-v2", "ViT", "ViT-IN", "MAE", "Ours"]
markers = ['-^', '-<', '-v', '->', '-p', '-8', '-*']
colors = ['k', 'b', 'g', 'purple', 'pink', 'orange', 'r']

fig = plt.figure()
axes = fig.subplots(nrows=2, ncols=4)
for num, ax in enumerate(fig.axes):
    dset_mode = dset_modes[num]
    if 'shot' not in dset_mode:
        src_dir = osp.join(base_folder, 'ratio')
    else:
        src_dir = osp.join(base_folder, 'few_shot')

    # compare the convergence speed in validation dataset.
    acc = []
    for method in methods:
        file_name = osp.join(src_dir, dset + f"_{dset_mode}_{method}", "log.txt")
        acc.append(read_acc(file_name))

    for i in range(len(acc)):
        ax.plot(range(1,47,5), acc[i], markers[i], color = colors[i], label=legend[i])

    if num == 0 or num == 4:
        ax.set_ylabel('Val acc')
    if num >= 4:
        ax.set_xlabel('Epoch')
    # if num < 4:
    #     ax.set_xticks([])
    # else:
    #     ax.set_xticks(np.arange(0, 46, 10))
    if num == 0 or num == 4:
        ax.set_yticks(np.arange(0, 101, 20))
    else:
        ax.set_yticks([])
    ax.set_title(titles[num])
    ax.set_xticks(np.arange(0, 51, 10))

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5,0.0), borderaxespad=0)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
fig.suptitle(str(dset_name[dset]))

plt.gcf().set_size_inches(10, 6)
# plt.show()
plt.savefig(osp.join(parser.vis_dir, f"{dset}.svg"))
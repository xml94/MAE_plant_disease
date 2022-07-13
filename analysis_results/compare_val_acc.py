import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

file_1 = "PlantVillage_train20_val20_test20_finetune_MAE"
file_2 = "PlantVillage_train20_val20_test20_finetune_PlantCLEF2022"

base_folder = "checkpoint"
file_1 = osp.join(base_folder, file_1, "log.txt")
file_2 = osp.join(base_folder, file_2, "log.txt")

def read_acc(file_name):
    with open(file_name) as file:
        meta_data = file.readlines()
        val_acc  = []
        for data in meta_data:
            val_acc.append(np.array(data.split(" ")[7].replace(',', ''), dtype=float))
        val_acc = np.array(val_acc)
    return val_acc

acc_1 = read_acc(file_1)
acc_2 = read_acc(file_2)

# plt.plot(np.arange(len(acc_1)), acc_1, 'b-')
# plt.plot(np.arange(len(acc_2)), acc_2, 'r-')
plt.plot(acc_1, 'b-*')
plt.plot(acc_2, 'r-o')
plt.xlabel('Epoch')
plt.ylabel('Val acc')
# plt.ylim(0, 100)
# plt.xlim(-1, 60)
# plt.show()
plt.legend(["ImageNet", "PlantCLEF2022"])
plt.savefig('PlantVillage_train20.jpg')
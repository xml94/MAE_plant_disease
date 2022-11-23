## Dataset

### Info of original plant related datasets

| Name         | env      | Plant             | img         | class | Paper                                                                                   | Dataset                                                                                                           |
|--------------|----------|-------------------|-------------|-------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| PlantVillage | lab      | Multiple leaf     | 54,305      | 38    | [Paper](https://arxiv.org/abs/1511.08060)                                               | [Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)                                |
| PlantDocCls  | internet | Multiple leaf     | 2,598       | 27    | [Paper](https://dl.acm.org/doi/pdf/10.1145/3371158.3371196)                             | [Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)                                                        |
| TaiwanTomato | real+lab | Tomato leaf       | 622         | 5     |                                                                                         | [Dataset](https://data.mendeley.com/datasets/ngdgg79rzb/1)                                                        |
| IVADLTomato  | real     | Tomato leaf       | 17,063      | 9     |                                                                                         | [Dataset](https://github.com/IVADL/tomato-disease-detector)                                                       |
| IVADLRose    | real     | Rose leaf         | 23,114      | 6     |                                                                                         | [Dataset](https://github.com/IVADL/tomato-disease-detector)                                                       |
| Apple2020    | real     | Apple leaf        | 3,642       | 4     | [Paper](https://bsapubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aps3.11390)       | [Dataset](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data)                                    |
| Apple2021    | real     | Apple leaf        | 18,632      | 6     | [Paper](https://vision.cornell.edu/se3/wp-content/uploads/2021/09/029.pdf)              | [Dataset](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data)                                    |
| Cassava      | real     | Cassava leaf      | 21,397      | 5     | [Paper](https://www.frontiersin.org/articles/10.3389/fpls.2017.01852/full)              | [Dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data)                           |
| Citrus       | lab      | Citrus fruit leaf | 105 + 609   | 5 + 5 | [Paper](https://www.sciencedirect.com/science/article/pii/S2352340919306948?via%3Dihub) | [Dataset](https://data.mendeley.com/datasets/3f83gxmv57/2)                                                        |
| Rice5932     | real     | Rice leaf         | 5,932       | 4     | [Paper](https://www.sciencedirect.com/science/article/pii/S0168169919326997)            | [Dataset](https://data.mendeley.com/datasets/fwcj7stb8r/1)                                                        |
| Rice1426     | real     | Rice leaf         | 1426        | 9     | [Paper](https://www.sciencedirect.com/science/article/pii/S1537511020300830?via%3Dihub) | [Dataset](https://drive.google.com/drive/folders/1ewBesJcguriVTX8sRJseCDbXAF_T4akK)                               |
| CGIARWheat   | real     | wheat             | 876         | 3     |                                                                                         | [Dataset](https://www.kaggle.com/datasets/shadabhussain/cgiar-computer-vision-for-crop-disease?resource=download) |
| PDD271       | real     | Multiple leaf     | 220,592     | 271   | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9325065&tag=1)             | [Sample](https://github.com/liuxindazz/PDD271)                                                                    |

Refer to [visualize_dataset/dataset.md](https://github.com/xml94/MAE_plant_disease/blob/main/visualize_dataset/dataset.md) to see the detail info:
* 3 random images for each label
* the number of images for each label

### Prepare the dataset for this project
You can download the original dataset used their links

To make the dataset for this project. After downloading the datasets: use ```./data/make_*.py```

To visualize the images for each datataset and each class
* use ```visualize_dataset/vis_dset.sh```

## PlantCLEF2022
* You can download the pretrained model [here](https://github.com/xml94/PlantCLEF2022) and see [the paper](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2022/paper-179.pdf)


## Train and test
* For CNN-based, normal case see ```/cnn_scripts/train.sh``` and ```/cnn_scripts/test.sh```
* For CNN-based, few-shot case see ```/cnn_scripts/few_shot.sh``` and ```/cnn_scripts/test_few_shot.sh```
* For ViT-based, see ```/scripts/train.sh``` and ```/scripts/test.sh```
* For ViT-based, few-shot case see ```/scripts/few_shot.sh``` and ```/scripts/test_few_shot.sh```

## Cite our paper
* [Published paper link](https://www.frontiersin.org/articles/10.3389/fpls.2022.1010981/full)

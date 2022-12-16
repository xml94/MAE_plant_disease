## Make a correct dataset to train and test
### Some basic information
Original dataset may already split the dataset.
* train
* train and test

MAE model uses ImageNet data format to train an image classifier.
* must with train and val datasets
* 

### In this project
* split the original dataset into train, val, test.
  * when the original dataset has test, split the original train dataset into train and val datasets.
  * the train dataset is used to train the model.
  * the val dataset is for validating the trained model and keep the best model.
  * the test dataset is finally used to test the best trained model.
* utilize different number of images in train dataset.
  * ratio: use different percentages to train the model, such as 80%, 60%, 40%, 20%.
  * few-shot: use only several image for each class to train, such as 1-shot, 5-shot, 10-shot, and 20-shot.
  * to have a fair comparison, we use more image to test, 20% for each class but 10% when using 80% for training.
  * all experiments use same test dataset except 80% for training.


### How to make dataset for this project
* make sure this kind of directory
```angular2html
├── dataset name
│   ├── raw
│   │   ├── class name 1
│   │   │   ├──img_1.jpg
│   │   │   ├──img_2.jpg
│   │   │   ├──img_3.jpg
│   │   ├── class name 2
│   │   ├── class name 3
│   ├── test
│   │   ├── train
│   │   └── val
```
```angular2html
│   ├── train10shot
│   │   ├── test
│   │   ├── class name 1
│   │   │   ├──img_1.jpg
│   │   │   ├──img_2.jpg
│   │   │   ├──img_3.jpg
│   │   ├── class name 2
│   │   ├── class name 3
│   │   ├── train
│   │   └── val
│   ├── train1shot
│   ├── train20
│   ├── train20shot
│   ├── train40
│   ├── train5shot
│   ├── train60
│   ├── train80

```
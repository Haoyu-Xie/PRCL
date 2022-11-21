# Boosting Pixel-Wise Contrastive Learning with Probabilitic Representations
This repository contains the source code of PRCL and baselines from the paper, [Boosting Pixel-Wise Contrastive Learning with Probabilitic Representations] (https://arxiv.org/abs/2210.14670), proposed by Haoyu Xie, Changqi Wang, Mingkai Zheng, Minjing Dong, Shan You, Chong Fu, and Chang Xu.

## Updates
**Nov. 2022** -- Upload the sorce code.

## Prepare
PRCL is evaluated with two datasets: PASCAL VOC 2012 and CityScapes. 
- For PASCAL VOC, please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar` and the augmented labels [here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip): `SegmentationClassAug.zip`. 
Extract the folder `JPEGImages` and `SegmentationClassAug` as follows:
```data
├── data
│   ├── VOCdevkit
│   │   ├──VOC2012
│   │   |   ├──JPEGImages
│   │   |   ├──SegmentationClassAug
│   │   |   ├──prefix
│   │   |   |   ├──val.txt
│   │   |   |   ├──train_aug.txt

```
- For CityScapes, please download the original images and labels from the [official CityScapes site](https://www.cityscapes-dataset.com/downloads/): `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
Extract the folder `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` as follows:
```data
├── data
│   ├── cityscapes
│   │   ├──leftImg8bit
│   │   |   ├──train
│   │   |   ├──val
│   │   ├──train
│   │   ├──val
```
Folders `train` and `val` under `leftImg8bit` contains training and validation images while folders `train` and `val` under `leftImg8bit` contains labels.
- For pretrained models, please download the model pretrained on Imagenet from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth) and change the dir in the train_res100.py.

## Run
Running the following script: 
```
run ./script/batch_train.sh
```

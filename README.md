# Boosting Pixel-Wise Contrastive Learning with Probabilitic Representations
## Prepare
Please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar` and the augmented labels [here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip): `SegmentationClassAug.zip`. 
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
Please download the model pretrained on Imagenet from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth) and change the dir in the train_res100.py.

## Run
Running the following script: 
```
run ./script/batch_train.sh
```

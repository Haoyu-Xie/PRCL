# Boosting Pixel-Wise Contrastive Learning with Probabilitic Representations (AAAI 2023)
![https://github.com/Haoyu-Xie/PRCL/blob/main/PRCL.gif](https://github.com/Haoyu-Xie/PRCL/blob/main/PRCL.gif)
This repository contains the source code of **PRCL** from the paper [Boosting Pixel-Wise Contrastive Learning with Probabilitic Representations](https://arxiv.org/abs/2210.14670).

In this paper, we redefine the representation in pixel-wise contrastive learning from a perspective of probability theory. We consider the probability and model the representation as random variable, namely **Probabilistic Representation**. 
## Updates
**Nov. 2022** -- Upload the sorce code.

## Prepare
PRCL is evaluated with two datasets: PASCAL VOC 2012 and CityScapes. 
- For PASCAL VOC, please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar` and the augmented labels [here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip): `SegmentationClassAug.zip`. 
Extract the folder `JPEGImages` and `SegmentationClassAug` as follows:
```
├── data
│   ├── VOCdevkit
│   │   ├──VOC2012
│   │   |   ├──JPEGImages
│   │   |   ├──SegmentationClassAug
```
- For CityScapes, please download the original images and labels from the [official CityScapes site](https://www.cityscapes-dataset.com/downloads/): `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
Extract the folder `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` as follows:
```
├── data
│   ├── cityscapes
│   │   ├──leftImg8bit
│   │   |   ├──train
│   │   |   ├──val
│   │   ├──train
│   │   ├──val
```
Folders `train` and `val` under `leftImg8bit` contains training and validation images while folders `train` and `val` under `leftImg8bit` contains labels.

The data split folder of VOC and CityScapes is as follows:
```
├── VOC(CityScapes)_split
│   ├── labeled number
│   │   ├──seed
│   │   |   ├──labeled_filename.txt
│   │   |   ├──unlabeled_filename.txt
│   │   |   ├──valid_filename.txt
```
You need to change the name of folders (labeled number and seed) according to your actual experiments.

PRCL uses ResNet-101 pretrained on ImageNet and ResNet-101 with deep stem block, please download from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth) for ResNet-101 and [here](https://drive.google.com/file/d/131dWv_zbr1ADUr_8H6lNyuGWsItHygSb/view?usp=sharing) for ResNet-101 stem. Remember to change the directory in corresponding python file.

In order to install the correct environment, please run the following script:
```
conda create -n prcl python=3.8.5
conda activate prcl
pip install -r requirements.txt
```
It may takes a long time, take a break and have a cup of coffee!
It is OK if you want to install environment manually, remember to check CAREFULLY!

## Run
You can run our code with a single GPU or multiple GPUs.
- For single GPU users, please run the following script:
```
python prcl_sig.py [--config]
```
You need to change the file name after --config according to your actual experiments.
- For multiple GPUs users, please run the following script: 
```
run ./script/batch_train.sh
```
We provide 662 labels for VOC and 150 labels for CityScapes, the seed in our experiments is 3407. You can change the label rate and seed as you like, remember to change the corresponding config files and data_split directory.
## Hyper-parameters
All hyper-parameters used in the code are shown below:
|Name        | Discription  |  Value |
| :-: |:-:| :-:|
| `alpha`     | hyper-parameter in EMA model  |  `0.99`  |
| `lr`     | learning rate of backbone, prediction head, and project head  |  `3.2e-3`  |
| `uncer_lr`     | learning rate of probability head  |  `5e-5`  |
| `un_threshold`     | threshold in unsupervised loss  |  `0.97`  |
| `weak_threshold`     | weak threshold in PRCL loss  |  `0.7`  |
| `strong_threshold`     | strong threshold in PRCL loss  |  `0.8`  |
| `temp`     | temperature in PRCL loss  |  `100`  |
| `num_queries`     | number of queries in PRCL loss  |  `256`  |
| `num_negatives`     | number of negatives in PRCL loss  |  `512`  |
| `begin_epoch`     | the begin epoch of scheduler $\lambda_c$  |  `0`  |
| `max_epoch`     | the end epoch of scheduler $\lambda_c$  |  `200`  |
| `max_value`     | the max value of scheduler $\lambda_c$  |  `1.0`  |
| `min_value`     | the min value of scheduler $\lambda_c$  |  `0`  |
| `ramp_mult`     | the $\alpha$ of scheduler $\lambda_c$  |  `-5.0`  |

**It is worth noting that uncer_lr is very sensitive and training may crash if uncer_lr is not fine-tuned CAREFULLY.**

## Acknowledgement
The data processing and augmentation (CutMix, CutOut, and ClassMix) are borrowed from ReCo.
- ReCo: https://github.com/lorenmt/reco

Thanks a lot for their splendid work!

## Citation
If you think this work is useful for you and your research, please considering citing the following:
```
@article{PRCL,
  title={Boosting Semi-Supervised Semantic Segmentation with Probabilistic Representations},
  author={Xie, Haoyu and Wang, Changqi and Zheng, Mingkai and Dong, Minjing and You, Shan and Xu, Chang},
  journal={arXiv preprint arXiv:2210.14670},
  year={2022}
}
```
More interesting works based on Probabilistic Representations are coming soon. 👣

## Contact
If you have any questions or meet any problems, please feel free to contact us.
- Haoyu Xie, [895852154@qq.com](mailto:895852154@qq.com)
- Changqi Wang, [wangchangqi98@gmail.com](mailto:wangchangqi98@gmail.com)

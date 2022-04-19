# Author: Changqi Wang 
import os
from cv2 import transform
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter
import h5py
import numpy as np
import torch    
from torchvision import transforms
import random
import torchvision.transforms.functional as transforms_f
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from generalframeworks.utils import class2one_hot

##### get index from list #####
def get_pascal_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_name = root + '/labeled_filename.txt'
    else:
        file_name = root + '/val_filename.txt'
    with open(file_name) as f:
        idx_list = f.read().splitlines()

    # return label index and unlabel index
    if train:
        label_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(21)
        label_fill = np.arange(21)
        while len(label_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []
            mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
            mask_unique = np.unique(mask)[:-1] if 225 in mask else np.unique(mask)
            unique_num = len(mask_unique)

            # sample image if it includes the lowest appeared class and with more than 3 distinctive classes
            if len(label_idx) == 0 and unique_num >= 3:
                label_idx.append(idx)
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
                label_idx.append(idx)
            else:
                save_idx.append(idx)

        return label_idx, [idx for idx in idx_list if idx not in label_idx]
    else:
        return idx_list

##### Data augmentation #####

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))    #  For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # To Tensor and fix label
    image = transforms_f.to_tensor(image)
    label = transforms_f.to_tensor(label)
    label = (label * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),\
                    augmentation=True, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
    
    def __getitem__(self, idx):
        image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[idx]))
        label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[idx]))
        image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
        return image, label.squeeze(0)

    def __len__(self):
        return len(self.idx_list)

class BuildDataloader:
    def __init__(self, dataset, data_path, batch_size, num_labels):
        self.dataset = dataset
        if self.dataset == 'pascal':
            self.data_path = data_path
            self.img_size = [513, 513]
            self.crop_size = [256, 256]
            self.num_segments = 21
            self.scale_size = (0.5, 1.5)
            self.batch_size = batch_size
            self.num_labels = num_labels
            self.train_lab_idx, self.train_unlab_idx = get_pascal_idx(self.data_path, train=True, label_num=self.num_labels)
            self.test_idx = get_pascal_idx(self.data_path, train=False)
        if num_labels == 0:
            self.train_lab_idx = self.train_unlab_idx
    def build(self, supervised=False):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_lab_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True)
        train_u_dataset = BuildDataset(self.data_path, self.dataset, self.train_unlab_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=True)
        test_dataset    = BuildDataset(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.img_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        train_l_loader = DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset),
            drop_last=True,
        )

        if not supervised:
            train_u_loader = DataLoader(
                train_u_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(data_source=train_u_dataset),
                drop_last=True,
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
        )
        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, test_loader
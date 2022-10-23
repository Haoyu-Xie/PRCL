from turtle import right
from cv2 import mean
from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import random
import glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import torch
import os
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)

from generalframeworks.augmentation.transform import denormalise

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Randomly rescale images
    raw_w, raw_h = image.size
    # print(scale_size)
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image is smaller than crop size
    if crop_size == -1: # Use original image size
        crop_size = (raw_w, raw_h)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad = max(crop_size[1] - resized_size[1], 0)
        bottom_pad = max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
    
    # Randomly crop images
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)
    
    if augmentation:
        # Random color jittering
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)
        
        # Random Gaussian filtering
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)
        
    # Transform to Tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1 # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply ImageNet normalization
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def tensor_to_pil(image, label, logits=None):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())
    if logits is not None:
        logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
        return image, label, logits
    else:
        return image, label


def batch_transform(images, labels, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label_list, logits_list = [], [], []
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label_pil, logits_pil = tensor_to_pil(images[k], labels[k], logits[k])
        aug_image, aug_label, aug_logits = transform(image_pil, label_pil, logits_pil, crop_size, scale_size, augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)
    if logits is not None:
        image_trans, label_trans, logits_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
        return image_trans, label_trans
    else:
        image_trans, label_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device)
        return image_trans, label_trans

def batch_transform_nologits(images, labels, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label_list = [], []
    # print(scale_size)
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label_pil = tensor_to_pil(images[k], labels[k])
        aug_image, aug_label = transform(image=image_pil, 
                                         label=label_pil, 
                                         crop_size=crop_size, 
                                         scale_size=scale_size, 
                                         augmentation=augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
    image_trans, label_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device)
    return image_trans, label_trans

def get_pascal_idx_via_txt(root, seed):
    '''
    Read idx list via generated txt, pre-perform make_list.py
    '''
    root = os.path.expanduser(root)
    with open(root + '/prefix/my_big_subset/' + str(seed) + '/labeled_filename.txt') as f:
        labeled_list = f.read().splitlines()
    f.close()
    with open(root + '/prefix/my_big_subset/' + str(seed) + '/unlabeled_filename.txt') as f:
        unlabeled_list = f.read().splitlines()
    f.close()
    with open(root + '/prefix/my_subset/val.txt') as f:
        test_list = f.read().splitlines()
    f.close()
    return labeled_list, unlabeled_list, test_list
    
def get_pascal_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_name = root + 'prefix/train_aug.txt'
    else:
        file_name = root + 'prefix/val.txt'
    with open(file_name) as f:
        idx_list = f.read().splitlines()

    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(21)
        label_fill = np.arange(21)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []
            mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            unique_num = len(mask_unique)   # number of unique classes

            # sample image if it includes the lowest appeared class and with more than 3 distinctive classes
            if len(labeled_idx) == 0 and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowest appearance
            label_fill = np.where(label_counter == label_counter.min())[0]

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list

    '''
    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(21)
        label_fill = np.arange(21)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []
            mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
            mask_unique = np.unique(mask)[:-1] if 255  in mask else np.unique(mask) # remove voild class
            unique_num = len(mask_unique) # num of unique classes

            # sample image if it includes the lowest appeared class and with more than 3 distinctive classes
            if len(labeled_idx) == 0 and unique_num > 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowesr appearance
            label_fill = np.where(label_counter == label_counter.min())[0]

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
        
    else:
        return idx_list'''

class Pascal_VOC_Dataset(Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0), augmentation=True, train=True,
                apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.scale_size = scale_size
        self.idx_list = idx_list
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        

    def __getitem__(self, index):
        image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
        if self.apply_partial is None:
            label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[index]))
        else:
            label_root = Image.open(self.root + '/SegmentationClassAug_{}_{}/{}.png'.format(self.apply_partial, self.partial_seed, self.idx_list[index], ))
        #print(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
        image, label = transform(image_root, label_root, None, crop_size=self.crop_size, scale_size=self.scale_size, augmentation=self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)

class BuildDataLoader():
    def __init__(self, batch_size=6, num_labels=5, distributed=False, seed=0):
        self.data_path = '/home/xiaoluoxi/PycharmProjects/Dirty/data/VOCdevkit/VOC2012'
        self.image_size = [513, 513]
        self.crop_size = [321, 321]
        self.num_segments = 21
        self.scale_size = (0.5, 1.5)
        self.batch_size = batch_size
        self.train_l_idx, self.train_u_idx = get_pascal_idx(self.data_path, num_labels=num_labels)
        #self.train_l_idx, self.train_u_idx = get_pascal_idx(self.data_path, train=True, label_num=num_labels)
        self.test_idx = get_pascal_idx(self.data_path, train=False)

        self.distributed = distributed
        
    def build(self, supervised=False, partial=None, partial_seed=None):
        self.train_l_dataset = Pascal_VOC_Dataset(self.data_path, self.train_l_idx, self.crop_size, self.scale_size,
                                             augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        self.train_u_dataset = Pascal_VOC_Dataset(self.data_path, self.train_u_idx, self.crop_size, scale_size=(1.0, 1.0),
                                                augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed)
        self.test_dataset = Pascal_VOC_Dataset(self.data_path, self.test_idx, self.crop_size, scale_size=(1.0, 1.0),augmentation=False,
                                          train=False)

        if supervised: # no unlabeled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2
        
        if self.distributed:
            #num_samples = self.batch_size * 200
            self.train_l_sampler = DistributedSampler(self.train_l_dataset)
            self.train_u_sampler = DistributedSampler(self.train_u_dataset)
        else:
            num_samples = self.batch_size * 200
            train_l_sampler = RandomSampler(self.train_l_dataset, replacement=True, num_samples=num_samples)
            train_u_sampler = RandomSampler(self.train_u_dataset, replacement=True, num_samples=num_samples)
        train_l_loader = torch.utils.data.DataLoader(self.train_l_dataset, batch_size=self.batch_size, sampler=self.train_l_sampler, num_workers=2, drop_last=True)
        train_u_loader = torch.utils.data.DataLoader(self.train_u_dataset, batch_size=self.batch_size, sampler=self.train_u_sampler, num_workers=2, drop_last=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4, shuffle=False, num_workers=2)
        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, test_loader

class my_DistributedSampler(DistributedSampler):
    ''' Redefine __init__() for sample cyclically'''
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
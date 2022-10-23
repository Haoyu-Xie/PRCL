import torch.utils.data as data
import torch
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import random
from PIL import Image, ImageFilter
# import mc
import io
import numpy as np
import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)

# class DatasetCache(data.Dataset):
#     def __init__(self):
#         super().__init__()
#         self.initialized = False
        
#     def _init_memcached(self):
#         server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf'
#         client_config_file = '/mnt/lustre/share/memcached_client/client.conf'
#         self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
#         self.initialized = True

#     def load_image(self, filename):
#         self._init_memcached()
#         value = mc.pyvector()
#         self.mclient.Get(filename, value)
#         value_str = mc.ConvertBuffer(value)

#         buff = io.BytesIO(value_str)
#         with Image.open(buff) as img:
#             img = img.convert('RGB')
#         return img

#     def load_label(self, filename):
#         self._init_memcached()
#         value = mc.pyvector()
#         self.mclient.Get(filename, value)
#         value_str = mc.ConvertBuffer(value)

#         buff = io.BytesIO(value_str)
#         with Image.open(buff) as label:
#             label = label
#         return label


class Cityscapes_Dataset_cache(data.Dataset):
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
        if self.train:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='train')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='train')
            label = Image.open(self.root + label_root)
            label = Image.fromarray(cityscapes_class_map(np.array(label)))
        else:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='val')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='val')
            label = Image.open(self.root + label_root)
            label = Image.fromarray(cityscapes_class_map(np.array(label)))
        image, label = transform(image, label, None, self.crop_size, self.scale_size, self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)

class Cityscapes_Dataset(data.Dataset):
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
        if self.train:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='train')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='train')
            label = Image.open(self.root + label_root)
            label = Image.fromarray(cityscapes_class_map(np.array(label)))
        else:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='val')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='val')
            label = Image.open(self.root + label_root)
            label = Image.fromarray(cityscapes_class_map(np.array(label)))
        image, label = transform(image, label, None, self.crop_size, self.scale_size, self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)
            
class City_BuildData():
    def __init__(self, data_path, txt_path, label_num, seed):
        self.data_path = data_path
        self.txt_path = txt_path
        self.label_num = label_num
        self.seed = seed
        self.im_size = [512, 1024]
        self.crop_size = [512, 512]
        self.num_segments = 19
        self.scale_size = (1.0, 1.0)
        self.train_l_idx, self.train_u_idx, self.test_idx= get_cityscapes_idx_via_txt(self.txt_path, self.label_num, self.seed)
        # self.train_l_idx, self.train_u_idx = get_cityscapes_idx(self.data_path, train=True, label_num=num_labels)
        # self.test_idx = get_cityscapes_idx(self.data_path, train=False)
        
    def build(self, supervised=False, partial=None, partial_seed=None):
        train_l_dataset = Cityscapes_Dataset(self.data_path, self.train_l_idx, self.crop_size, self.scale_size,
                                             augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        train_u_dataset = Cityscapes_Dataset(self.data_path, self.train_u_idx, self.crop_size, scale_size=(1.0, 1.0),
                                                augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed)
        test_dataset = Cityscapes_Dataset(self.data_path, self.test_idx, self.crop_size, scale_size=(1.0, 1.0),augmentation=False,
                                          train=False)
        return train_l_dataset, train_u_dataset, test_dataset

def get_cityscapes_idx_via_txt(root, label_num, seed):
    '''
    Read idx list via generated txt, pre-perform make_list.py
    '''
    root = root + '/' + str(label_num) + '/' + str(seed)
    with open(root + '/labeled_filename.txt') as f:
        labeled_list = f.read().splitlines()
    f.close()
    with open(root + '/unlabeled_filename.txt') as f:
        unlabeled_list = f.read().splitlines()
    f.close()
    with open(root + '/valid_filename.txt') as f:
        test_list = f.read().splitlines()
    f.close()
    return labeled_list, unlabeled_list, test_list

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Randomly rescale images
    raw_w, raw_h = image.size
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

def tensor_to_pil(image, label, logits):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())
    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return image, label, logits

def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def batch_transform(images, labels, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label_list, logits_list = [], [], []
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label_pil, logits_pil = tensor_to_pil(images[k], labels[k], logits[k])
        aug_image, aug_label, aug_logits = transform(image_pil, label_pil, logits_pil, crop_size, scale_size, augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)
    
    image_trans, label_trans, logits_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return image_trans, label_trans, logits_trans

def generate_cut(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, mode='cutout'):
    batch_size, _, image_h, image_w = image.shape
    device = image.device

    new_image = []
    new_label = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([image_h, image_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(label[i]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label.append((label[i] * mix_mask + label[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)

    return new_image, new_label.long(), new_logits

def generate_class_mask(pseudo_labels: torch.Tensor):
    # select the half classes and cover up them
    labels = torch.unique(pseudo_labels) # all unique labels
    labels_select: torch.Tensor = labels[torch.randperm(len(labels))][:len(labels) // 2] # Randmoly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(dim=-1)
    return mask.float()

def generate_cutout_mask(image_size, ratio=2):
    # Cutout: random generate mask where the region inside is 0, one ouside is 1
    cutout_area = image_size[0] * image_size[1] / ratio
    
    w = np.random.randint(image_size[1] / ratio + 1, image_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, image_size[1] - w + 1)
    y_start = np.random.randint(0, image_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(image_size)
    mask[y_start: y_end, x_start: x_end] = 0

    return mask.float()

def cityscapes_class_map(mask):
    # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = 255
    mask_map[np.isin(mask, [7])] = 0
    mask_map[np.isin(mask, [8])] = 1
    mask_map[np.isin(mask, [11])] = 2
    mask_map[np.isin(mask, [12])] = 3
    mask_map[np.isin(mask, [13])] = 4
    mask_map[np.isin(mask, [17])] = 5
    mask_map[np.isin(mask, [19])] = 6
    mask_map[np.isin(mask, [20])] = 7
    mask_map[np.isin(mask, [21])] = 8
    mask_map[np.isin(mask, [22])] = 9
    mask_map[np.isin(mask, [23])] = 10
    mask_map[np.isin(mask, [24])] = 11
    mask_map[np.isin(mask, [25])] = 12
    mask_map[np.isin(mask, [26])] = 13
    mask_map[np.isin(mask, [27])] = 14
    mask_map[np.isin(mask, [28])] = 15
    mask_map[np.isin(mask, [31])] = 16
    mask_map[np.isin(mask, [32])] = 17
    mask_map[np.isin(mask, [33])] = 18
    return mask_map

def label_root_transform(root: str, name: str, mode: str):
    label_root = root.strip()[0: -12] + '_gtFine_labelIds'
    return '/{}/{}/{}.png'.format(mode, name, label_root)

def image_root_transform(root: str, mode: str):
    name = root[0: root.find('_')]
    return '/leftImg8bit/{}/{}/{}.png'.format(mode, name, root), name
# class my_DistributedSampler(data.distributed.DistriutedSampler):
#     ''' Redefine __init__() for sample cyclically'''
#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False, iter_num=200) -> None:
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 "Invalid rank {}, rank should be in the interval"
#                 " [0, {}]".format(rank, num_replicas - 1))
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.drop_last = drop_last

#         # We rewrite here
#         self.num_samples = iter_num * len(self.dataset)
#         # If the dataset length is evenly divisible by # of replicas, then there
#         # is no need to drop any data, since the dataset will be split equally.
#         if self.drop_last and len(self.dataset) % self.num_replicas != 0:
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil(
#                 (self.num_samples - self.num_replicas) / self.num_replicas
#             )
#         else:
#             self.num_samples = math.ceil(self.num_samples / self.num_replicas)
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle
#         self.seed = seed
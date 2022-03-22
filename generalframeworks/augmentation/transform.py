from hashlib import new
from logging import raiseExceptions
from PIL import Image, ImageFilter
from matplotlib.pyplot import sca
import torch
from typing import Tuple
from torchvision import transforms
import torchvision.transforms.functional as transform_f
import random
import numpy as np

def batch_transform(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, crop_size: Tuple['h', 'w'], scale_size, 
                apply_augmentation=False):
    image_list, label_list, logits_list = [], [], []
    device = image.device

    for k in range(image.shape[0]):
        image_pil, label_pil, logits_pil = tensor_to_pil(image[k], label[k], logits[k])
        aug_image, aug_label, aug_logits = transform(image_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    image_trans, label_trans, logits_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), \
                                            torch.cat(logits_list).to(device)
    return image_trans, label_trans, logits_trans

def tensor_to_pil(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor):
    image = denormalise(image)
    image = transform_f.to_pil_image(image.cpu())

    label = label.float() / 255
    label = transform_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transform_f.to_pil_image(logits.unsqueeze(0).cpu())
    
    return image, label, logits


def denormalise(x, imagenet=False):
    if imagenet:
        x = transform_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transform_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), label_fill=255, augmentation=False):
    '''
    Only apply on the 3d image (one image not batch)
    '''
    # Random Rescale image
    raw_w, raw_h = image.size 
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transform_f.resize(image, resized_size, Image.NEAREST)
    label = transform_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transform_f.resize(logits, resized_size, Image.NEAREST)

    # Adding padding if rescaled image size is less than crop size
    if crop_size == -1: # Use original image size without rop or padding
        crop_size = (raw_h, raw_w)
    
    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transform_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transform_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=label_fill, padding_mode='constant')
        if logits is not None:
            logits = transform_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transform_f.crop(image, i, j, h, w)
    label = transform_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transform_f.crop(logits, i, j, h, w)
    
    if augmentation:
        # Random Color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Rnadmom Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal filpping
        if torch.rand(1) > 0.5:
            image = transform_f.hflip(image)
            label = transform_f.hflip(label)
            if logits is not None:
                logits = transform_f.hflip(logits)

        # Transform to Tensor
    image = transform_f.to_tensor(image)
    label = (transform_f.to_tensor(label) * 255).long()
    label[label == 255] = -1 # incalid pixels are re-mapping to index -1
    if logits is not None:
        logits = transform_f.to_tensor(logits)
    
    # Apply (ImageNet) normalization
    #image = transform_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform_f.normalize(image, mean=[0.5], std=[0.299])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

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
            mix_mask = generate_class_mask([image_h, image_w]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label.append((label[i] * mix_mask + label[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)

    return new_image, new_label.long(), new_logits



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

def generate_class_mask(pseudo_labels: torch.Tensor):
    # select the half classes and cover up them
    labels = torch.unique(pseudo_labels) # all unique labels
    labels_select: torch.Tensor = labels[torch.randperm(len(labels))][:len(labels) // 2] # Randmoly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(dim=-1)
    return mask.float()
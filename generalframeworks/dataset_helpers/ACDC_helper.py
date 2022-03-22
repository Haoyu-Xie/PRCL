
#author: Haoyu Xie 
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import h5py
import numpy as np
import torch    
from torchvision import transforms
import random
import torchvision.transforms.functional as transforms_f
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from generalframeworks.utils import class2one_hot
'''
class ACDC_Dataset(Dataset):
    def __init__(self, root_dir, idx_list=None):
        self.root_dir = root_dir
        self.idx_list = idx_list

    def __getitem__(self, index):
        sample1_dir = self.root_dir + '/data' + '/patient' + add_nulls(index, 3) + '_frame01.h5'
        sample2_dir = self.root_dir + '/data' + '/patient' + add_nulls(index, 3) + '_frame02.h5'
        h5f_1 = h5py.File(sample1_dir)
        image_1 = h5f_1['image'][:]
        label_1 = h5f_1['label'][:]
        h5f_2 = h5py.File(sample2_dir)
        image_2 = h5f_2['image'][:]
        label_2 = h5f_2['label'][:]
        image = np.concatenate([image_1, image_2])
        label = np.concatenate([label_1, label_2])
        return image, label
    def __len__(self):
        return len(self.idx_list)'''

class ACDC_Dataset(Dataset):
    def __init__(self, root_dir, save_dir: str, mode, meta_label=False):
        '''
        mode in ['label', 'unlabel', 'val']
        '''
        assert mode in ['label', 'unlabel', 'val'], 'mode must be in [label, unlabel, val]'
        self.root_dir = root_dir
        self.mode = mode 
        self.meta_label = meta_label
        self.save_dir = save_dir

        if self.mode == 'label':
            with open(self.save_dir + '/labeled_filename.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        
        if self.mode == 'unlabel':
            with open(self.save_dir + '/unlabeled_filename.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.mode == 'val':
            with open(self.save_dir + '/val_filename.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        case = self.sample_list[index]
        idx = int(case[7:10])
        h5f = h5py.File(self.root_dir + '/data/slices/{}'.format(case), 'r')
        image = torch.from_numpy(h5f['image'][:]).unsqueeze(0)
        label = torch.from_numpy(h5f['label'][:]).unsqueeze(0)
        if self.mode in ['label', 'unlabel']:
            image, label = my_transform(image, label, augmentation=True)
        else:
            image, label = my_transform(image, label)
        image = image.type(torch.float32)
        label = label.type(torch.int64)
        if self.meta_label:
            return image, label.squeeze(0), idx
        else:
            return image, label.squeeze(0)
       
#def transform(image, label, logits=None, size=(256, 256), augmentation=True):
    

def my_transform(image: torch.Tensor, label: torch.Tensor, size=(256, 256), augmentation=False):
    image = transforms.Resize(size)(image)
    label = transforms.Resize(size)(label)

    if augmentation:
        # Filp
        if random.random() > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
        if random.random() > 0.5:
            image = transforms_f.vflip(image)
            label = transforms_f.vflip(label)
            
        # Rotate 90
        if random.random() > 0.5:
            angle_90 = np.random.randint(0, 4) * 90
            image = transforms_f.rotate(image, float(angle_90))
            label = transforms_f.rotate(label, float(angle_90))

        # Rotate
        angle = random.randint(-20, 20)
        if random.random() > 0.5:
            image = transforms_f.rotate(image, float(angle))
            label = transforms_f.rotate(label, float(angle))
     
    return image, label


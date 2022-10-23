
from unittest.mock import patch
import numpy as np  
from typing import Dict, Tuple
from torch.utils.data import Dataset
from PIL import Image 
import h5py
import random
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data.sampler import Sampler
import itertools
from torchvision import transforms
import torchvision.transforms.functional as transforms_f

##### BaseDataSet #####

class BaseDataSet(Dataset):
    def __init__(self, config, mode='train'):
        self._base_dir = config['Dataset']['root_dir']
        self.sample_list = []
        self.mode = mode
        self.patch_size = config['Dataset']['patch_size']
        num = config['Dataset']['num_patient']
        #self.transform = AugmentGenerator([256, 256])
        self.transform = my_transform
        if self.mode == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.mode == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        if num is not None and self.mode == 'train':
            self.sample_list = self.sample_list[: num]
        print(mode + " dataset total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index: int) -> Dict['image', 'label']:
        case = self.sample_list[index]
        if self.mode == 'train':
            h5f = h5py.File(self._base_dir + '/data/slices/{}.h5'.format(case), 'r')
            print(self._base_dir + '/data/slices/{}.h5'.format(case))
        else:
            h5f = h5py.File(self._base_dir + '/data/{}.h5'.format(case), 'r')
        #image = h5f['image'][:]
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.mode == 'train':
            sample = self.transform(sample, self.patch_size)
        sample['idx'] = index
        return sample    


##### Data Augmentation ##### 
'''
class AugmentGenerator(object):
    def __init__(self, output_size: Tuple['H', 'W']):
        self.output_size = output_size
    
    def __call__(self, sample: Dict['image', 'label']) -> Dict['image', 'label']:
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        _, x, y = image.shape
        image = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        label = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
'''
def my_transform(sample: Dict['image', 'label'], patch_size):
    image = torch.from_numpy(sample['image'])
    label = torch.from_numpy(sample['label'])
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
        label = transforms_f.rotate(label, float(label))
    sape = image.size()
    patch_size = int(patch_size)
    image = transforms.Resize((patch_size, patch_size))(image.unsqueeze(0))
    label = transforms.Resize((patch_size, patch_size))(label.unsqueeze(0))
    sample['image'] = image
    sample['label'] = label
    
    return sample

    








##### Label and Unlabel Sampler ######

class TwoStreamBatchSampler(Sampler):
    
    def __init__(self, primary_indices: list, secondary_indices: list, batch_size: int):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size
        assert len(self.primary_indices) >= self.batch_size > 0, 'primary_indices length must > batch_size'
        assert len(self.secondary_indices) >= self.batch_size > 0, 'secondary_indices length must > batch_size'

    def __iter__(self):
        primary_iter = self._iterate_once(self.primary_indices)
        secondary_iter = self._iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch for (primary_batch, secondary_batch) 
                in zip(self._grouper(primary_iter, self.batch_size),
                    self._grouper(secondary_iter, self.batch_size)))
        
    def __len__(self):
        return len(self.primary_indices) // self.batch_size

    def _iterate_once(self, iterable):
        '''Disoder'''
        return np.random.permutation(iterable) 
    
    def _iterate_eternally(self, indices: list):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)
        return itertools.chain.from_iterable(infinite_shuffles())

    def _grouper(self, iterable, n: int):
        "Collect data into fixed-length n chunks or blocks. e.g. grouper('ABCDEF', 3) -> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)





 
            

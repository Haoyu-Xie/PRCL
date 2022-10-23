
import glob
import random
import os
from PIL import Image
import numpy as np
from generalframeworks.utils import fix_all_seed

def make_ACDC_list(dataset_dir: str, labeled_num: int, save_dir: str):
    if os.path.exists(save_dir + '/labeled_filename.txt'):
        os.remove(save_dir + '/labeled_filename.txt')
    ids = [i + 1 for i in range (90)]
    ids = np.random.permutation(ids)
    labeled_ids = list(ids[: labeled_num])
    unlabeled_ids = list(ids[labeled_num: ])
    labeled_str = [add_nulls(i, 3) for i in labeled_ids]
    sampleList = []
    for string in labeled_str:
        sampleList.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/labeled_filename.txt', 'a')
    for item in sampleList:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Labeled Dataset contains {} '.format(labeled_num) + 'patients and {} slices'.format(len(sampleList)))

    if os.path.exists(save_dir + '/unlabeled_filename.txt'):
        os.remove(save_dir + '/unlabeled_filename.txt')
    unlabeled_str = [add_nulls(i, 3) for i in unlabeled_ids]
    sampleList = []
    for string in unlabeled_str:
        sampleList.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/unlabeled_filename.txt', 'a')
    for item in sampleList:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Unlabeled Dataset contains {} '.format(len(unlabeled_ids)) + 'patients and {} slices'.format(len(sampleList)))

    if os.path.exists(save_dir + '/val_filename.txt'):
        os.remove(save_dir + '/val_filename.txt')
    val_ids = [i + 91 for i in range(10)]
    val_str = [add_nulls(i, 3) for i in val_ids]
    sampleList = []
    for string in val_str:
        sampleList.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/val_filename.txt', 'a')
    for item in sampleList:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Valid Dataset contains {} '.format(len(val_ids)) + 'patients and {} slices'.format(len(sampleList)))

def make_ACDC_list_llu(dataset_dir: str, labeled_num: int, save_dir: str):
    if os.path.exists(save_dir + '/labeled_filename.txt'):
        os.remove(save_dir + '/labeled_filename.txt')
    ids = [i + 1 for i in range (90)]
    ids = np.random.permutation(ids)
    labeled_ids = list(ids[: labeled_num])
    if len(labeled_ids) == 1:
        labeled_ids_0 = labeled_ids
        labeled_ids_1 = labeled_ids
    else:
        labeled_ids_0 = labeled_ids[: int(len(labeled_ids)/2)]
        labeled_ids_1 = labeled_ids[int(len(labeled_ids)/2):]
    unlabeled_ids = list(ids[labeled_num: ])
    labeled_str_0 = [add_nulls(i, 3) for i in labeled_ids_0]
    labeled_str_1 = [add_nulls(i, 3) for i in labeled_ids_1]
    sampleList_0 = []
    sampleList_1 = []
    for string in labeled_str_0:
        sampleList_0.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    for string in labeled_str_1:
        sampleList_1.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/labeled_0_filename.txt', 'a')
    for item in sampleList_0:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    f = open(save_dir + '/labeled_1_filename.txt', 'a')
    for item in sampleList_1:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Labeled Dataset 0 contains {} '.format(labeled_ids_0) + 'patients and {} slices'.format(len(sampleList_0)))
    print('Labeled Dataset 1 contains {} '.format(labeled_ids_1) + 'patients and {} slices'.format(len(sampleList_1)))
    if os.path.exists(save_dir + '/unlabeled_filename.txt'):
        os.remove(save_dir + '/unlabeled_filename.txt')
    unlabeled_str = [add_nulls(i, 3) for i in unlabeled_ids]
    sampleList = []
    for string in unlabeled_str:
        sampleList.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/unlabeled_filename.txt', 'a')
    for item in sampleList:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Unlabeled Dataset contains {} '.format(len(unlabeled_ids)) + 'patients and {} slices'.format(len(sampleList)))

    if os.path.exists(save_dir + '/val_filename.txt'):
        os.remove(save_dir + '/val_filename.txt')
    val_ids = [i + 91 for i in range(10)]
    val_str = [add_nulls(i, 3) for i in val_ids]
    sampleList = []
    for string in val_str:
        sampleList.extend(glob.glob(dataset_dir + '/data/slices/patient' + string + '*.h5'))
    f = open(save_dir + '/val_filename.txt', 'a')
    for item in sampleList:
        sample_name = item.split('/')[-1]
        f.write(sample_name + '\n')
    f.close()
    print('Valid Dataset contains {} '.format(len(val_ids)) + 'patients and {} slices'.format(len(sampleList)))


def add_nulls(wait_int, cnt):
    nulls = str(wait_int)
    for i in range(cnt - len(str(wait_int))):
        nulls = '0' + nulls
    return nulls

def make_VOC_list(dataset_dir: str, labeled_num: int, unlabeled_num: int, random_seed: int, save_dir: str):
    fix_all_seed(random_seed)
    root = os.path.expanduser(dataset_dir)
    filename_train = root + '/prefix/train_aug.txt'
    with open(filename_train) as f:
        idx_list = f.read().splitlines()
        random.shuffle(idx_list)
    labeled_idx = []
    save_idx = []
    idx_list_ = idx_list.copy()
    label_counter = np.zeros(21)
    label_fill = np.arange(21)
    while len(labeled_idx) < labeled_num:
        if len(idx_list_) > 0:
            idx = idx_list_.pop()
        else:
            idx_list_ = save_idx.copy()
            idx = idx_list_.pop()
            save_idx = []

        mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
        mask_unique = np.unique(mask)[:-1]
        unique_num = len(mask_unique)

        if len(labeled_idx) == 0 and unique_num >= 3:
            labeled_idx.append(idx)
            label_counter[mask_unique] += 1
        elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
            labeled_idx.append(idx)
            label_counter[mask_unique] += 1
        else:
            save_idx.append(idx)

        label_fill = np.where(label_counter == label_counter.min())[0]
    unlabeled_idx = [idx for idx in idx_list if idx not in labeled_idx]
    if not os.path.exists(save_dir + '/' + str(random_seed)):
        os.makedirs(save_dir + '/' + str(random_seed))
    f = open(save_dir + '/' + str(random_seed) + '/labeled_filename.txt', 'a')
    for item in labeled_idx:
        f.write(item + '\n')
    f.close()
    print('Labeled Dataset contains {}'.format(labeled_idx.__len__()))
    f = open(save_dir + '/' + str(random_seed) + '/unlabeled_filename.txt', 'a')
    for item in unlabeled_idx:
        f.write(item + '\n')
    f.close()

if __name__ =='__main__':
    make_VOC_list(dataset_dir='/home/xiaoluoxi/PycharmProjects/Dirty/data/VOCdevkit/VOC2012',
                    labeled_num=60, unlabeled_num=1102, random_seed=0, save_dir='/home/xiaoluoxi/PycharmProjects/Dirty/data/VOCdevkit/VOC2012/prefix/my_big_subset')
                    
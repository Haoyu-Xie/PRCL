
import glob
import random
import os
from tracemalloc import stop 
import numpy as np

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

if __name__ =='__main__':
    make_ACDC_list_llu('/home/xiaoluoxi/PycharmProjects/Dirty/data/ACDC', 3, save_dir='/home/xiaoluoxi/PycharmProjects/Dirty/tmp_')
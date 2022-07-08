import monai
import torch
import itk
import numpy as np
import glob
import os


def path_to_id(path):
    return os.path.basename(path).split('.')[0]


def split_data(img_path, seg_path, num_seg):
    total_img_paths = []
    total_seg_paths = []
    for i in sorted(glob.glob(img_path + '/*.nii.gz')):
        total_img_paths.append(i)

    for j in sorted(glob.glob(seg_path + '/*.nii.gz')):
        total_seg_paths.append(j)

    np.random.shuffle(total_img_paths)
    num_train = int(round(len(total_seg_paths)*0.8))
    num_test = len(total_seg_paths) - num_train
    seg_train = total_seg_paths[:num_train]
    seg_test = total_seg_paths[num_train:]
    img_train = []
    img_test = []
    test = []
    train = []
    img_ids = list(map(path_to_id, total_img_paths))
    img_ids1 = img_ids
    total_img_paths1 = total_img_paths
    seg_ids_test = map(path_to_id, seg_test)
    seg_ids_train = map(path_to_id, seg_train)
    for seg_index, seg_id in enumerate(seg_ids_test):
        data_item = {}
        if seg_id in img_ids:
            img_test.append(total_img_paths[img_ids.index(seg_id)])
            data_item['img'] = total_img_paths[img_ids.index(seg_id)]
            total_img_paths1.pop(img_ids1.index(seg_id))
            img_ids1.pop(img_ids1.index(seg_id))
        data_item['seg'] = seg_test[seg_index]
        test.append(data_item)

    img_train = total_img_paths1
    np.random.shuffle(seg_train)
    if num_seg < len(seg_train):    
        seg_train_available = seg_train[:num_seg]
    else:
        seg_train_available = seg_train
    seg_ids = list(map(path_to_id, seg_train_available))
    img_ids = map(path_to_id, img_train)
    for img_index, img_id in enumerate(img_ids):
        data_item = {'img': img_train[img_index]}
        if img_id in seg_ids:
            data_item['seg'] = seg_train_available[seg_ids.index(img_id)]
        train.append(data_item)

    num_train = len(img_train)
    return train, test, num_train, num_test


def load_seg_dataset(train, valid):
    transform_seg_available = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=True),
            monai.transforms.AddChannelD(keys=['img', 'seg']),
            monai.transforms.SpacingD(keys=['img', 'seg'], pixdim=(1., 1., 1.), mode=('trilinear', 'nearest')),
            monai.transforms.ToTensorD(keys=['img', 'seg'])
        ]
    )
    itk.ProcessObject.SetGlobalWarningDisplay(False)
    dataset_seg_available_train = monai.data.CacheDataset(
        data=train,
        transform=transform_seg_available,
        cache_num=16,
        hash_as_key=True
    )

    dataset_seg_available_valid = monai.data.CacheDataset(
        data=valid,
        transform=transform_seg_available,
        cache_num=16,
        hash_as_key=True
    )
    return dataset_seg_available_train, dataset_seg_available_valid


def load_reg_dataset(train, valid):
    transform_pair = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(
                keys=['img1', 'seg1', 'img2', 'seg2'], image_only=True, allow_missing_keys=True),
            monai.transforms.ToTensorD(
                keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
            monai.transforms.AddChannelD(
                keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
            monai.transforms.SpacingD(keys=['img1', 'seg1', 'img2', 'seg2'], pixdim=(1., 1., 1.), mode=(
                'trilinear', 'nearest', 'trilinear', 'nearest'), allow_missing_keys=True),
            monai.transforms.ConcatItemsD(
                keys=['img1', 'img2'], name='img12', dim=0),
            monai.transforms.DeleteItemsD(keys=['img1', 'img2'])
        ]
    )
    dataset_pairs_train_subdivided = {
        seg_availability: monai.data.CacheDataset(
            data=data_list,
            transform=transform_pair,
            cache_num=32,
            hash_as_key=True
        )
        for seg_availability, data_list in train.items()
    }

    dataset_pairs_valid_subdivided = {
        seg_availability: monai.data.CacheDataset(
            data=data_list,
            transform=transform_pair,
            cache_num=32,
            hash_as_key=True
        )
        for seg_availability, data_list in valid.items()
    }
    return dataset_pairs_train_subdivided, dataset_pairs_valid_subdivided


def take_data_pairs(data, symmetric=True):
    """Given a list of dicts that have keys for an image and maybe a segmentation,
    return a list of dicts corresponding to *pairs* of images and maybe segmentations.
    Pairs consisting of a repeated image are not included.
    If symmetric is set to True, then for each pair that is included, its reverse is also included"""
    data_pairs = []
    for i in range(len(data)):
        j_limit = len(data) if symmetric else i
        for j in range(j_limit):
            if j == i:
                continue
            d1 = data[i]
            d2 = data[j]
            pair = {
                'img1': d1['img'],
                'img2': d2['img']
            }
            if 'seg' in d1.keys():
                pair['seg1'] = d1['seg']
            if 'seg' in d2.keys():
                pair['seg2'] = d2['seg']
            data_pairs.append(pair)
    return data_pairs


def subdivide_list_of_data_pairs(data_pairs_list):
    out_dict = {'00': [], '01': [], '10': [], '11': []}
    for d in data_pairs_list:
        if 'seg1' in d.keys() and 'seg2' in d.keys():
            out_dict['11'].append(d)
        elif 'seg1' in d.keys():
            out_dict['10'].append(d)
        elif 'seg2' in d.keys():
            out_dict['01'].append(d)
        else:
            out_dict['00'].append(d)
    return out_dict

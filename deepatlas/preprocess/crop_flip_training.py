import numpy as np
import glob
import ants
import nibabel as nib
import os
import argparse
import sys
from crop import crop, cropV2, save_fileV2
from pathlib import Path

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for data preprocessing')                   
    parser.add_argument('-rs', metavar='shape after resizing', type=int, nargs='+',
                        help='shape after resizing the image and segmentation. Expected to be 2^N')
    parser.add_argument('-fp', action='store_true',
                        help='check if need to flip the data')
    parser.add_argument('-ti', metavar='task id and name', type=str,
                        help='task name and id')
    argv = parser.parse_args()
    return argv

def path_to_id(path):
    ids = []
    for i in glob.glob(path + '/*nii.gz'):
        id = os.path.basename(i).split('.')[0]
        ids.append(id)
    return ids

def pad(raw_image, bound_x, bound_y, bound_z, resize, seg=False):
    diff_x = resize[0] - (bound_x[1]-bound_x[0])
    diff_y = resize[1] - (bound_y[1]-bound_y[0])
    diff_z = resize[2] - (bound_z[1]-bound_z[0])
    if diff_x < 0 or diff_y < 0 or diff_z < 0:
        sys.exit(
            'the dimension of ROI is larger than the resizing dimension, please choose a different padding dimension')
    left_y, right_y = split(diff_y)
    left_z, right_z = split(diff_z)
    left_x, right_x = split(diff_x)
    new_bound_x_left = bound_x[0] - left_x
    new_bound_x_right = bound_x[1] + right_x
    new_bound_y_left = bound_y[0] - left_y
    new_bound_y_right = bound_y[1] + right_y
    new_bound_z_left = bound_z[0] - left_z
    new_bound_z_right = bound_z[1] + right_z
    # check if x_dim out of bounds
    if new_bound_x_left < 0:
        new_bound_x_left = 0
        new_bound_x_right = bound_x[1] + diff_x - bound_x[0]

    elif new_bound_x_right > raw_image.shape[0]:
        new_bound_x_right = raw_image.shape[0]
        new_bound_x_left = bound_x[0] - \
            (diff_x - (raw_image.shape[0] - bound_x[1]))
    # check if y_dim out of bounds
    if new_bound_y_left < 0:
        new_bound_y_left = 0
        new_bound_y_right = bound_y[1] + diff_y - bound_y[0]

    elif new_bound_y_right > raw_image.shape[1]:
        new_bound_y_right = raw_image.shape[1]
        new_bound_y_left = bound_y[0] - \
            (diff_y - (raw_image.shape[1] - bound_y[1]))
    # check if z_dim out of bounds
    if new_bound_z_left < 0:
        new_bound_z_left = 0
        new_bound_z_right = bound_z[1] + diff_z - bound_z[0]

    elif new_bound_z_right > raw_image.shape[2]:
        new_bound_z_right = raw_image.shape[2]
        new_bound_z_left = bound_z[0] - \
            (diff_z - (raw_image.shape[2] - bound_z[1]))

    assert new_bound_x_right - new_bound_x_left == resize[0]
    assert new_bound_y_right - new_bound_y_left == resize[1]
    assert new_bound_z_right - new_bound_z_left == resize[2]
    if not seg:
        return raw_image[new_bound_x_left:new_bound_x_right, new_bound_y_left:new_bound_y_right, new_bound_z_left:new_bound_z_right]
    else:
        new_seg = np.zeros_like(raw_image)
        new_seg[bound_x[0]:bound_x[1],
                bound_y[0]:bound_y[1], bound_z[0]:bound_z[1]] = raw_image[bound_x[0]:bound_x[1], bound_y[0]:bound_y[1], bound_z[0]:bound_z[1]]
        return new_seg[new_bound_x_left:new_bound_x_right, new_bound_y_left:new_bound_y_right, new_bound_z_left:new_bound_z_right]


def split(distance):
    if distance == 0:
        return 0, 0

    half_dist = int(distance / 2)
    left = int(half_dist * 0.8)
    right = distance - left
    return left, right


def crop_and_flip(nib_img, nib_seg, ants_img, ants_seg, resize):
    img = nib_img.get_fdata()
    seg = nib_seg.get_fdata()
    gem = ants.label_geometry_measures(ants_seg, ants_img)
    low_x = min(list(gem.loc[:, 'BoundingBoxLower_x']))
    upp_x = max(list(gem.loc[:, 'BoundingBoxUpper_x']))
    low_y = min(list(gem.loc[:, 'BoundingBoxLower_y']))
    upp_y = max(list(gem.loc[:, 'BoundingBoxUpper_y']))
    low_z = min(list(gem.loc[:, 'BoundingBoxLower_z']))
    upp_z = max(list(gem.loc[:, 'BoundingBoxUpper_z']))

    img = Zscore_normalization(img)
    #img = MinMax_normalization(img)
    # Compute mid point
    mid_x = int((low_x + upp_x) / 2)

    tuple_x_left = tuple([low_x, mid_x])
    tuple_x_right = tuple([mid_x, upp_x])
    tuple_y = tuple([low_y, upp_y])
    tuple_z = tuple([low_z, upp_z])
    left_img = pad(img, tuple_x_left, tuple_y, tuple_z, resize, seg=False)
    left_seg = pad(seg, tuple_x_left, tuple_y, tuple_z, resize, seg=True)
    right_img = pad(img, tuple_x_right, tuple_y, tuple_z, resize, seg=False)
    right_seg = pad(seg, tuple_x_right, tuple_y, tuple_z, resize, seg=True)
    flipped_right_img = np.flip(right_img, axis=0)
    flipped_right_seg = np.flip(right_seg, axis=0)

    return left_img, left_seg, flipped_right_img, flipped_right_seg


def crop_and_flip_V2(nib_img, ants_img, resize, geo_info):
    img = nib_img.get_fdata()
    tuple_x = geo_info[0]
    tuple_y = geo_info[1]
    tuple_z = geo_info[2]
    low_x = tuple_x[0]
    upp_x = tuple_x[1]
    img = Zscore_normalization(img)
    #img = MinMax_normalization(img)
    # Compute mid point
    mid_x = int((low_x + upp_x) / 2)

    tuple_x_left = tuple([low_x, mid_x])
    tuple_x_right = tuple([mid_x, upp_x])

    left_img = pad(img, tuple_x_left, tuple_y, tuple_z, resize, seg=False)
    right_img = pad(img, tuple_x_right, tuple_y, tuple_z, resize, seg=False)
    flipped_right_img = np.flip(right_img, axis=0)

    return left_img, flipped_right_img


def MinMax_normalization(scan):
    lb = np.amin(scan)
    ub = np.amax(scan)
    scan = (scan - lb) / (ub - lb)
    return scan


def Zscore_normalization(scan):
    mean = np.mean(scan)
    std = np.std(scan)
    lb = np.percentile(scan, 0.05)
    ub = np.percentile(scan, 99.5)
    scan = np.clip(scan, lb, ub)
    scan = (scan - mean) / std
    return scan


def load_data(img_path, seg_path):
    nib_seg = nib.load(seg_path)
    nib_img = nib.load(img_path)
    ants_seg = ants.image_read(seg_path)
    ants_img = ants.image_read(img_path)
    return nib_img, nib_seg, ants_img, ants_seg


def crop_flip_save_file(left_img, left_seg, flipped_right_img, flipped_right_seg, nib_img, nib_seg, output_img, output_seg, scan_id):
    left_img_nii = nib.Nifti1Image(
        left_img, affine=nib_img.affine, header=nib_img.header)
    left_seg_nii = nib.Nifti1Image(
        left_seg, affine=nib_seg.affine, header=nib_seg.header)
    right_img_nii = nib.Nifti1Image(
        flipped_right_img, affine=nib_img.affine, header=nib_img.header)
    right_seg_nii = nib.Nifti1Image(
        flipped_right_seg, affine=nib_seg.affine, header=nib_seg.header)
    left_img_nii.to_filename(os.path.join(
        output_img, 'right_' + scan_id + '.nii.gz'))
    left_seg_nii.to_filename(os.path.join(
        output_seg, 'right_' + scan_id + '.nii.gz'))
    right_img_nii.to_filename(os.path.join(
        output_img, 'left_' + scan_id + '.nii.gz'))
    right_seg_nii.to_filename(os.path.join(
        output_seg, 'left_' + scan_id + '.nii.gz'))

def get_geometry_info(seg_path, img_path):
    abs_low_x = np.Inf
    abs_upp_x = -np.Inf
    abs_low_y = np.Inf
    abs_upp_y = -np.Inf
    abs_low_z = np.Inf
    abs_upp_z = -np.Inf
    for i in sorted(glob.glob(os.path.join(img_path, '*.nii.gz'))):
        name = os.path.basename(i)
        if os.path.exists(os.path.join(seg_path, name)):
            seg = ants.image_read(os.path.join(seg_path, name))
            img = ants.image_read(i)
            gem = ants.label_geometry_measures(seg, img)
            low_x = min(list(gem.loc[:, 'BoundingBoxLower_x']))
            upp_x = max(list(gem.loc[:, 'BoundingBoxUpper_x']))
            low_y = min(list(gem.loc[:, 'BoundingBoxLower_y']))
            upp_y = max(list(gem.loc[:, 'BoundingBoxUpper_y']))
            low_z = min(list(gem.loc[:, 'BoundingBoxLower_z']))
            upp_z = max(list(gem.loc[:, 'BoundingBoxUpper_z']))
            if low_x < abs_low_x:
                abs_low_x = low_x
            if upp_x > abs_upp_x:
                abs_upp_x = upp_x
            if low_y < abs_low_y:
                abs_low_y = low_y
            if upp_y > abs_upp_y:
                abs_upp_y = upp_y
            if low_z < abs_low_z:
                abs_low_z = low_z
            if upp_z > abs_upp_z:
                abs_upp_z = upp_z
    
    tuple_x = tuple([abs_low_x, abs_upp_x])
    tuple_y = tuple([abs_low_y, abs_upp_y])
    tuple_z = tuple([abs_low_z, abs_upp_z])
    return [tuple_x, tuple_y, tuple_z]


def crop_flip_save_file_V2(left_img, flipped_right_img, nib_img, output_img, scan_id):
    left_img_nii = nib.Nifti1Image(
        left_img, affine=nib_img.affine, header=nib_img.header)
    right_img_nii = nib.Nifti1Image(
        flipped_right_img, affine=nib_img.affine, header=nib_img.header)
    left_img_nii.to_filename(os.path.join(
        output_img, 'right_' + scan_id + '.nii.gz'))
    right_img_nii.to_filename(os.path.join(
        output_img, 'left_' + scan_id + '.nii.gz'))

def crop_save_file(left_img, left_seg, nib_img, nib_seg, output_img, output_seg, scan_id):
    left_img_nii = nib.Nifti1Image(
        left_img, affine=nib_img.affine, header=nib_img.header)
    left_seg_nii = nib.Nifti1Image(
        left_seg, affine=nib_seg.affine, header=nib_seg.header)
    left_img_nii.to_filename(os.path.join(
        output_img, scan_id + '.nii.gz'))
    left_seg_nii.to_filename(os.path.join(
        output_seg, scan_id + '.nii.gz'))


def main():
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    resize_shape = args.rs
    flipped = args.fp
    deepatlas_path = ROOT_DIR
    task_id = args.ti
    base_path = os.path.join(deepatlas_path, 'deepatlas_raw_data_base', task_id, 'Training_dataset')
    image_path = os.path.join(base_path, 'images')
    seg_path = os.path.join(base_path, 'labels')
    output_path = os.path.join(deepatlas_path, 'deepatlas_preprocessed')
    task_path = os.path.join(deepatlas_path, 'deepatlas_preprocessed', task_id)
    training_data_path = os.path.join(deepatlas_path, 'deepatlas_preprocessed', task_id, 'Training_dataset')
    output_img = os.path.join(deepatlas_path, 'deepatlas_preprocessed', task_id, 'Training_dataset', 'images')
    output_seg = os.path.join(deepatlas_path, 'deepatlas_preprocessed', task_id, 'Training_dataset', 'labels')
    label_list = path_to_id(seg_path)
    geo_info = get_geometry_info(seg_path, image_path)
    print(geo_info)
    try:
        os.mkdir(output_path)
    except:
        print(f'{output_path} is already existed')

    try:
        os.mkdir(task_path)
    except:
        print(f'{task_path} is already existed')
    
    try:
        os.mkdir(training_data_path)
    except:
        print(f"{training_data_path} already exists")

    try:
        os.mkdir(output_path)
    except:
        print(f'{output_path} is already existed')

    try:
        os.mkdir(output_img)
    except:
        print(f'{output_img} is already existed')

    try:
        os.mkdir(output_seg)
    except:
        print(f'{output_seg} is already existed')

    for i in sorted(glob.glob(image_path + '/*nii.gz')):
        id = os.path.basename(i).split('.')[0]
        if id in label_list:
            label_path = os.path.join(seg_path, id + '.nii.gz')
            nib_img, nib_seg, ants_img, ants_seg = load_data(i, label_path)
            if flipped:
                left_img, left_seg, flipped_right_img, flipped_right_seg = crop_and_flip(
                    nib_img, nib_seg, ants_img, ants_seg, resize_shape)
                print(
                    'Scan ID: ' + id + f', img & seg before cropping: {nib_img.get_fdata().shape}, after cropping, flipping and padding: {left_img.shape} and {flipped_right_img.shape}')
                crop_flip_save_file(left_img, left_seg, flipped_right_img, flipped_right_seg,
                        nib_img, nib_seg, output_img, output_seg, id)
            else:
                left_img, left_seg = crop(
                    nib_img, nib_seg, ants_img, ants_seg, resize_shape)
                print(
                    'Scan ID: ' + id + f', img & seg before cropping: {nib_img.get_fdata().shape}, after cropping and padding the image and seg: {left_img.shape}')
                crop_save_file(left_img, left_seg, nib_img,
                        nib_seg, output_img, output_seg, id)
        else:
            nib_img = nib.load(i)
            ant_img = ants.image_read(i)
            if flipped:
                left_img, flipped_right_img = crop_and_flip_V2(nib_img, ant_img, resize_shape, geo_info)
                print(
                    'Scan ID: ' + id + f', img before cropping: {nib_img.get_fdata().shape}, after cropping, flipping and padding: {left_img.shape} and {flipped_right_img.shape}')
                crop_flip_save_file_V2(left_img, flipped_right_img, nib_img, output_img, id)
            else:
                outImg = cropV2(nib_img, ant_img, resize_shape, geo_info)
                print(
                    'Scan ID: ' + id + f', img before cropping: {nib_img.get_fdata().shape}, after cropping and padding the image: {outImg.shape}')
                save_fileV2(outImg, nib_img, output_img, id)

if __name__ == '__main__':
    main()

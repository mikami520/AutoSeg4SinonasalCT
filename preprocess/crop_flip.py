from turtle import right
import numpy as np
import glob
import ants
import nibabel as nib
import os
import argparse


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for data preprocessing')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="relative path of the image directory")
    parser.add_argument('-op', metavar='preprocessing result output path', type=str, default='output',
                        help='relative path of the preprocessing result directory')
    parser.add_argument('-rs', metavar='shape after resizing', type=int, nargs='+',
                        help='shape after resizing the image and segmentation. Expected to be 2^N')
    argv = parser.parse_args()
    return argv


def pad(raw_image, bound_x, bound_y, bound_z, resize, seg=False):
    diff_x = resize[0] - (bound_x[1]-bound_x[0])
    diff_y = resize[1] - (bound_y[1]-bound_y[0])
    diff_z = resize[2] - (bound_z[1]-bound_z[0])
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


def load_data(img_path, seg_path):
    nib_seg = nib.load(seg_path)
    nib_img = nib.load(img_path)
    ants_seg = ants.image_read(seg_path)
    ants_img = ants.image_read(img_path)
    return nib_img, nib_seg, ants_img, ants_seg


def save_file(left_img, left_seg, flipped_right_img, flipped_right_seg, nib_img, nib_seg, output_img, output_seg, scan_id):
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


def main():
    args = parse_command_line()
    base_path = args.bp
    image_path = os.path.join(base_path, args.ip)
    seg_path = os.path.join(base_path, args.sp)
    output_path = os.path.join(base_path, args.op)
    resize_shape = args.rs
    output_img = os.path.join(output_path, 'images')
    output_seg = os.path.join(output_path, 'labels')
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
        label_path = os.path.join(seg_path, id + '.nii.gz')
        nib_img, nib_seg, ants_img, ants_seg = load_data(i, label_path)
        left_img, left_seg, flipped_right_img, flipped_right_seg = crop_and_flip(
            nib_img, nib_seg, ants_img, ants_seg, resize_shape)
        print(
            'Scan ID: ' + id + f', before cropping: {nib_img.get_fdata().shape}, after cropping and padding: {left_img.shape} and {flipped_right_img.shape}')
        save_file(left_img, left_seg, flipped_right_img, flipped_right_seg,
                  nib_img, nib_seg, output_img, output_seg, id)


if __name__ == '__main__':
    main()

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
                        help="Absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-op', metavar='preprocessing result output path', type=str, default='output',
                        help='Relative path of the preprocessing result directory')
    parser.add_argument('-rs', metavar='shape after resizing', type=int, nargs='+',
                        help='shape after resizing the image and segmentation. Expected to be 2^N')
    argv = parser.parse_args()
    return argv


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset)
                   for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded


def crop_and_flip(nib_img, nib_seg, ants_img, ants_seg):
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

    left_img = img[low_x:mid_x, low_y:upp_y, low_z:upp_z]
    left_seg = seg[low_x:mid_x, low_y:upp_y, low_z:upp_z]

    flipped_right_img = np.flip(
        img[mid_x:upp_x, low_y:upp_y, low_z:upp_z], axis=0)
    flipped_right_seg = np.flip(
        seg[mid_x:upp_x, low_y:upp_y, low_z:upp_z], axis=0)
    return left_img, left_seg, flipped_right_img, flipped_right_seg


def load_data(img_path, seg_path):
    nib_seg = nib.load(seg_path)
    nib_img = nib.load(img_path)
    ants_seg = ants.image_read(seg_path)
    ants_img = ants.image_read(img_path)
    return nib_img, nib_seg, ants_img, ants_seg


def save_file(left_img, left_seg, flipped_right_img, flipped_right_seg, nib_img, nib_seg, output_img, output_seg, scan_id):
    left_img_nii = nib.Nifti1Image(
        left_img, affine=nib_img.affine)
    left_seg_nii = nib.Nifti1Image(
        left_seg, affine=nib_seg.affine)
    right_img_nii = nib.Nifti1Image(
        flipped_right_img, affine=nib_img.affine)
    right_seg_nii = nib.Nifti1Image(
        flipped_right_seg, affine=nib_seg.affine)
    left_img_nii.to_filename(os.path.join(
        output_img, 'left_' + scan_id + '.nii.gz'))
    left_seg_nii.to_filename(os.path.join(
        output_seg, 'left_' + scan_id + '.nii.gz'))
    right_img_nii.to_filename(os.path.join(
        output_img, 'right_' + scan_id + '.nii.gz'))
    right_seg_nii.to_filename(os.path.join(
        output_seg, 'right_' + scan_id + '.nii.gz'))


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
            nib_img, nib_seg, ants_img, ants_seg)
        print(
            'Scan ID: ' + id + f', before cropping: {nib_img.get_fdata().shape}, after cropping: {left_img.shape} and {flipped_right_img.shape}, padding to {tuple(resize_shape)}')
        left_img = pad(left_img, resize_shape)
        left_seg = pad(left_seg, resize_shape)
        flipped_right_img = pad(flipped_right_img, resize_shape)
        flipped_right_seg = pad(flipped_right_seg, resize_shape)
        save_file(left_img, left_seg, flipped_right_img, flipped_right_seg,
                  nib_img, nib_seg, output_img, output_seg, id)


if __name__ == '__main__':
    main()

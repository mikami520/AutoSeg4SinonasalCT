'''
python3 split_data.py RegisteredImageFolderPath RegisteredLabelFolderPath
Given the parameter as the path to the registered images, 
function creates two folders in the base directory (same level as this script), randomly putting in
70 percent of images into the train and 30 percent to the test
'''
import os
import glob
import random
import shutil
from pathlib import Path
from typing import Tuple
import numpy as np
from collections import OrderedDict
import json
import argparse


"""
creates a folder at a specified folder path if it does not exists
folder_path : relative path of the folder (from cur_dir) which needs to be created
over_write :(default: False) if True overwrite the existing folder 
 """
def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='pipeline for dataset split')
    parser.add_argument('-ti', metavar='task id and name', type=str,
                        help='task name and id')
    parser.add_argument('-kf', metavar='k-fold validation', type=int, default=5,
                        help='k-fold validation')
    parser.add_argument('-ns', metavar='number of segmentations', type=int, default=3,
                        help='number of segmentations used for training')
    argv = parser.parse_args()
    return argv


def make_if_dont_exist(folder_path, overwrite=False):

    if os.path.exists(folder_path):
        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")

def split(img, seg, seg_path):
    label = []
    unlabel = []
    for i in img:
        name = os.path.basename(i)
        seg_name = os.path.join(seg_path, name)
        if seg_name in seg:
            item = {"img": i,
                    "seg": seg_name}
            label.append(item)
        else:
            item = {"img": i}
            unlabel.append(item)
    
    return label, unlabel

def main():
    random.seed(2938649572)
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    task_id = args.ti
    k_fold = args.kf
    num_seg = args.ns
    deepatlas_path = ROOT_DIR
    base_path = os.path.join(deepatlas_path, "deepatlas_preprocessed")
    task_path = os.path.join(base_path, task_id)
    img_path = os.path.join(task_path, 'Training_dataset', 'images')
    seg_path = os.path.join(task_path, 'Training_dataset', 'labels')
    image_list = glob.glob(img_path + "/*.nii.gz")
    label_list = glob.glob(seg_path + "/*.nii.gz")
    label, unlabel = split(image_list, label_list, seg_path)
    piece_data = {}
    info_path = os.path.join(task_path, 'Training_dataset', 'data_info')
    try:
        os.mkdir(info_path)
    except:
        print(f'{info_path} is already existed !!!')
    
    if num_seg < len(label):
        piece_data['fold_0'] = random.sample(label, num_seg)
        unused = list(filter(lambda x: x not in piece_data['fold_0'], label))
        unlabel.extend(unused)
    else:
        piece_data['fold_0'] = label
        
    # compute number of data for each fold
    num_images = len(unlabel)
    num_each_fold = divmod(num_images, k_fold)[0]
    fold_num = np.repeat(num_each_fold, k_fold)
    num_remain = divmod(num_images, k_fold)[1]
    count = 0
    while num_remain > 0:
        fold_num[count] += 1
        count = (count+1) % k_fold 
        num_remain -= 1
    
    random.shuffle(unlabel)
    start_point = 0
    # select scans for each fold
    for m in range(k_fold):
        piece_data[f'fold_{m+1}'] = unlabel[start_point:start_point+fold_num[m]]
        start_point += fold_num[m]
    
    with open(os.path.join(info_path, f'info_{num_seg}gt.json'), 'w') as f:
        json.dump(piece_data, f, indent=4, sort_keys=True)

    if os.path.exists(os.path.join(info_path, f'info_{num_seg}gt.json')):
        print("new json file created!")

if __name__ == '__main__':
    main()
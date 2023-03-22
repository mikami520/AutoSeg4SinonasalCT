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
import sys
from collections import namedtuple

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/utils'))

from utils import (
    make_if_dont_exist, load_json
)
"""
creates a folder at a specified folder path if it does not exists
folder_path : relative path of the folder (from cur_dir) which needs to be created
over_write :(default: False) if True overwrite the existing folder 
 """
def parse_command_line():
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='pipeline for dataset split')
    parser.add_argument('--config', metavar='path to the configuration file', type=str,
                        help='absolute path to the configuration file')
    parser.add_argument('--train_only', action='store_true',
                        help='only training or training plus test')
    argv = parser.parse_args()
    return argv


def split(img, seg, seg_path):
    label = []
    unlabel = []
    total = []
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

        total.append(item)
    return label, unlabel, total

def main():
    random.seed(2938649572)
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    config = args.config
    config = load_json(config)
    config = namedtuple("config", config.keys())(*config.values())
    task_id = config.task_name
    k_fold = config.num_fold
    train_only = args.train_only
    deepatlas_path = ROOT_DIR
    base_path = os.path.join(deepatlas_path, "deepatlas_preprocessed")
    task_path = os.path.join(base_path, task_id)
    img_path = os.path.join(task_path, 'Training_dataset', 'images')
    seg_path = os.path.join(task_path, 'Training_dataset', 'labels')
    image_list = glob.glob(img_path + "/*.nii.gz")
    label_list = glob.glob(seg_path + "/*.nii.gz")
    label, unlabel, total = split(image_list, label_list, seg_path)
    piece_data = {}
    info_path = os.path.join(task_path, 'Training_dataset', 'data_info')
    make_if_dont_exist(info_path)
    
    if not train_only: 
        # compute number of scans for each fold
        num_images = len(image_list)
        num_each_fold_scan = divmod(num_images, k_fold)[0]
        fold_num_scan = np.repeat(num_each_fold_scan, k_fold)
        num_remain_scan = divmod(num_images, k_fold)[1]
        count = 0
        while num_remain_scan > 0:
            fold_num_scan[count] += 1
            count = (count+1) % k_fold 
            num_remain_scan -= 1
        
        # compute number of labels for each fold
        num_seg = len(label_list)
        num_each_fold_seg = divmod(num_seg, k_fold)[0]
        fold_num_seg = np.repeat(num_each_fold_seg, k_fold)
        num_remain_seg = divmod(num_seg, k_fold)[1]
        count = 0
        while num_remain_seg > 0:
            fold_num_seg[count] += 1
            count = (count+1) % k_fold 
            num_remain_seg -= 1
        
        random.shuffle(unlabel)
        random.shuffle(label)
        start_point = 0
        start_point1 = 0
        # select scans for each fold
        for m in range(k_fold):
            piece_data[f'fold_{m+1}'] = label[start_point:start_point+fold_num_seg[m]]
            fold_num_unlabel = fold_num_scan[m] - fold_num_seg[m]
            piece_data[f'fold_{m+1}'].extend(unlabel[start_point1:start_point1+fold_num_unlabel])
            start_point += fold_num_seg[m]
            start_point1 += fold_num_unlabel
        
        info_json_path = os.path.join(info_path, f'info.json')
    else:
        piece_data = total
        info_json_path = os.path.join(info_path, f'info_train_only.json')
    
    with open(info_json_path, 'w') as f:
        json.dump(piece_data, f, indent=4, sort_keys=True)

    if os.path.exists(info_json_path):
        print("new info json file created!")

if __name__ == '__main__':
    main()
from pkg_resources import add_activation_listener
import monai
import torch
import itk
import numpy as np
import os.path
import argparse
import sys
from pathlib import Path
import deep_atlas_train
import glob

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/test'))

from test import (
    seg_training_inference, load_json, reg_training_inference
)

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas test')
    parser.add_argument('-gpu', metavar='id of gpu', type=str, default='0',
                        help='id of gpu device to use')
    parser.add_argument('-ti', metavar='task id and name', type=str,
                        help='task name and id')
    parser.add_argument('-nf', metavar='number of fold', type=int,
                        help='number of fold for testing')
    parser.add_argument('-op', metavar='output path for prediction step', type=str,
                        help="relative path of the output directory, should be same name in the registration, crop and final prediction steps")
    argv = parser.parse_args()
    return argv


def path_to_id(path):
    return os.path.basename(path).split('.')[0]


def main():
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    gpu = args.gpu
    task = args.ti
    num_fold = f'fold_{args.nf}'
    output_path = os.path.join(ROOT_DIR, 'deepatlas_results', task, 'customize_predicted_results')
    fold_path = os.path.join(output_path, num_fold)
    out_path = os.path.join(fold_path, args.op)
    json_path = os.path.join(
        ROOT_DIR, 'deepatlas_results', task, "training_results", num_fold, 'dataset.json')
    seg_model_path = os.path.join(
        ROOT_DIR, 'deepatlas_results', task, 'training_results', num_fold, 'SegNet', 'seg_net_best.pth')
    reg_model_path = os.path.join(
        ROOT_DIR, 'deepatlas_results', task, 'training_results', num_fold, 'RegNet', 'reg_net_best.pth')
    json_file = load_json(json_path)
    labels = json_file['labels']
    num_label = len(labels.keys())
    network_info = json_file['network']
    spatial_dim = network_info['spatial_dim']
    dropout = network_info['dropout']
    activation_type = network_info['activation_type']
    normalization_type = network_info['normalization_type']
    num_res = network_info['num_res']
    device = torch.device("cuda:" + gpu)
    output_seg_path = os.path.join(out_path, 'SegNet')
    output_reg_path = os.path.join(out_path, 'RegNet')
    try:
        os.mkdir(output_path)
    except:
        print(f'{output_path} is already existed !!!')

    try:
        os.mkdir(fold_path)
    except:
        print(f'{fold_path} is already existed !!!')
    
    try:
        os.mkdir(out_path)
    except:
        print(f'{out_path} is already existed !!!')

    try:
        os.mkdir(output_seg_path)
    except:
        print(f'{output_seg_path} is already existed !!!')

    try:
        os.mkdir(output_reg_path)
    except:
        print(f'{output_reg_path} is already existed !!!')

    seg_net = deep_atlas_train.get_seg_net(
        spatial_dim, num_label, dropout, activation_type, normalization_type, num_res)
    reg_net = deep_atlas_train.get_reg_net(
        spatial_dim, spatial_dim, dropout, activation_type, normalization_type, num_res)
    
    img_path = os.path.join(ROOT_DIR, 'deepatlas_preprocessed', task, 'customize_test_data', args.op, 'images')
    seg_path = os.path.join(ROOT_DIR, 'deepatlas_preprocessed', task, 'customize_test_data', args.op, 'labels')
    total_img_paths = []
    total_seg_paths = []
    for i in sorted(glob.glob(img_path + '/*.nii.gz')):
        total_img_paths.append(i)

    for j in sorted(glob.glob(seg_path + '/*.nii.gz')):
        total_seg_paths.append(j)
    
    seg_ids = list(map(path_to_id, total_seg_paths))
    img_ids = map(path_to_id, total_img_paths)
    data = []
    for img_index, img_id in enumerate(img_ids):
        data_item = {'img': total_img_paths[img_index]}
        if img_id in seg_ids:
            data_item['seg'] = total_seg_paths[seg_ids.index(img_id)]
        data.append(data_item)

    seg_training_inference(seg_net, device, seg_model_path, output_seg_path, num_label, json_path=None, data=data)
    reg_training_inference(reg_net, device, reg_model_path, output_reg_path, num_label, json_path=None, data=data)


if __name__ == '__main__':
    main()

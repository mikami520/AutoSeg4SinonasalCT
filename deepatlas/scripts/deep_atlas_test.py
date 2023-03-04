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
from collections import namedtuple

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/test'))

from test import (
    seg_training_inference, load_json, reg_training_inference
)

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas test')
    parser.add_argument('--config', metavar='path to the configuration file', type=str,
                        help='absolute path to the configuration file')
    argv = parser.parse_args()
    return argv


def main():
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    config = args.config
    config = load_json(config)
    config = namedtuple("config", config.keys())(*config.values())
    task = config.task_name
    for i in range(1, config.num_fold+1):
        num_fold = f'fold_{i}'
        output_path = os.path.join(ROOT_DIR, 'deepatlas_results', task, 'training_predicted_results')
        json_path = os.path.join(
            ROOT_DIR, 'deepatlas_results', task, 'training_results', num_fold, 'dataset.json')
        #num_fold = json_file['num_fold']
        output_fold_path = os.path.join(output_path, num_fold)
        seg_model_path = os.path.join(
            ROOT_DIR, 'deepatlas_results', task, 'training_results', num_fold, 'SegNet', 'seg_net_best.pth')
        reg_model_path = os.path.join(
            ROOT_DIR, 'deepatlas_results', task, 'training_results', num_fold, 'RegNet', 'reg_net_best.pth')
        labels = config.labels
        num_label = len(labels.keys())
        network_info = config.network
        spatial_dim = network_info['spatial_dim']
        dropout = network_info['dropout']
        activation_type = network_info['activation_type']
        normalization_type = network_info['normalization_type']
        num_res = network_info['num_res']
        device = torch.device("cuda:" + config.gpu)
        seg_path = os.path.join(output_fold_path, 'SegNet')
        reg_path = os.path.join(output_fold_path, 'RegNet')
        try:
            os.mkdir(output_path)
        except:
            print(f'{output_path} is already existed !!!')

        try:
            os.mkdir(output_fold_path)
        except:
            print(f'{output_fold_path} is already existed !!!')
        
        try:
            os.mkdir(seg_path)
        except:
            print(f'{seg_path} is already existed !!!')

        try:
            os.mkdir(reg_path)
        except:
            print(f'{reg_path} is already existed !!!')

        seg_net = deep_atlas_train.get_seg_net(
            spatial_dim, num_label, dropout, activation_type, normalization_type, num_res)
        reg_net = deep_atlas_train.get_reg_net(
            spatial_dim, spatial_dim, dropout, activation_type, normalization_type, num_res)
        seg_training_inference(seg_net, device, seg_model_path, seg_path, num_label, json_path=json_path, data=None)
        reg_training_inference(reg_net, device, reg_model_path, reg_path, num_label, json_path=json_path, data=None)


if __name__ == '__main__':
    main()

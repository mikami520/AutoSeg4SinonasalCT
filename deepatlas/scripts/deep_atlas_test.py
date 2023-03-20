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
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/utils'))
from test import (
    seg_training_inference, reg_training_inference
)
from utils import (
    make_dir, load_json
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
    monai.utils.set_determinism(seed=2938649572)
    config = args.config
    config = load_json(config)
    config = namedtuple("config", config.keys())(*config.values())
    task = config.task_name
    info_name = config.info_name
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    output_path = os.path.join(ROOT_DIR, 'deepatlas_results', task, f'set_{config.exp_set}',f'{config.num_seg_used}gt', 'training_predicted_results')
    make_dir(output_path)
    for i in range(1, config.num_fold+1):
        num_fold = f'fold_{i}'
        json_path = os.path.join(
            ROOT_DIR, 'deepatlas_results', task, f'set_{config.exp_set}',f'{config.num_seg_used}gt', 'training_results', num_fold, 'dataset.json')
        #num_fold = json_file['num_fold']
        output_fold_path = os.path.join(output_path, num_fold)
        seg_model_path = os.path.join(Path(json_path).parent.absolute(), 'SegNet', 'model', 'seg_net_best.pth')
        reg_model_path = os.path.join(Path(json_path).parent.absolute(), 'RegNet', 'model', 'reg_net_best.pth')
        labels = config.labels
        num_label = len(labels.keys())
        network_info = config.network
        spatial_dim = network_info['spatial_dim']
        dropout = network_info['dropout']
        activation_type = network_info['activation_type']
        normalization_type = network_info['normalization_type']
        num_res = network_info['num_res']
        seg_path = os.path.join(output_fold_path, 'SegNet')
        reg_path = os.path.join(output_fold_path, 'RegNet')
        make_dir(output_fold_path)
        make_dir(seg_path)
        make_dir(reg_path)
        seg_net = deep_atlas_train.get_seg_net(
            spatial_dim, num_label, dropout, activation_type, normalization_type, num_res)
        reg_net = deep_atlas_train.get_reg_net(
            spatial_dim, spatial_dim, dropout, activation_type, normalization_type, num_res)
        seg_training_inference(seg_net, device, seg_model_path, seg_path, num_label, json_path=json_path, data=None)
        reg_training_inference(reg_net, device, reg_model_path, reg_path, num_label, json_path=json_path, data=None)


if __name__ == '__main__':
    main()

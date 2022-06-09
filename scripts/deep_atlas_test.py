import monai
import torch
import itk
import numpy as np
import os.path
import argparse
import sys
from pathlib import Path
import deep_atlas_train

sys.path.insert(0, '/home/ameen/DeepAtlas/test')

from test import (
    seg_inference, load_json
)


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas test')
    parser.add_argument('-jp', metavar='path to the dataset.json file', type=str,
                        help="absolute path of the dataset.json file")
    parser.add_argument('-mp', metavar='path to the saved best model', type=str,
                        help="absolute path of the saved best model")
    parser.add_argument('-gpu', metavar='id of gpu', type=str, default='0',
                        help='id of gpu device to use')
    parser.add_argument('-op', metavar='prediction result output path', type=str, default='prediction',
                        help='relative path of the prediction result directory')
    parser.add_argument('-ti', metavar='task id and name', type=str, 
                        help='task name and id')
    parser.add_argument('-sd', metavar='spatial dimension', type=int, default=3,
                        help='spatial dimension of dataset')
    parser.add_argument('-dr', metavar='value of dropout', type=float, default=0.0,
                        help='dropout ratio. Defaults to no dropout.')
    parser.add_argument('-at', metavar='activation type and arguments', type=str, default='prelu',
                        help='activation type and arguments. Defaults to PReLU.')
    parser.add_argument('-nm', metavar='feature normalization type and arguments', type=str, default='instance',
                        help='feature normalization type and arguments. Defaults to instance norm.')
    parser.add_argument('-nr', metavar='number of residual units', type=int, default=0,
                        help='number of residual units. Defaults to 0.')
    
    argv = parser.parse_args()
    return argv

def main():
    ROOT_DIR = str(Path(os.getcwd()).parent.absolute())
    args = parse_command_line()
    json_path = args.jp
    model_path = args.mp
    gpu = args.gpu
    output_path = args.op
    task = args.ti
    spatial_dim = args.sd
    dropout = args.dr
    activation_type = args.at
    normalization_type = args.nm
    num_res = args.nr
    json_file = load_json(json_path)
    labels = json_file['labels']
    num_label = len(labels.keys())
    device = torch.device("cuda:" + gpu)

    output_path = os.path.join(ROOT_DIR, 'DeepAtlas_dataset', task, output_path)
    
    try:
        os.mkdir(output_path)
    except:
        print(f'{output_path} is already existed !!!')

    seg_net = deep_atlas_train.get_seg_net(spatial_dim, num_label, dropout, activation_type, normalization_type, num_res)

    seg_inference(seg_net, device, model_path, json_path, output_path)


if __name__ == '__main__':
    monai.utils.set_determinism(seed=2938649572)
    main()
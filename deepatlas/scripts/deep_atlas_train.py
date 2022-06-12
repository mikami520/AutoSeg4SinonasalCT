from process_data import (
    split_data, load_seg_dataset, load_reg_dataset, take_data_pairs, subdivide_list_of_data_pairs
)
from network import (
    regNet, segNet
)
from train import (
    train_network
)
import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os.path
import argparse
import sys
import json
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0, '/home/ameen/DeepAtlas/deepatlas/preprocess')
sys.path.insert(0, '/home/ameen/DeepAtlas/deepatlas/network')
sys.path.insert(0, '/home/ameen/DeepAtlas/deepatlas/train')


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas train')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="Relative path of the image directory")
    # parser.add_argument('-op', metavar='preprocessing result output path', type=str, default='preprocessing',
    # help='Relative path of the preprocessing result directory')
    parser.add_argument('-sl', metavar='segmentation information list', type=str, nargs='+',
                        help='a list of label name and corresponding value')
    parser.add_argument('-ns', metavar='number of segmentations', type=int, default=3,
                        help='number of segmentations used for training')
    parser.add_argument('-sd', metavar='spatial dimension', type=int, default=3,
                        help='spatial dimension of dataset')
    # parser.add_argument('-ch', metavar='sequence of channels', type=int, nargs='+',
    # help='sequence of channels. Top block first. The length of `channels` should be no less than 2.')
    # parser.add_argument('-st', metavar='sequence of strides', type=int, nargs='+',
    # help='sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1')
    parser.add_argument('-dr', metavar='value of dropout', type=float, default=0.0,
                        help='dropout ratio. Defaults to no dropout.')
    parser.add_argument('-gpu', metavar='id of gpu', type=str, default='0',
                        help='id of gpu device to use')
    parser.add_argument('-at', metavar='activation type and arguments', type=str, default='prelu',
                        help='activation type and arguments. Defaults to PReLU.')
    parser.add_argument('-nm', metavar='feature normalization type and arguments', type=str, default='instance',
                        help='feature normalization type and arguments. Defaults to instance norm.')
    parser.add_argument('-nr', metavar='number of residual units', type=int, default=0,
                        help='number of residual units. Defaults to 0.')
    parser.add_argument('-lr', metavar='learning rate of registration network', type=float, default=0.01,
                        help='learning rate of registration network. Defaults to 0.01.')
    parser.add_argument('-ls', metavar='learning rate of segmentation network', type=float, default=0.01,
                        help='learning rate of segmentation network. Defaults to 0.01.')
    parser.add_argument('-ba', metavar='anatomy loss weight', type=float, default=1.0,
                        help='anatomy loss weight. Defaults to 1.0.')
    parser.add_argument('-bs', metavar='supervised segmentation loss weight', type=float, default=1.0,
                        help='supervised segmentation loss weight. Defaults to 1.0.')
    parser.add_argument('-me', metavar='maximum number of training epochs', type=int, default=100,
                        help='maximum number of training epochs. Defaults to 100.')
    parser.add_argument('-vs', metavar='validation steps per epoch', type=int, default=5,
                        help='validation steps per epoch. Defaults to 5.')
    parser.add_argument('-ti', metavar='task id and name', type=str,
                        help='task name and id')
    argv = parser.parse_args()
    return argv


def get_seg_net(spatial_dims, num_label, dropout, activation_type, normalization_type, num_res):
    seg_net = segNet(
        spatial_dim=spatial_dims,  # spatial dims
        in_channel=1,  # input channels
        out_channel=num_label,  # output channels
        channel=(8, 16, 16, 32, 32, 64, 64),  # channel sequence
        stride=(1, 2, 1, 2, 1, 2),  # convolutional strides
        dropouts=dropout,
        acts=activation_type,
        norms=normalization_type,
        num_res_unit=num_res
    )
    return seg_net


def get_reg_net(spatial_dims, num_label, dropout, activation_type, normalization_type, num_res):
    reg_net = regNet(
        spatial_dim=spatial_dims,  # spatial dims
        in_channel=2,  # input channels
        out_channel=num_label,  # output channels
        channel=(16, 32, 32, 32, 32, 32, 32, 32,
                 32, 16, 16),  # channel sequence
        stride=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),  # convolutional strides
        dropouts=dropout,
        acts=activation_type,
        norms=normalization_type,
        num_res_unit=num_res
    )
    return reg_net


def main():
    args = parse_command_line()
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    data_path = os.path.join(ROOT_DIR, 'DeepAtlas_dataset')
    base_path = args.bp
    seg_list = args.sl
    img_path = os.path.join(base_path, args.ip)
    seg_path = os.path.join(base_path, args.sp)
    task = os.path.join(data_path, args.ti)
    result_path = os.path.join(task, 'results')
    result_seg_path = os.path.join(result_path, 'SegNet')
    result_reg_path = os.path.join(result_path, 'RegNet')
    #output_path = os.path.join(task, args.op)
    num_seg = args.ns
    spatial_dim = args.sd
    dropout = args.dr
    activation_type = args.at
    normalization_type = args.nm
    num_res = args.nr
    gpu = args.gpu
    lr_reg = args.lr
    lr_seg = args.ls
    lam_a = args.ba
    lam_sp = args.bs
    max_epoch = args.me
    val_step = args.vs
    device = torch.device("cuda:" + gpu)

    try:
        os.mkdir(data_path)
    except:
        print('---'*10)
        print(f'{data_path} is already existed !!!')

    try:
        os.mkdir(task)
    except:
        print('---'*10)
        print(f'{task} is already existed !!!')

    try:
        os.mkdir(result_path)
    except:
        print('---'*10)
        print(f'{result_path} is already existed !!!')

    try:
        os.mkdir(result_seg_path)
    except:
        print('---'*10)
        print(f'{result_seg_path} is already existed !!!')

    try:
        os.mkdir(result_reg_path)
    except:
        print('---'*10)
        print(f'{result_reg_path} is already existed !!!')

    print('---'*10)
    print('split dataset into train and test')
    json_dict = OrderedDict()
    json_dict['name'] = os.path.basename(task).split('_')[0]
    json_dict['description'] = os.path.basename(task).split('_')[1]
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "MODIFY"
    json_dict['licence'] = "MODIFY"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
    }
    for i in range(0, len(seg_list), 2):
        assert(seg_list[i].isdigit() == True)
        assert(seg_list[i + 1].isdigit() == False)
        json_dict['labels'].update({
            seg_list[i]: seg_list[i + 1]
        })

    train, test, num_train, num_test = split_data(img_path, seg_path, num_seg)
    json_dict['total_numScanTraining'] = num_train
    json_dict['total_numLabelTraining'] = num_seg
    json_dict['total_numTest'] = num_test
    json_dict['toal_train'] = train
    json_dict['total_test'] = test
    # prepare segmentation dataset
    print('---'*10)
    print('prepare segmentation dataset')
    data_seg_available = list(filter(lambda d: 'seg' in d.keys(), train))
    data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), train))
    data_seg_available_train, data_seg_available_valid = \
        monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))
    json_dict['seg_numTrain'] = len(data_seg_available)
    json_dict['seg_train'] = data_seg_available
    dataset_seg_available_train, dataset_seg_available_valid = load_seg_dataset(
        data_seg_available_train, data_seg_available_valid)
    data_item = random.choice(dataset_seg_available_train)
    num_label = len(torch.unique(data_item['seg']))
    print('---'*10)
    print('prepare segmentation network')
    seg_net = get_seg_net(spatial_dim, num_label, dropout,
                          activation_type, normalization_type, num_res)
    print(seg_net)
    '''
    print('---'*10)
    print('check foward pass for segmentation network')
    seg_net_example_output = seg_net(data_item['img'].unsqueeze(0))
    print(f"Segmentation classes: {torch.unique(data_item['seg'])}")
    print(f"Shape of ground truth label: {data_item['seg'].unsqueeze(0).shape}")
    print(f"Shape of seg_net output: {seg_net_example_output.shape}")
    '''
    # prepare registration dataset
    print('---'*10)
    print('prepare registration dataset')
    data_without_seg_valid = data_seg_unavailable + data_seg_available_train
    data_valid, data_train = monai.data.utils.partition_dataset(
        data_without_seg_valid,  # Note the order
        ratios=(2, 8),  # Note the order
        shuffle=False
    )
    data_paires_without_seg_valid = take_data_pairs(data_without_seg_valid)
    data_pairs_valid = take_data_pairs(data_valid)
    data_pairs_train = take_data_pairs(data_train)
    data_pairs_valid_subdivided = subdivide_list_of_data_pairs(
        data_pairs_valid)
    data_pairs_train_subdivided = subdivide_list_of_data_pairs(
        data_pairs_train)
    num_train_reg_net = len(data_pairs_train)
    num_valid_reg_net = len(data_pairs_valid)
    num_train_both = len(data_pairs_train_subdivided['01']) +\
        len(data_pairs_train_subdivided['10']) +\
        len(data_pairs_train_subdivided['11'])
    json_dict['reg_numTrain'] = num_train_reg_net + num_valid_reg_net
    json_dict['reg_train'] = data_paires_without_seg_valid
    print('---'*10)
    print(f"""We have {num_train_both} pairs to train reg_net and seg_net together,
    and an additional {num_train_reg_net - num_train_both} to train reg_net alone.""")
    print(f"We have {num_valid_reg_net} pairs for reg_net validation.")
    with open(os.path.join(task, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=False)

    dataset_pairs_train_subdivided, dataset_pairs_valid_subdivided = load_reg_dataset(
        data_pairs_train_subdivided, data_pairs_valid_subdivided)
    print('---'*10)
    print('prepare registration network')
    reg_net = get_reg_net(spatial_dim, spatial_dim, dropout,
                          activation_type, normalization_type, num_res)
    print(reg_net)
    datasets = list(dataset_pairs_train_subdivided.values())
    datasets_combined = sum(datasets[1:], datasets[0])
    data_item = random.choice(datasets_combined)
    reg_net_example_input = data_item['img12'].unsqueeze(0)
    image_scale = reg_net_example_input.shape[-1]
    '''
    print('---'*10)
    print('check foward pass for segmentation network')
    datasets = list(dataset_pairs_train_subdivided.values())
    datasets_combined = sum(datasets[1:], datasets[0])
    data_item = random.choice(datasets_combined)
    reg_net_example_input = data_item['img12'].unsqueeze(0)
    reg_net_example_output = reg_net(reg_net_example_input)
    print(f"Shape of reg_net input: {reg_net_example_input.shape}")
    print(f"Shape of reg_net output: {reg_net_example_output.shape}")
    '''
    dataloader_train_seg = monai.data.DataLoader(
        dataset_seg_available_train,
        batch_size=4,
        num_workers=4,
        shuffle=True
    )
    dataloader_valid_seg = monai.data.DataLoader(
        dataset_seg_available_valid,
        batch_size=8,
        num_workers=4,
        shuffle=False
    )
    dataloader_train_reg = {
        seg_availability: monai.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=4,
            shuffle=True
        )
        # empty dataloaders are not a thing-- put an empty list if needed
        if len(dataset) > 0 else []
        for seg_availability, dataset in dataset_pairs_train_subdivided.items()
    }
    dataloader_valid_reg = {
        seg_availability: monai.data.DataLoader(
            dataset,
            batch_size=8,
            num_workers=4,
            shuffle=True  # Shuffle validation data because we will only take a sample for validation each time
        )
        # empty dataloaders are not a thing-- put an empty list if needed
        if len(dataset) > 0 else []
        for seg_availability, dataset in dataset_pairs_valid_subdivided.items()
    }

    train_network(dataloader_train_reg,
                  dataloader_valid_reg,
                  dataloader_train_seg,
                  dataloader_valid_seg,
                  device,
                  seg_net,
                  reg_net,
                  image_scale,
                  num_label,
                  lr_reg,
                  lr_seg,
                  lam_a,
                  lam_sp,
                  max_epoch,
                  val_step,
                  result_seg_path,
                  result_reg_path
                  )


if __name__ == '__main__':
    # Set deterministic training for reproducibility
    monai.utils.set_determinism(seed=2938649572)
    main()

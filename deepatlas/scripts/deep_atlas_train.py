import seg_train
from pathlib import Path
from collections import OrderedDict
import json
import sys
import argparse
import os.path
import glob
import random
import matplotlib.pyplot as plt
import torch
import monai
import logging
import shutil
from collections import namedtuple
import numpy as np
import datetime

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/preprocess'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/network'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/train'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/utils'))
from train import (
    train_network
)
from network import (
    regNet, segNet
)
from process_data import (
    split_data, load_seg_dataset, load_reg_dataset, take_data_pairs, subdivide_list_of_data_pairs
)
from utils import (
    load_json, make_if_dont_exist
)

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas train')
    parser.add_argument('--config', metavar='path to the configuration file', type=str,
                        help='absolute path to the configuration file')
    parser.add_argument('--continue_training', action='store_true',
                        help='use this if you want to continue a training')
    parser.add_argument('--train_only', action='store_true',
                        help='only training or training plus test')
    parser.add_argument('--plot_network', action='store_true',
                        help='whether to plot the network')
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
        channel=(16, 32, 32, 32, 32),  # channel sequence
        stride=(1, 2, 2, 2),  # convolutional strides
        dropouts=dropout,
        acts=activation_type,
        norms=normalization_type,
        num_res_unit=num_res
    )
    return reg_net


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)

def classify_data(data_info, fold):
    lab_each_fold = {}
    lab = []
    unlab = []
    total_seg = 0
    total_seg_each_fold = {}
    for key, value in data_info.items():
        if key != f'fold_{fold}':
            lab_each_fold[key] = []
            total_seg_each_fold[key] = 0
            for val in value:
                if 'seg' not in val.keys():
                    unlab.append(val)
                else:
                    lab_each_fold[key].append(val)
                    lab.append(val)
                    total_seg += 1
                    total_seg_each_fold[key] += 1
                    
    return lab_each_fold, lab, unlab, total_seg, total_seg_each_fold

def select_n_seg(lab, fold, num, total_seg_each_fold):
    seg_items = lab[f'fold_{fold}']
    num_seg = len(seg_items)
    rand_num = random.sample(range(num_seg), num)
    seg_item = np.array(seg_items)[np.array(rand_num)]
    seg_items.pop(rand_num[0])
    total_seg_each_fold[f'fold_{fold}'] -= 1
    lab[f'fold_{fold}'] = seg_items
    return list(seg_item), lab, total_seg_each_fold

def combine_data(data_info, fold, exp, num_seg):
    all_fold = np.arange(len(data_info.keys())) + 1
    num_train_fold = len(data_info.keys()) - 1
    fake_train_fold = np.delete(all_fold, fold-1)
    fake_train_fold = np.tile(fake_train_fold, 2)
    real_train_fold = fake_train_fold[fold-1:fold+num_train_fold-1]
    train = []
    test = []
    for j in data_info[f'fold_{fold}']:
        if 'seg' in j.keys():
            test.append(j)
    
    lab_each_fold, lab, unlab, total_seg, total_seg_each_fold = classify_data(data_info, fold)
    if total_seg < num_seg:
        num_seg = total_seg
    
    num_each_fold_seg = divmod(num_seg, num_train_fold)[0]
    fold_num_seg = np.repeat(num_each_fold_seg, num_train_fold)
    num_remain_seg = divmod(num_seg, num_train_fold)[1]
    count = 0
    while num_remain_seg > 0:
        fold_num_seg[count] += 1
        count = (count+1) % num_train_fold
        num_remain_seg -= 1
    
    train = unlab
    k = 0
    while num_seg > 0:
        next_fold = real_train_fold[k]
        if total_seg_each_fold[f'fold_{next_fold}'] > 0:
            seg_items, lab_each_fold, total_seg_each_fold = select_n_seg(lab_each_fold, next_fold, 1, total_seg_each_fold)
            train.extend(seg_items)     
            num_seg -= 1
        k = (k+1) % 4
            
    num_segs = 0
    if exp != 1:
        for key, value in total_seg_each_fold.items():
            if value != 0:
                for j in lab_each_fold[key]:
                    item = {'img': j['img']}
                    train.append(item)
                    total_seg_each_fold[key] -= 1
        for key, value in total_seg_each_fold.items():
            num_segs += value
        
        assert num_segs == 0
    
    return train, test


def main():
    args = parse_command_line()
    config = args.config
    continue_training = args.continue_training
    train_only = args.train_only
    config = load_json(config)
    config = namedtuple("config", config.keys())(*config.values())
    folder_name = config.folder_name
    num_seg_used = config.num_seg_used
    experiment_set = config.exp_set
    monai.utils.set_determinism(seed=2938649572)
    data_path = os.path.join(ROOT_DIR, 'deepatlas_results')
    base_path = os.path.join(ROOT_DIR, 'deepatlas_preprocessed')
    task = os.path.join(data_path, config.task_name)
    exp_path = os.path.join(task, f'set_{experiment_set}')
    gt_path = os.path.join(exp_path, f'{num_seg_used}gt')
    folder_path = os.path.join(gt_path, folder_name)
    result_path = os.path.join(folder_path, 'training_results')
    if train_only:
        info_name = 'info_train_only'
    else:
        info_name = 'info'
    info_path = os.path.join(base_path, config.task_name, 'Training_dataset', 'data_info', folder_name, info_name+'.json')
    info = load_json(info_path)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    
    spatial_dim = config.network['spatial_dim']
    dropout = config.network['dropout']
    activation_type = config.network['activation_type']
    normalization_type = config.network['normalization_type']
    num_res = config.network['num_res']
    lr_reg = config.network["registration_network_learning_rate"]
    lr_seg = config.network["segmentation_network_learning_rate"]
    lam_a = config.network["anatomy_loss_weight"]
    lam_sp = config.network["supervised_segmentation_loss_weight"]
    lam_re = config.network["regularization_loss_weight"]
    max_epoch = config.network["number_epoch"]
    val_step = config.network["validation_step"]
    make_if_dont_exist(data_path)
    make_if_dont_exist(task)
    make_if_dont_exist(exp_path)
    make_if_dont_exist(gt_path)
    make_if_dont_exist(folder_path)
    make_if_dont_exist(result_path)
    
    if not continue_training:
        start_fold = 1
    else:
        folds = sorted(os.listdir(result_path))
        if len(folds) == 0:
            continue_training = False
            start_fold = 1
        else:
            last_fold_num = folds[-1].split('_')[-1]
            start_fold = int(last_fold_num)
    
    if train_only:
        num_fold = 1
    else:
        num_fold = config.num_fold
    
    for i in range (start_fold, num_fold+1):
        if not train_only:
            fold_path = os.path.join(result_path, f'fold_{i}')
            result_seg_path = os.path.join(fold_path, 'SegNet')
            result_reg_path = os.path.join(fold_path, 'RegNet')
        else:
            fold_path = os.path.join(result_path, f'all')
            result_seg_path = os.path.join(fold_path, 'SegNet')
            result_reg_path = os.path.join(fold_path, 'RegNet')
        
        make_if_dont_exist(fold_path)
        make_if_dont_exist(result_reg_path)
        make_if_dont_exist(result_seg_path)
        datetime_object = 'training_log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log'
        log_path = os.path.join(fold_path, datetime_object)
        
        if not train_only:
            if not continue_training:
                setup_logger(f'log_{i}', log_path)
                logger = logging.getLogger(f'log_{i}')
                logger.info(f"Start Pipeline with fold_{i}")
            else:
                setup_logger(f'log_{i+1}', log_path)
                logger = logging.getLogger(f'log_{i+1}')
                logger.info(f"Resume Pipeline with fold_{i}")
        else:
            setup_logger(f'all', log_path)
            logger = logging.getLogger(f'all')
            logger.info(f"Start Pipeline with all data")

        if not os.path.exists(os.path.join(fold_path, 'dataset.json')):
            logger.info('prepare dataset into train and test')
            json_dict = OrderedDict()
            json_dict['name'] = os.path.basename(task).split('_')[0]
            json_dict['description'] = '_'.join(os.path.basename(task).split('_')[1:])
            json_dict['tensorImageSize'] = "4D"
            json_dict['reference'] = "MODIFY"
            json_dict['licence'] = "MODIFY"
            json_dict['release'] = "0.0"
            json_dict['modality'] = {
                "0": "CT"
            }
            json_dict['labels'] = config.labels
            json_dict['network'] = config.network
            json_dict['experiment_set'] = experiment_set
            if not train_only:
                json_dict['num_fold'] = f'fold_{i}'
                train, test = combine_data(info, i, experiment_set, num_seg_used)
            else:
                json_dict['num_fold'] = 'all'
                train = info
                test = []
                num_seg_used = len(list(filter(lambda d: 'seg' in d.keys(), train)))
            #num_seg = 15
            #train, test, num_train, num_test = split_data(img_path, seg_path, num_seg) 
            #print(type(train))
            
            num_seg = num_seg_used
            num_train = len(train)
            num_test = len(test)
            #print(train.keys())
            json_dict['total_numScanTraining'] = num_train
            json_dict['total_numLabelTraining'] = num_seg
            json_dict['total_numTest'] = num_test
            json_dict['total_train'] = train
            json_dict['total_test'] = test
            # prepare segmentation dataset
            logger.info('prepare segmentation dataset')
            data_seg_available = list(filter(lambda d: 'seg' in d.keys(), train))
            data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), train))
            data_seg_available_train, data_seg_available_valid = \
                monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))
            json_dict['seg_numTrain'] = len(data_seg_available_train)
            json_dict['seg_train'] = data_seg_available_train
            json_dict['seg_numValid'] = len(data_seg_available_valid)
            json_dict['seg_valid'] = data_seg_available_valid
            dataset_seg_available_train, dataset_seg_available_valid = load_seg_dataset(
                data_seg_available_train, data_seg_available_valid)
            data_item = random.choice(dataset_seg_available_train)
            img_shape = data_item['seg'].unsqueeze(0).shape[2:]
            num_label = len(torch.unique(data_item['seg']))
            logger.info('prepare segmentation network')
            seg_net = get_seg_net(spatial_dim, num_label, dropout,
                                activation_type, normalization_type, num_res)
            # prepare registration dataset
            logger.info('prepare registration dataset')
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
            json_dict['reg_seg_numTrain'] = num_train_reg_net
            json_dict['reg_seg_numTrain_00'] = len(data_pairs_train_subdivided['00'])
            json_dict['reg_seg_train_00'] = data_pairs_train_subdivided['00']
            json_dict['reg_seg_numTrain_01'] = len(data_pairs_train_subdivided['01'])
            json_dict['reg_seg_train_01'] = data_pairs_train_subdivided['01']
            json_dict['reg_seg_numTrain_10'] = len(data_pairs_train_subdivided['10'])
            json_dict['reg_seg_train_10'] = data_pairs_train_subdivided['10']
            json_dict['reg_seg_numTrain_11'] = len(data_pairs_train_subdivided['11'])
            json_dict['reg_seg_train_11'] = data_pairs_train_subdivided['11']
            json_dict['reg_numValid'] = num_valid_reg_net
            json_dict['reg_numValid_00'] = len(data_pairs_valid_subdivided['00'])
            json_dict['reg_valid_00'] = data_pairs_valid_subdivided['00']
            json_dict['reg_numValid_01'] = len(data_pairs_valid_subdivided['01'])
            json_dict['reg_valid_01'] = data_pairs_valid_subdivided['01']
            json_dict['reg_numValid_10'] = len(data_pairs_valid_subdivided['10'])
            json_dict['reg_valid_10'] = data_pairs_valid_subdivided['10']
            json_dict['reg_numValid_11'] = len(data_pairs_valid_subdivided['11'])
            json_dict['reg_valid_11'] = data_pairs_valid_subdivided['11']
            print(f"""We have {num_train_both} pairs to train reg_net and seg_net together, and an additional {num_train_reg_net - num_train_both} to train reg_net alone.""")
            print(f"We have {num_valid_reg_net} pairs for reg_net validation.")

            dataset_pairs_train_subdivided, dataset_pairs_valid_subdivided = load_reg_dataset(
                data_pairs_train_subdivided, data_pairs_valid_subdivided)
            logger.info('prepare registration network')
            reg_net = get_reg_net(spatial_dim, spatial_dim, dropout,
                                activation_type, normalization_type, num_res)
            logger.info('generate dataset json file')
            with open(os.path.join(fold_path, 'dataset.json'), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=False)

        else:
            dataset_json = load_json(os.path.join(fold_path, 'dataset.json'))
            
            data_seg_available_train = dataset_json['seg_train']
            data_seg_available_valid = dataset_json['seg_valid']
            dataset_seg_available_train, dataset_seg_available_valid = load_seg_dataset(data_seg_available_train, data_seg_available_valid)
            data_item = random.choice(dataset_seg_available_train)
            img_shape = data_item['seg'].unsqueeze(0).shape[2:]
            num_label = len(torch.unique(data_item['seg']))
            logger.info('prepare segmentation network')
            seg_net = get_seg_net(spatial_dim, num_label, dropout, activation_type, normalization_type, num_res)
            
            data_pairs_train_subdivided = {
                '00': dataset_json['reg_seg_train_00'],
                '01': dataset_json['reg_seg_train_01'],
                '10': dataset_json['reg_seg_train_10'],
                '11': dataset_json['reg_seg_train_11']
            }
            data_pairs_valid_subdivided = {
                '00': dataset_json['reg_valid_00'],
                '01': dataset_json['reg_valid_01'],
                '10': dataset_json['reg_valid_10'],
                '11': dataset_json['reg_valid_11']
            }
            num_train_reg_net = dataset_json['reg_seg_numTrain']
            num_valid_reg_net = dataset_json['reg_numValid']
            num_train_both = len(data_pairs_train_subdivided['01']) +\
                len(data_pairs_train_subdivided['10']) +\
                len(data_pairs_train_subdivided['11'])
            print(f"""We have {num_train_both} pairs to train reg_net and seg_net together,
            and an additional {num_train_reg_net - num_train_both} to train reg_net alone.""")
            print(f"We have {num_valid_reg_net} pairs for reg_net validation.")

            dataset_pairs_train_subdivided, dataset_pairs_valid_subdivided = load_reg_dataset(
                data_pairs_train_subdivided, data_pairs_valid_subdivided)
            logger.info('prepare registration network')
            reg_net = get_reg_net(spatial_dim, spatial_dim, dropout,
                                activation_type, normalization_type, num_res)

        
        dataloader_train_seg = monai.data.DataLoader(
            dataset_seg_available_train,
            batch_size=2,
            num_workers=4,
            shuffle=True
        )
        dataloader_valid_seg = monai.data.DataLoader(
            dataset_seg_available_valid,
            batch_size=4,
            num_workers=4,
            shuffle=False
        )
        dataloader_train_reg = {
            seg_availability: monai.data.DataLoader(
                dataset,
                batch_size=1,
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
                batch_size=2,
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
                      num_label,
                      lr_reg,
                      lr_seg,
                      lam_a,
                      lam_sp,
                      lam_re,
                      max_epoch,
                      val_step,
                      result_seg_path,
                      result_reg_path,
                      logger,
                      img_shape,
                      plot_network=args.plot_network,
                      continue_training=continue_training
                      )
    '''
    seg_train.train_seg(
        dataloader_train_seg,
        dataloader_valid_seg,
        device,
        seg_net,
        lr_seg,
        max_epoch,
        val_step,
        result_seg_path
    )
    '''

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

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

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/preprocess'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/network'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/train'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/test'))
from train import (
    train_network
)
from network import (
    regNet, segNet
)
from process_data import (
    split_data, load_seg_dataset, load_reg_dataset, take_data_pairs, subdivide_list_of_data_pairs
)

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas train')
    parser.add_argument('--config', metavar='path to the configuration file', type=str,
                        help='absolute path to the configuration file')
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

def load_json(json_path):
    assert type(json_path) == str
    fjson = open(json_path, 'r')
    json_file = json.load(fjson)
    return json_file

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
  

def main():
    args = parse_command_line()
    config = args.config
    config = load_json(config)
    config = namedtuple("config", config.keys())(*config.values())
    #monai.utils.set_determinism(seed=2938649572)
    data_path = os.path.join(ROOT_DIR, 'deepatlas_results')
    base_path = os.path.join(ROOT_DIR, 'deepatlas_preprocessed')
    task = os.path.join(data_path, config.task_name)
    gt_path = os.path.join(task, config.info_name.split('_')[1])
    img_path = os.path.join(base_path, config.task_name, 'Training_dataset', 'images')
    seg_path = os.path.join(base_path, config.task_name, 'Training_dataset', 'labels')
    info_path = os.path.join(base_path, config.task_name, 'Training_dataset', 'data_info', config.info_name+'.json')
    info = load_json(info_path)
    result_path = os.path.join(gt_path, 'training_results')
    try:
        os.mkdir(data_path)
    except:
        print(f'{data_path} is already existed !!!')

    try:
        os.mkdir(task)
    except:
        print(f'{task} is already existed !!!')
    
    try:
        os.mkdir(gt_path)
    except:
        print(f'{gt_path} is already existed !!!')
    
    try:
        os.mkdir(result_path)
    except:
        print(f'{result_path} is already existed !!!')
    for i in range (1, config.num_fold+1):
        fold_path = os.path.join(result_path, f'fold_{i}')
        result_seg_path = os.path.join(fold_path, 'SegNet')
        result_reg_path = os.path.join(fold_path, 'RegNet')
        log_path = os.path.join(base_path, config.task_name, 'Training_dataset', 'training_log.log')
        setup_logger(f'log_{i}', log_path)
        logger = logging.getLogger(f'log_{i}')
        logger.info("Start Pipeline")
        spatial_dim = config.network['spatial_dim']
        dropout = config.network['dropout']
        activation_type = config.network['activation_type']
        normalization_type = config.network['normalization_type']
        num_res = config.network['num_res']
        gpu = config.gpu
        lr_reg = config.network["registration_network_learning_rate"]
        lr_seg = config.network["segmentation_network_learning_rate"]
        lam_a = config.network["anatomy_loss_weight"]
        lam_sp = config.network["supervised_segmentation_loss_weight"]
        lam_re = config.network["regularization_loss_weight"]
        max_epoch = config.network["number_epoch"]
        val_step = config.network["validation_step"]
        device = torch.device("cuda:" + gpu)
        logger.info('create necessary folders')

        try:
            os.mkdir(fold_path)
        except:
            logger.info(f'{fold_path} is already existed !!!')
        
        try:
            os.mkdir(result_seg_path)
        except:
            logger.info(f'{result_seg_path} is already existed !!!')

        try:
            os.mkdir(result_reg_path)
        except:
            logger.info(f'{result_reg_path} is already existed !!!')

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
        json_dict['num_fold'] = f'fold_{i}'
        #num_seg = 15
        #train, test, num_train, num_test = split_data(img_path, seg_path, num_seg) 
        #print(type(train))
        train = []
        test = info[f'fold_{i}']
        train.extend(info['fold_0'])
        for key, value in info.items():
            if key != f'fold_{i}' and key != 'fold_0':
                for item in value:
                    it = {'img': item['img']}
                    train.append(it)
        
        num_seg = len(info['fold_0'])
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
        json_dict['seg_numValid'] = len(data_seg_available_valid)
        json_dict['seg_valid'] = data_seg_available_valid
        dataset_seg_available_train, dataset_seg_available_valid = load_seg_dataset(
            data_seg_available_train, data_seg_available_valid)
        data_item = random.choice(dataset_seg_available_train)
        num_label = len(torch.unique(data_item['seg']))
        logger.info('prepare segmentation network')
        seg_net = get_seg_net(spatial_dim, num_label, dropout,
                            activation_type, normalization_type, num_res)
        print(seg_net)

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
        print(f"""We have {num_train_both} pairs to train reg_net and seg_net together,
        and an additional {num_train_reg_net - num_train_both} to train reg_net alone.""")
        print(f"We have {num_valid_reg_net} pairs for reg_net validation.")
        logger.info('generate dataset json file')
        with open(os.path.join(fold_path, 'dataset.json'), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=False)

        dataset_pairs_train_subdivided, dataset_pairs_valid_subdivided = load_reg_dataset(
            data_pairs_train_subdivided, data_pairs_valid_subdivided)
        logger.info('prepare registration network')
        reg_net = get_reg_net(spatial_dim, spatial_dim, dropout,
                            activation_type, normalization_type, num_res)
        print(reg_net)

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
        if os.path.exists(os.path.join(fold_path, 'training_log.log')):
            os.remove(os.path.join(fold_path, 'training_log.log'))
        shutil.move(os.path.join(base_path, config.task_name, 'Training_dataset', 'training_log.log'), fold_path)
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
                    logger
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

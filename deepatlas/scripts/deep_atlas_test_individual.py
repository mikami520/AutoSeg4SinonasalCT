from pkg_resources import add_activation_listener
import monai
import torch
import itk
import numpy as np
import os.path
import argparse
import sys
from pathlib import Path
ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/test'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/preprocess'))


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for deep atlas test')
    parser.add_argument('-gpu', metavar='id of gpu', type=str, default='0',
                        help='id of gpu device to use')
    parser.add_argument('-template_scan', metavar='template scan path', type=str,
                        help='path to the template scan for initial registration')
    parser.add_argument('-target_scan', metavar='target scan path', type=str,
                        help='path to the target scan for initial registration')
    parser.add_argument('-target_label', metavar='target label path', type=str,
                        help='path to the target labels for initial registration')
    parser.add_argument('-ti', metavar='task id and name', type=str,
                        help='task name and id')
    argv = parser.parse_args()
    return argv

def main():
    ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
    args = parse_command_line()
    gpu = args.gpu
    task = args.ti
    output_path = os.path.join(ROOT_DIR, 'deepatlas_results', task_id, 'individual_predicted_results')
    template_path = args.template_scan
    target_path = args.target_scan
    target_label = args.target_label



if __name__ == '__main__':
    main()
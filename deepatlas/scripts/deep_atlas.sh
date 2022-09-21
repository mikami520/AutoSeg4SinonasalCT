#!/bin/bash
source ~/proj_MONAI/bin/activate
cd ~/DeepAtlas/deepatlas/preprocess
python3 registration_training.py -bp ~/Nasal\ Cavity/similarity -ip images -sp labels -sl 1 Septum 2 IT 3 MS -ti Task001_NC
python3 crop_flip_training.py -rs 240 240 240 -ti Task001_NC
cd ~/DeepAtlas/deepatlas/scripts
python3 deep_atlas_train.py -ti Task001_NC -sl 1 Septum 2 IT 3 MS -ns 10 -sd 3 -dr 0.2 -gpu 0 -at leakyrelu -nm batch -nr 0 -lr 1e-3 -ls 5e-4 -bg 5e-6 -ba 15.0 -bs 3.0 -me 5
python3 deep_atlas_test.py -gpu 0 -ti Task001_NC153_sim_Zsc 
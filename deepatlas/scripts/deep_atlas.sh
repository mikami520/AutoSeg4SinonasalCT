#!/bin/bash
source ~/proj_MONAI/bin/activate
cd ~/DeepAtlas/deepatlas/preprocess
python3 registration_training.py -bp ~/Test_data -ip images -sp labels -sl 1 Ear 2 Mid 3 Nasal -ti Task001_SepET
python3 crop_flip_training.py -rs 128 128 128 -fp -ti Task001_SepET
cd ~/DeepAtlas/deepatlas/scripts
python3 deep_atlas_train.py -ti Task001_SepET -sl 1 Ear 2 Mid 3 Nasal -ns 5 -sd 3 -dr 0.2 -gpu 0 -at leakyrelu -nm batch -nr 0 -lr 1e-3 -ls 5e-4 -bg 5e-6 -ba 3.0 -bs 3.0 -me 5
python3 deep_atlas_test.py -gpu 0 -ti Task002_rnmw153_sim_Zsc 
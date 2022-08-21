#!/bin/bash
source ~/proj_MONAI/bin/activate
cd ~/DeepAtlas/deepatlas/preprocess
python3 registration_test.py -op firstTest -bp ~/Test_data -template template -target_scan target -target_seg target_seg -sl 1 Ear 2 Mid 3 Nasal -ti Task001_SepET
python3 crop_flip_test.py -rs 128 128 128 -fp -ti Task001_SepET -op firstTest
cd ~/DeepAtlas/deepatlas/scripts
python3 deep_atlas_test_customized.py -gpu 0 -ti Task001_SepET -op firstTest
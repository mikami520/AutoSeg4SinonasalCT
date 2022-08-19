#!/bin/bash
source ~/proj_MONAI/bin/activate
cd ~/DeepAtlas/deepatlas/preprocess
python3 registration.py -bp ~/Test_data -ip images -sp labels -sl 1 Ear 2 Mid 3 Nasal -dp ~/DeepAtlas -ti Task001_SepET
python3 crop_flip.py -bp ~/Test_data -ip imagesRS -sp labelsRS -rs 128 128 128 -fp -dp ~/DeepAtlas -ti Task001_SepET
cd ~/DeepAtlas/deepatlas/scripts
python3 deep_atlas_train.py -bp ~/Test_data/output -ip images -sp labels -sl 1 Ear 2 Mid 3 Nasal -ns 5 -sd 3 -dr 0.2 -gpu 0 -at leakyrelu -nm batch -nr 0 -lr 1e-3 -ls 5e-4 -bg 5e-6 -ba 3.0 -bs 3.0 -me 5 -ti Task001_SepET
python3 deep_atlas_test.py -gpu 0 -ti Task001_SepET 
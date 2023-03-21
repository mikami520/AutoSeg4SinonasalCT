#!/bin/bash
source ~/proj_MONAI/bin/activate
cd ~/DeepAtlasV2/DeepAtlas/deepatlas/preprocess
#python3 registration_training.py -bp ~/HN_data/modified/data -ip images -sp labels -ti Task003_HN33_numres2_25lab
#python3 crop_flip_training.py -rs 240 240 144 -ti Task003_HN33_numres2_25lab
#python3 generate_info.py -ti Task003_test -kf 5
cd ~/DeepAtlasV2/DeepAtlas/deepatlas/scripts
CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_train.py --config ~/DeepAtlasV2/DeepAtlas/deepatlas_config/config_test.json --continue_training
CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_test.py --config ~/DeepAtlasV2/DeepAtlas/deepatlas_config/config_test.json
#CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_train.py --config ~/DeepAtlas/deepatlas_config/config_NC_2gt.json 
#CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_test.py --config ~/DeepAtlas/deepatlas_config/config_NC_2gt.json
#CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_train.py --config ~/DeepAtlas/deepatlas_config/config_NC_4gt.json 
#CUDA_VISIBLE_DEVICES=1 python3 deep_atlas_test.py --config ~/DeepAtlas/deepatlas_config/config_NC_4gt.json
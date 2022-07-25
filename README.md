# Deep Atlas
Deep Atlas: Joint Semi-supervised Learning of Image Registration and Segmentation

## Step 0: Fork This GitHub Repository 
```
git clone https://github.com/mikami520/DeepAtlas.git
```

## Step 1: Set Up Two Environments Using requirements.txt Files (virtual environment is recommended)
For Python 2
```
pip install -r requirements.txt
```
For Python 3
```
pip3 install -r requirements.txt
```

## Step 2: Preprocess Datasets
### Step 2.1: Co-align the data (make sure scan and segmentation are co-aligned)
Activate scripting environment
```
cd <path to repo>/deepatlas/preprocessing
```
Co-align the scans and segmentations (recommendation: Similarity registration)
```
python3 registration.py 
-bp <full path of base dir> 
-ip <relative path to nifti images dir> 
-sp <relative path to segmentations dir> 
```
If you want to make sure correspondence of the name and value of segmentations, you can add the following commands after above command (**Option for nrrd format**)
```
-sl LabelValue1 LabelName1 LabelValue2 LabelName2 LabelValue3 LabelName3 ...
```
For example, if I have two labels for maxillary sinus named ```L-MS``` and ```R-MS``` and I want ```L-MS``` matched to ```label 1``` and ```R-MS``` to ```label 2``` (**Pay attention to the order**)
```
python3 registration.py -bp /Users/mikamixiao/Desktop -ip images -sp labels -sl 1 L-MS 2 R-MS
```
Final output of registered images and segmentations will be saved in 
```
base_dir/imagesRS/ && base_dir/labelsRS/
```
### Step 2.2: Crop Normalize and Flip Data (if needed)
Crop normalize and flip data to extract region of interest (ROI). **Notice: the images and segmentations should be co-registered. We recommend to use the outputs of Step 2.1**
```
python3 crop_flip.py 
-bp <full path of base dir> 
-ip <relative path to nifti images dir> 
-sp <relative path to segmentations dir> 
-op <relative path to output dir> 
-rs <customized resized shape>
``` 
or **only crop and no need to flip**
```
python3 crop.py 
-bp <full path of base dir> 
-ip <relative path to nifti images dir> 
-sp <relative path to segmentations dir> 
-op <relative path to output dir> 
-rs <customized resized shape>
```
**Pay attention to the resized dimension which should not smaller than cropped dimension**\
Final output of ROI will be saved in
```
base_dir/output/images/ && base_dir/output/labels
```

## Step 3: Model Training
Activate scripting environment
```
cd <path to repo>/deepatlas/scripts
```
Start training
```
python3 deep_atlas_train.py
-bp <full path of base dir> 
-ip <relative path to nifti images dir> 
-sp <relative path to segmentations dir>
-ns <number of segmentations used for training>
-sd <spatial dimension of dataset>
-dr <dropout ratio of the network>
-gpu <id of gpu device to use>
-at <activation type and arguments>
-nm <feature normalization type and arguments>
-nr <number of residual units>
-lr <learning rate of registration network>
-ls <learning rate of segmentation network>
-lg <regularization loss weight>
-ba <anatomy loss weight>
-bs <supervised segmentation loss weight>
-me <maximum number of training epochs>
-vs <validation steps per epoch>
-ti <task id and name>
```
**For detailed information, use ```-h``` to see more instructions**
Before training, a folder named ```DeepAtlas_dataset``` is created under the repository directory. All training results are stored in this folder. A clear structure is shown below:
```
DeepAtlas_dataset/
    ├── Task001_ET
    |   └── results
    |       └── RegNet
    |           |── anatomy_loss_reg.txt
    |           |── anatomy_reg_losses.png
    |           |── reg_net_best.pth
    |           |── reg_net_training_losses.png
    |           |── regularization_loss.txt
    |           |── regularization_reg_losses.png
    |           |── similarity_loss_reg.txt
    |           |── similarity_reg_losses.png
    |       └── SegNet
    |           |── anatomy_loss_seg.txt
    |           |── anatomy_seg_losses.png
    |           |── seg_net_best.pth
    |           |── seg_net_training_losses.png
    |           |── supervised_loss_seg.txt
    |           |── supervised_seg_losses.png
    |       └── training_log.txt
    |   └── dataset.json
    ├── Task002_Nasal_Cavity
```
## Step 4: Run Inference
```
python3 deep_atlas_test.py
-gpu <id of gpu device to use>
-op <relative path of the prediction result directory>
-ti <task id and name>
```
The final prediction results will be saved in the ```DeepAtlas_dataset/Task_id_and_Name``` directory
For example,
```
DeepAtlas_dataset/
    ├── Task001_ET
    |   └── results
    |       └── RegNet
    |           |── anatomy_loss_reg.txt
    |           |── anatomy_reg_losses.png
    |           |── reg_net_best.pth
    |           |── reg_net_training_losses.png
    |           |── regularization_loss.txt
    |           |── regularization_reg_losses.png
    |           |── similarity_loss_reg.txt
    |           |── similarity_reg_losses.png
    |       └── SegNet
    |           |── anatomy_loss_seg.txt
    |           |── anatomy_seg_losses.png
    |           |── seg_net_best.pth
    |           |── seg_net_training_losses.png
    |           |── supervised_loss_seg.txt
    |           |── supervised_seg_losses.png
    |       └── training_log.txt
    |   └── prediction
    |       └── RegNet
    |           |── reg_img_losses.txt
    |           |── reg_seg_dsc.txt
    |           |── figures containing fixed, moving, warped scans deformation field and jacobian determinant
    |           |── warped scans and labels in nifti format
    |       └── SegNet
    |           |── seg_dsc.txt
    |           |── predicted labels in nifti format
    |   └── dataset.json
    ├── Task002_Nasal_Cavity
```


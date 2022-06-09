import generators
import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import os.path
import nibabel as nib
import sys
import json

sys.path.insert(0, '/home/ameen/DeepAtlas/utils')
sys.path.insert(0, '/home/ameen/DeepAtlas/loss_function')

from losses import (
    warp_func, warp_nearest_func, lncc_loss_func, dice_loss_func, dice_loss_func2
)
from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)

def load_dataset(data_list):
    transform_seg_available = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=True),
            monai.transforms.TransposeD(
                keys=['img', 'seg'], indices=(2, 1, 0)),
            monai.transforms.AddChannelD(keys=['img', 'seg']),
            monai.transforms.SpacingD(keys=['img', 'seg'], pixdim=(
                1., 1., 1.), mode=('trilinear', 'nearest')),
            monai.transforms.OrientationD(keys=['img', 'seg'], axcodes='RAS'),
            monai.transforms.ToTensorD(keys=['img', 'seg'])
        ]
    )
    itk.ProcessObject.SetGlobalWarningDisplay(False)
    dataset_seg_available_train = monai.data.CacheDataset(
        data=data_list,
        transform=transform_seg_available,
        cache_num=16,
        hash_as_key=True
    )
    return dataset_seg_available_train

def load_json(json_path):
    with open(json_path) as f:
        json_file = json.load(f)
    return json_file

def get_nii_info(data):
    headers = []
    affines = []
    ids = []
    for i in range(len(data)):
        item = data[i]
        id = os.path.basename(item['seg']).split('.')[0]
        seg = nib.load(item['seg'])
        headers.append(seg.header)
        affines.append(seg.affine)
        ids.append(id)
    return headers, affines, ids

def seg_inference(seg_net, device, model_path, json_path, output):
    json_file = load_json(json_path)
    raw_data = json_file['total_test']
    headers, affines, ids = get_nii_info(raw_data)
    seg_net.load_state_dict(torch.load(model_path))
    seg_net.to(device)
    dice_loss = dice_loss_func2()
    data = load_dataset(raw_data)
    k = 0
    for i in data:
        header = headers[k]
        affine = affines[k]
        id = ids[k]
        data_item = i
        test_input = data_item['img']
        test_gt = data_item['seg']
        seg_net.eval()
        with torch.no_grad():
            test_seg_predicted = seg_net(test_input.unsqueeze(0).cuda()).cpu()
            loss = dice_loss(test_seg_predicted, test_gt.unsqueeze(0)).item()
        
        print(f"Scan ID: {id}, dice loss: {loss}")
        prediction = torch.argmax(torch.softmax(test_seg_predicted, dim=1), dim=1, keepdim=True)[0, 0]
        k += 1
        pred_np = prediction.detach().cpu().numpy()
        #print(np.unique(pred_np))
        nii = nib.Nifti1Image(pred_np, affine=affine, header=header)
        #preview_image(prediction, normalize_by='slice')
        nii.to_filename(os.path.join(output, id + '.nii.gz'))

        del test_seg_predicted
    torch.cuda.empty_cache()
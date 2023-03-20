import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import nibabel as nib
import sys
import json
from pathlib import Path
mpl.rc('figure', max_open_warning = 0)
ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/utils'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/loss_function'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/preprocess'))

from process_data import (
    take_data_pairs, subdivide_list_of_data_pairs
)
from utils import (
    plot_2D_vector_field, jacobian_determinant, plot_2D_deformation
)
from losses import (
    warp_func, warp_nearest_func, lncc_loss_func, dice_loss_func2, dice_loss_func
)

def load_seg_dataset(data_list):
    transform_seg_available = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=True, allow_missing_keys=True),
            #monai.transforms.TransposeD(
                #keys=['img', 'seg'], indices=(2, 1, 0)),
            monai.transforms.AddChannelD(keys=['img', 'seg'], allow_missing_keys=True),
            monai.transforms.SpacingD(keys=['img', 'seg'], pixdim=(1., 1., 1.), mode=('trilinear', 'nearest'), allow_missing_keys=True),
            #monai.transforms.OrientationD(keys=['img', 'seg'], axcodes='RAS'),
            monai.transforms.ToTensorD(keys=['img', 'seg'], allow_missing_keys=True)
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


def load_reg_dataset(data_list):
    transform_pair = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(
                keys=['img1', 'seg1', 'img2', 'seg2'], image_only=True, allow_missing_keys=True),
            #monai.transforms.TransposeD(keys=['img1', 'seg1', 'img2', 'seg2'], indices=(2, 1, 0), allow_missing_keys=True),
            # if resize is not None else monai.transforms.Identity()
            monai.transforms.ToTensorD(
                keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
            monai.transforms.AddChannelD(
                keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
            monai.transforms.SpacingD(keys=['img1', 'seg1', 'img2', 'seg2'], pixdim=(1., 1., 1.), mode=(
                'trilinear', 'nearest', 'trilinear', 'nearest'), allow_missing_keys=True),
            #monai.transforms.OrientationD(
                #keys=['img1', 'seg1', 'img2', 'seg2'], axcodes='RAS', allow_missing_keys=True),
            monai.transforms.ConcatItemsD(
                keys=['img1', 'img2'], name='img12', dim=0),
            monai.transforms.DeleteItemsD(keys=['img1', 'img2']),
        ]
    )
    dataset_pairs_train_subdivided = {
        seg_availability: monai.data.CacheDataset(
            data=data,
            transform=transform_pair,
            cache_num=32,
            hash_as_key=True
        )
        for seg_availability, data in data_list.items()
    }

    return dataset_pairs_train_subdivided


def load_json(json_path):
    assert type(json_path) == str
    fjson = open(json_path, 'r')
    json_file = json.load(fjson)
    return json_file


def get_nii_info(data, reg=False):
    headers = []
    affines = []
    ids = []
    if not reg:
        for i in range(len(data)):
            item = data[i]
            if 'seg' in item.keys():
                id = os.path.basename(item['seg']).split('.')[0]
                seg = nib.load(item['seg'])
                num_labels = len(np.unique(seg.get_fdata()))
                headers.append(seg.header)
                affines.append(seg.affine)
                ids.append(id)
            else:
                id = os.path.basename(item['img']).split('.')[0]
                img = nib.load(item['img'])
                headers.append(img.header)
                affines.append(img.affine)
                ids.append(id)
    else:
        headers = {'00': [], '01': [], '10': [], '11': []}
        affines = {'00': [], '01': [], '10': [], '11': []}
        ids = {'00': [], '01': [], '10': [], '11': []}
        for i in range(len(data)):
            header = {}
            affine = {}
            id = {}
            item = data[i]
            keys = item.keys()
            if 'seg1' in keys and 'seg2' in keys:
                for key in keys:
                    idd = os.path.basename(item[key]).split('.')[0]
                    ele = nib.load(item[key])
                    header[key] = ele.header
                    affine[key] = ele.affine
                    id[key] = idd

                headers['11'].append(header)
                affines['11'].append(affine)
                ids['11'].append(id)
            elif 'seg1' in keys:
                for key in keys:
                    idd = os.path.basename(item[key]).split('.')[0]
                    ele = nib.load(item[key])
                    header[key] = ele.header
                    affine[key] = ele.affine
                    id[key] = idd

                headers['10'].append(header)
                affines['10'].append(affine)
                ids['10'].append(id)
            elif 'seg2' in keys:
                for key in keys:
                    idd = os.path.basename(item[key]).split('.')[0]
                    ele = nib.load(item[key])
                    header[key] = ele.header
                    affine[key] = ele.affine
                    id[key] = idd

                headers['01'].append(header)
                affines['01'].append(affine)
                ids['01'].append(id)
            else:
                for key in keys:
                    idd = os.path.basename(item[key]).split('.')[0]
                    ele = nib.load(item[key])
                    header[key] = ele.header
                    affine[key] = ele.affine
                    id[key] = idd
                
                headers['00'].append(header)
                affines['00'].append(affine)
                ids['00'].append(id)

    return headers, affines, ids


def seg_training_inference(seg_net, device, model_path, output_path, num_label, json_path=None, data=None):
    if json_path is not None:
        assert data is None
        json_file = load_json(json_path)
        raw_data = json_file['total_test']
    else:
        assert data is not None
        raw_data = data
    headers, affines, ids = get_nii_info(raw_data, reg=False)
    seg_net.load_state_dict(torch.load(model_path, map_location=device))
    seg_net.to(device)
    print(next(seg_net.parameters()).device)
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction='none')
    data_seg = load_seg_dataset(raw_data)
    k = 0
    eval_losses = []
    eval_los = []
    for i in data_seg:
        has_seg = False
        header1 = headers[k]
        affine1 = affines[k]
        id = ids[k]
        data_item = i
        test_input = data_item['img']
        if 'seg' in data_item.keys():
            test_gt = data_item['seg']
            has_seg = True
        seg_net.eval()
        with torch.no_grad():
            test_seg_predicted = seg_net(test_input.unsqueeze(0).to(device)).cpu()

        prediction = torch.argmax(torch.softmax(
            test_seg_predicted, dim=1), dim=1, keepdim=True)[0, 0]
        prediction1 = torch.argmax(torch.softmax(
            test_seg_predicted, dim=1), dim=1, keepdim=True)
        
        onehot_pred = monai.networks.one_hot(prediction1, num_label)
        if has_seg:
            onehot_gt = monai.networks.one_hot(test_gt.unsqueeze(0), num_label)
            dsc = dice_metric(onehot_pred, onehot_gt).numpy()
            eval_los.append(dsc)
            eval_loss = f"Scan ID: {id}, dice score: {dsc}"
            eval_losses.append(eval_loss)
        
        pred_np = prediction.detach().cpu().numpy()
        print(f'{id}: {np.unique(pred_np)}')

        pred_np = pred_np.astype('int16')
        nii = nib.Nifti1Image(pred_np, affine=affine1, header=header1)
        nii.header.get_xyzt_units()
        nib.save(nii, (os.path.join(output_path, id + '.nii.gz')))
        k += 1

        del test_seg_predicted

    average = np.mean(eval_los, 0)
   
    with open(os.path.join(output_path, 'seg_dsc.txt'), 'w') as f:
        for s in eval_losses:
            f.write(s + '\n')
        f.write('\n\nAverage Dice Score: ' + str(average))
    torch.cuda.empty_cache()


def reg_training_inference(reg_net, device, model_path, output_path, num_label, json_path=None, data=None):
    if json_path is not None:
        assert data is None
        json_file = load_json(json_path)
        raw_data = json_file['total_test']
    else:
        assert data is not None
        raw_data = data
    # Run this cell to try out reg net on a random validation pair
    reg_net.load_state_dict(torch.load(model_path, map_location=device))
    reg_net.to(device)
    reg_net.eval()
    data_list = take_data_pairs(raw_data)
    headers, affines, ids = get_nii_info(data_list, reg=True)
    subvided_data_list = subdivide_list_of_data_pairs(data_list)
    subvided_dataset = load_reg_dataset(subvided_data_list)
    warp = warp_func()
    warp_nearest = warp_nearest_func()
    lncc_loss = lncc_loss_func()
    k = 0
    if len(subvided_data_list['01']) != 0:
        dataset01 = subvided_dataset['01']
        #test_len = int(len(dataset01) / 4)
        for j in range(len(dataset01)):
            data_item = dataset01[j]
            img12 = data_item['img12'].unsqueeze(0).to(device)
            moving_raw_seg = data_item['seg2'].unsqueeze(0).to(device)
            moving_seg = monai.networks.one_hot(moving_raw_seg, num_label)
            id = ids['01'][k]
            affine = affines['01'][k]
            header = headers['01'][k]
            with torch.no_grad():
                reg_net_example_output = reg_net(img12)

            example_warped_image = warp(
                img12[:, [1], :, :, :],  # moving image
                reg_net_example_output  # warping
            )
            example_warped_seg = warp_nearest(
                moving_seg,
                reg_net_example_output
            )
            moving_img = img12[0, 1, :, :, :]
            target_img = img12[0, 0, :, :, :]
            id_target_img = id['img1']
            id_moving_img = id['img2']
            head_target_img = header['img1']
            head_target_seg = header['img1']
            aff_target_img = affine['img1']
            aff_target_seg = affine['img1']
            prediction = torch.argmax(torch.softmax(
                example_warped_seg, dim=1), dim=1, keepdim=True)[0, 0]
            prediction1 = torch.argmax(torch.softmax(
                example_warped_seg, dim=1), dim=1, keepdim=True)
            warped_img_np = example_warped_image[0, 0].detach().cpu().numpy()
            #warped_img_np = np.transpose(warped_img_np, (2, 1, 0))
            warped_seg_np = prediction.detach().cpu().numpy()
            #warped_seg_np = np.transpose(warped_seg_np, (2, 1, 0))
            nii_seg = nib.Nifti1Image(
                warped_seg_np, affine=aff_target_seg, header=head_target_seg)
            nii = nib.Nifti1Image(
                warped_img_np, affine=aff_target_img, header=head_target_img)
            nii.to_filename(os.path.join(
                output_path, id_moving_img + '_to_' + id_target_img + '.nii.gz'))
            nii_seg.to_filename(os.path.join(
                output_path, id_moving_img + '_to_' + id_target_img + '_seg.nii.gz'))
            grid_spacing = 5
            det = jacobian_determinant(reg_net_example_output.cpu().detach()[0])
            visualize(target_img.cpu(),
                        id_target_img,
                        moving_img.cpu(),
                        id_moving_img,
                        example_warped_image[0, 0].cpu(),
                        reg_net_example_output.cpu().detach()[0],
                        det,
                        grid_spacing,
                        normalize_by='slice',
                        cmap='gray',
                        threshold=None,
                        linewidth=1,
                        color='darkblue',
                        downsampling=None,
                        threshold_det=0,
                        output=output_path
                        )
            k += 1
            del reg_net_example_output, img12, example_warped_image, example_warped_seg
    
    if len(subvided_data_list['11']) != 0:
        dataset11 = subvided_dataset['11']
        k = 0
        eval_losses_img = []
        eval_losses_seg = []
        eval_los = []
        #test_len = int(len(dataset11) / 4)
        for i in range(len(dataset11)):
            data_item = dataset11[i]
            img12 = data_item['img12'].unsqueeze(0).to(device)
            gt_raw_seg = data_item['seg1'].unsqueeze(0).to(device)
            moving_raw_seg = data_item['seg2'].unsqueeze(0).to(device)
            moving_seg = monai.networks.one_hot(moving_raw_seg, num_label)
            gt_seg = monai.networks.one_hot(gt_raw_seg, num_label)
            id = ids['11'][k]
            affine = affines['11'][k]
            header = headers['11'][k]
            with torch.no_grad():
                reg_net_example_output = reg_net(img12)

            example_warped_image = warp(
                img12[:, [1], :, :, :],  # moving image
                reg_net_example_output  # warping
            )
            example_warped_seg = warp_nearest(
                moving_seg,
                reg_net_example_output
            )
            moving_img = img12[0, 1, :, :, :]
            target_img = img12[0, 0, :, :, :]
            id_target_img = id['img1']
            id_moving_img = id['img2']
            head_target_img = header['img1']
            head_target_seg = header['seg1']
            aff_target_img = affine['img1']
            aff_target_seg = affine['seg1']
            dice_metric = monai.metrics.DiceMetric(include_background=False, reduction='none')
            prediction = torch.argmax(torch.softmax(
                example_warped_seg, dim=1), dim=1, keepdim=True)[0, 0]
            prediction1 = torch.argmax(torch.softmax(
                example_warped_seg, dim=1), dim=1, keepdim=True)
            onehot_pred = monai.networks.one_hot(prediction1, num_label)
            dsc = dice_metric(onehot_pred, gt_seg).detach().cpu().numpy()
            eval_los.append(dsc)
            eval_loss_seg = f"Scan {id_moving_img} to {id_target_img}, dice score: {dsc}"
            eval_losses_seg.append(eval_loss_seg)
            warped_img_np = example_warped_image[0, 0].detach().cpu().numpy()
            #warped_img_np = np.transpose(warped_img_np, (2, 1, 0))
            warped_seg_np = prediction.detach().cpu().numpy()
            #warped_seg_np = np.transpose(warped_seg_np, (2, 1, 0))
            nii_seg = nib.Nifti1Image(
                warped_seg_np, affine=aff_target_seg, header=head_target_seg)
            nii = nib.Nifti1Image(
                warped_img_np, affine=aff_target_img, header=head_target_img)
            nii.to_filename(os.path.join(
                output_path, id_moving_img + '_to_' + id_target_img + '.nii.gz'))
            nii_seg.to_filename(os.path.join(
                output_path, id_moving_img + '_to_' + id_target_img + '_seg.nii.gz'))
            grid_spacing = 5
            det = jacobian_determinant(reg_net_example_output.cpu().detach()[0])
            visualize(target_img.cpu(),
                    id_target_img,
                    moving_img.cpu(),
                    id_moving_img,
                    example_warped_image[0, 0].cpu(),
                    reg_net_example_output.cpu().detach()[0],
                    det,
                    grid_spacing,
                    normalize_by='slice',
                    cmap='gray',
                    threshold=None,
                    linewidth=1,
                    color='darkblue',
                    downsampling=None,
                    threshold_det=0,
                    output=output_path
                    )
            loss = lncc_loss(example_warped_image, img12[:, [0], :, :, :]).item()
            eval_loss_img = f"Warped {id_moving_img} to {id_target_img}, similarity loss: {loss}, number of folds: {(det<=0).sum()}"
            eval_losses_img.append(eval_loss_img)
            k += 1
            del reg_net_example_output, img12, example_warped_image, example_warped_seg

        with open(os.path.join(output_path, "reg_img_losses.txt"), 'w') as f:
            for s in eval_losses_img:
                f.write(s + '\n')

        average = np.mean(eval_los, 0)
        with open(os.path.join(output_path, "reg_seg_dsc.txt"), 'w') as f:
            for s in eval_losses_seg:
                f.write(s + '\n')
            f.write('\n\nAverage Dice Score: ' + str(average))
    torch.cuda.empty_cache()

def visualize(target,
              target_id,
              moving,
              moving_id,
              warped,
              vector_field,
              det,
              grid_spacing,
              normalize_by='volume',
              cmap=None,
              threshold=None,
              linewidth=1,
              color='red',
              downsampling=None,
              threshold_det=None,
              output=None
              ):
    if normalize_by == "slice":
        vmin = None
        vmax_moving = None
        vmax_target = None
        vmax_warped = None
        vmax_det = None
    elif normalize_by == "volume":
        vmin = 0
        vmax_moving = moving.max().item()
        vmax_target = target.max().item()
        vmax_warped = warped.max().item()
        vmax_det = det.max().item()
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    plt.figure(figsize=(24, 24))
    x, y, z = np.array(moving.shape)//2
    moving_imgs = (moving[x, :, :], moving[:, y, :], moving[:, :, z])
    target_imgs = (target[x, :, :], target[:, y, :], target[:, :, z])
    warped_imgs = (warped[x, :, :], warped[:, y, :], warped[:, :, z])
    det_imgs = (det[x, :, :], det[:, y, :], det[:, :, z])
    for i in range(3):
        im = moving_imgs[i]
        plt.subplot(6, 3, i+1)
        plt.axis('off')
        plt.title(f'moving image: {moving_id}')
        plt.imshow(im, origin='lower', vmin=vmin, vmax=vmax_moving, cmap=cmap)
        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            plt.imshow(red, origin='lower')

    for k in range(3):
        j = k + 4
        im = target_imgs[k]
        plt.subplot(6, 3, j)
        plt.axis('off')
        plt.title(f'target image: {target_id}')
        plt.imshow(im, origin='lower', vmin=vmin, vmax=vmax_target, cmap=cmap)
        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            plt.imshow(red, origin='lower')

    for m in range(3):
        j = 7 + m
        im = warped_imgs[m]
        plt.subplot(6, 3, j)
        plt.axis('off')
        plt.title(f'warped image: {moving_id} to {target_id}')
        plt.imshow(im, origin='lower', vmin=vmin, vmax=vmax_warped, cmap=cmap)
        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            plt.imshow(red, origin='lower')

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.subplot(6, 3, 10)
    plt.axis('off')
    plt.title(f'deformation vector field: {moving_id} to {target_id}')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(6, 3, 11)
    plt.axis('off')
    plt.title(f'deformation vector field: {moving_id} to {target_id}')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(6, 3, 12)
    plt.axis('off')
    plt.title(f'deformation vector field: {moving_id} to {target_id}')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)

    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.subplot(6, 3, 13)
    plt.axis('off')
    plt.title(f'deformation vector field on grid: {moving_id} to {target_id}')
    plot_2D_deformation(
        vector_field[[1, 2], x, :, :], grid_spacing, linewidth=linewidth, color=color)
    plt.subplot(6, 3, 14)
    plt.axis('off')
    plt.title(f'deformation vector field on grid: {moving_id} to {target_id}')
    plot_2D_deformation(
        vector_field[[0, 2], :, y, :], grid_spacing, linewidth=linewidth, color=color)
    plt.subplot(6, 3, 15)
    plt.axis('off')
    plt.title(f'deformation vector field on grid: {moving_id} to {target_id}')
    plot_2D_deformation(
        vector_field[[0, 1], :, :, z], grid_spacing, linewidth=linewidth, color=color)

    for n in range(3):
        o = n + 16
        im = det_imgs[n]
        plt.subplot(6, 3, o)
        plt.axis('off')
        plt.title(f'jacobian determinant: {moving_id} to {target_id}')
        plt.imshow(im, origin='lower', vmin=vmin, vmax=vmax_det, cmap=None)
        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold_det is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold_det] = [1, 0, 0, 1]
            plt.imshow(red, origin='lower')

    plt.savefig(os.path.join(
        output, f'reg_net_infer_{moving_id}_to_{target_id}.png'))

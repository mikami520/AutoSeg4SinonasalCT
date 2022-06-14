import monai
import torch
import numpy as np
import matplotlib.pyplot as plt

def warp_func():
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
    return warp

def warp_nearest_func():
    warp_nearest = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")
    return warp_nearest

def lncc_loss_func():
    lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
        spatial_dims=3,
        kernel_size=3,
        kernel_type='rectangular',
        reduction="mean",
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
    return lncc_loss

def similarity_loss(displacement_field, image_pair):
    warp = warp_func()
    lncc_loss = lncc_loss_func()
    """ Accepts a batch of displacement fields, shape (B,3,H,W,D),
        and a batch of image pairs, shape (B,2,H,W,D). """
    warped_img2 = warp(image_pair[:, [1], :, :, :], displacement_field)
    return lncc_loss(
        warped_img2,  # prediction
        image_pair[:, [0], :, :, :]  # target
    )

def regularization_loss_func():
    return monai.losses.BendingEnergyLoss()

def dice_loss_func():
    dice_loss = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=False,
        softmax=False,
        reduction="mean"
    )
    return dice_loss

def dice_loss_func2():
    dice_loss = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        reduction="mean"
    )
    return dice_loss

def anatomy_loss(displacement_field, image_pair, seg_net, gt_seg1=None, gt_seg2=None, num_segmentation_classes=None):
    """
    Accepts a batch of displacement fields, shape (B,3,H,W,D),
    and a batch of image pairs, shape (B,2,H,W,D).
    seg_net is the model used to segment an image,
      mapping (B,1,H,W,D) to (B,C,H,W,D) where C is the number of segmentation classes.
    gt_seg1 and gt_seg2 are ground truth segmentations for the images in image_pair, if ground truth is available;
      if unavailable then they can be None.
      gt_seg1 and gt_seg2 are expected to be in the form of class labels, with shape (B,1,H,W,D).
    """
    if gt_seg1 is not None:
        # ground truth seg of target image
        seg1 = monai.networks.one_hot(
            gt_seg1, num_segmentation_classes
        )
    else:
        # seg_net on target image, "noisy ground truth"
        seg1 = seg_net(image_pair[:, [0], :, :, :]).softmax(dim=1)

    if gt_seg2 is not None:
        # ground truth seg of moving image
        seg2 = monai.networks.one_hot(
            gt_seg2, num_segmentation_classes
        )
    else:
        # seg_net on moving image, "noisy ground truth"
        seg2 = seg_net(image_pair[:, [1], :, :, :]).softmax(dim=1)

    # seg1 and seg2 are now in the form of class probabilities at each voxel
    # The trilinear interpolation of the function `warp` is then safe to use;
    # it will preserve the probabilistic interpretation of seg2.
    dice_loss = dice_loss_func()
    warp = warp_func()
    return dice_loss(
        warp(seg2, displacement_field),  # warp of moving image segmentation
        seg1  # target image segmentation
    )

def reg_losses(batch, device, reg_net, seg_net, num_segmentation_classes):
    img12 = batch['img12'].to(device)
    displacement_field12 = reg_net(img12)

    loss_sim = similarity_loss(displacement_field12, img12)
    regularization_loss = regularization_loss_func()
    loss_reg = regularization_loss(displacement_field12)

    gt_seg1 = batch['seg1'].to(device) if 'seg1' in batch.keys() else None
    gt_seg2 = batch['seg2'].to(device) if 'seg2' in batch.keys() else None
    loss_ana = anatomy_loss(displacement_field12, img12, seg_net, gt_seg1, gt_seg2, num_segmentation_classes)

    return loss_sim, loss_reg, loss_ana




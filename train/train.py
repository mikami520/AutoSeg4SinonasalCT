import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os.path
import argparse
import sys

sys.path.insert(0, '/home/ameen/DeepAtlas/utils')
sys.path.insert(0, '/home/ameen/DeepAtlas/loss_function')

from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)

from losses import (
    warp_func, warp_nearest_func, lncc_loss_func, dice_loss_func, reg_losses, dice_loss_func2
)
import generators

def swap_training(network_to_train, network_to_not_train):
    """
        Switch out of training one network and into training another
    """

    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()

def train_network(dataloader_train_reg, 
         dataloader_valid_reg, 
         dataloader_train_seg, 
         dataloader_valid_seg, 
         device, 
         seg_net, 
         reg_net, 
         image_scale, 
         num_segmentation_classes, 
         lr_reg, 
         lr_seg, 
         lam_a, 
         lam_sp, 
         max_epoch, 
         val_step
        ):
    # Training cell
    # (if already done then you may skip this and uncomment the loading checkpoint cell below)
    seg_availabilities = ['00', '01', '10', '11']
    batch_generator_train_reg = generators.create_batch_generator(dataloader_train_reg)
    batch_generator_valid_reg = generators.create_batch_generator(dataloader_valid_reg)
    seg_train_sampling_weights = [0] + [len(dataloader_train_reg[s]) for s in seg_availabilities[1:]]
    print('---'*10)
    print(f"""When training seg_net alone, segmentation availabilities {seg_availabilities}
    will be sampled with respective weights {seg_train_sampling_weights}""")
    batch_generator_train_seg = generators.create_batch_generator(dataloader_train_reg, seg_train_sampling_weights)
    seg_net.to(device)
    reg_net.to(device)

    learning_rate_reg = lr_reg
    optimizer_reg = torch.optim.Adam(reg_net.parameters(), learning_rate_reg)

    learning_rate_seg = lr_seg
    optimizer_seg = torch.optim.Adam(seg_net.parameters(), learning_rate_seg)

    lambda_a = lam_a  # anatomy loss weight
    lambda_sp = lam_sp  # supervised segmentation loss weight

    # regularization loss weight
    # This often requires some careful tuning. Here we suggest a value, which unfortunately needs to
    # depend on image scale. This is because the bending energy loss is not scale-invariant.
    # 7.5 worked well with the above hyperparameters for images of size 128x128x128.
    lambda_r = 7.5 * (image_scale / 128)**2

    max_epochs = max_epoch
    reg_phase_training_batches_per_epoch = 10
    seg_phase_training_batches_per_epoch = 5  # Fewer batches needed, because seg_net converges more quickly
    reg_phase_num_validation_batches_to_use = 10
    val_interval = val_step

    training_losses_reg = []
    validation_losses_reg = []
    training_losses_seg = []
    validation_losses_seg = []

    best_seg_validation_loss = float('inf')
    best_reg_validation_loss = float('inf')

    for epoch_number in range(max_epochs):

        print(f"Epoch {epoch_number+1}/{max_epochs}:")

        # ------------------------------------------------
        #         reg_net training, with seg_net frozen
        # ------------------------------------------------

        # Keep computational graph in memory for reg_net, but not for seg_net, and do reg_net.train()
        swap_training(reg_net, seg_net)

        losses = []
        for batch in batch_generator_train_reg(reg_phase_training_batches_per_epoch):
            optimizer_reg.zero_grad()
            loss_sim, loss_reg, loss_ana = reg_losses(batch, device, reg_net, seg_net, num_segmentation_classes)
            loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
            loss.backward()
            optimizer_reg.step()
            losses.append(loss.item())

        training_loss = np.mean(losses)
        print(f"\treg training loss: {training_loss}")
        training_losses_reg.append([epoch_number, training_loss])

        if epoch_number % val_interval == 0:
            reg_net.eval()
            losses = []
            with torch.no_grad():
                for batch in batch_generator_valid_reg(reg_phase_num_validation_batches_to_use):
                    loss_sim, loss_reg, loss_ana = reg_losses(batch, device, reg_net, seg_net, num_segmentation_classes)
                    loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
                    losses.append(loss.item())

            validation_loss = np.mean(losses)
            print(f"\treg validation loss: {validation_loss}")
            validation_losses_reg.append([epoch_number, validation_loss])

            if validation_loss < best_reg_validation_loss:
                best_reg_validation_loss = validation_loss
                torch.save(reg_net.state_dict(), 'reg_net_best.pth')

        # Free up memory
        del loss, loss_sim, loss_reg, loss_ana
        torch.cuda.empty_cache()

        # ------------------------------------------------
        #         seg_net training, with reg_net frozen
        # ------------------------------------------------

        # Keep computational graph in memory for seg_net, but not for reg_net, and do seg_net.train()
        swap_training(seg_net, reg_net)

        losses = []
        dice_loss = dice_loss_func()
        warp_nearest = warp_nearest_func()
        dice_loss2 = dice_loss_func2()
        for batch in batch_generator_train_seg(seg_phase_training_batches_per_epoch):
            optimizer_seg.zero_grad()

            img12 = batch['img12'].to(device)

            displacement_fields = reg_net(img12)
            seg1_predicted = seg_net(img12[:, [0], :, :, :]).softmax(dim=1)
            seg2_predicted = seg_net(img12[:, [1], :, :, :]).softmax(dim=1)

            # Below we compute the following:
            # loss_supervised: supervised segmentation loss; compares ground truth seg with predicted seg
            # loss_anatomy: anatomy loss; compares warped seg of moving image to seg of target image
            # loss_metric: a single supervised seg loss, as a metric to track the progress of training

            if 'seg1' in batch.keys() and 'seg2' in batch.keys():
                seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
                seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
                loss_metric = dice_loss(seg2_predicted, seg2)
                loss_supervised = loss_metric
                # The above supervised loss looks a bit different from the one in the paper
                # in that it includes predictions for both images in the current image pair;
                # we might as well do this, since we have gone to the trouble of loading
                # both segmentations into memory.

            elif 'seg1' in batch.keys():  # seg1 available, but no seg2
                seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
                loss_metric = dice_loss(seg1_predicted, seg1)
                loss_supervised = loss_metric
                seg2 = seg2_predicted  # Use this in anatomy loss

            else:  # seg2 available, but no seg1
                assert('seg2' in batch.keys())
                seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
                loss_metric = dice_loss(seg2_predicted, seg2)
                loss_supervised = loss_metric
                seg1 = seg1_predicted  # Use this in anatomy loss

            # seg1 and seg2 should now be in the form of one-hot class probabilities

            loss_anatomy = dice_loss(warp_nearest(seg2, displacement_fields), seg1)\
                if 'seg1' in batch.keys() or 'seg2' in batch.keys()\
                else 0.  # It wouldn't really be 0, but it would not contribute to training seg_net

            # (If you want to refactor this code for *joint* training of reg_net and seg_net,
            #  then use the definition of anatomy loss given in the function anatomy_loss above,
            #  where differentiable warping is used and reg net can be trained with it.)

            loss = lambda_a * loss_anatomy + lambda_sp * loss_supervised
            loss.backward()
            optimizer_seg.step()

            losses.append(loss_metric.item())

        training_loss = np.mean(losses)
        print(f"\tseg training loss: {training_loss}")
        training_losses_seg.append([epoch_number, training_loss])

        if epoch_number % val_interval == 0:
            # The following validation loop would not do anything in the case
            # where there is just one segmentation available,
            # because data_seg_available_valid would be empty.
            seg_net.eval()
            losses = []
            with torch.no_grad():
                for batch in dataloader_valid_seg:
                    imgs = batch['img'].to(device)
                    true_segs = batch['seg'].to(device)
                    predicted_segs = seg_net(imgs)
                    loss = dice_loss2(predicted_segs, true_segs)
                    losses.append(loss.item())

            validation_loss = np.mean(losses)
            print(f"\tseg validation loss: {validation_loss}")
            validation_losses_seg.append([epoch_number, validation_loss])

            if validation_loss < best_seg_validation_loss:
                best_seg_validation_loss = validation_loss
                torch.save(seg_net.state_dict(), 'seg_net_best.pth')

        # Free up memory
        del loss, seg1, seg2, displacement_fields, img12, loss_supervised, loss_anatomy, loss_metric,\
            seg1_predicted, seg2_predicted
        torch.cuda.empty_cache()

    
    print(f"\n\nBest reg_net validation loss: {best_reg_validation_loss}")
    print(f"Best seg_net validation loss: {best_seg_validation_loss}")
    plot(training_losses_reg, validation_losses_reg, training_losses_seg, validation_losses_seg)

def plot(
    training_losses_reg,
    validation_losses_reg,
    training_losses_seg,
    validation_losses_seg
):
    # Plot the training and validation losses
    plot_against_epoch_numbers(training_losses_reg, label="training")
    plot_against_epoch_numbers(validation_losses_reg, label="validation")
    plt.legend()
    plt.ylabel('loss')
    plt.title('Alternating training: registration loss')
    plt.savefig('reg_net_losses.png')
    plt.show()

    plot_against_epoch_numbers(training_losses_seg, label="training")
    plt.ylabel('training loss')
    plt.title('Alternating training: segmentation loss (training)')
    plt.savefig('seg_net_training_losses.png')
    plt.show()

    plot_against_epoch_numbers(validation_losses_seg, label="validation", color='orange')
    plt.ylabel('validation loss')
    plt.title('Alternating training: segmentation loss (validation)')
    plt.savefig('seg_net_validation_losses.png')
    plt.show()
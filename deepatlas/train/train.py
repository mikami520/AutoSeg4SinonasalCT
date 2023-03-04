import generators
import monai
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(os.getcwd()).parent.parent.absolute())
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/utils'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'deepatlas/loss_function'))
from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)
from losses import (
    warp_func, warp_nearest_func, lncc_loss_func, dice_loss_func, reg_losses, dice_loss_func2
)

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
                  num_segmentation_classes,
                  lr_reg,
                  lr_seg,
                  lam_a,
                  lam_sp,
                  lam_re,
                  max_epoch,
                  val_step,
                  result_seg_path,
                  result_reg_path,
                  logger
                  ):
    # Training cell
    ROOT_DIR = str(Path(result_reg_path).parent.absolute())
    seg_availabilities = ['00', '01', '10', '11']
    batch_generator_train_reg = generators.create_batch_generator(
        dataloader_train_reg)
    batch_generator_valid_reg = generators.create_batch_generator(
        dataloader_valid_reg)
    seg_train_sampling_weights = [
        0] + [len(dataloader_train_reg[s]) for s in seg_availabilities[1:]]
    print('---'*10)
    print(f"""When training seg_net alone, segmentation availabilities {seg_availabilities}
    will be sampled with respective weights {seg_train_sampling_weights}""")
    batch_generator_train_seg = generators.create_batch_generator(
        dataloader_train_reg, seg_train_sampling_weights)
    seg_net.to(device)
    reg_net.to(device)

    learning_rate_reg = lr_reg
    optimizer_reg = torch.optim.Adam(reg_net.parameters(), learning_rate_reg)
    scheduler_reg = torch.optim.lr_scheduler.StepLR(optimizer_reg, step_size=70, gamma=0.2, verbose=True)
    learning_rate_seg = lr_seg
    optimizer_seg = torch.optim.Adam(seg_net.parameters(), learning_rate_seg)
    scheduler_seg = torch.optim.lr_scheduler.StepLR(optimizer_seg, step_size=50, gamma=0.2, verbose=True)
    lambda_a = lam_a  # anatomy loss weight
    lambda_sp = lam_sp  # supervised segmentation loss weight

    # regularization loss weight
    # monai has provided normalized bending energy loss
    # no need to modify the weight according to the image size
    lambda_r = lam_re

    max_epochs = max_epoch
    reg_phase_training_batches_per_epoch = 10
    # Fewer batches needed, because seg_net converges more quickly
    seg_phase_training_batches_per_epoch = 5
    reg_phase_num_validation_batches_to_use = 10
    val_interval = val_step

    training_losses_reg = []
    validation_losses_reg = []
    training_losses_seg = []
    validation_losses_seg = []
    regularization_loss_reg = []
    anatomy_loss_reg = []
    similarity_loss_reg = []
    supervised_loss_seg = []
    anatomy_loss_seg = []
    best_seg_validation_loss = float('inf')
    best_reg_validation_loss = float('inf')
    logger.info('Start Training')

    for epoch_number in range(max_epochs):

        logger.info(f"Epoch {epoch_number+1}/{max_epochs}:")
            # ------------------------------------------------
            #         reg_net training, with seg_net frozen
            # ------------------------------------------------

            # Keep computational graph in memory for reg_net, but not for seg_net, and do reg_net.train()
        swap_training(reg_net, seg_net)

        losses = []
        regularization_loss = []
        similarity_loss = []
        anatomy_loss = []
        for batch in batch_generator_train_reg(reg_phase_training_batches_per_epoch):
            optimizer_reg.zero_grad()
            loss_sim, loss_reg, loss_ana, df = reg_losses(
                batch, device, reg_net, seg_net, num_segmentation_classes)
            loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
            loss.backward()
            optimizer_reg.step()
            losses.append(loss.item())
            regularization_loss.append(loss_reg.item())
            similarity_loss.append(loss_sim.item())
            anatomy_loss.append(loss_ana.item())
        
        #preview_3D_vector_field(df.cpu().detach()[0], ep=epoch_number, path=result_reg_path)

        training_loss = np.mean(losses)
        regularization_loss_reg.append(
            [epoch_number, np.mean(regularization_loss)])
        similarity_loss_reg.append([epoch_number, np.mean(similarity_loss)])
        anatomy_loss_reg.append([epoch_number, np.mean(anatomy_loss)])
        logger.info(f"\treg training loss: {training_loss}")
        training_losses_reg.append([epoch_number, training_loss])

        if epoch_number % val_interval == 0:
            reg_net.eval()
            losses = []
            with torch.no_grad():
                for batch in batch_generator_valid_reg(reg_phase_num_validation_batches_to_use):
                    loss_sim, loss_reg, loss_ana, dv = reg_losses(
                        batch, device, reg_net, seg_net, num_segmentation_classes)
                    loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
                    losses.append(loss.item())
            
            validation_loss = np.mean(losses)
            logger.info(f"\treg validation loss: {validation_loss}")
            validation_losses_reg.append([epoch_number, validation_loss])

            if validation_loss < best_reg_validation_loss:
                best_reg_validation_loss = validation_loss
                torch.save(reg_net.state_dict(), os.path.join(
                    result_reg_path, 'reg_net_best.pth'))
        
        # scheduler_reg.step()
        # Free up memory
        del loss, loss_sim, loss_reg, loss_ana
        torch.cuda.empty_cache()

        # ------------------------------------------------
        #         seg_net training, with reg_net frozen
        # ------------------------------------------------

        # Keep computational graph in memory for seg_net, but not for reg_net, and do seg_net.train()
        swap_training(seg_net, reg_net)
        losses = []
        supervised_loss = []
        anatomy_loss = []
        dice_loss = dice_loss_func()
        warp = warp_func()
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
                seg1 = monai.networks.one_hot(
                    batch['seg1'].to(device), num_segmentation_classes)
                seg2 = monai.networks.one_hot(
                    batch['seg2'].to(device), num_segmentation_classes)
                loss_metric = dice_loss(seg2_predicted, seg2)
                loss_supervised = loss_metric + dice_loss(seg1_predicted, seg1)
                # The above supervised loss looks a bit different from the one in the paper
                # in that it includes predictions for both images in the current image pair;
                # we might as well do this, since we have gone to the trouble of loading
                # both segmentations into memory.

            elif 'seg1' in batch.keys():  # seg1 available, but no seg2
                seg1 = monai.networks.one_hot(
                    batch['seg1'].to(device), num_segmentation_classes)
                loss_metric = dice_loss(seg1_predicted, seg1)
                loss_supervised = loss_metric
                seg2 = seg2_predicted  # Use this in anatomy loss

            else:  # seg2 available, but no seg1
                assert('seg2' in batch.keys())
                seg2 = monai.networks.one_hot(
                    batch['seg2'].to(device), num_segmentation_classes)
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
            supervised_loss.append(loss_supervised.item())
            anatomy_loss.append(loss_anatomy.item())

        training_loss = np.mean(losses)
        supervised_loss_seg.append([epoch_number, np.mean(supervised_loss)])
        anatomy_loss_seg.append([epoch_number, np.mean(anatomy_loss)])
        logger.info(f"\tseg training loss: {training_loss}")
        training_losses_seg.append([epoch_number, training_loss])

        if len(dataloader_valid_seg) == 0:
            torch.save(seg_net.state_dict(), os.path.join(
                    result_seg_path, 'seg_net_best.pth'))
        else:
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
                logger.info(f"\tseg validation loss: {validation_loss}")
                validation_losses_seg.append([epoch_number, validation_loss])

                if validation_loss < best_seg_validation_loss:
                    best_seg_validation_loss = validation_loss
                    torch.save(seg_net.state_dict(), os.path.join(
                        result_seg_path, 'seg_net_best.pth'))
            
        # scheduler_seg.step()
        # Free up memory
        del loss, seg1, seg2, displacement_fields, img12, loss_supervised, loss_anatomy, loss_metric,\
            seg1_predicted, seg2_predicted
        torch.cuda.empty_cache()

    logger.info(f"Best reg_net validation loss: {best_reg_validation_loss}")
    if len(validation_losses_seg) == 0:
        logger.info('Only one label is used for training, no need to do validation. Replace best validation loss with training loss !!!')
        logger.info(f'Best seg_net validation loss: {training_loss}')
    else:
        logger.info(f"Best seg_net validation loss: {best_seg_validation_loss}")
    
    plot_fig(training_losses_reg,
             validation_losses_reg,
             training_losses_seg,
             validation_losses_seg,
             result_seg_path,
             result_reg_path,
             regularization_loss=regularization_loss_reg,
             anatomy_loss_reg=anatomy_loss_reg,
             anatomy_loss_seg=anatomy_loss_seg,
             similarity_loss=similarity_loss_reg,
             supervised_loss=supervised_loss_seg
             )
    regularization_loss_reg = np.array(regularization_loss_reg)
    anatomy_loss_reg = np.array(anatomy_loss_reg)
    anatomy_loss_seg = np.array(anatomy_loss_seg)
    similarity_loss_reg = np.array(similarity_loss_reg)
    supervised_loss_seg = np.array(supervised_loss_seg)
    np.savetxt(os.path.join(result_reg_path, 'regularization_loss.txt'),
               regularization_loss_reg)
    np.savetxt(os.path.join(result_reg_path, 'anatomy_loss_reg.txt'),
               anatomy_loss_reg)
    np.savetxt(os.path.join(result_seg_path, 'anatomy_loss_seg.txt'),
               anatomy_loss_seg)
    np.savetxt(os.path.join(result_reg_path, 'similarity_loss_reg.txt'),
               similarity_loss_reg)
    np.savetxt(os.path.join(result_seg_path, 'supervised_loss_seg.txt'),
               supervised_loss_seg)


def plot_fig(
    training_losses_reg,
    validation_losses_reg,
    training_losses_seg,
    validation_losses_seg,
    result_seg_path,
    result_reg_path,
    regularization_loss=None,
    anatomy_loss_reg=None,
    anatomy_loss_seg=None,
    similarity_loss=None,
    supervised_loss=None
):
    # Plot the training and validation losses
    plot_against_epoch_numbers(train_epoch_and_value_pairs=training_losses_reg,
                               validation_epoch_and_value_pairs=validation_losses_reg, train_label="training", val_label='validation')
    plt.legend()
    plt.ylabel('loss')
    plt.title('Alternating training: registration training loss')
    plt.savefig(os.path.join(result_reg_path, 'reg_net_training_losses.png'))

    plot_against_epoch_numbers(
        train_epoch_and_value_pairs=regularization_loss, train_label='regularization loss')
    plt.ylabel('loss')
    plt.title('Alternating training: registration regularization loss')
    plt.savefig(os.path.join(result_reg_path, 'regularization_reg_losses.png'))

    plot_against_epoch_numbers(
        train_epoch_and_value_pairs=anatomy_loss_reg, train_label='anatomy loss')
    plt.ylabel('loss')
    plt.title('Alternating training: registration anatomy loss')
    plt.savefig(os.path.join(result_reg_path, 'anatomy_reg_losses.png'))

    plot_against_epoch_numbers(
        train_epoch_and_value_pairs=similarity_loss, train_label='similarity loss')
    plt.ylabel('loss')
    plt.title('Alternating training: registration similarity loss')
    plt.savefig(os.path.join(result_reg_path, 'similarity_reg_losses.png'))

    if len(validation_losses_seg) == 0:
        plot_against_epoch_numbers(train_epoch_and_value_pairs=training_losses_seg,
                                validation_epoch_and_value_pairs=None, train_label="training", val_label=None)
        plt.legend()
        plt.ylabel('loss')
        plt.title('Alternating training: segmentation training loss')
        plt.savefig(os.path.join(result_seg_path, 'seg_net_training_losses.png'))
    else:
        plot_against_epoch_numbers(train_epoch_and_value_pairs=training_losses_seg,
                                validation_epoch_and_value_pairs=validation_losses_seg, train_label="training", val_label='validaiton')
        plt.legend()
        plt.ylabel('loss')
        plt.title('Alternating training: segmentation training loss')
        plt.savefig(os.path.join(result_seg_path, 'seg_net_training_losses.png'))
    
    plot_against_epoch_numbers(
        train_epoch_and_value_pairs=supervised_loss, train_label='supervised loss')
    plt.ylabel('loss')
    plt.title('Alternating training: segmentation supervised loss')
    plt.savefig(os.path.join(result_seg_path, 'supervised_seg_losses.png'))

    plot_against_epoch_numbers(
        train_epoch_and_value_pairs=anatomy_loss_seg, train_label='anatomy loss')
    plt.ylabel('loss')
    plt.title('Alternating training: segmentation anatomy loss')
    plt.savefig(os.path.join(result_seg_path, 'anatomy_seg_losses.png'))

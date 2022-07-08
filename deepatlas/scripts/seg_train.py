#import generators
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

def train_seg(dataloader_train_seg,
              dataloader_valid_seg,
              device,
              seg_net,
              lr_seg,
              max_epoch,
              val_step,
              result_seg_path
              ):
# (if already done then you may skip to and uncomment the checkpoint loading cell below)

    seg_net.to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(seg_net.parameters(), learning_rate)

    max_epochs = 300
    training_losses = []
    validation_losses = []
    val_interval = 5
    dice_loss = dice_loss_func2()
    for epoch_number in range(max_epochs):

        print(f"Epoch {epoch_number+1}/{max_epochs}:")

        seg_net.train()
        losses = []
        for batch in dataloader_train_seg:
            imgs = batch['img'].to(device)
            true_segs = batch['seg'].to(device)

            optimizer.zero_grad()
            predicted_segs = seg_net(imgs)
            loss = dice_loss(predicted_segs, true_segs)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        training_loss = np.mean(losses)
        print(f"\ttraining loss: {training_loss}")
        training_losses.append([epoch_number, training_loss])

        if epoch_number % val_interval == 0:
            seg_net.eval()
            losses = []
            with torch.no_grad():
                for batch in dataloader_valid_seg:
                    imgs = batch['img'].to(device)
                    true_segs = batch['seg'].to(device)
                    predicted_segs = seg_net(imgs)
                    loss = dice_loss(predicted_segs, true_segs)
                    losses.append(loss.item())

            validation_loss = np.mean(losses)
            print(f"\tvalidation loss: {validation_loss}")
            validation_losses.append([epoch_number, validation_loss])
    
    # Free up some memory
    del loss, predicted_segs, true_segs, imgs
    torch.cuda.empty_cache()
    torch.save(seg_net.state_dict(), os.path.join(result_seg_path, 'seg_net_best.pth'))
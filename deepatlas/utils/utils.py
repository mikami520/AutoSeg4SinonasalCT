import numpy as np
import matplotlib.pyplot as plt
import monai
import torch
import os
import json
import matplotlib
import shutil


def make_if_dont_exist(folder_path, overwrite=False):
    if os.path.exists(folder_path):
        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path, ignore_errors = True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")

def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    plt.figure()
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    x, y, z = np.array(image_array.shape)//2
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im in zip(axs, imgs):
        ax.axis('off')
        ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin='lower')

    plt.savefig('test.png')


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, downsampling=None, ep=None, path=None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    plt.savefig(os.path.join(path, f'df_{ep}.png'))


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H, W))
    grid_img[np.arange(0, H, grid_spacing), :] = 1
    grid_img[:, np.arange(0, W, grid_spacing)] = 1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(
        0)  # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin='lower', cmap='gist_gray')


def preview_3D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.show()


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return np.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det

def load_json(json_path):
    assert type(json_path) == str
    fjson = open(json_path, 'r')
    json_file = json.load(fjson)
    return json_file

def plot_progress(logger, save_dir, train_loss, val_loss, name):
    """
    Should probably by improved
    :return:
    """
    assert len(train_loss) != 0
    train_loss = np.array(train_loss)
    try:
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax.plot(train_loss[:,0], train_loss[:,1], color='b', ls='-', label="loss_tr")
        if len(val_loss) != 0:
            val_loss = np.array(val_loss)
            ax.plot(val_loss[:, 0], val_loss[:, 1], color='r', ls='-', label="loss_val")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)
    except:
        logger.info(f"failed to plot {name} training progress")

def save_reg_checkpoint(network, optimizer, epoch, best_loss, sim_loss=None, regular_loss=None, ana_loss=None, total_loss=None, save_dir=None, name=None):
    all_loss = {
        'best_loss': best_loss,
        'total_loss': total_loss,
    }
    if sim_loss is not None:
        all_loss['sim_loss'] = sim_loss
    if regular_loss is not None:
        all_loss['regular_loss'] = regular_loss
    if ana_loss is not None:
        all_loss['ana_loss'] = ana_loss
    
    torch.save({
        'epoch': epoch,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'all_loss': all_loss,
    }, os.path.join(save_dir, name+'_checkpoint.pth'))


def save_seg_checkpoint(network, optimizer, epoch, best_loss, super_loss=None, ana_loss=None, total_loss=None, save_dir=None, name=None):
    all_loss = {
        'best_loss': best_loss,
        'total_loss': total_loss,
    }
    if super_loss is not None:
        all_loss['super_loss'] = super_loss
    if ana_loss is not None:
        all_loss['ana_loss'] = ana_loss
    
    torch.save({
        'epoch': epoch,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'all_loss': all_loss,
    }, os.path.join(save_dir, name+'_checkpoint.pth'))


def load_latest_checkpoint(path, network, optimizer, device):
    checkpoint_path = os.path.join(path, 'latest_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    all_loss = checkpoint['all_loss']
    return network, optimizer, all_loss

def load_valid_checkpoint(path, device):
    checkpoint_path = os.path.join(path, 'valid_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    all_loss = checkpoint['all_loss']
    return all_loss

def load_best_checkpoint(path, device):
    checkpoint_path = os.path.join(path, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_loss = checkpoint['all_loss']['best_loss']
    return best_loss


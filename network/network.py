import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.factories import Act, Norm
import random
import glob
import os.path


def segNet(
    spatial_dim: int,
    in_channel: int,
    out_channel: int,
    channel: Sequence[int],
    stride: Sequence[int],
    num_res_unit: int = 0,
    acts: Union[Tuple, str] = Act.PRELU,
    norms: Union[Tuple, str] = Norm.INSTANCE,
    dropouts: float = 0.0,
):
    seg_net = monai.networks.nets.UNet(
        spatial_dims=spatial_dim,  # spatial dims
        in_channels=in_channel,  # input channels
        out_channels=out_channel,  # output channels
        channels=channel,  # channel sequence
        strides=stride,  # convolutional strides
        dropout=dropouts,
        act=acts,
        norm=norms,
        num_res_units=num_res_unit
    )
    return seg_net

def regNet(
    spatial_dim: int,
    in_channel: int,
    out_channel: int,
    channel: Sequence[int],
    stride: Sequence[int],
    num_res_unit: int = 0,
    acts: Union[Tuple, str] = Act.PRELU,
    norms: Union[Tuple, str] = Norm.INSTANCE,
    dropouts: float = 0.0,
):
    reg_net = monai.networks.nets.UNet(
        spatial_dims=spatial_dim,  # spatial dims
        in_channels=in_channel,  # input channels
        out_channels=out_channel,  # output channels
        channels=channel,  # channel sequence
        strides=stride,  # convolutional strides
        dropout=dropouts,
        act=acts,
        norm=norms,
        num_res_units=num_res_unit
    )
    return reg_net

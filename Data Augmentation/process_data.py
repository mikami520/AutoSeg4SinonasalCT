from msilib.schema import Directory
import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os.path
import tempfile


from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)
# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=2938649572)

root_dir = '/home/ameen'
data_dir = os.path.join(root, 'ET')

print(root)
print(data_dir)

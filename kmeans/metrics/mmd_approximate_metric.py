import torch
import numpy as np

import os
import sys
sys.path.append(os.path.realpath(".."))
from losses.mmd.mmd_loss import MMDApproximate


class MMDpace:
    def __init__(self, num_features, spatial_mode, patch_mode):
        self.name = f"MMDApprox-dist-(r-{num_features}_s-{spatial_mode}_p-{patch_mode})"
        self.feature_extractor = MMDWrapper(num_features, patch_mode)
        # self.metric = lambda x,y: self.feature_extractor.spatial_reduction((x - y).pow(2).mean(dim=1, keepdim=True), dim=(1, 2, 3))
        self.metric = SpatialDist(spatial_mode)

class SpatialDist:
    def __init__(self, spatial_mode):
        self.spatial_mode = spatial_mode

    def __call__(self, x,y):
        mean = np.mean((x - y)**2, axis=0, keepdims=True)

        if self.spatial_mode == "mean":
            return np.mean(mean, axis=(0, 1, 2))
        elif self.spatial_mode == "sum":
            return np.sum(mean, axis=(0, 1, 2))
        else:
            return mean


class MMDWrapper(torch.nn.Module):
    def __init__(self, num_features, normalize_patch):
        super(MMDWrapper, self).__init__()
        self.mmd = MMDApproximate(r=num_features, normalize_patch=normalize_patch, pool_size=32, pool_strides=16)

    def get_fv(self, x):
        return self.mmd.get_activations(x)

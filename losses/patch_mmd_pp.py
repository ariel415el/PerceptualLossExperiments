import torch
import numpy as np

from losses.l2 import L2
from losses.patch_loss import PatchRBFLoss
from losses.patch_mmd_loss import MMDApproximate


class MMD_PP(torch.nn.Module):
    def __init__(self, patch_size=3, sigma=0.06, strides=1, r=512, pool_size=32, pool_strides=16
                 , normalize_patch='channel_mean', pad_image=True, weights=None, batch_reduction='mean', local_patch_size=None, local_sigma=None) -> object:
        super(MMD_PP, self).__init__()
        if weights is None:
            self.weights = [0.001, 0.05, 1.0]
        else:
            self.weights = weights

        if local_sigma is None:
            local_sigma = sigma
        if local_patch_size is None:
            local_patch_size = patch_size

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=local_patch_size, sigma=local_sigma, pad_image=pad_image, batch_reduction=batch_reduction),
            MMDApproximate(patch_size=patch_size, sigma=sigma, strides=strides, r=r, pool_size=pool_size,
                           pool_strides=pool_strides, batch_reduction=batch_reduction,
                           normalize_patch=normalize_patch, pad_image=pad_image)
        ])

        # self.name = f"MMD++(p={patch_size},win={pool_size}:{pool_strides},rf={r},w={self.weights},s={sigma},ls={local_sigma},lp={local_patch_size})"
        self.name = f"MMD++"

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])

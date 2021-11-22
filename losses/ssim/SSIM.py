# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    """Create 2-D gauss kernel"""
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    w = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    w /= w.sum()

    w = (w.reshape(size, 1) * w.reshape(1, size))

    w = w.unsqueeze(0).unsqueeze(0)

    return w


def gaussian_filter(input, win):
    C = input.shape[-3]
    win = win.repeat(C, 1, 1, 1)
    return F.conv2d(input, weight=win, stride=1, padding=0, groups=C)


def _ssim(X, Y, filter, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y"""
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape

    C1 = K1 ** 2
    C2 = K2 ** 2

    filter = filter.to(X.device, dtype=X.dtype)

    mu_x = gaussian_filter(X, filter)
    mu_y = gaussian_filter(Y, filter)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_x_mu_y = mu_x * mu_y

    sigma_x_sq = (gaussian_filter(X * X, filter) - mu_x_sq)
    sigma_y_sq = (gaussian_filter(Y * Y, filter) - mu_y_sq)
    sigma_xy = (gaussian_filter(X * Y, filter) - mu_x_mu_y)

    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu_x_mu_y + C1) / (mu_x_sq + mu_y_sq + C1))

    ssim_map *= cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)

    return ssim_per_channel, cs


class SSIM(torch.nn.Module):
    def __init__(
            self,
            patch_size=11,
            sigma=1.5,
            K=(0.01, 0.03),
            batch_reduction='none'
    ):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(SSIM, self).__init__()
        self.patch_size = patch_size
        self.K = K
        self.filter = _fspecial_gauss_1d(patch_size, sigma)
        self.batch_reduction = batch_reduction
        self.name = f'SSIM(p={patch_size},s={sigma})'

    def forward(self, X, Y):
        if not X.shape == Y.shape:
            raise ValueError("Input images should have the same dimensions.")

        X = (X + 1) / 2
        Y = (Y + 1) / 2

        similarity, _ = _ssim(X, Y, filter=self.filter, K=self.K)

        if self.batch_reduction == 'mean':
            similarity = similarity.mean()
        else:
            similarity = similarity.mean(-1)

        # shift from -1,1 to 0,1 and flip to be loss
        loss = 0.5 - similarity/2

        return loss


if __name__ == '__main__':
    loss = SSIM()
    x = torch.ones(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)

    loss(x,y)
    # import losses
    # # loss = losses.PatchRBFLoss(patch_size=11, normalize_patch='channel_mean')
    # from time import time
    # start = time()
    # for i in range(1000):
    #     loss(x, y)
    # print(f"{(time()-start)/1000} s per inference")

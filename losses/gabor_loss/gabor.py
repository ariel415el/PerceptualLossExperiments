import os
import random

import numpy as np
import torch
from skimage.filters import gabor_kernel
from torch.nn.functional import conv2d
import cv2

def get_gabors_sklearn():
    # prepare filter bank kernels
    kernels = []
    for theta in np.arange(8) / 8. * np.pi: # only half a turn due to symetry
        for sigma in [1.5]:
            for frequency in (0.2, 0.4, 0.7):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    max_s = max([x.shape[0] for x in kernels])
    for i in range(len(kernels)):
        if kernels[i].shape[0] < max_s:
            n_pad = (max_s - kernels[i].shape[0]) // 2
            kernels[i] = np.pad(kernels[i],pad_width=n_pad,constant_values=0)
    assert(len(np.unique([x.shape[0] for x in kernels])) == 1)
    return kernels

def get_gabors_cv(kernel_size=11, ch=1):
    range_theta = np.arange(8) / 8. * np.pi
    range_gaussian_sigma = np.array([kernel_size * 0.2, kernel_size * 0.4])
    range_wave_length = np.array([kernel_size*0.25, kernel_size*0.5])
    range_aspect_ratio = [1, 2]
    range_phase_offset = [np.pi]

    kernels = []
    import itertools
    for (theta, sigma, lam, gamma, psi) in list(
            itertools.product(range_theta, range_gaussian_sigma, range_wave_length, range_aspect_ratio,range_phase_offset)):
        kernel = cv2.getGaborKernel((kernel_size, kernel_size),
                                    sigma,
                                    theta,
                                    lam,
                                    gamma,
                                    psi)
        kernels.append(kernel)
    # random.shuffle(kernels)
    kernels = torch.tensor(kernels, dtype=torch.float32).reshape(-1, 1, kernel_size, kernel_size)
    return kernels


class GaborPerceptualLoss(torch.nn.Module):
    def __init__(self, pool_window=128, pool_stride=1, batch_reduction='none'):
        super(GaborPerceptualLoss, self).__init__()
        # self.kernels = get_gabors_cv(kernel_size=11)
        self.kernels = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alexnet_1_weights.pth'))
        self.biases = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alexnet_1_biases.pth'))
        # self.kernels = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg_1_weight.pth'))
        # self.biases = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg_1_biases.pth'))
        self.non_linearity = torch.nn.ReLU()
        # self.non_linearity = lambda x: x
        self.pool = torch.nn.AvgPool2d(pool_window, pool_stride)
        # self.pool = torch.nn.MaxPool2d(pool_window, pool_stride)
        self.batch_reduction = batch_reduction
        self.name = f'GaborPerceptualLoss'

    def forward(self, x, y):
        # x = torch.mean(x, dim=1, keepdim=True)
        # y = torch.mean(y, dim=1, keepdim=True)
        self.kernels = self.kernels.to(x.device)
        self.biases = self.biases.to(x.device)
        x_features = self.pool(self.non_linearity(conv2d(x, self.kernels, padding=0)))#, bias=self.biases)))
        y_features = self.pool(self.non_linearity(conv2d(y, self.kernels, padding=0)))#, bias=self.biases)))
        dist = (x_features - y_features).pow(2)
        if self.batch_reduction == 'mean':
            return dist.mean()
        else:
            return dist.view(x.shape[0], -1).mean(1)


if __name__ == '__main__':
    p = 11
    img = cv2.imread(
        '/style_transfer/imgs/faces/00009.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    img = img.reshape(1, 3, 128, 128)
    img = torch.mean(img, dim=1, keepdim=True)

    kernels = torch.tensor(get_gabors_cv(kernel_size=p), dtype=torch.float32)
    import torchvision.utils as vutils

    vutils.save_image(kernels, 'gabor.png', normalize=True)
    output = conv2d(img, kernels, padding=p//2)

    import matplotlib.pyplot as plt

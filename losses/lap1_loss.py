import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

def kernel_gauss(size=5, sigma=1.0, n_channels=1):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")

    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyramid = []

    for level in range(max_levels):
        blurred = conv_gauss(current, kernel)
        diff = current - blurred
        pyramid.append(diff)
        current = F.avg_pool2d(blurred, 2)

    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=3, k_size=5, sigma=2.0, n_channels=3, batch_reduction='mean', weightening_mode=0, no_last_layer=False):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = kernel_gauss(size=k_size, sigma=sigma, n_channels=n_channels)
        self.name = f"Lap1(L-{max_levels},M-{weightening_mode},{'NL' if no_last_layer else ''})"
        self.no_last_layer = no_last_layer
        if batch_reduction == 'mean':
            self.metric = F.l1_loss
        else:
            self.metric = lambda x,y: torch.abs(x-y).view(x.shape[0], -1).mean(1)
        if weightening_mode == 0:
            self.weight = lambda j: (2 ** (2 * j))
        if weightening_mode == 1:
            self.weight = lambda j: (2 ** (-2 * j))
        if weightening_mode == 2:
            self.weight = lambda j: (2 ** (2 * (max_levels - j)))
        if weightening_mode == 3:
            self.weight = lambda j: 1

    def forward(self, output, target):
        self._gauss_kernel = self._gauss_kernel.to(output.device)
        pyramid_output = laplacian_pyramid(output, self._gauss_kernel, self.max_levels)
        pyramid_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        if self.no_last_layer:
            pyramid_target = pyramid_target[:-1]
            pyramid_output = pyramid_output[:-1]
        lap1_loss = 0
        for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)):
            lap1_loss += self.metric(a,b) * self.weight(j)
        l2_loss = self.metric(output, target)

        return lap1_loss + l2_loss

if __name__ == '__main__':
    x = cv2.imread()
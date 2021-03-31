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
    def __init__(self, max_levels=5, k_size=5, sigma=2.0, n_channels=1):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = kernel_gauss(size=k_size, sigma=sigma, n_channels=n_channels)
        self.name = f"Lap1_ML-{max_levels}"

    def forward(self, output, target):
        self._gauss_kernel = self._gauss_kernel.to(output.device)
        pyramid_output = laplacian_pyramid(output, self._gauss_kernel, self.max_levels)
        pyramid_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        # lap1_loss = sum(F.l1_loss(a, b)*(2 ** (2 * j)) for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)))
        # lap1_loss = sum(F.l1_loss(a, b)*(2 ** (-2 * j)) for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)))
        lap1_loss = sum(F.l1_loss(a, b) for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)))
        l2_loss = nn.MSELoss()(output, target)

        return lap1_loss + l2_loss

if __name__ == '__main__':
    x = cv2.imread()
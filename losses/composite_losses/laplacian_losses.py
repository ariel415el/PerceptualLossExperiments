import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn


######## Method 1 ###########
def get_kernel_gauss(size=5, sigma=1.0, n_channels=1):
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


######## Method 2 ###########
def get_gaussian_kernel_2(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel_2(device=device)
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1, :, :], gaussian_k, padding=2, stride=2) for i in range(3)]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def gaussian_pyramid(original, n_pyramids, device="cpu"):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x, device=device)
        pyramids.append(x)
    return pyramids


def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel_2(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1, :, :], gaussian_k, padding=2) for i in range(3)]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def laplacian_pyramid_2(original, n_pyramids, device="cpu"):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid_2(x.to(device), n_pyramids, device=device)
        p = [x.cpu() for x in p]
        pyramids.append(p)
    del x
    result = []
    for i in range(n_pyramids + 1):
        x = []
        for j in range(n):
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result


class LaplacyanLoss(nn.Module):
    def __init__(self, metric, max_levels=3, k_size=5, sigma=1.0, weightening_mode=0):
        super(LaplacyanLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = get_kernel_gauss(size=k_size, sigma=sigma, n_channels=3)
        # self._gauss_kernel = get_gaussian_kernel_2(device=torch.device('cuda')).repeat(3,1,1,1)

        self.metric = metric

        self.name = f"Laplacian(L-{max_levels},M-{weightening_mode})-{self.metric.name}"

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

        lap1_loss = 0
        for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)):
            lap1_loss += self.metric(a, b) * self.weight(j)

        return lap1_loss



if __name__ == '__main__':
    k1 = get_gaussian_kernel_2()
    k2 = get_kernel_gauss(size=5, sigma=2, n_channels=3)

    x = torch.ones((5, 3, 128, 128))
    import cv2

    x = cv2.imread(
        '/style_transfer/imgs/content/green_eye.jpg')
    x = cv2.resize(x, (256, 256)).transpose(2, 0, 1)
    print(x.shape)
    x = torch.from_numpy(x).unsqueeze(0).float()
    print(x.shape)
    z = laplacian_pyramid(x, get_kernel_gauss(size=7, sigma=3, n_channels=3), 5)
    w = laplacian_pyramid_2(x, 5, device="cpu")

    for n, (i, j) in enumerate(zip(z, w)):
        from torchvision.utils import save_image

        save_image(i, f"i-{n}.png", normalize=True)
        save_image(j, f"j-{n}.png", normalize=True)
        # print(torch.all(i == j))
        print((i - j).mean())
        # print(i.shape, j.shape)

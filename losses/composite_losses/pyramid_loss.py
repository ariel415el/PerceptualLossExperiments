from torch import nn
import torch.nn.functional as F


def get_gaussian_pyramid(img, max_levels=5):
    current = img
    pyramid = [current]

    for level in range(max_levels):
        current = F.avg_pool2d(current, 2)
        pyramid.append(current)

    return pyramid


class PyramidLoss(nn.Module):
    def __init__(self, metric, max_levels=2, weightening_mode=0):
        super(PyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.metric = metric
        self.name = f"Pyramid(L-{max_levels},M-{weightening_mode})-{self.metric.name}"

        if weightening_mode == 0:
            self.weight = lambda j: (2 ** (2 * j))
        if weightening_mode == 1:
            self.weight = lambda j: (2 ** (-2 * j))
        if weightening_mode == 2:
            self.weight = lambda j: (2 ** (2 * (max_levels - j)))
        if weightening_mode == 3:
            self.weight = lambda j: 1

    def forward(self, output, target):
        pyramid_output = get_gaussian_pyramid(output, self.max_levels)
        pyramid_target = get_gaussian_pyramid(target, self.max_levels)

        lap1_loss = 0
        for j, (a, b) in enumerate(zip(pyramid_output, pyramid_target)):
            lap1_loss += self.metric(a, b) * self.weight(j)

        return lap1_loss
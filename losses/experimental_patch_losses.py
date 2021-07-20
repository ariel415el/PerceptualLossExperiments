import torch.nn

from losses.composite_losses.window_loss import WindowLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.classic_losses.lap1_loss import *
from losses.patch_loss import *
from losses.swd.patch_swd import PatchSWDLoss
from torch.nn.functional import conv2d, mse_loss, avg_pool2d


def get_reduction_fn(reduction):
    if reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        def no_reduce(tensor, *args, **kwargs):  # support the reduction API
            return tensor

        return no_reduce
    else:
        raise ValueError("Invalid reduction type")


class MMD_PP(torch.nn.Module):
    def __init__(self, patch_size=3, sigma=0.06, strides=1, r=512, pool_size=32, pool_strides=16
                 , normalize_patch='channel_mean', pad_image=True, weights=None, batch_reduction='mean',
                 local_patch_size=3, local_sigma=0.1) -> object:
        super(MMD_PP, self).__init__()
        if weights is None:
            self.weights = [0.001, 0.05, 1.0]
        else:
            self.weights = weights

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=local_patch_size, sigma=local_sigma, pad_image=pad_image,
                         batch_reduction=batch_reduction, normalize_patch='none'),
            MMDApproximate(patch_size=patch_size, sigma=sigma, strides=strides, r=r, pool_size=pool_size,
                           pool_strides=pool_strides, batch_reduction=batch_reduction,
                           normalize_patch=normalize_patch, pad_image=pad_image)
        ])

        # self.name = f"MMD++(p={patch_size},win={pool_size}:{pool_strides},rf={r},w={self.weights},s={sigma},ls={local_sigma},lp={local_patch_size})"
        self.name = f"MMD++"

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class MMD_PPP(torch.nn.Module):
    def __init__(self, r=512, normalize_patch='channel_mean', weights=None, batch_reduction='mean'):
        super(MMD_PPP, self).__init__()
        if weights is None:
            # self.weights = [0.001, 0.05, 0.1, 1.0]
            self.weights = [0.001, 0.05, 1.0]
        else:
            self.weights = weights

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            # PatchRBFLoss(patch_size=3, sigma=0.06, pad_image=True, batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction=batch_reduction),
            MMDApproximate(patch_size=3, sigma=0.03, strides=1, r=r, pool_size=32, pool_strides=16,
                           batch_reduction=batch_reduction,
                           normalize_patch=normalize_patch, pad_image=True)
        ])

        # self.name = f"MMD+++(rf={r},w={self.weights})"
        self.name = f"MMD+++"

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class SWD_PPP(torch.nn.Module):
    def __init__(self, r=512, normalize_patch='channel_mean', weights=None, batch_reduction='mean'):
        super(SWD_PPP, self).__init__()
        if weights is None:
            self.weights = [0.01, 0.2, 1, 0.5]
        else:
            self.weights = weights

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            # PatchRBFLoss(patch_size=3, sigma=0.06, pad_image=True, batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction=batch_reduction),
            WindowLoss(PatchSWDLoss(patch_size=3, num_proj=128, n_samples=128), batch_reduction=batch_reduction,
                       stride=16)
        ])

        # self.name = f"MMD+++(rf={r},w={self.weights})"
        self.name = f"SWD+++(w={self.weights})"
        # self.name = f"SWD+++"

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class EngeneeredPerceptualLoss(torch.nn.Module):
    def __init__(self, relu_bias=0, pool_window=128, pool_stride=1, batch_reduction='none'):
        super(EngeneeredPerceptualLoss, self).__init__()
        self.kernels = torch.tensor(
            [
                [[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]],
                [[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]
            ],
            dtype=torch.float32)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(pool_window, pool_stride)
        self.batch_reduction = batch_reduction
        self.relu_bias = relu_bias
        self.name = f'EngeneeredPerceptualLoss(b={relu_bias})'

    def forward(self, x, y):
        x = torch.mean(x, dim=1, keepdim=True)
        y = torch.mean(y, dim=1, keepdim=True)
        self.kernels = self.kernels.to(x.device)
        x_features = self.pool(self.relu(conv2d(x, self.kernels, padding=1) + self.relu_bias))
        y_features = self.pool(self.relu(conv2d(y, self.kernels, padding=1) + self.relu_bias))
        dist = (x_features - y_features).pow(2)
        if self.batch_reduction == 'mean':
            return dist.mean()
        else:
            return dist.view(x.shape[0], -1).mean(1)


class SimplePatchLoss(torch.nn.Module):
    def __init__(self, patch_size=3, sigma=0.06, batch_reduction='none'):
        super(SimplePatchLoss, self).__init__()
        self.kernels = torch.ones((1, 3, patch_size, patch_size), dtype=torch.float32)
        self.pool = torch.nn.AvgPool2d(patch_size, 1)
        self.batch_reduction = batch_reduction
        self.sigma = sigma * patch_size**2
        self.name = f'simplePatchLoss(p={patch_size},s={sigma})'

    def forward(self, x, y):
        pixel_diff = (x-y)**2
        patch_diff = conv2d(pixel_diff, self.kernels.to(x.device))
        patch_rbf = 1 -1 * (patch_diff / (-1 * self.sigma)).exp()
        loss = torch.mean(patch_rbf, dim=(1, 2, 3))
        if self.batch_reduction == 'mean':
            return loss.mean()
        else:
            return loss.view(x.shape[0], -1).mean(1)

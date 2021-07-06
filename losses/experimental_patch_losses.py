from losses.composite_losses.window_loss import WindowLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.lap1_loss import *
from losses.patch_loss import *
from losses.swd.swd import PatchSWDLoss
import torch.nn.functional as F


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


class GradLoss(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(GradLoss, self).__init__()
        self.ky = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.kx = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.batch_reduction = batch_reduction
        self.name = 'GradLoss'

    def forward(self, im1, im2):
        from torch.nn.functional import conv2d, pad

        b = im1.size(0)
        self.kx = self.kx.to(im1.device)
        self.ky = self.ky.to(im1.device)
        im1_gray = torch.mean(im1, 1, keepdim=True)
        im2_gray = torch.mean(im2, 1, keepdim=True)
        # im1_gray = pad(torch.mean(im1, 1, keepdim=True), (1, 1, 1, 1), mode='reflect')
        # im2_gray = pad(torch.mean(im2, 1, keepdim=True), (1, 1, 1, 1), mode='reflect')
        diff_x = F.l1_loss(conv2d(im1_gray, self.kx), conv2d(im2_gray, self.kx), reduction=self.batch_reduction)
        diff_y = F.l1_loss(conv2d(im1_gray, self.ky), conv2d(im2_gray, self.ky), reduction=self.batch_reduction)

        if self.batch_reduction == 'none':
            diff_x = torch.mean(diff_x, (1,2,3))
            diff_y = torch.mean(diff_y, (1,2,3))

        return (diff_x + diff_y) / 2

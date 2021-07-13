from losses.composite_losses.window_loss import WindowLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.classic_losses.lap1_loss import *
from losses.patch_loss import *
from losses.swd.patch_swd import PatchSWDLoss


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
                 , normalize_patch='channel_mean', pad_image=True, weights=None, batch_reduction='mean', local_patch_size=3, local_sigma=0.1) -> object:
        super(MMD_PP, self).__init__()
        if weights is None:
            self.weights = [0.001, 0.05, 1.0]
        else:
            self.weights = weights

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



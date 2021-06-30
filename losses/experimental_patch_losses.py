from losses.l2 import L2
from losses.patch_mmd_loss import MMDApproximate
from losses.lap1_loss import *
from losses.patch_loss import *
from losses.laplacian_losses import get_kernel_gauss, laplacian_pyramid


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
            self.weights = [0.001, 0.05, 0.1, 1.0]
        else:
            self.weights = weights

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=3, sigma=0.06, pad_image=True, batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction=batch_reduction),
            MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=r, pool_size=32,
                           pool_strides=16, batch_reduction=batch_reduction,
                           normalize_patch=normalize_patch, pad_image=True)
        ])

        # self.name = f"MMD+++(rf={r},w={self.weights})"
        self.name = f"MMD+++"

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class LossesList(torch.nn.Module):
    def __init__(self, losses, weights):
        super(LossesList, self).__init__()
        self.weights = weights

        self.losses = torch.nn.ModuleList(losses)

        self.name = "+".join([f"{w}*{l.name}" for l,w in zip(self.losses, self.weights)])

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])


class LapPatchLoss(torch.nn.Module):
    def __init__(self, max_levels=3, k_size=5, sigma=2.0, n_channels=3, batch_reduction='mean', weightening_mode=0):
        super(LapPatchLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = get_kernel_gauss(size=k_size, sigma=sigma, n_channels=n_channels)
        self.name = f"PatchLap1(L-{max_levels},M-{weightening_mode})"

        self.metric = PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction=batch_reduction)

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
            lap1_loss += self.metric(a,b) * self.weight(j)
        l2_loss = self.metric(output, target)

        return lap1_loss + l2_loss


if __name__ == '__main__':
    x = cv2.imread()
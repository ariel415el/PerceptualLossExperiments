import torch
from torch.nn.functional import conv2d, pad

from losses.l2 import L2
from losses.patch_loss import PatchRBFLoss
from losses.patch_mmd_loss import MMDApproximate


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


class DoubleMMDAprox(torch.nn.Module):
    def _get_w_b(self, n_channels):
        self.w1 = torch.randn(self.r, n_channels, self.ksize, self.ksize)
        self.w2 = torch.randn(self.r, self.r, self.ksize, self.ksize)
        if self.normalize_patch == 'channel_mean':
            self.w1 -= self.w1.mean(dim=(2, 3), keepdim=True)
            self.w2 -= self.w2.mean(dim=(2, 3), keepdim=True)
        elif self.normalize_patch == 'mean':
            self.w1 -= self.w1.mean(dim=(1, 2, 3), keepdim=True)
            self.w2 -= self.w2.mean(dim=(1, 2, 3), keepdim=True)
        self.b1 = torch.rand(self.r)
        self.b2 = torch.rand(self.r)
        return self.w1 ,self.w2, self.b1, self.b2

    def __init__(self,
                 patch_size=3,
                 strides=1,
                 sigma=0.06,
                 r=1024,
                 pool_size=32,
                 pool_strides=16,
                 batch_reduction='mean',
                 spatial_reduction='mean',
                 pad_image=True,
                 normalize_patch='none'):
        super(DoubleMMDAprox, self).__init__()
        self.r = r
        self.pool_size = pool_size
        if pool_size > 1:
            self.pool = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides)
        else:
            def no_op(x):
                return x

            self.pool = no_op
        self.ksize = patch_size
        self.strides = strides
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.normalize_patch = normalize_patch
        self.averaging_kernel = None
        self.batch_reduction = get_reduction_fn(batch_reduction)
        self.spatial_reduction = get_reduction_fn(spatial_reduction)
        self.sigma = sigma * patch_size ** 2  # sigma is defined per pixel

        self.padding = self.ksize // 2 if pad_image else 0

        self.name = f"DoubleMMD-Approx(p={patch_size},win={pool_size},rf={r})"

    def forward(self, x, y):
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
            y = pad(y, (self.padding, self.padding, self.padding, self.padding), mode='reflect')

        c = x.shape[1]
        w1, w2, b1, b2 = self._get_w_b(c)
        w1 = w1.to(x.device)
        w2 = w2.to(x.device)
        b1 = b1.to(x.device)
        b2 = b2.to(x.device)


        act_x1 = conv2d(x, w1, b1, self.strides, padding=1)
        # act_x2 = conv2d(act_x1, w2, b2, self.strides, padding=1)
        act_x2 = conv2d(torch.relu(act_x1), w2, b2, self.strides, padding=1)
        x_feats1 = self.pool(torch.cos(act_x1))
        x_feats2 = self.pool(torch.cos(act_x2))
        x_feats = torch.cat([x_feats1, x_feats2], dim=1)
        # x_feats = x_feats1

        act_y1 = conv2d(y, w1, b1, self.strides, padding=1)
        # act_y2 = conv2d(act_y1, w2, b2, self.strides, padding=1)
        act_y2 = conv2d(torch.relu(act_y1), w2, b2, self.strides, padding=1)
        y_feats1 = self.pool(torch.cos(act_y1))
        y_feats2 = self.pool(torch.cos(act_y2))
        y_feats = torch.cat([y_feats1, y_feats2], dim=1)
        # y_feats = y_feats1

        distance = self.spatial_reduction((x_feats - y_feats).pow(2).mean(dim=1, keepdim=True), dim=(1, 2, 3))

        return self.batch_reduction(distance, dim=0)


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



from losses.lap1_loss import *
from losses.patch_loss import *
class LapPatchLoss(torch.nn.Module):
    def __init__(self, max_levels=3, k_size=5, sigma=2.0, n_channels=3, batch_reduction='mean', weightening_mode=0):
        super(LapPatchLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = kernel_gauss(size=k_size, sigma=sigma, n_channels=n_channels)
        self.name = f"PatchLap1(L-{max_levels},M-{weightening_mode})"

        # self.metric = PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction=batch_reduction)
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
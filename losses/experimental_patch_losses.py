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


class SWD_PP(torch.nn.Module):
    def __init__(self, num_proj=128, normalize_patch='channel_mean', weights=None, batch_reduction='mean'):
        super(SWD_PP, self).__init__()
        if weights is None:
            self.weights = [0.01, 0.2, 1, 0.5]
        else:
            self.weights = weights

        self.losses = torch.nn.ModuleList([
            L2(batch_reduction=batch_reduction),
            PatchRBFLoss(patch_size=11, sigma=0.015, pad_image=True, batch_reduction=batch_reduction),
            # WindowLoss(PatchSWDLoss(patch_size=11, stride=5, num_proj=num_proj, normalize_patch=normalize_patch), batch_reduction=batch_reduction, window_size=32, stride=16)
            PatchSWDLoss(patch_size=11, stride=5, num_proj=num_proj, normalize_patch=normalize_patch)
        ])

        self.name = f"SWD++(num_proj={num_proj})"

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
    """
    Minimize gaussian distance over spatially matching patches of two images.
    RBf forces the patches to be exactly like the optimized image. As sigma gets closer, any deviation leads immediatly
        to a higher loss. In other words, averaging is not the way to go here
    """
    def __init__(self, patch_size=3, sigma=0.06, batch_reduction='none'):
        super(SimplePatchLoss, self).__init__()
        self.patch_summation_kernel = torch.ones((1, 3, patch_size, patch_size), dtype=torch.float32)
        self.pool = torch.nn.AvgPool2d(patch_size, 1)
        self.batch_reduction = batch_reduction
        self.sigma = sigma * patch_size**2
        self.name = f'simplePatchLoss(p={patch_size},s={sigma})'

    def forward(self, x, y):
        pixel_diff = (x-y)**2
        patch_l2_norm = conv2d(pixel_diff, self.patch_summation_kernel.to(x.device))
        patch_rbf_dist = 1 - 1 * (patch_l2_norm / (-1 * self.sigma)).exp()
        loss = torch.mean(patch_rbf_dist, dim=(1, 2, 3))
        if self.batch_reduction == 'mean':
            return loss.mean()
        else:
            return loss.view(x.shape[0], -1).mean(1)


# class PerPatchLoss(torch.nn.Module):
#     """
#     Minimize gaussian distance over spatially matching patches of two images.
#     RBf forces the patches to be exactly like the optimized image. As sigma gets closer, any deviation leads immediatly
#         to a higher loss. In other words, averaging is not the way to go here
#     """
#     def __init__(self, receptive_field, stride):
#         super(PerPatchLoss, self).__init__()
#         self.name = f'PerPatchLoss'
#         self.receptive_field = receptive_field
#         self.stride = stride
#         from classic_losses.grad_loss import GradLoss3Channels
#         self.loss = GradLoss3Channels()
#
#     def forward(self, x, y):
#         x_patches = torch.nn.functional.unfold(x, kernel_size=self.receptive_field, padding=0, stride=self.stride)
#         y_patches = torch.nn.functional.unfold(y, kernel_size=self.receptive_field, padding=0, stride=self.stride)
#         dists = 0.05 * ((x_patches - y_patches)**2).mean()
#         d = int(np.sqrt(x_patches.shape[1] / 3))
#         x_patches = x_patches.transpose(2, 1).reshape(-1, 3, d, d)
#         y_patches = y_patches.transpose(2, 1).reshape(-1, 3, d, d)
#         dists += self.loss(x_patches, y_patches)
#         return dists

if __name__ == '__main__':

    loss = SimplePatchLoss(11, sigma=0.02)
    img = cv2.imread('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/green_waves.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    x = torch.zeros((1, 3, 128, 128))
    y = torch.zeros((1, 3, 128, 128))

    patch = img[:, 20:32, 20:32]
    print(patch.shape)
    x[0, :, 50:62, 50:62] = patch
    # y[0, :, 70:82, 70:82] = patch

    loss(x,y)

    # import torchvision.utils as vutils
    # vutils.save_image(x, f"x.png", normalize=True)
    # vutils.save_image(y, f"y.png", normalize=True)


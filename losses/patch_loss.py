import os
import sys
import torch
from torch.nn.functional import conv2d
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from losses.patch_mmd_loss import get_reduction_fn


class PatchLoss(torch.nn.Module):
    def _sum_over_patches(self, diffs):
        return conv2d(diffs, self.sum_w, stride=self.strides, padding=self.padding)

    def _extract_patches(self, x):
        patches = self.unfolder(x)
        bs, c, h, w = x.shape
        ps = self.patch_size if isinstance(self.patch_size, int) else self.patch_size[0]
        nh = int(np.floor(((h + 2 * self.padding - 1 - (ps - 1)) / self.strides) + 1))
        nw = int(np.floor(((w + 2 * self.padding - 1 - (ps - 1)) / self.strides) + 1))
        return patches.view(bs, c, ps, ps, nh, nw)


    def __init__(self, patch_size, strides=1, scale=1., spatial_reduction='mean', batch_reduction='mean', pad_image=False, normalize_patch='none', ignore_patch_norm=False, device=torch.device("cpu")):
        super(PatchLoss, self).__init__()
        self.patch_size = (patch_size, patch_size) if type(patch_size) == int else patch_size
        self.strides = strides
        self.scale = scale
        w_np = np.ones((1, 1, self.patch_size[0], self.patch_size[1]))
        self.sum_w = torch.from_numpy(w_np).to(device).float()
        self.sum_w.requires_grad_ = False
        self.spatial_reduction = get_reduction_fn(spatial_reduction)
        self.batch_reduction = get_reduction_fn(batch_reduction)
        self.pad_image = pad_image
        self.padding = 0 if not self.pad_image else (self.patch_size[0] // 2)
        self.normalize_patch = normalize_patch
        self.unfolder = torch.nn.Unfold(kernel_size=patch_size, stride=strides, padding=self.padding)
        self.ignore_patch_norm = ignore_patch_norm
        self.name = f"PatchLoss(p={patch_size})"

    def _channel_reduction_op(self, diffs):
        raise NotImplementedError()

    def _post_patch_summation(self, patch_sums):
        raise NotImplementedError()

    def forward(self, x, y):
        x = x * self.scale
        y = y * self.scale
        s = x.shape[2]
        if self.normalize_patch == 'none' and not self.ignore_patch_norm:
            xy_elem_diff = (x - y) ** 2
            xy_elem_diff = self._channel_reduction_op(xy_elem_diff)
            patch_diff_sums = self._sum_over_patches(xy_elem_diff)
            patch_losses = self._post_patch_summation(patch_diff_sums)
        else:
            norm_dims = (2, 3) if self.normalize_patch == 'channel_mean' else (1, 2, 3)
            def prepare_patches(t: torch.Tensor):
                t = self._extract_patches(t)
                if self.normalize_patch != 'none':
                    t = t - t.mean(dim=norm_dims, keepdim=True)
                if self.ignore_patch_norm:
                    t = t / safe_sqrt(t.pow(2).sum(dim=(1,2,3), keepdim=True))
                return t
            x_patches = prepare_patches(x)
            y_patches = prepare_patches(y)
            xy_elem_diff = (x_patches - y_patches) ** 2
            xy_elem_diff = self._channel_reduction_op(xy_elem_diff)
            patch_diff_sums = torch.sum(xy_elem_diff, dim=(2,3))
            patch_losses = self._post_patch_summation(patch_diff_sums)
        patch_losses = self.spatial_reduction(patch_losses, dim=(1, 2, 3))
        return self.batch_reduction(patch_losses)


class PatchRBFLoss(PatchLoss):
    def _channel_reduction_op(self, diffs):
        return diffs.sum(dim=1, keepdims=True)

    def _post_patch_summation(self, patch_sums):
        return 1 - 1 * (patch_sums / (-2 * self.sigma ** 2)).exp()

    def __init__(self, patch_size, strides=1, scale=1., spatial_reduction='mean', batch_reduction='mean', sigma=0.5, pad_image=False, **patch_loss_kwargs):
        super(PatchRBFLoss, self).__init__(patch_size=patch_size, strides=strides, scale=scale, spatial_reduction=spatial_reduction, batch_reduction=batch_reduction, pad_image=pad_image, **patch_loss_kwargs)
        self.sigma = sigma * self.patch_size[0] * self.patch_size[1]
        self.name = f"PatchLoss(p={patch_size},s={sigma})"


class PatchRBFLaplacianLoss(PatchLoss):
    def _channel_reduction_op(self, diffs):
        return diffs.sum(dim=1, keepdims=True)

    def _post_patch_summation(self, patch_sums):
        patch_sums = safe_sqrt(patch_sums)
        return 1 -1 * (patch_sums / (-1 * self.sigma)).exp()

    def __init__(self, patch_size, strides=1, scale=1., spatial_reduction='mean', batch_reduction='mean', sigma=0.5, pad_image=False, **patch_loss_kwargs):
        super(PatchRBFLaplacianLoss, self).__init__(patch_size=patch_size, strides=strides, scale=scale, spatial_reduction=spatial_reduction, batch_reduction=batch_reduction, pad_image=pad_image, **patch_loss_kwargs)
        self.sigma = sigma * self.patch_size[0] * self.patch_size[1]


def safe_sqrt(tensor: torch.Tensor):
    return tensor.clamp(min=1e-30).sqrt()


if __name__ == '__main__':
    loss = PatchRBFLoss(3, batch_reduction='none')
    x = torch.ones((1, 3, 64, 64), dtype=float)*2
    y = torch.ones((4, 3, 64, 64), dtype=float)*5
    z = loss(x, y)
    print(z)
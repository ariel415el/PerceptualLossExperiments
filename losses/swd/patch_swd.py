import torch

from losses.composite_losses.compare_patch_distribution import PatchdistributionsLoss


def compute_swd(x, y, num_proj=256):
    """Compute a Sliced Wasserstein distance between two equal size sets of vectors using num_proj projections"""
    assert len(x.shape) == len(y.shape) and len(y.shape) == 2
    rand = torch.randn(x.size(1), num_proj).to(x.device)  # (slice_size**2*ch)
    rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
    # projection into (batch-zie, num_projections)
    proj1 = torch.matmul(x, rand)
    proj2 = torch.matmul(y, rand)

    # sort by first dimension means each column is sorted separately
    proj1, _ = torch.sort(proj1, dim=0)
    proj2, _ = torch.sort(proj2, dim=0)
    # proj1 = proj1[:proj2.shape[0]]
    d = torch.abs(proj1 - proj2)
    return torch.mean(d)


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, n_samples=None, num_proj=256, sample_same_locations=True,
                 batch_reduction='mean', normalize_patch='none'):
        super(PatchSWDLoss, self).__init__()
        patch_metric = lambda x,y: compute_swd(x,y, num_proj)
        self.loss = PatchdistributionsLoss(patch_metric, patch_size, stride, n_samples, sample_same_locations, batch_reduction, normalize_patch)
        self.name = f"PatchSWD(p-{patch_size}:{stride})"

    def forward(self, x, y):
        return self.loss(x, y)

class FastPatchSWDLoss(torch.nn.Module):
    def _get_w_b(self, n_channels):
        self.w = torch.randn(self.r, n_channels, self.ksize, self.ksize) / self.sigma
        if self.normalize_patch == 'channel_mean':
            self.w -= self.w.mean(dim=(2, 3), keepdim=True)
        elif self.normalize_patch == 'mean':
            self.w -= self.w.mean(dim=(1, 2, 3), keepdim=True)
        self.b = torch.rand(self.r) * (2 * np.pi)
        return self.w, self.b

    def __init__(self, patch_size=7, stride=1, n_samples=None, num_proj=256, sample_same_locations=True,
                 batch_reduction='mean', normalize_patch='none'):
        super(FastPatchSWDLoss, self).__init__()
        self.kernel = torch.randn(self.r, n_channels, self.ksize, self.ksize) / self.sigma

        self.name = f"PatchSWD(p-{patch_size}:{stride})"

    def forward(self, x, y):
        return self.loss(x, y)



if __name__ == '__main__':
    x = torch.ones((5, 3, 64, 64))
    y = torch.ones((5, 3, 64, 64)) * 3
    loss = PatchSWDLoss(batch_reduction='none')

    print(loss(x, y))

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
    d = torch.abs(proj1 - proj2)
    return torch.mean(d)


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, n_samples=None, num_proj=256, sample_same_locations=True, batch_reduction='mean'):
        super(PatchSWDLoss, self).__init__()
        patch_metric = lambda x,y: compute_swd(x,y, num_proj)
        self.loss = PatchdistributionsLoss(patch_metric, patch_size, stride, n_samples, sample_same_locations, batch_reduction)
        self.name = f"PatchSWD(p-{patch_size}:{stride})"

    def forward(self, x, y):
        return self.loss(x, y)


if __name__ == '__main__':
    x = torch.ones((5, 3, 64, 64))
    y = torch.ones((5, 3, 64, 64)) * 3
    loss = PatchSWDLoss(batch_reduction='none')

    print(loss(x, y))

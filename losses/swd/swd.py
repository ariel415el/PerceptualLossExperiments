import torch


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


def extract_patches(x, patch_size=7, stride=1):
    """Extract normalized patches from an image"""
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 3, patch_size, patch_size)
    x_std, x_mean = torch.std_mean(x_patches, dim=(0, 1, 3, 4), keepdim=True)
    x_patches = (x_patches - x_mean) / (x_std + 1e-8)
    return x_patches.view(b, -1, 3 * patch_size ** 2)


def compute_patch_swd(x, y, num_proj=256, patch_size=7, stride=1, n_samples=None):
    """Compute a SWD between distribution of c x patch_size x patch_size patches from both feature-maps / images"""
    results = []
    b, c, h, w = x.shape
    # patches are of size (b, k x k x 3, num_patches)
    x_patches = extract_patches(x, patch_size, stride)
    y_patches = extract_patches(y, patch_size, stride)

    if n_samples:
        indices = torch.randperm(x_patches.shape[1])[:n_samples]
        x_patches = x_patches[:, indices]
        y_patches = y_patches[:, indices]

    # y_patches = self.unfold(y).view(b, -1, 3*self.k**2)
    for i in range(b):
        results.append(compute_swd(x_patches[i], y_patches[i], num_proj))

    return torch.stack(results)


class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, n_samples=None, num_proj=256, batch_reduction='mean'):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_samples = n_samples
        self.num_proj = num_proj
        self.batch_reduction = batch_reduction
        self.name = f"PatchSWD(p-{patch_size}-{stride})"

    def forward(self, x, y):
        results = compute_patch_swd(x, y, patch_size=self.patch_size, stride=self.stride, n_samples=self.n_samples, num_proj=self.num_proj)
        # results *= 1e3
        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results


if __name__ == '__main__':
    x = torch.ones((5, 3, 64, 64))
    y = torch.ones((5, 3, 64, 64)) * 3
    loss = PatchSWDLoss(batch_reduction='none')

    print(loss(x, y))

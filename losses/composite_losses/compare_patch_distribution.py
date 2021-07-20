import torch


def extract_patches(x, patch_size, stride, normalize_patch):
    """Extract normalized patches from an image"""
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 3, patch_size, patch_size)
    if normalize_patch != 'none':
        dims = (0, 1, 2, 3, 4) if normalize_patch == 'mean' else (0, 1, 3, 4)
        x_std, x_mean = torch.std_mean(x_patches, dim=dims, keepdim=True)
        x_patches = (x_patches - x_mean)
        # x_patches /= (x_std + 1e-8)
    return x_patches.view(b, -1, 3 * patch_size ** 2)


def apply_ditribution_metric_to_patches(x, y, dist_metric, patch_size, stride, n_samples=None, sample_same_locations=True, normalize_patch='mean'):
    """Compute a SWD between distribution of c x patch_size x patch_size patches from both feature-maps / images"""
    results = []
    b, c, h, w = x.shape
    # patches are of size (b, k x k x 3, num_patches)
    x_patches = extract_patches(x, patch_size, stride, normalize_patch)
    y_patches = extract_patches(y, patch_size, stride, normalize_patch)

    if n_samples:
        indices = torch.randperm(x_patches.shape[1])[:n_samples]
        x_patches = x_patches[:, indices]
        if sample_same_locations:
            indices = torch.randperm(y_patches.shape[1])[:n_samples]
        y_patches = y_patches[:, indices]

    for i in range(b):
        results.append(dist_metric(x_patches[i], y_patches[i]))

    return torch.stack(results)


class PatchdistributionsLoss(torch.nn.Module):
    def __init__(self, dist_metric, patch_size=7, stride=1, n_samples=None, sample_same_locations=True, batch_reduction='mean', normalize_patch='none'):
        super(PatchdistributionsLoss, self).__init__()
        self.dist_metric = dist_metric
        self.patch_size = patch_size
        self.stride = stride
        self.n_samples = n_samples
        self.batch_reduction = batch_reduction
        self.sample_same_locations = sample_same_locations
        self.normalize_patch = normalize_patch
        self.name = f"PDistLoss(p-{patch_size}-{stride})"

    def forward(self, x, y):
        results = apply_ditribution_metric_to_patches(x, y, self.dist_metric,
                                                      patch_size=self.patch_size,
                                                      stride=self.stride,
                                                      n_samples=self.n_samples,
                                                      normalize_patch=self.normalize_patch)
        # results *= 1e3
        if self.batch_reduction == 'mean':
            return results.mean()
        else:
            return results

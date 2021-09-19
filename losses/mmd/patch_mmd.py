import numpy as np
import torch

import losses
from losses.composite_losses.compare_patch_distribution import PatchdistributionsLoss


def get_distance_matrix(X):
    XX = torch.matmul(X, X.t())

    X_norms = torch.sum(X ** 2, 1, keepdim=True)

    # exp[a,b] = (X[a] @ X[a])^2 -2(X[a] @ X[b]) + (X[b] @ X[b])^2 = || X[a] - X[b] ||^2
    return X_norms - 2 * XX + X_norms.t()


class MultiBandWitdhRbfKernel:
    def __init__(self, sigmas=None):
        self.name = '-MultiBandWitdhRbfKernel'
        if sigmas is None:
            self.sigmas = [2, 5, 10, 20, 40, 80]
        else:
            self.sigmas = sigmas

    def __call__(self, X, S, **kwargs):
        squared_l2_dist_mat = get_distance_matrix(X)
        loss = 0
        for s in self.sigmas:
            rbf_gram_matrix = torch.exp(squared_l2_dist_mat / (-2 * s ** 2))
            # rbf_gram_matrix = torch.exp(1.0 / v * squared_l2_dist_mat)
            loss += torch.sum(S * rbf_gram_matrix)
        # return torch.sqrt(loss)
        return loss


class DotProductKernel:
    def __init__(self):
        self.name = '-DotProductKernel'

    def __call__(self, X, S, **kwargs):
        XX = torch.matmul(X, X.t())
        loss = torch.sum(S * XX)
        return loss


class SSIMKernel:
    def __init__(self, win_size=11, win_sigma=1.5, K=(0.01, 0.03),):
        from losses.ssim.SSIM import _fspecial_gauss_1d
        self.win_size = win_size
        self.K = K
        self.win = _fspecial_gauss_1d(win_size, win_sigma).reshape(-1)
        self.name = '-SSIMKernel'

    def __call__(self, X, S, **kwargs):

        self.win = self.win.to(X.device)
        C1 = self.K[0] ** 2
        C2 = self.K[1] ** 2

        X = (X + 1) / 2
        X = X.reshape(X.shape[0], 3, -1)
        mus = (X * self.win).sum(-1)
        mus_sq = mus.pow(2)
        simgas = (X*X * self.win).sum(-1) - mus_sq

        simgas_sum = simgas[:,None] + simgas[None,:]
        mus_sum = mus_sq[:,None] + mus_sq[None,:]
        mus_mult = mus_sq[:,None] * mus_sq[None,:]
        sigmas_xy = (X[:,None] * X[None,:] * self.win).sum(-1) - mus_mult

        cs_map = (2 * sigmas_xy + C2) / (simgas_sum + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mus_mult + C1) / (mus_sum + C1))

        ssim_map *= cs_map

        return ssim_map.mean(-1) * S


def get_scale_matrix(M, N):
    """
    return an (N+M)x(N+M) matrix where the the TL and BR NxN and MxM blocks are 1/N^2 and 1/M^2 equivalently
    and the other two blocks are -1/NM
    """
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    s = torch.cat((s1, s2), 0)
    return torch.matmul(s, s.t())


def compute_MMD(x_patches, y_patches, kernel):
    """
    :param x_patches:
    :param y_patches:
    :param kernel: a function f: (M+N,3xpxp) -> (M+N, M+N) appliess a kernel for all ofh the (M+N)x(M+N) pairs of patches
    """
    # Compute signed scale matrix to sum up the right entries in the gram matrix for MMD loss
    M = x_patches.size()[0]
    N = y_patches.size()[0]
    S = get_scale_matrix(M, N).to(x_patches.device)
    all_patches = torch.cat((x_patches, y_patches), 0)
    # S[:N,:N] *= 0.9
    # S[N:,N:] = 0
    return kernel(all_patches, S)


class PatchMMD(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, n_samples=None, sample_same_locations=True, batch_reduction='mean',
                 normalize_patch='none'):
        super(PatchMMD, self).__init__()

        patch_metric = lambda x, y: compute_MMD(x, y, self.kernel)
        self.loss = PatchdistributionsLoss(patch_metric, patch_size, stride, n_samples, sample_same_locations,
                                           batch_reduction, normalize_patch)
        self.name = f"PatchMMD{self.kernel.name if self.kernel else ''}(p-{patch_size}:{stride})"

    def forward(self, x, y):
        if not self.kernel:
            raise NotImplementedError
        else:
            return self.loss(x, y)


class PatchMMD_RBF(PatchMMD):
    def __init__(self, patch_size=7, stride=1, n_samples=None, sample_same_locations=True, batch_reduction='mean',
                 normalize_patch='none', sigmas=None):
        if sigmas == None:
            sigmas = [0.1, 0.05, 0.025, 0.01]
        sigmas = np.array(sigmas) * patch_size ** 2
        self.kernel = MultiBandWitdhRbfKernel(sigmas)

        super(PatchMMD_RBF, self).__init__(patch_size=patch_size, stride=stride, n_samples=n_samples,
                                              sample_same_locations=sample_same_locations,
                                              batch_reduction=batch_reduction,
                                              normalize_patch=normalize_patch)


class PatchMMD_DotProd(PatchMMD):
    def __init__(self, patch_size=7, stride=1, n_samples=None, sample_same_locations=True, batch_reduction='mean',
                 normalize_patch='none'):
        self.kernel = DotProductKernel()

        super(PatchMMD_DotProd, self).__init__(patch_size=patch_size, stride=stride, n_samples=n_samples,
                                              sample_same_locations=sample_same_locations,
                                              batch_reduction=batch_reduction,
                                              normalize_patch=normalize_patch)


class PatchMMD_SSIM(PatchMMD):
    def __init__(self, patch_size=11, stride=1, n_samples=None, sample_same_locations=True, batch_reduction='mean',
                 normalize_patch='none'):
        self.kernel = SSIMKernel(win_size=patch_size)

        super(PatchMMD_SSIM, self).__init__(patch_size=patch_size, stride=stride, n_samples=n_samples,
                                              sample_same_locations=sample_same_locations,
                                              batch_reduction=batch_reduction,
                                              normalize_patch=normalize_patch)


if __name__ == '__main__':
    from tqdm import tqdm
    from time import time
    for loss in [
        PatchMMD_RBF(patch_size=11, stride=5),
        PatchMMD_SSIM(patch_size=11, stride=5),
        losses.MMDApproximate(patch_size=11, strides=5, r=128),
        losses.PatchSWDLoss(patch_size=11, stride=5),
    ]:

        x = torch.randn((16, 3, 128, 128)).cuda()
        y = torch.ones(16, 3, 128, 128).cuda()
        print(loss(x, y))

        start = time()
        for i in range(10):
            loss(x, y)
        print(f"{loss.name}: {(time()-start)/10} s per inference")

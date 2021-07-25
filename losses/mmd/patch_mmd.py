import numpy as np
import torch

from losses.composite_losses.compare_patch_distribution import PatchdistributionsLoss


def get_distance_matrix(X):
    XX = torch.matmul(X, X.t())

    X_norms = torch.sum(X ** 2, 1, keepdim=True)

    # exp[a,b] = (X[a] @ X[a])^2 -2(X[a] @ X[b]) + (X[b] @ X[b])^2 = || X[a] - X[b] ||^2
    return X_norms - 2 * XX + X_norms.t()


def multi_bandwitdh_rbf_kernel(X, S, sigmas=None):
    if sigmas is None:
        sigmas = [2, 5, 10, 20, 40, 80]
        # sigmas = [0.5, 0.2, 0.1, 0.05, 0.025, 0.0125]
    squared_l2_dist_mat = get_distance_matrix(X)
    loss = 0
    for s in sigmas:
        rbf_gram_matrix = torch.exp(squared_l2_dist_mat / (-2 * s**2))
        # rbf_gram_matrix = torch.exp(1.0 / v * squared_l2_dist_mat)
        loss += torch.sum(S * rbf_gram_matrix)
    # return torch.sqrt(loss)
    return loss


def dot_product_kernel(X, S):
    XX = torch.matmul(X, X.t())
    loss = torch.sum(S * XX)
    return loss


def get_scale_matrix(M, N):
    """
    return an (N+M)x(N+M) matrix where the the TL and BR NxN and MxM blocks are 1/N^2 and 1/M^2 equivalently
    and the other two blocks are -1/NM
    """
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    s = torch.cat((s1, s2), 0)
    return torch.matmul(s, s.t())


def compute_MMD(x, y, sigmas):
    # Compute signed scale matrix to sum up the right entries in the gram matrix for MMD loss
    M = x.size()[0]
    N = y.size()[0]
    S = get_scale_matrix(M, N).to(x.device)

    # return dot_product_kernel(X, S)
    return multi_bandwitdh_rbf_kernel(torch.cat((x, y), 0), S, sigmas)


class PatchMMDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, n_samples=None, sample_same_locations=True, sigmas=None,
                 batch_reduction='mean', normalize_patch='none'):
        super(PatchMMDLoss, self).__init__()
        if sigmas == None:
            sigmas = [0.1, 0.05, 0.025, 0.01]
        sigmas = np.array(sigmas) * patch_size**2
        patch_metric = lambda x, y: compute_MMD(x, y, sigmas)
        self.loss = PatchdistributionsLoss(patch_metric, patch_size, stride, n_samples, sample_same_locations, batch_reduction, normalize_patch)
        self.name = f"PatchMMD(p-{patch_size}-{stride})"

    def forward(self, x, y):
        return self.loss(x, y)

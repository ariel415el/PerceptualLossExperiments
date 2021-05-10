import torch


def get_distance_matrix(X):
    XX = torch.matmul(X, X.t())

    X_norms = torch.sum(X ** 2, 1, keepdim=True)

    # exp[a,b] = (X[a] @ X[a])^2 -2(X[a] @ X[b]) + (X[b] @ X[b])^2 = || X[a] - X[b] ||^2
    return X_norms - 2 * XX + X_norms.t()


def multi_bandwitdh_rbf_kernel(X, S):
    sigmas = [2, 5, 10, 20, 40, 80]
    squared_l2_dist_mat = get_distance_matrix(X)
    loss = 0
    for v in sigmas:
        rbf_gram_matrix = torch.exp(squared_l2_dist_mat / (-2 * v))
        # rbf_gram_matrix = torch.exp(1.0 / v * squared_l2_dist_mat)
        loss += torch.sum(S * rbf_gram_matrix)
    return torch.sqrt(loss)


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


class MMD(torch.nn.Module):
    def __init__(self):
        super(MMD, self).__init__()
        self.name = f"MMD"

    def forward(self, output, target):
        """
        Computes the sample estimator for MMD between two sample sets for various sigmas, sums the resutls
        """
        X = torch.cat((output.reshape(output.shape[0], -1), target.reshape(target.shape[0], -1)), 0)

        # Compute signed scale matrix to sum up the right entries in the gram matrix for MMD loss
        M = output.size()[0]
        N = target.size()[0]
        S = get_scale_matrix(M, N).to(output.device)

        # return dot_product_kernel(X, S)
        return multi_bandwitdh_rbf_kernel(X, S)
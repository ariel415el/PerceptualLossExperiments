import torch


class L2(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(L2, self).__init__()
        self.name = 'L2'
        self.batch_reduction = batch_reduction

    def __call__(self, x, y):
        dists = (x - y).pow(2)
        if self.batch_reduction == 'mean':
            return dists.mean()
        else:
            return dists.view(x.shape[0], -1).mean(1)

class L1(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(L1, self).__init__()
        self.name = 'L1'
        self.batch_reduction = batch_reduction

    def __call__(self, x, y):
        dists = torch.abs(x - y)
        if self.batch_reduction == 'mean':
            return dists.mean()
        else:
            return dists.view(x.shape[0], -1).mean(1)
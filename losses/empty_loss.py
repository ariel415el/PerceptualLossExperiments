import torch


class NoLoss(torch.nn.Module):
    def __init__(self):
        super(NoLoss, self).__init__()
        self.name = 'No-loss'

    def __call__(self, x, y):
        return 0
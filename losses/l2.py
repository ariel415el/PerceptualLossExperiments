import torch


class L2(torch.nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.name = 'L2'
        self.l = torch.nn.MSELoss()

    def __call__(self, x, y):
        return self.l(x, y)
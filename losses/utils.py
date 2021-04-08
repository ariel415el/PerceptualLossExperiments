import numpy as np
import torch

class ListOfLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super(ListOfLosses, self).__init__()
        self.losses = torch.nn.ModuleList(losses)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones((len(losses),))

        self.name = "#".join([x.name for x in losses])

    def forward(self, x, y):
        return sum([self.losses[i](x,y) * self.weights[i] for i in range(len(self.losses))])
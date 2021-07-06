import torch


class LossesList(torch.nn.Module):
    def __init__(self, losses, weights, name=None):
        super(LossesList, self).__init__()
        self.weights = weights

        self.losses = torch.nn.ModuleList(losses)

        self.name = name if name else "+".join([f"{w}*{l.name}" for l,w in zip(self.losses, self.weights)])

    def forward(self, x, y):
        return sum([self.losses[i](x, y) * self.weights[i] for i in range(len(self.losses))])
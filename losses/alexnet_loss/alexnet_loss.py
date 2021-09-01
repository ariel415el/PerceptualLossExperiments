import os

import torch
from torch import nn


class AlexNetLoss(nn.Module):
    def __init__(self, pretrained=True, n_maxpools=5, batch_reduction='none'):
        super(AlexNetLoss, self).__init__()
        self.batch_reduction = batch_reduction
        self.name = f'AlexNet_{n_maxpools}' + ('_pt' if pretrained else '_rand')
        layers = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (256 x 6x6)
        ][:n_maxpools]
        self.features = nn.Sequential(*layers)

        if pretrained:
            state_dict = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "alexnet-owt-7be5be79.pth"))
            state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
            self.load_state_dict(state_dict)

    def get_features(self, x):
        # layer_indices = [1, 4, 7, 9, 11]
        layer_indices = [0, 3, 6, 8, 10]
        activations = dict()
        for i, f in enumerate(self.features):
            x = f(x)
            if i in layer_indices:
                activations[f"conv{1 +layer_indices.index(i)}"] = x.clone()

        return activations

    def forward(self, x, y):
        x_features = self.get_features(x)
        # x_features.update({"pixels":x})
        y_features = self.get_features(y)
        # y_features.update({"pixels":y})
        loss = 0
        for k in x_features.keys():
            loss += ((x_features[k] - y_features[k])**2).view(x.shape[0], -1).mean(1)

        if self.batch_reduction == 'mean':
            loss = loss.mean()

        return loss


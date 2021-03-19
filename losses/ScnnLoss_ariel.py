import torch
import torch.nn as nn
import torch.nn.functional


class ScnnLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.features = SCNNNetwork()

    def forward(self, x, y):
        f1 = self.features(x)[0]
        f2 = self.features(y)[0]
        return self.l2(f1, f2)


def normalize_batch(batch):
    if batch.shape[1] == 1:
        batch = batch.repeat((1,3,1,1))
    # normalize using imagenet mean and std assuming input in range [-1,1]
    if normalize_batch.mean is None:
        normalize_batch.mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        normalize_batch.std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    batch = batch + 1
    batch = batch / 2
    return (batch - normalize_batch.mean) / normalize_batch.std


class SCNNNetwork(nn.Module):
    def __init__(self,
                 n_channels=3,
                 hdim=1024,
                 ksize=3,
                 strides=1,
                 pool_size=32,
                 pool_strides=16):
        super().__init__()
        self.normalize_spatial_filter = False

        self.spatial_conv = nn.Conv2d(n_channels, hdim, ksize, stride=strides, padding=ksize // 2)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(hdim, hdim, 1)
        self.conv_2 = nn.Conv2d(hdim, hdim, 1)
        self.conv_3 = nn.Conv2d(hdim, hdim, 1)
        self.conv_4 = nn.Conv2d(hdim, hdim, 1)
        self.conv_5 = nn.Conv2d(hdim, hdim, 1)
        self.conv_6 = nn.Conv2d(hdim, hdim, 1)
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides)
        self.l2 = nn.MSELoss()

        self.initialize()

    def get_activations(self, x):
        x = self.spatial_conv(self.relu(x))
        x = self.conv_1(self.relu(x))
        x = self.conv_2(self.relu(x))
        x = self.conv_3(self.relu(x))
        x = self.conv_4(self.relu(x))
        x = self.conv_5(self.relu(x))
        x = self.conv_6(self.relu(x))
        x = self.avg_pool(x)
        return x

    def forward(self, x, y):
        f1 = self.get_activations(x)
        f2 = self.get_activations(y)
        return self.l2(f1, f2)

    def initialize(self,):
        a = 0

        if self.normalize_spatial_filter:
            self.spatial_conv.weight.data -= torch.mean(self.spatial_conv.weight.data, dim=(2, 3), keepdim=True)
        nn.init.kaiming_normal_(self.conv_1.weight, a=a)
        nn.init.zeros_(self.conv_1.bias.data)
        nn.init.kaiming_normal_(self.conv_2.weight, a=a)
        nn.init.zeros_(self.conv_2.bias.data)
        nn.init.kaiming_normal_(self.conv_3.weight, a=a)
        nn.init.zeros_(self.conv_3.bias.data)
        nn.init.kaiming_normal_(self.conv_4.weight, a=a)
        nn.init.zeros_(self.conv_4.bias.data)
        nn.init.kaiming_normal_(self.conv_5.weight, a=a)
        nn.init.zeros_(self.conv_5.bias.data)
        nn.init.kaiming_normal_(self.conv_6.weight, a=a)
        nn.init.zeros_(self.conv_6.bias.data)


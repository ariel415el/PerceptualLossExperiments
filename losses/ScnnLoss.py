import torch
import torch.nn as nn
import torch.nn.functional


class ScnnLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.features = FeaturesNetwork(SCNNNetwork())

    def forward(self, x, y):
        return self.l2(self.features(x)[0], self.features(y)[0])


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


class FeaturesNetwork(torch.nn.Module):
    """
    General loss module for netowrk features based loss function
    """
    def reinitialize(self):
        self.base_network.reinitialize()

    def __init__(self, base_network: torch.nn.Module, layers=[14], normalize_input=False, ignore_patch_norm=True):
        """

        :param base_network: should be a module for which running forward iteratively through its named children gives
        the same value as running forward directly.
        :param layers: list of sub modules (by indices or names) for which outputs will be collected during forward pass.
        """
        super().__init__()
        self.base_network = base_network
        self.layers = []
        for i, (child_name, child) in enumerate(self.base_network.named_children()):
            print(i)
            for l in layers:
                if (isinstance(l, int) and l == i) or (isinstance(l, str) and l == child_name):
                    self.layers.append(i)
        self.max_layer = max(self.layers)
        self.num_features = len(self.layers)
        self.normalize_input = normalize_input
        self.ignore_patch_norm = ignore_patch_norm

    def forward(self, x):
        if self.normalize_input:
            x = normalize_batch(x)
        features = []
        for i, l in enumerate(self.base_network.children()):
            if i == 0 and self.ignore_patch_norm: # assumes first layer is convolution!
                norm_kernel = torch.ones((1, x.shape[1], l.kernel_size[0], l.kernel_size[1]))
                if norm_kernel.device != x.device:
                    norm_kernel = norm_kernel.to(x.device)
                xx = torch.nn.functional.conv2d(x.pow(2), norm_kernel, padding=l.kernel_size[0] // 2)
                xx_sqrt = xx.clamp(min=1e-20).pow(0.5)
                x = l(x)
                x = x / xx_sqrt
            else:
                x = l(x)
            if i > self.max_layer:
                break

            if i in self.layers:
                features.append(x)
        return tuple(features)


class SCNNNetwork(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=1,
                 hdim=1024,
                 ksize=3,
                 n_local_layers=6,
                 activation="relu",
                 activation_kwargs={},
                 normalize_spatial_filter=False,
                 strides=1,
                 pool_size=32,
                 pool_strides=16):
        super().__init__()
        spatial_conv = nn.Conv2d(n_channels, hdim, ksize, stride=strides, padding=ksize // 2)
        layers = [spatial_conv]
        for i in range(n_local_layers):
            layers.append(nn.ReLU())
            out_dim = hdim
            in_dim = hdim
            layers.append(nn.Conv2d(in_dim, out_dim, 1))
        layers.append(nn.ReLU())

        if pool_size is not None:
            layers.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides))
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hdim, out_features=n_classes))
        for i, l in enumerate(layers):
            setattr(self, "layer{}".format(i), l)
        self.layers = layers
        self.ksize = ksize
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.normalize_spatial_filter = normalize_spatial_filter

        self.initialize()

    def forward(self, x):
        for i, l in self.children():
            x = l(x)
        return x

    def initialize(self, gain=None):
        a = 0
        for i, layer in enumerate(self.children()):
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                if i < (len(self.layers) - 1):
                    nn.init.kaiming_normal_(layer.weight, a=a)
                    if i == 0 and self.normalize_spatial_filter:
                        layer.weight.data -= torch.mean(layer.weight.data, dim=(2, 3), keepdim=True)
                else:
                    nn.init.xavier_normal_(layer.weight, gain=gain if gain is not None else 1.)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias.data)

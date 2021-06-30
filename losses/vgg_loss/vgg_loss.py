import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from losses.swd.swd import compute_swd
from losses.vgg_loss.blur_pool import MaxBlurPool
from losses.vgg_loss.contextual_loss import contextual_loss
from losses.vgg_loss.gram_loss import gram_loss, gram_trace_loss

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'AP1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'AP2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'AP3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'AP4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'AP5', ]


def layer_names_to_indices(names):
    return [layer_names.index(n) for n in names]


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def get_features_metric(features_metric_name, **kwargs):
    if features_metric_name == 'l2':
        return lambda x, y: ((x - y) ** 2).view(x.shape[0], -1).mean(1)
    elif features_metric_name == 'l1':
        return lambda x, y: torch.abs(x - y).view(x.shape[0], -1).mean(1)
    elif features_metric_name == 'dot':
        return lambda x, y: 1 - torch.bmm(x.view(x.shape[0], 1, -1), y.view(x.shape[0], -1, 1))  # TODO normalize
    elif features_metric_name == 'gram':
        return gram_loss
    elif features_metric_name == 'gram_trace':
        return gram_trace_loss
    elif features_metric_name == 'cx':
        return lambda x, y: contextual_loss(x, y, **kwargs)
    elif features_metric_name == 'l1+gram':
        return lambda x, y: gram_loss(x, y) * 0.5 + torch.abs(x - y).view(x.shape[0], -1).mean(1)
    elif features_metric_name == 'swd':
        def features_swd(x, y):
            b, c, h, w = x.shape
            x = x.view(b, h * w, c)
            y = y.view(b, h * w, c)
            return torch.stack([compute_swd(x[i], y[i], num_proj=5*c) for i in range(b)])
        return features_swd
    else:
        raise ValueError(f"No such feature metric: {features_metric_name}")


class VGGFeatures(nn.Module):
    def __init__(self, pretrained=False, norm_first_conv=False):
        super(VGGFeatures, self).__init__()
        self.pretrained = pretrained
        self.norm_first_conv = norm_first_conv

        features = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                features += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # features += [MaxBlurPool(in_channels)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1)
                features += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*features)

        self.name = 'VGG'

        if pretrained:
            self.name += '-PT'
            weigths = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "vgg16_head.pth"))
            self.load_state_dict(weigths)
        else:
            self.name += '-random' + '-NFC' if self.norm_first_conv else ''
            self.initialize_weights_randomly()

    def initialize_weights_randomly(self):
        i = 0
        for feat in self.features:
            if type(feat) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(feat.weight)
                # torch.nn.init.normal_(feat.weight, 0, 0.015)
                # torch.nn.init.uniform_(feat.weight, -0.1, 0.1)
                if i == 0 and self.norm_first_conv:
                    i += 1
                    feat.weight.data -= torch.mean(feat.weight.data, dim=(2, 3), keepdim=True)
                torch.nn.init.constant_(feat.bias, 0.0)
                # torch.nn.init.constant_(feat.bias, 0.5)

    def get_activations(self, z, normalize=False):
        if normalize:
            normalize_input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            z = normalize_input(z)

        activations = dict()
        for i, f in enumerate(self.features):
            z = f(z)
            activations[layer_names[i]] = z
        return activations


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers_and_weights=None, pretrained=False, reinit=False, norm_first_conv=False,
                 features_metric_name='l1', batch_reduction='mean'):
        super(VGGPerceptualLoss, self).__init__()
        assert not (reinit and pretrained), "better not reinit pretrained weights"
        self.VGGFeatures = VGGFeatures(pretrained, norm_first_conv)
        self.norm_first_conv = norm_first_conv
        self.reinit = reinit
        self.features_metric = get_features_metric(features_metric_name)
        if layers_and_weights:
            self.layers_and_weights = layers_and_weights
        else:
            self.layers_and_weights = [('pixels', 1.0), ('conv1_2', 1.0), ('conv2_2', 1.0), ('conv3_3', 1.0), ('conv4_3', 1.0),
                                       ('conv5_3', 1.0)]
        self.batch_reduction = batch_reduction
        self.name = f"VGG({'_reinit' if reinit else ''}{'_NormFC' if norm_first_conv else ''}{'_PT' if pretrained else ''}_M-{features_metric_name})"

    def forward(self, x, y):
        if self.reinit:
            self.VGGFeatures.initialize_weights_randomly()

        fx = self.VGGFeatures.get_activations(x)
        fy = self.VGGFeatures.get_activations(y)
        fx.update({"pixels": x})
        fy.update({"pixels": y})

        loss = 0
        for layer_name, w in self.layers_and_weights:
            loss += self.features_metric(fx[layer_name], fy[layer_name]) * w

        if self.batch_reduction == 'mean':
            return loss.mean()
        else:
            return loss


if __name__ == '__main__':
    model = VGGFeatures(3)
    model(torch.zeros((8, 3, 64, 64)), torch.ones((8, 3, 64, 64)))

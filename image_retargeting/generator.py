import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        num_layer = 5
        nfc = 32
        min_nfc = 32
        ker_size = 3
        padd_size = 1
        stride = 1
        nc_im = 3
        N = nfc
        self.head = ConvBlock(in_channel=nc_im, out_channel=nfc, ker_size=ker_size, padd=padd_size, stride=stride)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, stride)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, min_nfc), nc_im, kernel_size=ker_size, stride=stride, padding=padd_size),
            nn.Tanh()
        )

    def forward(self, x, y=None):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if y is not None:
            ind = int((y.shape[2] - x.shape[2]) / 2)
            y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
            return x + y
        return x
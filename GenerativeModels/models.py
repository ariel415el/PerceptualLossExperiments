import numpy as np
import torch.nn.init as init
from torch import nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))


class DCGANGenerator(nn.Module):
    def __init__(self, input_dim, channels, output_img_dim=28):
        self.input_dim = input_dim
        self.output_img_dim = output_img_dim
        super(DCGANGenerator, self).__init__()
        if output_img_dim == 64:
            layer_depths = [input_dim, 512, 256, 128, 64]
            kernel_dim = [4, 4, 4, 4, 4]
            strides = [1, 2, 2, 2, 2]
            padding = [0, 1, 1, 1, 1]
        elif output_img_dim == 128:
            layer_depths = [input_dim, 512, 512, 256, 128, 64]
            kernel_dim = [4, 4, 4, 4, 4, 4]
            strides = [1, 2, 2, 2, 2, 2]
            padding = [0, 1, 1, 1, 1, 1]
        else:
            raise ValueError("Image dim supports only 28, 64, 128")
        layers = []
        for i in range(len(layer_depths) - 1):
            layers += [
                nn.ConvTranspose2d(layer_depths[i], layer_depths[i+1], kernel_dim[i], strides[i], padding[i], bias=False),
                nn.BatchNorm2d(layer_depths[i+1]),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1], bias=False),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input)
        return output


class DCGANEncoder(nn.Module):
    def __init__(self, input_img_dim, channels, output_latent_dim):
        super(DCGANEncoder, self).__init__()
        if input_img_dim == 64:
            layer_depth = [channels, 64, 128, 256, 512]
        elif input_img_dim == 128:
            layer_depth = [channels, 64, 128, 256, 512, 512]
        else:
            raise ValueError("Image dim supports only 28, 64, 128")
        layers = []
        for i in range(len(layer_depth)-1):
            layers += [
                nn.Conv2d(layer_depth[i], layer_depth[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(layer_depth[i+1]),
                nn.ReLU(True)
                ]
        layers.append(nn.Conv2d(layer_depth[-1], output_latent_dim, 4, 1, 0, bias=False))
        self.convs = nn.Sequential(*layers)


    def forward(self, input):
            input = input
            output = self.convs(input).view(input.size(0), -1)
            return output


class LatentMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatentMapper, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin1 = nn.Linear(input_dim, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin_out = nn.Linear(128, output_dim, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.lin1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.lin_out(z)
        return z



if __name__ == '__main__':
    ### Optimize the model to output a single specific image
    import torch
    import cv2
    import torchvision.utils as vutils

    target = cv2.imread('/home/ariel/universirty/data/FFHQ/thumbnails128x128/00009.png')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target = cv2.resize(target, (64, 64)) / 255.0
    target = 2 * target - 1  # transform to -1 1
    target = target.transpose(2, 0, 1)
    target = torch.from_numpy(target).unsqueeze(0).float()
    vutils.save_image(target[0], 'generated.png', normalize=True)
    input = torch.randn(64).unsqueeze(0)
    input.requires_grad_(True)

    decoder = DCGANGenerator(64, 3, 128)
    print(sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    encoder = DCGANEncoder(128, 3, 64)
    print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))

    optimizer = torch.optim.SGD(list(decoder.parameters()) + [input], lr=0.01)
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=0.1)

    for i in range(10000):
        if (i+1) % 100 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.8
            print(i, loss.item())
            vutils.save_image(output[0], 'generated.png',normalize=True)
        output = decoder(input)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()



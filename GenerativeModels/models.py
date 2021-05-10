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


class InfoGanGenerator(nn.Module):
    def __init__(self, nz, sz, nc, do_bn=False):
        super(InfoGanGenerator, self).__init__()
        self.sz = sz
        self.dim_im = 128 * (sz // 4) * (sz // 4)
        self.lin_in = nn.Linear(nz, 1024, bias=False)
        self.bn_in = nn.BatchNorm1d(1024)
        self.lin_im = nn.Linear(1024, self.dim_im, bias=False)
        self.bn_im = nn.BatchNorm1d(self.dim_im)

        self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True)
        self.bn_conv = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=True)
        # self.output_nl = nn.Sigmoid()  # to [0,1]
        self.output_nl = nn.Tanh()  # tp [-1,1]
        self.do_bn = do_bn
        # self.nonlin = nn.SELU(True)
        self.nonlin = nn.LeakyReLU(0.2, inplace=True)

    def main(self, z):
        z = self.lin_in(z)
        z = self.nonlin(z)
        z = self.lin_im(z)
        if self.do_bn:
            z = self.bn_im(z)
        z = self.nonlin(z)
        z = z.view(-1, 128, self. sz // 4, self.sz // 4)
        z = self.conv1(z)
        if self.do_bn:
            z = self.bn_conv(z)
        z = self.nonlin(z)
        z = self.conv2(z)
        z = self.output_nl(z)
        return z

    def forward(self, z):
        zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
        z = z.div(zn)
        output = self.main(z)
        return output


class DCGANGenerator(nn.Module):
    def __init__(self, input_dim, channels, output_dim=28):
        self.input_dim = input_dim
        self.output_dim = output_dim
        ngf = 64
        super(DCGANGenerator, self).__init__()
        if output_dim == 28:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(input_dim, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # nn.Sigmoid()

            )
        elif output_dim == 64:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # nn.Sigmoid()

            )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input)
        return output


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

    model = DCGANGenerator(64,3,64)

    optimizer = torch.optim.SGD(list(model.parameters()) + [input], lr=0.01)
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=0.1)

    for i in range(10000):
        if (i+1) % 100 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.8
            print(i, loss.item())
            vutils.save_image(output[0], 'generated.png',normalize=True)
        output = model(input)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()


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
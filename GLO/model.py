import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        init.normal_(m.weight, mean=0, std=0.01)


class _netZ(nn.Module):
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz

    def get_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze()
        return z


class _netG(nn.Module):
    def __init__(self, nz, sz, nc, do_bn=False):
        super(_netG, self).__init__()
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
    def __init__(self, latent_dim, channels, output_dim=28):
        ngf = 64
        super(DCGANGenerator, self).__init__()
        if output_dim == 28:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
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
            )
        elif output_dim == 64:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
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
            )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input)
        return output

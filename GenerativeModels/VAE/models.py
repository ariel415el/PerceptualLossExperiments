import numpy as np
import torch
from torch import nn
from torch.nn import init


class VAEGenerator(nn.Module):
    def __init__(self, input_dim, channels, output_img_dim=28):
        super(VAEGenerator, self).__init__()
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(input_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],kernel_size=3, stride=2, padding=1,  output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class VAEEncoder(nn.Module):
    def __init__(self, input_img_dim, channels, output_latent_dim):
        super(VAEEncoder, self).__init__()
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        in_channels = channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, output_latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, output_latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


class Decoder(nn.Module):
    def __init__(self, nz,  nc, output_img_dim, ndf=32, isize=64):
        super(Decoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
            nn.ReLU(True)
        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf * 4, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf * 2, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.decoder_dense(input).view(
            batch_size, self.ndf * 8, self.out_size, self.out_size)
        output = self.decoder_conv(hidden)
        return output


class Encoder(nn.Module):
    def __init__(self, img_dim, nc, nz, nef=32, isize=64):
        super(Encoder, self).__init__()

        # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef*2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*2),

            nn.Conv2d(nef*2, nef*4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*4),

            nn.Conv2d(nef*4, nef*8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*8)
        )

        # Map the encoded feature map to the latent vector of mean, (log)variance
        out_size = isize // 16
        self.mean = nn.Linear(nef*8*out_size*out_size, nz)
        self.logvar = nn.Linear(nef*8*out_size*out_size, nz)


    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        # Encoded feature map
        hidden = self.encoder(inputs)
        # Reshape
        hidden = hidden.view(batch_size, -1)
        # Calculate mean and (log)variance
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        # latent_z = self.reparametrize(mean, logvar)

        return [mean, logvar]


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.zeros_(m.weight)
        init.zeros_(m.bias)
    elif classname.find('Conv') != -1:
        init.uniform_(m.weight, -0.08, 0.08)
    elif classname.find('Emb') != -1:
        init.uniform_(m.weight, -0.08, 0.08)


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
            # layer_depths = [input_dim, 64, 64, 64, 64, 64]
            kernel_dim = [4, 4, 4, 4, 4, 4]
            strides = [1, 2, 2, 2, 2, 2]
            padding = [0, 1, 1, 1, 1, 1]
        else:
            raise ValueError("Image dim supports only 28, 64, 128")
        layers = []
        for i in range(len(layer_depths) - 1):
            layers += [
                nn.ConvTranspose2d(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i],
                                   bias=False),
                nn.BatchNorm2d(layer_depths[i + 1]),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1], bias=False),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*layers)
        print("DC generator params: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

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
            # layer_depth = [channels, 64, 64, 64, 64, 64]
        else:
            raise ValueError("Image dim supports only 28, 64, 128")
        layers = []
        for i in range(len(layer_depth) - 1):
            layers += [
                nn.Conv2d(layer_depth[i], layer_depth[i + 1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(layer_depth[i + 1]),
                nn.ReLU(True)
            ]
        layers.append(nn.Conv2d(layer_depth[-1], output_latent_dim, 4, 1, 0, bias=False))
        self.convs = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(output_latent_dim, output_latent_dim)
        self.fc_var = nn.Linear(output_latent_dim, output_latent_dim)
        print("DC encoder params: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, input):
        result = self.convs(input).view(input.size(0), -1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
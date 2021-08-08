import os

import numpy as np
import torch.optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import utils as vutils
import torch.nn.init as init
from tqdm import tqdm

from Experiments.all import load_models
from GenerativeModels.config import default_config
from GenerativeModels.utils.data_utils import get_dataset, MemoryDataset, get_dataloader
from losses.classic_losses.grad_loss import GradLoss, GradLoss3Channels
from losses.classic_losses.l2 import L2, L1
from losses.composite_losses.window_loss import WindowLoss
from losses.experimental_patch_losses import MMD_PP
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.patch_loss import PatchRBFLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

device = torch.device("cuda")


# def weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear):
#         init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
#     elif classname.find('Conv') != -1:
#         init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
#     elif classname.find('Linear') != -1:
#         init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
#     elif classname.find('Emb') != -1:
#         init.normal_(m.weight, mean=0, std=0.01)
#
#
# class Encoder(torch.nn.Module):
#     def __init__(self, z_dim):
#         super(Encoder, self).__init__()
#         self.linear1 = torch.nn.Linear(64 ** 2, 128)
#         self.linear2 = torch.nn.Linear(128, 64)
#         self.linear3 = torch.nn.Linear(64, z_dim)
#
#     def forward(self, x):
#         x = x.reshape(-1, 64 ** 2)
#         x = self.linear1(x)
#         x = torch.relu(x)
#         x = self.linear2(x)
#         x = torch.relu(x)
#         x = self.linear3(x)
#         x = torch.relu(x)
#         return x
#
#
# class Generator(torch.nn.Module):
#     def __init__(self, z_dim):
#         super(Generator, self).__init__()
#         self.linear1 = torch.nn.Linear(z_dim, 64)
#         self.linear2 = torch.nn.Linear(64, 128)
#         self.linear3 = torch.nn.Linear(128, 64 ** 2)
#
#     def forward(self, z):
#         x = self.linear1(z)
#         x = torch.relu(x)
#         x = self.linear2(x)
#         x = torch.relu(x)
#         x = self.linear3(x)
#         x = torch.relu(x)
#         x = x.reshape(-1, 1, 64, 64)
#         return x
#
#
# class DCGANGenerator(nn.Module):
#     def __init__(self, input_dim, channels, output_img_dim, n_c):
#         self.input_dim = input_dim
#         self.output_img_dim = output_img_dim
#         super(DCGANGenerator, self).__init__()
#         if output_img_dim == 64:
#             layer_depths = [input_dim, n_c, n_c, n_c, n_c]
#             kernel_dim = [4, 4, 4, 4, 4]
#             strides = [1, 2, 2, 2, 2]
#             padding = [0, 1, 1, 1, 1]
#         layers = []
#         for i in range(len(layer_depths) - 1):
#             layers += [
#                 nn.ConvTranspose2d(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i],
#                                    bias=False),
#                 nn.BatchNorm2d(layer_depths[i + 1]),
#                 nn.ReLU(True),
#             ]
#         layers += [
#             nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1], bias=False),
#             nn.Tanh()
#         ]
#         self.network = nn.Sequential(*layers)
#         print("DC generator params: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
#
#     def forward(self, input):
#         input = input.view(input.size(0), input.size(1), 1, 1)
#         output = self.network(input)
#         return output
#
#
# class DCGANEncoder(nn.Module):
#     def __init__(self, input_img_dim, channels, output_latent_dim, n_c):
#         super(DCGANEncoder, self).__init__()
#         if input_img_dim == 64:
#             layer_depth = [channels, n_c, n_c, n_c, n_c]
#         layers = []
#         for i in range(len(layer_depth) - 1):
#             layers += [
#                 nn.Conv2d(layer_depth[i], layer_depth[i + 1], 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(layer_depth[i + 1]),
#                 nn.ReLU(True)
#             ]
#         layers.append(nn.Conv2d(layer_depth[-1], output_latent_dim, 4, 1, 0, bias=False))
#         self.convs = nn.Sequential(*layers)
#         print("DC encoder params: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
#
#     def forward(self, input):
#         input = input
#         output = self.convs(input).view(input.size(0), -1)
#         return output


def optimize_model(criterion, outputs_dir, train_dataset, z_dim, n_c):
    n = 9
    # encoder = Encoder(z_dim)
    # generator = Generator(z_dim)
    from GenerativeModels.models import DCGANEncoder
    from GenerativeModels.models import DCGANGenerator
    from GenerativeModels.models import weights_init
    encoder = DCGANEncoder(128, 3, z_dim).to(device)
    weights_init(encoder)
    generator = DCGANGenerator(z_dim, 3, 128).to(device)
    weights_init(generator)
    encoder.train()
    generator.train()

    criterion = criterion.to(device)

    debug_images = torch.from_numpy(np.array([train_dataset[i][1] for i in range(n)])).to(device).float()
    vutils.save_image(debug_images, os.path.join(outputs_dir, f"orig.png"), normalize=True, nrow=int(np.sqrt(n)))
    # Define optimizers
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizerE = torch.optim.Adam(encoder.parameters(), lr=0.001)
    dataloader = get_dataloader(dataset, 64, device, drop_last=False, shuffle=True)
    i = 0
    pbar = tqdm()
    losses = []
    for epoch in range(1000):
        for _, images in dataloader:
            images = images.float().to(device)
            if i % 50 == 0:
                recons = generator(encoder(debug_images))
                vutils.save_image(recons, os.path.join(outputs_dir, f"recons{i}.png"), normalize=True, nrow=int(np.sqrt(n)))
                plt.plot(range(len(losses)), losses)
                plt.savefig(os.path.join(outputs_dir, f"loss.png"))
                plt.clf()
            latent_codes = encoder(images)
            fake_images = generator(latent_codes)

            rec_loss = criterion(fake_images, images).mean()
            rec_loss.backward()
            optimizerE.step()
            encoder.zero_grad()
            optimizerG.step()
            generator.zero_grad()
            losses.append(rec_loss.item())
            pbar.set_description(f"loss: {rec_loss}")
            i += 1
            pbar.update(1)

            if (i + 1) % 500 == 0:
                for g, e in zip(optimizerG.param_groups, optimizerE.param_groups):
                    e['lr'] *= 0.75
                    g['lr'] *= 0.75

    torch.save(encoder.state_dict(), f"{outputs_dir}/encoder.pth")
    torch.save(generator.state_dict(), f"{outputs_dir}/generator.pth")


if __name__ == '__main__':
    device = torch.device("cuda")

    # criterion = L2().to(device)
    # criterion = L1().to(device)
    # criterion = GradLoss3Channels().to(device)
    # criterion = PatchRBFLoss(patch_size=11, sigma=0.02).to(device)
    # criterion = MMDApproximate(patch_size=11, pool_size=32, pool_strides=16, sigma=0.02, r=64)
    # criterion = MMD_PP(r=64)
    criterion = VGGPerceptualLoss(pretrained=True)
    # criterion = WindowLoss(PatchMMDLoss(patch_size=3, n_samples=None, sigmas=[0.02]), window_size=32, stride=16)

    outputs_dir = f'optimize_model/{criterion.name}'

    os.makedirs(outputs_dir, exist_ok=True)
    params = default_config

    # paths = [os.path.join('box_dataset', x) for x in os.listdir('box_dataset')]
    paths = [os.path.join('/home/ariel/university/data/FFHQ/thumbnails128x128', x) for x in os.listdir('/home/ariel/university/data/FFHQ/thumbnails128x128')]
    paths = paths[:64]
    dataset = MemoryDataset(paths, resize=128)
    z_dim = 128
    optimize_model(criterion, outputs_dir, dataset, z_dim, 128)

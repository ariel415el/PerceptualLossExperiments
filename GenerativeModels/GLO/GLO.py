import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch import nn as nn
from tqdm import tqdm
from time import time

import GenerativeModels.GLO.utils
import GenerativeModels.utils.data_utils
import os
import matplotlib.pyplot as plt
from GenerativeModels.models import weights_init


class GLOTrainer:
    def __init__(self, glo_params, generator, criterion, dataset, device):
        self.device = device
        self.glo_params = glo_params
        self.generator = generator.to(device)
        self.generator.apply(weights_init)
        self.criterion = criterion.to(device)

        self.dataloader = GenerativeModels.utils.data_utils.get_dataloader(dataset, self.glo_params.batch_size, self.device)

        # Define the learnable input codes
        self.latent_codes = LatentCodesDict(self.glo_params.z_dim, len(dataset)).to(device)
        self.latent_codes.apply(weights_init)

        # self.latent_codes.load_state_dict(torch.load('outputs/batch_3-spinoffs/l2_norm1-50-epochs/latent_codes.pth', map_location=device))
        # self.generator.load_state_dict(torch.load('outputs/batch_3-spinoffs/l2_norm1-50-epochs/generator.pth', map_location=device))

        # Define optimizers
        self.optimizerG = optim.Adam(self.generator.parameters(),
                                     lr=self.glo_params.lr * self.glo_params.generator_lr_factor,
                                     betas=(0.5, 0.999))
        self.optimizerZ = optim.Adam(self.latent_codes.parameters(), lr=self.glo_params.lr, betas=(0.5, 0.999))

        self.name = f"GLO{'-FN' if glo_params.force_norm else ''}" \
                    f"(LR-{glo_params.lr}/{glo_params.decay_rate}/{glo_params.decay_epochs}_BS-{glo_params.batch_size})"

    def train(self, outptus_dir, epochs=200, vis_freq=1):
        start = time()
        loss_means = []
        losses = []
        pbar = tqdm(total=epochs)
        step = 0
        for epoch in range(epochs):
            for indices, images in self.dataloader:
                step += 1
                indices = indices.long().to(self.device)
                images = images.float().to(self.device)

                zi = self.latent_codes(indices)
                fake_images = self.generator(zi)

                rec_loss = self.criterion(fake_images, images).mean()
                rec_loss.backward()

                losses.append(rec_loss.item())

                self.optimizerZ.step()
                self.latent_codes.zero_grad()
                if step % self.glo_params.num_z_steps == 0:
                    self.optimizerG.step()
                    self.generator.zero_grad()

            if epoch % vis_freq == 0:
                self._visualize(epoch, self.dataloader.dataset, outptus_dir)
                loss_means.append(np.mean(losses))
                losses = []
                plot_epoch(loss_means, "Loss", outptus_dir)
                torch.save(self.latent_codes.state_dict(), f"{outptus_dir}/latent_codes.pth")
                torch.save(self.generator.state_dict(), f"{outptus_dir}/generator.pth")
                pbar.set_description(f"Epoch: {epoch}: step {step}, im/sec: {(step + 1) * self.glo_params.batch_size / (time() - start):.2f}")

            if (epoch + 1) % self.glo_params.decay_epochs == 0:
                for g, z in zip(self.optimizerG.param_groups, self.optimizerZ.param_groups):
                    z['lr'] *= self.glo_params.decay_rate
                    g['lr'] *= self.glo_params.decay_rate

    def _visualize(self, epoch, dataset, outptus_dir):
        os.makedirs(outptus_dir, exist_ok=True)
        debug_indices = np.arange(25)
        if epoch == 0:
            os.makedirs(os.path.join(outptus_dir, "imgs", "reconstructions"), exist_ok=True)
            # dumpy original images fro later reconstruction debug images
            first_imgs = torch.from_numpy(dataset[debug_indices][1])
            vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', 'originals.png'), normalize=True, nrow=5)

        # Reconstructed images
        idx = torch.from_numpy(debug_indices).to(self.device)
        Irec = self.generator(self.latent_codes(idx))
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', f"step_{epoch}.png"), normalize=True, nrow=5)


class LatentCodesDict(nn.Module):
    def __init__(self, nz, n):
        super(LatentCodesDict, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz
        torch.nn.init.normal_(self.emb.weight, mean=0, std=0.01)

    def force_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze()
        return z

def endless_iterator(dataloader):
    while True:
        for x in iter(dataloader): yield x

def plot_epoch(arr, y_name, outptus_dir):
    plt.plot(np.arange(len(arr)), arr)
    plt.xlabel("Epoch")
    plt.ylabel(y_name)
    plt.title(f"{y_name} per epoch. last: {arr[-1]:.2f}")
    plt.savefig(os.path.join(outptus_dir, f"{y_name}.png"))
    plt.clf()
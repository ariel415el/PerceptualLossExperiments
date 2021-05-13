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


class AutoEncoderTraniner:
    def __init__(self, params, encoder, generator, criterion, dataset, device):
        self.device = device
        self.params = params
        self.generator = generator.to(device)
        self.encoder = encoder.to(device)
        self.generator.apply(weights_init)
        self.criterion = criterion.to(device)

        self.dataloader = GenerativeModels.utils.data_utils.get_dataloader(dataset, self.params.batch_size, self.device)


        # Define optimizers
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.params.lr, betas=(0.5, 0.999))
        self.optimizerE = optim.Adam(self.encoder.parameters(), lr=self.params.lr, betas=(0.5, 0.999))

    def train(self, outptus_dir, epochs=200, vis_freq=1):
        start = time()
        loss_means = []
        losses = []
        pbar = tqdm(total=epochs)
        step = 0
        for epoch in range(epochs):
            for _, images in self.dataloader:
                step += 1
                images = images.to(self.device).float()
                latent_codes = self.encoder(images)
                fake_images = self.generator(latent_codes)

                rec_loss = self.criterion(fake_images, images).mean()
                rec_loss.backward()

                losses.append(rec_loss.item())

                self.optimizerE.step()
                self.encoder.zero_grad()
                if step % self.params.num_z_steps == 0:
                    self.optimizerG.step()
                    self.generator.zero_grad()

            if epoch % vis_freq == 0:
                self._visualize(epoch, self.dataloader.dataset, outptus_dir)
                loss_means.append(np.mean(losses))
                losses = []
                plot_epoch(loss_means, "Loss", outptus_dir)
                torch.save(self.encoder.state_dict(), f"{outptus_dir}/encoder.pth")
                torch.save(self.generator.state_dict(), f"{outptus_dir}/generator.pth")
                pbar.set_description(f"Epoch: {epoch}: step {step}, im/sec: {(step + 1) * self.params.batch_size / (time() - start):.2f}")

            if (epoch + 1) % self.params.decay_epochs == 0:
                for g, e in zip(self.optimizerG.param_groups, self.optimizerE.param_groups):
                    e['lr'] *= self.params.decay_rate
                    g['lr'] *= self.params.decay_rate

    def _visualize(self, epoch, dataset, outptus_dir):
        os.makedirs(outptus_dir, exist_ok=True)
        debug_indices = np.arange(25)
        first_imgs = torch.from_numpy(dataset[debug_indices][1])
        if epoch == 0:
            os.makedirs(os.path.join(outptus_dir, "imgs", "reconstructions"), exist_ok=True)
            # dumpy original images fro later reconstruction debug images
            vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', 'originals.png'), normalize=True, nrow=5)

        # Reconstructed images
        Irec = self.generator(self.encoder(first_imgs.to(self.device).float()))
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', f"step_{epoch}.png"), normalize=True, nrow=5)


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
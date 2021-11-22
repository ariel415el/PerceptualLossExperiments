import os
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import torchvision.utils as vutils

import GenerativeModels.utils.data_utils as data_utils
from GenerativeModels.AutoEncoders.autoencoder import plot_epoch


class VAETrainer():
    def __init__(self, params, encoder, generator, criterion, dataset, device):
        super(VAETrainer, self).__init__()
        self.device = device
        self.params = params
        self.generator = generator.to(device)
        self.encoder = encoder.to(device)
        self.criterion = criterion.to(device)

        self.dataloader = data_utils.get_dataloader(dataset, self.params.batch_size, self.device)

        # Define optimizers
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.params.lr, betas=(0.5, 0.999))
        self.optimizerE = optim.Adam(self.encoder.parameters(), lr=self.params.lr, betas=(0.5, 0.999))

        self.step = 0
        self.epoch = 0
        self.loss_means = []

    def train(self, outptus_dir, epochs=200, vis_freq=1):
        start = time()
        losses = []
        pbar = tqdm(total=epochs)

        while self.epoch < epochs:
            for _, images in self.dataloader:
                self.encoder.zero_grad()
                self.generator.zero_grad()

                self.step += 1
                images = images.to(self.device).float()

                recons, input, mu, log_var = self.forward(images)
                recons_loss = self.criterion(recons, input)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
                loss = recons_loss + 0.0001 * kld_loss
                # loss = recons_loss

                loss.backward()
                self.optimizerE.step()
                self.optimizerG.step()

                losses.append(np.log(loss.item()))

            if self.epoch % vis_freq == 0:
                self._visualize(self.epoch, self.dataloader.dataset, outptus_dir)
                self.loss_means.append(np.mean(losses))
                losses = []
                plot_epoch(self.loss_means, "Loss", outptus_dir)
                self._save_state(outptus_dir)
                pbar.set_description(f"Epoch: {self.epoch}: step {self.step}, im/sec: {(self.step + 1) * self.params.batch_size / (time() - start):.2f}")

            self.epoch += 1

            if self.epoch % self.params.decay_epochs == 0:
                for g, e in zip(self.optimizerG.param_groups, self.optimizerE.param_groups):
                    e['lr'] *= self.params.decay_rate
                    g['lr'] *= self.params.decay_rate

    def _save_state(self, folder_path):
        torch.save(self.encoder.state_dict(), f"{folder_path}/encoder.pth")
        torch.save(self.generator.state_dict(), f"{folder_path}/generator.pth")
        torch.save({"optimizerG": self.optimizerG.state_dict(),
                    "optimizerE": self.optimizerE.state_dict(),
                    'step': self.step,
                    'epoch': self.epoch,
                    'loss_means': self.loss_means},
                    f"{folder_path}/train_state.pth")
        self.params.save(f"{folder_path}/config.pth")

    def _load_ckpt(self, folder_path):
        self.encoder.load_state_dict(torch.load(f"{folder_path}/encoder.pth"))
        self.generator.load_state_dict(torch.load(f"{folder_path}/generator.pth"))
        train_state_path = f"{folder_path}/train_state.pth"
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path)
            self.optimizerE.load_state_dict(train_state['optimizerE'])
            self.optimizerG.load_state_dict(train_state['optimizerG'])
            self.step = train_state['step']
            self.epoch = train_state['epoch']
            self.loss_means = train_state['loss_means']

    def _visualize(self, epoch, dataset, outptus_dir):
        os.makedirs(outptus_dir, exist_ok=True)
        debug_indices = np.arange(25)
        images = np.stack([dataset[x][1] for x in debug_indices])
        first_imgs = torch.from_numpy(images)
        if epoch == 0:
            os.makedirs(os.path.join(outptus_dir, "reconstructions"), exist_ok=True)
            # dumpy original images fro later reconstruction debug images
            vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'reconstructions', 'originals.png'), normalize=True, nrow=5)

        # Reconstructed images
        Irec = self.reconstruct(first_imgs.to(self.device).float())
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'reconstructions', f"step_{epoch}.png"), normalize=True, nrow=5)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        logvar = torch.clamp(logvar, -10,10)
        std = torch.exp(0.5 * logvar.double())
        eps = torch.randn_like(std)
        return (eps * std + mu).float()

    def forward(self, input):
        mu, log_var = self.encoder(input)
        # log_var = torch.clamp(log_var, -10, 10)
        z = self.reparameterize(mu, log_var)
        return self.generator(z), input, mu, log_var


    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.params.z_dim).to(self.device)
        samples = self.generator(z)
        return samples

    def reconstruct(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


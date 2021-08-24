import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm
from time import time

import GenerativeModels.GLO.utils
import GenerativeModels.utils.data_utils
import os
import matplotlib.pyplot as plt


class AutoEncoderTraniner:
    def __init__(self, params, encoder, generator, criterion, dataset, device):
        self.device = device
        self.params = params
        self.generator = generator.to(device)
        self.encoder = encoder.to(device)
        self.criterion = criterion.to(device)

        self.dataloader = GenerativeModels.utils.data_utils.get_dataloader(dataset, self.params.batch_size, self.device)

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
                self.step += 1
                images = images.to(self.device).float()
                latent_codes = self.encoder(images)
                fake_images = self.generator(latent_codes)

                rec_loss = self.criterion(fake_images, images).mean()
                rec_loss.backward()

                losses.append(rec_loss.item())

                self.optimizerE.step()
                self.encoder.zero_grad()
                self.optimizerG.step()
                self.generator.zero_grad()

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
        Irec = self.generator(self.encoder(first_imgs.to(self.device).float()))
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'reconstructions', f"step_{epoch}.png"), normalize=True, nrow=5)


def endless_iterator(dataloader):
    while True:
        for x in iter(dataloader): yield x


def plot_epoch(arr, y_name, outptus_dir):
    plt.plot(np.arange(len(arr)), arr)
    plt.xlabel("Epoch")
    plt.ylabel(y_name)
    plt.title(f"{y_name} per epoch. last: {arr[-1]:.4f}")
    plt.savefig(os.path.join(outptus_dir, f"{y_name}.png"))
    plt.clf()
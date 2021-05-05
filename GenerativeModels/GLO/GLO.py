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
        self.latent_codes = LatentCodesDict(self.glo_params.z_dim, len(self.dataloader.dataset)).to(device)
        self.latent_codes.apply(weights_init)

        self.latent_codes.load_state_dict(torch.load('outputs/batch_3-spinoffs/l2-50-epochs/latent_codes.pth', map_location=device))
        self.generator.load_state_dict(torch.load('outputs/batch_3-spinoffs/l2-50-epochs/generator.pth', map_location=device))

        # Define optimizers
        self.optimizerG = optim.Adam(self.generator.parameters(),
                                     lr=self.glo_params.lr * self.glo_params.generator_lr_factor,
                                     betas=(0.5, 0.999))
        self.optimizerZ = optim.Adam(self.latent_codes.parameters(), lr=self.glo_params.lr, betas=(0.5, 0.999))

        self.num_debug_imgs = 64
        self.fixed_noise = torch.FloatTensor(self.num_debug_imgs, glo_params.z_dim).normal_(0, 1).to(self.device)
        self.name = f"GLO{'-FN' if glo_params.force_norm else ''}" \
                    f"(LR-{glo_params.lr}/{glo_params.decay_rate}/{glo_params.decay_epochs}_BS-{glo_params.batch_size})"

    def train(self, outptus_dir, vis_epochs=1):
        os.makedirs(outptus_dir, exist_ok=True)

        losses = []
        latent_codes_norms = []
        latent_codes_change = []
        for epoch in range(self.glo_params.num_epochs):
            old_latent_codes = self.latent_codes.emb.weight.clone()

            loss = self._train_epoch(self.dataloader)

            losses.append(loss)
            latent_codes_norms.append(torch.mean(torch.norm(self.latent_codes.emb.weight, dim=1)).item())
            latent_codes_change.append(torch.mean(torch.norm(self.latent_codes.emb.weight - old_latent_codes, dim=1)).item())
            print("Epoch: %d Error: %f" % (epoch, loss))
            if epoch % vis_epochs == 0:
                self._visualize(epoch, self.dataloader.dataset, outptus_dir)
                plot_epoch(losses, "Loss", outptus_dir)
                plot_epoch(latent_codes_norms, "Latent codes norm", outptus_dir)
                plot_epoch(latent_codes_change, "Latent codes norm of change", outptus_dir)
                torch.save(self.latent_codes.state_dict(), f"{outptus_dir}/latent_codes.pth")
                torch.save(self.generator.state_dict(), f"{outptus_dir}/generator.pth")
            if (epoch + 1) % self.glo_params.decay_epochs == 0:
                for g,z in zip(self.optimizerG.param_groups, self.optimizerZ.param_groups):
                    g['lr'] *= self.glo_params.decay_rate
                    z['lr'] *= self.glo_params.decay_rate

    def _train_epoch(self, dataloader):
        # Start optimizing
        er = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader.dataset) // dataloader.batch_size)
        start = time()
        for i, (indices, images) in pbar:
            # Put numpy data into tensors
            indices = indices.long().to(self.device)
            images = images.float().to(self.device)

            for j in range(self.glo_params.num_opt_steps):
                # Forward pass
                self.latent_codes.zero_grad()
                self.generator.zero_grad()
                zi = self.latent_codes(indices)
                fake_images = self.generator(zi)

                rec_loss = self.criterion(fake_images, images)
                rec_loss = rec_loss.mean()

                # Backward pass and optimization step
                rec_loss.backward()
                self.optimizerG.step()
                self.optimizerZ.step()

                er += rec_loss.item()
            pbar.set_description(f"im/sec: {(i+1) * self.glo_params.batch_size * self.glo_params.num_opt_steps / (time() - start):.2f}")
        if self.glo_params.force_norm:
            self.latent_codes.force_norm()
        er = er / (i + 1) / self.glo_params.num_opt_steps
        return er

    def _visualize(self, epoch, dataset, outptus_dir):
        debug_indices = np.arange(self.num_debug_imgs)
        if epoch == 0:
            os.makedirs(os.path.join(outptus_dir, "imgs", "generate_fixed"), exist_ok=True)
            os.makedirs(os.path.join(outptus_dir, "imgs", "generate_sampled"), exist_ok=True)
            os.makedirs(os.path.join(outptus_dir, "imgs", "reconstructions"), exist_ok=True)
            # dumpy original images fro later reconstruction debug images
            first_imgs = torch.from_numpy(np.array([dataset[x][1] for x in debug_indices]))
            vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', 'originals.png'), normalize=True)
        # fixed latent Generated images
        Igen = self.generator(self.fixed_noise)
        vutils.save_image(Igen.data, os.path.join(outptus_dir, 'imgs', 'generate_fixed',f"epoch_{epoch}.png"), normalize=True)

        # Generated from sampled latent vectors
        z = GenerativeModels.GLO.utils.sample_gaussian(self.latent_codes.emb.weight.clone().cpu(), self.num_debug_imgs).to(self.device)
        Igauss = self.generator(z)
        vutils.save_image(Igauss.data, os.path.join(outptus_dir, 'imgs', 'generate_sampled',f"epoch_{epoch}.png"), normalize=True)

        # Reconstructed images
        idx = torch.from_numpy(debug_indices).to(self.device)
        Irec = self.generator(self.latent_codes(idx))
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', f"epoch_{epoch}.png"), normalize=True)


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


def plot_epoch(arr, y_name, outptus_dir):
    plt.plot(np.arange(len(arr)), arr)
    plt.xlabel("Epoch")
    plt.ylabel(y_name)
    plt.title(f"{y_name} per epoch. last: {arr[-1]:.2f}")
    plt.savefig(os.path.join(outptus_dir, f"{y_name}.png"))
    plt.clf()
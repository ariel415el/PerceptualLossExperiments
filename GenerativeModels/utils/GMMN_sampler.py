import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from GenerativeModels.models import weights_init
from tqdm import tqdm

from losses.mmd.patch_mmd import compute_MMD


class GMMNSamplerTrainer:
    def __init__(self, mapping, lr, batch_size, device):
        self.device = device
        self.mapping = mapping.to(device)
        self.mapping.apply(weights_init)
        self.optimizer = optim.Adam(self.mapping.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        self.batch_size = batch_size
        self.name = f"GMMN-mapping"

    def train(self, samples, train_dir, epochs=50):
        losses = []
        pbar = tqdm(range(epochs))
        pbar.set_description("Training #")
        for epoch in pbar:
            error = self.train_epoch(samples)
            pbar.set_description(f"GMMN-sampler: Epoch: {epoch} Error: {error}")

            losses.append(error)
            plt.plot(range(len(losses)), losses)
            plt.savefig(f"{train_dir}/GMMN-train-loss.png")
            plt.clf()

    def train_epoch(self, samples):
        batch_size = self.batch_size
        n, d = samples.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        total_loss = 0
        for i in range(batch_n):
            random_vecs = torch.randn((batch_size, self.mapping.input_dim)).to(self.device)
            mapped_vecs = self.mapping(random_vecs)
            random_indices = rp[i * batch_size: (i + 1) * batch_size]
            loss = compute_MMD(mapped_vecs, samples[random_indices], sigmas=[1, 10])
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / batch_n

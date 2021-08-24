from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import faiss
from GenerativeModels.models import weights_init
from tqdm import tqdm


def find_nearest_neighbors(mapped_vecs, samples, batch_size=16):
    n, d = samples.shape
    nn_indices = torch.zeros(n).long()
    for i in range(int(np.ceil(n / batch_size))):
        dist_mat = ((samples[i * batch_size: (i + 1) * batch_size, None] - mapped_vecs[None, :])**2).mean(2)  # batch_size x n_mapped_vecs matrix
        nn_indices[i * batch_size: (i + 1) * batch_size] = torch.argmin(dist_mat, dim=1)

    return nn_indices


def find_nearest_neighbors_faiss(mapped_vecs, samples):
    nbrs = faiss.IndexFlatL2(samples.shape[1])
    nbrs.add(mapped_vecs.detach().cpu().numpy().astype('float32'))
    _, indices = nbrs.search(samples.detach().cpu().numpy().astype('float32'), 1)
    nn_indices = torch.from_numpy(indices.squeeze(1))

    return nn_indices


class IMLESamplerTrainer:
    def __init__(self, mapping, lr, batch_size, device):
        self.device = device
        self.mapping = mapping.to(device)
        self.mapping.apply(weights_init)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.mapping.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        self.batch_size = batch_size
        self.name = f"IMLE-mapping"

    def train(self, samples, train_dir, epochs=50):
        losses = []
        pbar = tqdm(range(epochs))
        pbar.set_description("Training #")
        for epoch in pbar:
            error, timing = self.train_epoch(samples)
            pbar.set_description(f"IMLE-sampler: Epoch: {epoch} Error: {error}, Timing: {timing}")

            losses.append(error)
            plt.plot(range(len(losses)), losses)
            plt.savefig(f"{train_dir}/IMLE-train-loss.png")
            plt.clf()

    def train_epoch(self, samples):
        batch_size = self.batch_size
        times = {}
        n, d = samples.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        random_vecs = torch.randn((batch_n * batch_size, self.mapping.input_dim)).to(self.device)

        # # Compute current mapping nearest neighbors
        start = time()
        mapped_vecs = torch.zeros((batch_n * batch_size, self.mapping.output_dim)).to(self.device)
        for i in range(batch_n):
            mapped_vecs[i * batch_size: (i + 1) * batch_size] = self.mapping(random_vecs[i * batch_size: (i + 1) * batch_size])

        # nn_indices = find_nearest_neighbors_faiss(mapped_vecs, samples)
        nn_indices = find_nearest_neighbors(mapped_vecs, samples, batch_size)

        times['nn'] = f"{n / (time() - start):.3f} im/sec"

        # Start optimizing
        er = 0
        for i in range(batch_n):
            self.mapping.zero_grad()
            indices = rp[i * batch_size + torch.arange(batch_size).long()]
            e = random_vecs[nn_indices[indices]]
            fake_samples = self.mapping(e)  # map again instead of using mapped_vecs
            real_samples = samples[indices]
            loss = self.criterion(fake_samples, real_samples)
            loss.backward()
            er += loss.item()
            self.optimizer.step()

        return er / batch_n, times
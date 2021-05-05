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


class IMLE:
    def __init__(self, mapping, lr, batch_size, device):
        self.device = device
        self.mapping = mapping.to(device)
        self.mapping.apply(weights_init)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.mapping.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        self.batch_size = batch_size
        self.name = f"IMLE-mapping"

    def train(self, z_np, train_dir, epochs=50):

        pbar = tqdm(range(epochs))
        losses = []
        for epoch in pbar:
            error = self.train_epoch(z_np)
            pbar.set_description("IMLE: Epoch: %d Error: %f" % (epoch, error))
            losses.append(error)

        plt.plot(range(len(losses)), losses)
        plt.savefig(f"{train_dir}/IMLE-train-loss.png")
        plt.clf()

    def train_epoch(self, z_np):
        # Compute batch size
        batch_size = self.batch_size
        n, d = z_np.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        random_vecs = np.zeros((batch_n * batch_size, self.mapping.input_dim))
        mapped_vecs = np.zeros((batch_n * batch_size, self.mapping.output_dim))

        for i in range(batch_n):
            random_vectors = torch.randn(batch_size, self.mapping.input_dim).to(self.device)
            mapped_vecs[i * batch_size: (i + 1) * batch_size] = self.mapping(random_vectors).cpu().data.numpy()
            random_vecs[i * batch_size: (i + 1) * batch_size] = random_vectors.cpu().data.numpy()

        nbrs = faiss.IndexFlatL2(self.mapping.output_dim)
        nbrs.add(mapped_vecs.astype('float32'))
        _, indices = nbrs.search(z_np.astype('float32'), 1)
        indices = indices.squeeze(1)

        # Start optimizing
        er = 0

        for i in range(batch_n):
            self.mapping.zero_grad()
            # Put numpy data into tensors
            idx_np = i * batch_size + np.arange(batch_size)
            e = torch.from_numpy(random_vecs[indices[rp[idx_np]]]).float().to(self.device)
            z_act = torch.from_numpy(z_np[rp[idx_np]]).float().to(self.device)
            z_est = self.mapping(e)
            loss = self.criterion(z_est, z_act)
            loss.backward()
            er += loss.item()
            self.optimizer.step()

        return er / batch_n
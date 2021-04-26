import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import weights_init
import numpy as np
from losses.mmd_loss import MMD


class GMMN():
    def __init__(self, mapping, lr, batch_size, device):
        self.device = device
        self.mapping = mapping.to(device)
        self.mapping.apply(weights_init)
        self.loss = MMD()
        self.optimizer = torch.optim.Adam(self.mapping.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        self.batch_size = batch_size
        self.name = f"GMMN-mapping"

    def train(self, Zs, train_dir, epochs=50):
        pbar = tqdm(range(epochs))
        losses = []
        for epoch in pbar:
            loss = self.train_epoch(Zs)
            pbar.set_description("GMMN: Epoch: %d loss: %f" % (epoch, loss))
            losses.append(loss)
            if (epoch + 1) % 5 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.85
        plt.plot(range(len(losses)), losses)
        plt.savefig(f"{train_dir}/GMMN-train-loss.png")
        plt.clf()

    def train_epoch(self, Zs):
        # Compute batch size
        n, d = Zs.shape
        num_batchs = n // self.batch_size
        shuffled_Zs = Zs[np.random.permutation(n)]

        loss = 0
        for b in range(num_batchs):
            data_batch = shuffled_Zs[b * self.batch_size: (b + 1) * self.batch_size]
            noise_batch = torch.FloatTensor(self.batch_size, self.mapping.input_dim).uniform_(-1.0, 1.0).to(self.device)
            # noise_batch = torch.randn(self.batch_size, self.mapping.input_dim).to(self.device)
            fake_batch = self.mapping(noise_batch)

            loss = self.loss(fake_batch, data_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss += loss.item()

        return loss / self.batch_size

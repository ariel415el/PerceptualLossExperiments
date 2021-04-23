import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from IMLE import _netT
import numpy as np
from losses.mmd_loss import MMD


class GMMN():
    def __init__(self, e_dim, z_dim, device):
        self.device = device
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.netT = _netT(e_dim, z_dim).train().to(device)
        self.loss = MMD()
        self.name = f"GMMN-sampler"

    def train(self, Zs, train_dir, lr=0.0001, batch_size=1024, epochs=50):
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.netT.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        losses = []
        pbar = tqdm(range(epochs))
        for epoch in range(epochs):
            loss = self.train_epoch(Zs, epoch)
            pbar.set_description("GMMN: Epoch: %d loss: %f" % (epoch, loss))
            losses.append(loss)
        torch.save(self.netT.state_dict(), f"{train_dir}/netT-MMD.pth")
        plt.plot(range(len(losses)), losses)
        plt.savefig(f"{train_dir}/imgs/GMMN-train-loss.png")

    def train_epoch(self, Zs, epoch):
        # Compute batch size
        batch_size = self.batch_size
        n, d = Zs.shape
        num_batchs = n // batch_size
        shuffled_Zs = Zs[np.random.permutation(n)]

        loss = 0
        for b in range(num_batchs):
            data_batch = shuffled_Zs[b * batch_size: (b + 1) * batch_size]
            noise_batch = torch.from_numpy(np.random.uniform(low=-1.0, high=1.0, size=(batch_size, self.e_dim))).to(self.device).float()
            fake_batch = self.netT(noise_batch)

            loss = self.loss(fake_batch, data_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss += loss.item()

        return loss / batch_size

    def load_weights(self, ckp_dir, device):
        self.netT.load_state_dict(torch.load(os.path.join(ckp_dir, 'netT-MMD.pth'), map_location=device))
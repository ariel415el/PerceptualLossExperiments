import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import collections
import faiss
import os


class _netT(nn.Module):
    def __init__(self, xn, yn):
        super(_netT, self).__init__()
        self.xn = xn
        self.yn = yn
        self.lin1 = nn.Linear(xn, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin_out = nn.Linear(128, yn, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.lin1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.lin_out(z)
        return z

class IMLE():
    def __init__(self, e_dim, z_dim):
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.netT = _netT(e_dim, z_dim).cuda()

    def train(self, z_np, train_dir, lr=1e-3, batch_size=128, epochs=50):
        self.lr = lr
        self.batch_size = batch_size
        for epoch in range(epochs):
            self.train_epoch(z_np, epoch)

        torch.save(self.netT.state_dict(), f"{train_dir}/netT_nag.pth")

    def train_epoch(self, z_np, epoch):
        # Compute batch size
        batch_size = self.batch_size
        n, d = z_np.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        # Compute learning rate
        # Initialize optimizers
        optimizerT = optim.Adam(self.netT.parameters(), lr=self.lr ,
                                betas=(0.5, 0.999), weight_decay=1e-5)
        criterion = nn.MSELoss().cuda()
        self.netT.train()

        M = batch_n
        random_vecs = np.zeros((M * batch_size, self.e_dim))
        mapped_vecs = np.zeros((M * batch_size, self.z_dim))
        indices = []
        for i in range(M):
            random_vectors = torch.randn(batch_size, self.e_dim).cuda()
            mapped_vecs[i * batch_size: (i + 1) * batch_size] = self.netT(random_vectors).cpu().data.numpy()
            random_vecs[i * batch_size: (i + 1) * batch_size] = random_vectors.cpu().data.numpy()

        nbrs = faiss.IndexFlatL2(self.z_dim)
        nbrs.add(mapped_vecs.astype('float32'))
        _, indices = nbrs.search(z_np.astype('float32'), 1)
        indices = indices.squeeze(1)

        # Start optimizing
        er = 0

        for i in range(batch_n):
            self.netT.zero_grad()
            # Put numpy data into tensors
            idx_np = i * batch_size + np.arange(batch_size)
            e = torch.from_numpy(random_vecs[indices[rp[idx_np]]]).float().cuda()
            z_act = torch.from_numpy(z_np[rp[idx_np]]).float().cuda()
            z_est = self.netT(e)
            loss = criterion(z_est, z_act)
            loss.backward()
            er += loss.item()
            optimizerT.step()

        print("Epoch: %d Error: %f" % (epoch, er / batch_n))

    def load_weights(self, ckp_dir, device):
        self.netT.load_state_dict(torch.load(os.path.join(ckp_dir, 'netT_nag.pth'), map_location=device))
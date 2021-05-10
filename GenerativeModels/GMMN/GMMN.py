import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from time import time

from GenerativeModels.models import weights_init
from GenerativeModels.utils.data_utils import get_dataloader
import sys
sys.path.append(os.path.realpath("../.."))
from losses.mmd_loss import MMD
from losses.vgg_loss.vgg_loss import VGGFeatures
import numpy as np


class VGG_features_MMD(torch.nn.Module):
    def __init__(self, device):
        super(VGG_features_MMD, self).__init__()
        self.name = f"VGG_features_MMD"
        self.vgg = VGGFeatures(5, pretrained=True, post_relu=False).to(device)

    def forward(self, output, target):
        batch_size = output.shape[0]
        output_feature = self.vgg.get_activations(output)
        output_feature = torch.cat([output.reshape(batch_size, -1)] + [o.reshape(batch_size, -1) for o in output_feature], dim=1)
        batch_size = target.shape[0]
        target_feature = self.vgg.get_activations(target)
        target_feature = torch.cat([output.reshape(batch_size, -1)] + [o.reshape(batch_size, -1) for o in target_feature], dim=1)

        return torch.sum((output_feature.mean(0) - target_feature.mean(0))**2)


class GMMN():
    def __init__(self, mapping, lr, batch_size, device):
        self.device = device
        self.mapping = mapping.to(device)
        self.mapping.apply(weights_init)
        # self.mapping.load_state_dict(torch.load('../GLO/outputs/batch_3-spinoffs/l2_norm1-50-epochs/generator.pth', map_location=device))
        self.loss = MMD()
        self.optimizer = torch.optim.Adam(self.mapping.parameters(), lr=lr)
        self.batch_size = batch_size
        self.name = f"GMMN-mapping"

    def train(self, dataset, train_dir, epochs=50):
        os.makedirs(train_dir, exist_ok=True)
        dataloader = get_dataloader(dataset, self.batch_size, self.device)
        start = time()
        pbar = tqdm(range(epochs))
        mean_losses = []
        for epoch in pbar:
            epoch_losses = self.train_epoch(dataloader)
            mean_losses.append(np.mean(epoch_losses))
            pbar.set_description(f"GMMN: Epoch: {epoch} loss: {mean_losses[-1]:.2f} imgs/sec: {(epoch+1)*len(dataloader.dataset) / (time()-start):.2f}")
            if (epoch + 1) % 5 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.75
            self.visualize(train_dir, epoch)
            plt.plot(range(len(mean_losses)), mean_losses)
            plt.savefig(f"{train_dir}/GMMN-train-loss.png")
            plt.clf()

    def train_epoch(self, dataloader):
        # Compute batch size
        losses = []

        for (indices, data_batch) in dataloader:
            data_batch = data_batch.to(self.device).float()
            noise_batch = torch.FloatTensor(self.batch_size, self.mapping.input_dim).uniform_(-1.0, 1.0).to(self.device)
            # noise_batch = torch.randn(self.batch_size, self.mapping.input_dim).to(self.device)
            fake_batch = self.mapping(noise_batch)

            loss = self.loss(fake_batch, data_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss += loss.item()
            losses.append(loss.item())

        return losses

    def visualize(self, train_dir, epoch):
        noise_batch = torch.FloatTensor(64, self.mapping.input_dim).uniform_(-1.0, 1.0).to(self.device)
        fake_batch = self.mapping(noise_batch)
        vutils.save_image(fake_batch, os.path.join(train_dir, f"Generated-{epoch}.png"), normalize=True)
import dataclasses
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

        self.vgg = VGGFeatures(5, pretrained=True, post_relu=False).to(device)

    def compute_data_vgg_features(self, dataset):
        data_features = []
        batch_size = 128
        dataloader = get_dataloader(dataset, batch_size, self.device)
        avg = 0
        with torch.no_grad():
            for _, data in tqdm(dataloader):
                data = data.to(self.device).float()
                features = self.vgg.get_activations(data)
                features = torch.cat([data.reshape(batch_size, -1)] + [o.reshape(batch_size, -1) for o in features], dim=1)
                avg += features

        avg /= len(dataset)

        return avg

    def train(self, dataset, train_dir, num_steps):
        os.makedirs(train_dir, exist_ok=True)
        data_feature_mean = self.compute_data_vgg_features(dataset)
        start = time()
        pbar = tqdm(range(num_steps))
        losses = []
        diff_ma = 1
        alpha = 1/self.batch_size
        for step in pbar:

            noise_batch = torch.randn(self.batch_size, self.mapping.input_dim).to(self.device)
            fake_data = self.mapping(noise_batch)
            fake_features = self.vgg.get_activations(fake_data)
            fake_features = torch.cat([fake_data.reshape(self.batch_size, -1)] + [o.reshape(self.batch_size, -1) for o in fake_features],dim=1)

            # loss = self.loss(fake_features, data_features)
            diff = data_feature_mean - fake_features.mean(0)
            diff_ma = ((1-alpha) * diff_ma + alpha * diff).detach()
            loss = torch.sum(diff * diff_ma)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss += loss.item()
            losses.append(loss.item())

            pbar.set_description(f"GMMN: Step: {step} loss: {losses[-1]:.2f} imgs/sec: {(step+1)*self.batch_size / (time()-start):.2f}")
            if (step + 1) % 100 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.75
                self.visualize(train_dir, step)
                plt.plot(range(len(losses)), losses)
                plt.savefig(f"{train_dir}/GMMN-train-loss.png")
                plt.clf()

    def visualize(self, train_dir, step):
        noise_batch = torch.randn(25, self.mapping.input_dim).to(self.device)
        fake_batch = self.mapping(noise_batch)
        vutils.save_image(fake_batch, os.path.join(train_dir, f"Generated-{step}.png"), normalize=True, nrow=5)
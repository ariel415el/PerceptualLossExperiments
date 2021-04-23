import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm
from time import time
import models
import utils
import os
import matplotlib.pyplot as plt

import sys


sys.path.append(os.path.realpath(".."))
from losses.l2 import L2
from losses.lap1_loss import LapLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.mmd_loss import MMD
from losses.utils import ListOfLosses
from losses.vgg_loss.vgg_loss import VGGFeatures


class GLO:
    def __init__(self, glo_params, dataset_size, device):
        self.dataset_size = dataset_size
        self.device = device
        self.netZ = models.LatentCodesDict(glo_params.z_dim, dataset_size)
        self.netZ.apply(models.weights_init)
        self.netZ.to(self.device)

        # self.netG = model._netG(glo_params.z_dim, glo_params.img_dim, glo_params.channels, glo_params.use_bn)
        self.netG = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim)
        self.netG.apply(models.weights_init)
        self.netG.to(self.device)

        self.num_debug_imgs = 64
        self.fixed_noise = torch.FloatTensor(self.num_debug_imgs, glo_params.z_dim).normal_(0, 1).to(self.device)

        self.duplicate_channels = glo_params.channels == 1
        self.pad_imgs = glo_params.img_dim == 28

        self.loss = ListOfLosses([
                        # L2().to(device),
                        VGGFeatures(3 if glo_params.img_dim == 28 else 5, pretrained=True, post_relu=True).to(device),
                        # LapLoss(max_levels=3 if glo_params.img_dim == 28 else 5, n_channels=glo_params.channels).to(device),
                        # MMD()
                        # PatchRBFLoss(3, device=self.device).to(self.device),
                        # MMDApproximate(r=1024, pool_size=32, pool_strides=16, normalize_patch='mean').to(self.device),
                        # self.dist = ScnnLoss().to(self.device)
        ])

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=glo_params.lr * glo_params.generator_lr_factor, betas=(0.5, 0.999))
        self.optimizerZ = optim.Adam(self.netZ.parameters(), lr=glo_params.lr, betas=(0.5, 0.999))

    def train(self, dataloader, opt_params, vis_epochs=1, outptus_dir='runs', start_epoch=0):
        os.makedirs(outptus_dir, exist_ok=True)
        errs = []

        for epoch in range(start_epoch, opt_params.num_epochs):
            er = self.train_epoch(dataloader, epoch, opt_params)
            errs.append(er)
            print("Epoch: %d Error: %f" % (epoch, er))
            torch.save(self.netZ.state_dict(), f"{outptus_dir}/netZ_nag.pth")
            torch.save(self.netG.state_dict(), f"{outptus_dir}/netG_nag.pth")
            if epoch % vis_epochs == 0:
                self.visualize(epoch, dataloader.dataset, outptus_dir)
                plt.plot(np.arange(len(errs)), errs)
                plt.xlabel("Epoch")
                plt.ylabel(f"loss ({er:.2f})")
                plt.savefig(os.path.join(outptus_dir, "train_loss.png"))
            if (epoch + 1) % opt_params.decay_epochs == 0:
                for g,z in zip(self.optimizerG.param_groups, self.optimizerZ.param_groups):
                    g['lr'] *= opt_params.decay_rate
                    z['lr'] *= opt_params.decay_rate

    def train_epoch(self, dataloader, epoch, opt_params):
        # Start optimizing
        er = 0
        pbar = tqdm(enumerate(dataloader))
        start = time()
        for i, (indices, images) in pbar:
            # Put numpy data into tensors
            indices = indices.long().to(self.device)
            images = images.float().to(self.device)

            # Forward pass
            self.netZ.zero_grad()
            self.netG.zero_grad()
            zi = self.netZ(indices)
            Ii = self.netG(zi)

            rec_loss = self.loss(Ii, images)
            rec_loss = rec_loss.mean()

            # Backward pass and optimization step
            rec_loss.backward()
            self.optimizerG.step()
            self.optimizerZ.step()

            er += rec_loss.item()
            pbar.set_description(f"im/sec: {i * opt_params.batch_size / (time() - start):.2f}")
        self.netZ.get_norm()
        er = er / (i + 1)
        return er

    def visualize(self, epoch, dataset, outptus_dir):
        debug_indices = np.arange(self.num_debug_imgs)
        if epoch == 0:
            os.makedirs(os.path.join(outptus_dir, "imgs", "generate_fixed"), exist_ok=True)
            os.makedirs(os.path.join(outptus_dir, "imgs", "generate_sampled"), exist_ok=True)
            os.makedirs(os.path.join(outptus_dir, "imgs", "reconstructions"), exist_ok=True)
            # dumpy original images fro later reconstruction debug images
            first_imgs = torch.from_numpy(np.array([dataset[x][1] for x in debug_indices]))
            vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', 'originals.png'), normalize=True)
        # fixed latent Generated images
        Igen = self.netG(self.fixed_noise)
        vutils.save_image(Igen.data, os.path.join(outptus_dir, 'imgs', 'generate_fixed',f"epoch_{epoch}.png"), normalize=True)

        # Generated from sampled latent vectors
        z = utils.sample_gaussian(self.netZ.emb.weight.clone().cpu(), self.num_debug_imgs).to(self.device)
        Igauss = self.netG(z)
        vutils.save_image(Igauss.data, os.path.join(outptus_dir, 'imgs', 'generate_sampled',f"epoch_{epoch}.png"), normalize=True)

        # Reconstructed images
        idx = torch.from_numpy(debug_indices).to(self.device)
        Irec = self.netG(self.netZ(idx))
        vutils.save_image(Irec.data, os.path.join(outptus_dir, 'imgs', 'reconstructions', f"epoch_{epoch}.png"), normalize=True)

    def load_weights(self, ckp_dir, device):
        print("GLO: Loading Z and G weights")
        self.netZ.load_state_dict(torch.load(os.path.join(ckp_dir, 'netZ_nag.pth'), map_location=device))
        self.netG.load_state_dict(torch.load(os.path.join(ckp_dir, 'netG_nag.pth'), map_location=device))
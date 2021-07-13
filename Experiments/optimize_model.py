import os

import numpy as np
import torch.optim
from torchvision import utils as vutils
from tqdm import tqdm

from Experiments.all import load_models
from GenerativeModels.config import default_config
from GenerativeModels.utils.data_utils import get_dataset
from losses.classic_losses.grad_loss import GradLoss
from losses.classic_losses.l2 import L2
from losses.composite_losses.window_loss import WindowLoss
from losses.experimental_patch_losses import MMD_PP
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.patch_loss import PatchRBFLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

device = torch.device("cuda")

def optimize_model(criterion, outputs_dir, train_dataset, n=25):
    # encoder, generator = load_models('GenerativeModels/Aotuencoders/outputs/ffhq_128/L2/')
    encoder, generator = load_models(device, '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/GenerativeModels/Aotuencoders/outputs/ffhq_128/L1/')
    # encoder, generator = load_models(device)
    encoder.train()
    generator.train()

    criterion = criterion.to(device)

    images = torch.from_numpy(np.array([train_dataset[i][1] for i in range(n)])).to(device).float()
    # images = torch.from_numpy(np.array([train_dataset[i][1] for i in [1, 56475, 11821, 20768,  6, 52650,  7563]])).to(device).float()
    vutils.save_image(images, os.path.join(outputs_dir, f"orig.png"), normalize=True, nrow=int(np.sqrt(n)))

    # Define optimizers
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizerE = torch.optim.Adam(encoder.parameters(), lr=0.001)
    pbar = tqdm(range(1000))
    for i in pbar:
        if i % 1 == 0:
            recons = generator(encoder(images))
            vutils.save_image(recons, os.path.join(outputs_dir, f"recons{i}.png"), normalize=True, nrow=int(np.sqrt(n)))

        latent_codes = encoder(images)
        fake_images = generator(latent_codes)

        rec_loss = criterion(fake_images, images).mean()
        rec_loss.backward()

        optimizerE.step()
        encoder.zero_grad()
        optimizerG.step()
        generator.zero_grad()
        pbar.set_description(f"loss: {rec_loss}")
        if (i + 1) % 200 == 0:
            for g, e in zip(optimizerG.param_groups, optimizerE.param_groups):
                e['lr'] *= 0.9
                g['lr'] *= 0.9

    torch.save(encoder.state_dict(), f"{outputs_dir}/encoder.pth")
    torch.save(generator.state_dict(), f"{outputs_dir}/generator.pth")

if __name__ == '__main__':
    device = torch.device("cuda")

    # criterion = MMD_PP(device, patch_size=7, pool_size=32, pool_strides=16, r=128, normalize_patch='channel_mean',
    #                    weights=[0.001, 0.05, 1.0]).to(device)
    # criterion = L2().to(device)
    # criterion = VGGPerceptualLoss(pretrained=False, norm_first_conv=True, reinit=True, layers_and_weights=[('conv4_3', 1.0)]).to(device)
    # criterion = VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv4_3', 1.0)]).to(device)
    # criterion = MMDApproximate(patch_size=3, sigma=0.06, pool_size=32, pool_strides=16, r=128)
    # criterion = MMDApproximate(patch_size=11, sigma=0.02, pool_size=32, pool_strides=16, r=128, normalize_patch='channel_mean')
    # criterion = PatchMMDLoss(patch_size=11, n_samples=64)
    # criterion = PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True)
    criterion = MMDApproximate(patch_size=11, sigma=0.02, pool_size=32, pool_strides=32, r=256, batch_reduction='none', normalize_patch='channel_mean')
    # criterion = MMD_PP()
    # criterion = WindowLoss(PatchMMDLoss(patch_size=11, n_samples=64), window_size=32, stride=16)
    # criterion = VGGFeatures(pretrained=True).to(device)
    # criterion = VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[1,1,1,1,1,1]).to(device)

    outputs_dir = f'Experiments/optimize_model/{criterion.name}'

    os.makedirs(outputs_dir, exist_ok=True)
    params = default_config
    train_dataset = get_dataset('ffhq', split='test', resize=params.img_dim, val_percent=0.15)
    optimize_model(criterion, outputs_dir, train_dataset, n=32)

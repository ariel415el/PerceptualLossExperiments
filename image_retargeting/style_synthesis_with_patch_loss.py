import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

import sys

from image_retargeting.utils import aspect_ratio_resize
from losses.composite_losses.laplacian_losses import LaplacyanLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.composite_losses.window_loss import WindowLoss
from losses.swd.patch_swd import PatchSWDLoss

from losses.mmd.patch_mmd import PatchMMDLoss, compute_MMD

sys.path.append(os.path.realpath(".."))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# def cv2pt(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float64) / 255.
#     img = img * 2 - 1
#     img = torch.from_numpy(img.transpose(2, 0, 1)).float()
#
#     return img


def optimize_texture(target_texture, criterias, output_dir=None, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (H, W, C)
    """
    scale_x, scale_y = 1.5,0.7
    # scale_x, scale_y = 1,1
    H,W = target_texture.shape[0], target_texture.shape[1]

    optimized_variable = cv2.resize(target_texture, (int(W * scale_x), int(H * scale_y)))
    optimized_variable = cv2.blur(optimized_variable, (11,11))
    optimized_variable = cv2pt(optimized_variable).unsqueeze(0).to(device)
    # optimized_variable = torch.ones((1, 3, int(H * scale_y), int(W * scale_x))).to(device)# * torch.mean(target_texture, dim=(0, 2, 3), keepdim=True)
    # optimized_variable *= 0
    # optimized_variable += torch.randn(optimized_variable.shape).to(device).clamp(-1, 1) * 0.1
    optimized_variable.requires_grad_(True)
    optim = torch.optim.Adam([optimized_variable], lr=lr)

    target_texture = cv2pt(target_texture).unsqueeze(0).to(device)
    vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-0.png", normalize=True)

    all_losses = {loss.name:[] for loss in criterias}
    all_means = {loss.name:[] for loss in criterias}
    for i in tqdm(range(1, num_steps + 1)):

        optim.zero_grad()

        total_loss = 0
        for criteria in criterias:
            loss = criteria(target_texture, optimized_variable)
            all_losses[criteria.name].append(loss.item())
            total_loss += loss

        total_loss.backward()
        optim.step()

        if i % 1000 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.9
        if i % 500 == 0 and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-{i}.png", normalize=True)
        if i % 100 == 0 and output_dir is not None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            for criteria in criterias:
                losses = all_losses[criteria.name]
                all_means[criteria.name].append(np.mean(all_losses[criteria.name][-100:]))
                # ax.set_title(f"last-Loss: {losses[-1]}")
                ax.plot(np.arange(len(losses)), np.log(losses), label=f'{criteria.name}: {all_means[criteria.name][-1]:.6f}')
                ax.plot((1 + np.arange(len(all_means[criteria.name])))*100, np.log(all_means[criteria.name]), c='y')
            ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    optimized_image = torch.clip(optimized_variable, -1, 1)

    return optimized_image


def crop_center(img, perc=0.5):
    y, x, c = img.shape
    cropx = int(perc * x)
    cropy = int(perc * y)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def run_sigle():
    images = [
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/ball-on-grass.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/cobbles.jpeg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/long_grass.jpeg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/rug.jpeg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/patch_dist/building.bmp',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/patch_dist/fruit.png',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/patch_dist/girafs.png',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/singan/balloons.png',
    ]

    losses = [
        # PatchSWDLoss(patch_size=9, num_proj=256, n_samples=2048),
        # PatchMMDLoss(patch_size=9, n_samples=2048)
        # LaplacyanLoss(PatchMMDLoss(patch_size=15, n_samples=512), weightening_mode=3, max_levels=2),

        # LossesList([
        #     PatchMMDLoss(patch_size=31, stride=15, n_samples=4096),
        #     PatchMMDLoss(patch_size=11, stride=5, n_samples=4096),
        # PatchMMDLoss(patch_size=3, stride=2, n_samples=4096),
        # LossesList([
        #     PatchMMDLoss(patch_size=61, stride=30, n_samples=4096),
        #     PatchMMDLoss(patch_size=51, stride=25, n_samples=4096),
        #     PatchMMDLoss(patch_size=41, stride=20, n_samples=4096),
        #     PatchMMDLoss(patch_size=31, stride=15, n_samples=4096),
        #     PatchMMDLoss(patch_size=21, stride=10, n_samples=4096),
        #     PatchMMDLoss(patch_size=11, stride=5, n_samples=4096),
        # ], weights=[1,1,1,1,1,1], name=f'multipatchMMD2'),
        # ], weights=[0.3, 0.3, 0.3])
        # PatchMMDLoss(patch_size=3, stride=2, n_samples=4096),
        # PatchMMDLoss(patch_size=31, stride=1, n_samples=4096),
        # PatchMMDLoss(patch_size=11, stride=1, n_samples=4096),
        # WindowLoss(PatchSWDLoss(patch_size=11, stride=1, n_samples=4096, num_proj=1024), window_size=32, stride=16),
        # PatchSWDLoss(patch_size=11, stride=1, n_samples=4096, num_proj=1024, sample_same_locations=False),
        # PatchSWDLoss(patch_size=31, stride=15, n_samples=4096, sample_same_locations=False),
        # PatchMMDLoss(patch_size=15, stride=8, n_samples=4096, sample_same_locations=False),
        # PatchMMDLoss(patch_size=31, stride=15, n_samples=4096, sample_same_locations=False),
        # PatchMMDLoss(patch_size=64, stride=32, n_samples=4096, sample_same_locations=False),
        # PatchMMDLoss(patch_size=11, stride=5, n_samples=4096, sample_same_locations=False)
        PyramidLoss(PatchMMDLoss(patch_size=11, stride=5, n_samples=4096, sample_same_locations=False), weightening_mode=3, max_levels=3),
        # PatchSWDLoss(patch_size=5, stride=1, n_samples=4096, num_proj=1024),
        # PatchMMDLoss(patch_size=5, stride=2, n_samples=1024),
        # PatchMMDLoss(patch_size=7, stride=3, n_samples=1024),
        # PatchMMDLoss(patch_size=5, stride=1, n_samples=1024)

        # PatchSWDLoss(patch_size=11, stride=1, n_samples=4096, num_proj=1024),
        # LaplacyanLoss(PatchMMDLoss(patch_size=19, n_samples=1024), weightening_mode=3, max_levels=2),
        # WindowLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), window_size=32, stride=16),
        # PyramidLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), weightening_mode=3, max_levels=3),
        # LaplacyanLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), weightening_mode=3, max_levels=3)
    ]
    exp_name = "_".join([l.name for l in losses])
    for path in images:
        img_name = os.path.basename(os.path.splitext(path)[0])
        input_dir = f"Experiments/style_synthesis/{img_name}"
        os.makedirs(input_dir, exist_ok=True)

        texture_image = cv2.imread(path)
        texture_image = aspect_ratio_resize(texture_image, max_dim=256)
        # texture_image = crop_center(texture_image, 0.9)

        cv2.imwrite(os.path.join(input_dir, f"org.png"), texture_image)

        train_dir = f"{input_dir}/{exp_name}"
        os.makedirs(train_dir, exist_ok=True)

        img = optimize_texture(texture_image, losses, output_dir=train_dir, num_steps=100000, lr=0.005)
        vutils.save_image(img, os.path.join(input_dir, f"{exp_name}.png"), normalize=True)


if __name__ == '__main__':
    run_sigle()

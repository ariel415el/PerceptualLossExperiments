import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

import sys

from losses.composite_losses.laplacian_losses import LaplacyanLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.composite_losses.window_loss import WindowLoss
from losses.swd.patch_swd import PatchSWDLoss

from losses.mmd.patch_mmd import PatchMMDLoss

sys.path.append(os.path.realpath(".."))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1

    # img = np.clip(img, -1 + 1e-9, 1 - 1e-9)
    # img = np.arctanh(img)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def pt2cv(img):
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def comupute_loss(criteria, x, x_squared):
    s = x.shape[2]
    unfold = torch.nn.Unfold(kernel_size=s, stride=s // 2)
    windows = unfold(x_squared)[0].transpose(0, 1).reshape(-1, 3, s, s)
    return criteria(x.repeat(windows.shape[0], 1, 1, 1), windows).mean()


def optimize_texture(target_texture, criteria, output_dir=None, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    s = target_texture.shape[2]
    assert s == target_texture.shape[3]
    optimized_variable = torch.ones((1, 3, 2 * s, 2 * s)).to(device) * \
                         torch.mean(target_texture, dim=(0, 2, 3), keepdim=True)
    optimized_variable += torch.randn(optimized_variable.shape).to(device).clamp(-1, 1) * 0.1

    optimized_variable.requires_grad_(True)
    optim = torch.optim.Adam([optimized_variable], lr=lr)
    losses = []
    for i in tqdm(range(num_steps + 1)):
        if i % 500 == 0 and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            optimized_image = optimized_variable
            vutils.save_image(optimized_image, f"{output_dir}/output-{i}.png", normalize=True)

        optim.zero_grad()

        # loss = comupute_loss(criteria, target_texture, optimized_variable)  # + calc_TV_Loss(optimized_variable.unsqueeze(0))
        loss = criteria(target_texture, optimized_variable)

        loss.backward()
        optim.step()
        losses.append(loss.item())

        # if i % 200 == 0:
        #     for g in optim.param_groups:
        #         g['lr'] *= 0.5
        if i % 100 == 0 and output_dir is not None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.set_title(f"last-Loss: {losses[-1]}")
            ax.plot(np.arange(len(losses)), losses)
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    optimized_image = torch.clip(optimized_variable, -1, 1)

    return optimized_image


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def run_sigle():
    images = [
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/green_waves.jpg',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/cobbles.jpeg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures/soil.jpeg',
    ]

    losses = [
        # PatchSWDLoss(patch_size=9, num_proj=256, n_samples=2048),
        # PatchMMDLoss(patch_size=9, n_samples=2048)
        # LaplacyanLoss(PatchMMDLoss(patch_size=15, n_samples=512), weightening_mode=3, max_levels=2),
        # PatchMMDLoss(patch_size=7, n_samples=1024),
        # PatchSWDLoss(patch_size=19, n_samples=4096),
        # LaplacyanLoss(PatchMMDLoss(patch_size=19, n_samples=1024), weightening_mode=3, max_levels=2),
        WindowLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), window_size=32, stride=16),
        PyramidLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), weightening_mode=3, max_levels=3),
        LaplacyanLoss(PatchMMDLoss(patch_size=11, n_samples=1024, sample_same_locations=False), weightening_mode=3, max_levels=3)
    ]
    s = 128
    for path in images:
        for loss in losses:
            img_name = os.path.basename(os.path.splitext(path)[0])
            input_dir = f"Experiments/style_synthesis/{img_name}(crop-{s})"
            os.makedirs(input_dir, exist_ok=True)

            texture_image = cv2.imread(path)
            texture_image = crop_center(texture_image, s, s)

            cv2.imwrite(os.path.join(input_dir, f"org.png"), texture_image)

            train_dir = f"{input_dir}/{loss.name}"
            os.makedirs(train_dir, exist_ok=True)

            texture_image = cv2pt(texture_image).unsqueeze(0).to(device)
            img = optimize_texture(texture_image, loss, output_dir=train_dir, num_steps=1000, lr=0.05)
            vutils.save_image(img, os.path.join(input_dir, f"{loss.name}.png"), normalize=True)


if __name__ == '__main__':
    run_sigle()

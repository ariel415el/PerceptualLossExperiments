import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

import sys

from image_retargeting.utils import aspect_ratio_resize
from perceptual_mean_optimization.utils import cv2pt
import losses
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
    scale_x, scale_y = 1.5, 0.7
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
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/image_retargeting/images/fruit.png',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/patch_dist/girafs.png',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/singan/balloons.png',
    ]

    criterias = [

        losses.PyramidLoss(losses.PatchMMD_RBF(patch_size=11, stride=5, n_samples=4096, sample_same_locations=False), weightening_mode=3, max_levels=3),

    ]
    exp_name = "_".join([l.name for l in criterias])
    for path in images:
        img_name = os.path.basename(os.path.splitext(path)[0])
        input_dir = f"style_synthesis/{img_name}"
        os.makedirs(input_dir, exist_ok=True)

        texture_image = cv2.imread(path)
        texture_image = aspect_ratio_resize(texture_image, max_dim=256)
        # texture_image = crop_center(texture_image, 0.9)

        cv2.imwrite(os.path.join(input_dir, f"org.png"), texture_image)

        train_dir = f"{input_dir}/{exp_name}"
        os.makedirs(train_dir, exist_ok=True)

        img = optimize_texture(texture_image, criterias, output_dir=train_dir, num_steps=100000, lr=0.005)
        vutils.save_image(img, os.path.join(input_dir, f"{exp_name}.png"), normalize=True)


if __name__ == '__main__':
    run_sigle()

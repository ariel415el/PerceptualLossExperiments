import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.realpath(".."))
# from losses.mmd_exact_loss import MMDExact
from losses.utils import ListOfLosses
from losses.vgg_loss.vgg_loss import VGGFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1

    # img = np.clip(img, -1 + 1e-9, 1 - 1e-9)
    # img = np.arctanh(img)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def pt2cv(img):
    img = img
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_images(dir_path):
    images = []
    for fn in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, fn))
        # img = img[125 - 90:125 + 80, 125 - 75:125 + 75]
        img = cv2pt(img)
        images.append(img)

    return torch.stack(images)


def optimize_for_mean(images, criteria, output_dir, batch_size=1, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    N = images.shape[0]
    os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    criteria = criteria.to(device)

    optimized_variable = images.mean(0)[None, :]
    # optimized_variable *= 0
    # optimized_variable = torch.randn(optimized_variable.shape).to(device).clamp(-1,1)
    optimized_variable.requires_grad_(True)

    losses = []

    for i in tqdm(range(num_steps)):
        optim = torch.optim.Adam([optimized_variable], lr=lr)
        optim.zero_grad()

        for b in range(N // batch_size):
            batch_images = images[b * batch_size: (b+1) * batch_size]
            # batch_input = torch.tanh(optimized_variable.repeat(batch_size, 1, 1, 1))
            batch_input = optimized_variable.repeat(batch_size, 1, 1, 1)
            loss = criteria(batch_input, batch_images)
            loss = loss.mean()
            loss.backward()
            optim.step()
        losses.append(loss.item())

        if i % 50 == 0:
            lr *= 0.5

            plt.plot(np.arange(len(losses)), losses)
            plt.savefig(f"{output_dir}/train_loss.png")
            plt.clf()

            # optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
            optimized_image = optimized_variable.detach().cpu().numpy()[0]
            optimized_image = pt2cv(optimized_image)
            cv2.imwrite(f"{output_dir}/output-{i}.png", optimized_image)

    # optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
    optimized_image = optimized_variable.detach().cpu().numpy()[0]
    optimized_image = pt2cv(optimized_image)

    return optimized_image


def optimize_for_mean_fv(images, criteria, output_dir, batch_size=1, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    N = images.shape[0]
    os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    criteria = criteria.to(device)

    optimized_variable = images.mean(0)[None, :]
    # optimized_variable *= 0
    # optimized_variable = torch.randn(optimized_variable.shape).to(device).clamp(-1,1)
    optimized_variable.requires_grad_(True)

    losses = []

    for i in range(len(images)):
        activations = criteria.get_activations(images[i].unsqueeze(0))

    for i in tqdm(range(num_steps)):
        optim = torch.optim.Adam([optimized_variable], lr=lr)
        optim.zero_grad()

        for b in range(N // batch_size):
            batch_images = images[b * batch_size: (b+1) * batch_size]
            # batch_input = torch.tanh(optimized_variable.repeat(batch_size, 1, 1, 1))
            batch_input = optimized_variable.repeat(batch_size, 1, 1, 1)
            loss = criteria(batch_input, batch_images)
            loss = loss.mean()
            loss.backward()
            optim.step()
        losses.append(loss.item())

        if i % 50 == 0:
            lr *= 0.5

            plt.plot(np.arange(len(losses)), losses)
            plt.savefig(f"{output_dir}/train_loss.png")
            plt.clf()

            # optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
            optimized_image = optimized_variable.detach().cpu().numpy()[0]
            optimized_image = pt2cv(optimized_image)
            cv2.imwrite(f"{output_dir}/output-{i}.png", optimized_image)

    # optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
    optimized_image = optimized_variable.detach().cpu().numpy()[0]
    optimized_image = pt2cv(optimized_image)

    return optimized_image


if __name__ == '__main__':
    images_dir = 'clusters/2_faces/images'
    # output_dir = "VGG-Random"
    images = load_images(images_dir)

    creiterion = ListOfLosses([
            # LapLoss(n_channels=3, max_levels=1),
            # L2(),
            # torch.nn.MSELoss(),
            VGGFeatures(5, pretrained=True, post_relu=False),
            # MMDApproximate(batch_reduction='none', normalize_patch='channel_mean', pool_size=32, pool_strides=16),
            # SCNNNetwork(),
            # PatchRBFLaplacianLoss(patch_size=3, batch_reduction='none', normalize_patch='none', ignore_patch_norm=False, device=device)
    ]#, weights=[0.8, 0.2]
    )
    output_dir = images_dir + "_" + criterion.name
    img = optimize_for_mean(images, creiterion, output_dir, num_steps=1000, lr=0.001, batch_size=2)


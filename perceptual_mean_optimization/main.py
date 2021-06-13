import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

import sys

from GenerativeModels.utils.swd import swd
from losses.experimental_patch_losses import DoubleMMDAprox, DoubleMMD_PP, MMD_PPP
from losses.patch_mmd_pp import MMD_PP

sys.path.append(os.path.realpath(".."))
from losses.patch_mmd_loss import MMDApproximate
from losses.patch_loss import PatchRBFLaplacianLoss, PatchRBFLoss
from losses.l2 import L2
from losses.mmd_loss import MMD
from losses.lap1_loss import LapLoss
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
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_images(dir_path):
    images = []
    # for fn in sorted(os.listdir(dir_path)):
    for fn in [f"{i}.png" for i in range(6)]:
        img = cv2.imread(os.path.join(dir_path, fn))
        # img = img[125 - 90:125 + 80, 125 - 75:125 + 75]
        img = cv2pt(img)
        images.append(img)

    return torch.stack(images)


def optimize_for_mean(images, criteria, output_dir, weights, batch_size=1, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    criteria = criteria.to(device)
    swd_score = None
    optimized_variable = images.mean(0)[None, :]
    # optimized_variable *= 0
    # optimized_variable = torch.randn(optimized_variable.shape).to(device).clamp(-1,1)
    optimized_variable.requires_grad_(True)
    # optim = torch.optim.SGD([optimized_variable], lr=lr)
    optim = torch.optim.Adam([optimized_variable], lr=lr)
    losses = []
    swd_scores = []
    for i in tqdm(range(num_steps + 1)):
        if i % 500 == 0:
            optimized_image = optimized_variable
            vutils.save_image(optimized_image, f"{output_dir}/output-{i}.png", normalize=True)
            # Compute Sliced wassestein distances
            swd_score = swd(optimized_variable.repeat(len(images), 1, 1, 1), images, device='cpu')
            swd_scores.append((i, swd_score))

        optim.zero_grad()
        # batch_images = images[np.random.choice(range(len(images)), batch_size, replace=False)]
        batch_images = images
        batch_input = optimized_variable.repeat(batch_size, 1, 1, 1)
        loss = criteria(batch_input, batch_images)
        loss *= torch.tensor(weights).to(device)
        loss = loss.mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if i % 200 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.5
        if i % 100 == 0:
            plt.title(f"last-Loss: {losses[-1]}")
            plt.plot(np.arange(len(losses)), losses)
            plt.savefig(f"{output_dir}/train_loss.png")
            plt.clf()

            swd_scores_np = np.array(swd_scores)
            plt.plot(swd_scores_np[:, 0], swd_scores_np[:, 1])
            plt.title(f"last: {swd_scores_np[-1][1]}")
            plt.savefig(f"{output_dir}/swd_scores.png")
            plt.clf()

    optimized_image = torch.clip(optimized_variable, -1, 1)

    return optimized_image, swd_score


def batch_run():
    image_dirs = [
        'clusters/z_samples_new/1/data_neighbors1',
        'clusters/z_samples_new/4/data_neighbors4',
        'clusters/z_samples_new/6/data_neighbors6',
        # 'clusters/z_samples/10/data_neighbors10',
        # 'clusters/z_samples/6/latent_neighbors_direction6',
        # 'clusters/z_samples/10/data_neighbors10',
        # 'clusters/z_samples/10/data_neighbors10',
        # 'clusters/z_samples/10/data_neighbors10',
        # 'clusters/z_samples/10/reconstructions10',
        # 'clusters/z_samples/10/latent_neighbors_direction10',
        # 'clusters/z_samples/16/data_neighbors',
        # 'clusters/z_samples/16/reconstructions',
        # 'clusters/z_samples/16/latent_neighbors_direction',
        # 'clusters/stylegan/stylegan_128/images',
        # 'clusters/ffhq_jitters/00083_64/images',
        # 'clusters/increasing_variation/36126_s128_c_64/images',
        # 'clusters/increasing_variation/36096_s128_c_64/images'
        # 'clusters/increasing_variation/36096_s64_c_64/images'
        # 'clusters/z_samples/latent_neighbors_direction10'
        # 'clusters/00068_128/images',
    ]
    losses = [
        # L2(batch_reduction='none'),
        # VGGFeatures(pretrained=True),
        # VGGFeatures(pretrained=True, weights=[0,1,0,0,0,0]),
        # VGGFeatures(pretrained=True, weights=[0,0,1,0,0,0]),
        # VGGFeatures(pretrained=True, weights=[0,0,0,1,0,0]),
        # LapLoss(),
        # VGGFeatures(pretrained=True),
        # VGGFeatures(pretrained=True, weights=[0, 1, 1, 1, 1, 1]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True),
        # VGGFeatures(pretrained=False, norm_first_conv=False, reinit=True),
        # VGGFeatures(pretrained=False, norm_first_conv=False, reinit=False),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[0, 1, 0, 0, 0, 0]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[0, 1, 1, 0, 0, 0]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[0, 1, 1, 1, 0, 0]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[0, 1, 1, 1, 1, 0]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[0, 1, 1, 1, 1, 1]),
        # VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[1, 1, 1, 1, 1, 1]),


        # MMD_PP(device, patch_size=3, pool_size=32, pool_strides=16, r=512, sigma=0.06, normalize_patch='mean',
        #        weights=[0.001, 0.05, 1.0], batch_reduction='none'),
        MMD_PP(device, patch_size=7, pool_size=32, pool_strides=16, r=512, sigma=0.06, normalize_patch='mean',
               weights=[0.001, 0.5, 1.0], batch_reduction='none'),
        MMD_PP(device, patch_size=7, pool_size=32, pool_strides=16, r=512, sigma=0.06, normalize_patch='mean',
               weights=[0.01, 0.05, 1.0], batch_reduction='none'),
        MMD_PP(device, patch_size=7, pool_size=32, pool_strides=16, r=512, sigma=0.06, normalize_patch='mean',
               weights=[0.01, 0.5, 1.0], batch_reduction='none'),
    ]
    weights = [1, 1, 1, 1, 1, 1]
    # weights = [1, 1, 1]
    # weights = [1.0, 1.0, 1.0]
    num_images = 6
    root = 'batch_outputs'
    tag = f"_6{weights}"

    all_inputs = []
    all_results = []
    for images_dir in image_dirs:
        results = []
        images = load_images(images_dir)[:num_images]
        images_name = tag + "_" + os.path.basename(images_dir)
        os.makedirs(os.path.join(root, images_name), exist_ok=True)
        for loss in losses:
            output_dir = os.path.join(root, images_name, loss.name)
            img, swd_score = optimize_for_mean(images, loss, output_dir, weights=weights, num_steps=1000, lr=0.1,
                                               batch_size=num_images)
            vutils.save_image(img, os.path.join(root, images_name, f"{loss.name}.png"), normalize=True)
            results.append(img.cpu())

        vutils.save_image(images.mean(0)[None, :], os.path.join(root, images_name, "l2_mean.png"), normalize=True)

        all_inputs.append(images)
        all_results.append(torch.cat(results))
    vutils.save_image(torch.cat(all_inputs), os.path.join(root, f"{tag}_inputs.png"), normalize=True, nrow=len(images))
    vutils.save_image(torch.cat(all_results), os.path.join(root, f"{tag}_results.png"), normalize=True, nrow=len(results))


if __name__ == '__main__':
    batch_run()

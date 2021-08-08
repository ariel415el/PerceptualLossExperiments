import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
import sys

from losses.SSIM_1 import MS_SSIM
from losses.classic_losses.l2 import L2, L1
from losses.composite_losses.laplacian_losses import LaplacyanLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.composite_losses.window_loss import WindowLoss
from losses.gabor_loss.gabor import GaborPerceptualLoss

from losses.experimental_patch_losses import EngeneeredPerceptualLoss, SimplePatchLoss, MMD_PP, MMD_PPP
from losses.classic_losses.grad_loss import GradLoss, GradLoss3Channels
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.patch_loss import PatchRBFLoss
from losses.swd.lap_swd_loss import compute_lap_swd
from losses.swd.patch_swd import PatchSWDLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

sys.path.append(os.path.realpath(".."))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    # img *= 2
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def pt2cv(img):
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)

    return img


def load_images(dir_path, sort=True):
    images = []
    paths = os.listdir(dir_path) if not sort else sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
    for fn in paths:
        img = cv2.imread(os.path.join(dir_path, fn))
        img = cv2pt(img)
        # from style_transfer.utils import imload
        # img = imload(os.path.join(dir_path, fn))[0]

        images.append(img)

    return torch.stack(images)


def get_starting_point(images, mode):
    res = images.mean(0)
    for name in mode.split("+"):
        if 'gray' in name:
            res = torch.ones(res.shape) * torch.mean(res, dim=(0), keepdim=True)
        elif 'zeros' in name:
            res *= 0
        if 'blur' in name:
            from losses.composite_losses.laplacian_losses import conv_gauss, get_kernel_gauss
            res = conv_gauss(res.unsqueeze(0), get_kernel_gauss(size=5, sigma=3, n_channels=3))[0]
        if 'noise' in name:
            res += torch.randn(res.shape) * 0.15


    return res


def optimize_for_mean(optimized_variable, images, criteria, output_dir=None, batch_size=None,
                      num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    optimized_variable = optimized_variable.to(device)
    optimized_variable.requires_grad_(True)
    criteria = criteria.to(device)

    optim = torch.optim.Adam([optimized_variable], lr=lr)
    # optim = torch.optim.SGD([optimized_variable], lr=lr)
    losses = []
    for i in tqdm(range(num_steps + 1)):
        if i % 50 == 0 and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(optimized_variable.clone(), f"{output_dir}/output-{i}.png", normalize=True)

        optim.zero_grad()
        if batch_size is None:
            batch_size = len(images)
            batch_images = images
        else:
            batch_images = images[np.random.choice(range(len(images)), batch_size, replace=False)]

        loss = criteria(optimized_variable.repeat(batch_size, 1, 1, 1), batch_images)

        loss = loss.mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if i % 256 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.75
        if i % 100 == 0 and output_dir is not None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.set_title(f"last-Loss: {losses[-1]}")
            ax.plot(np.arange(len(losses)), losses)
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    optimized_image = torch.clip(optimized_variable, -1, 1)
    # optimized_image = optimized_variable

    return optimized_image


class Plotter:
    def __init__(self, n_losses, n_dirs):
        self.fig, self.axs = plt.subplots(n_dirs, n_losses + 1, figsize=(3 + 3 * n_losses, 3 * n_dirs))
        self.input_images = []

    def set_result(self, i, j, img, title=None):
        self.axs[i, j].imshow(img)
        if title:
            self.axs[i, j].set_title(title)
        self.axs[i, j].axis('off')
        self.axs[i, j].set_aspect('equal')

    def save_fig(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)
        plt.clf()

def split_to_unmatching_squares(images, starting_img):
    new_starting_img = torch.zeros(starting_img.shape)
    new_starting_img[:, 48:-16, 48:-16] = starting_img[:, 32:-32, 32:-32].clone()
    # starting_img = new_starting_img
    new_images = torch.zeros(images.shape)
    new_images[0, :, 16:-48, 16:-48] = images[0, :, 32:-32, 32:-32]
    # images = new_images

    return new_images, new_starting_img

def batch_run():
    image_dirs = [
        # Similar faces:
        # 'clusters/z_samples_new/1/data_neighbors1',
        # 'clusters/z_samples_new/4/data_neighbors4',
        # 'clusters/z_samples_new/6/data_neighbors6',
        # 'clusters/z_samples_new/10/data_neighbors10',
        'clusters/z_samples_new/16/data_neighbors16',
        # 'clusters/z_samples_new/51/data_neighbors51',
        # 'clusters/z_samples_new/55/data_neighbors55',
        # 'clusters/stylegan/stylegan_128/images',

        # Synthetic:
        # 'clusters/synthetic/box_dataset',
        # 'clusters/synthetic/color_box_dataset',

        # Jitters:
        # 'clusters/ffhq_jitters/00068_128',
        # 'clusters/ffhq_jitters/00083_128',
        # 'clusters/increasing_variation/36096_s128_c_64',
        # 'clusters/increasing_variation/36096_s128_c_128',
        # 'clusters/increasing_variation/36126_s128_c_64',
        # 'clusters/increasing_variation/36126_s128_c_128',

        # Textures:
        # 'clusters/textures/ball-on-grass_e-7_z-0.2',
        # 'clusters/textures/dry_grass_e-7_z-0.2',
        # 'clusters/textures/dry_needle_e-7_z-0.2',
        # 'clusters/textures/cobbles_e-7_z-0.2',
        # 'clusters/textures/long_grass_e-7_z-0.2',
        'clusters/textures/mixed/dry+longgrass',
        # 'clusters/textures/mixed/ferrs',
        # 'clusters/textures/mixed/wood_mix',

    ]
    pool = 32
    stride = 16
    from losses.SSIM_1 import SSIM
    from losses.SSIM_2 import SSIM_2
    losses = [
        # L2(batch_reduction='none'),
        # L1(batch_reduction='none'),
        # SSIM(nonnegative_ssim=False),
        # MS_SSIM(),
        # MS_SSIM(),
        # GradLoss(batch_reduction='none'),
        # GradLoss3Channels(batch_reduction='none'),
        # PatchRBFLoss(patch_size=3, sigma=0.1),
        # PatchRBFLoss(patch_size=11, sigma=0.02),
        # PatchRBFLoss(patch_size=11, sigma=0.02, normalize_patch='channel_mean'),
        # VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv1_1', 1.0)]),
        # VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv1_2', 1.0)])

        # SimplePatchLoss(patch_size=11, sigma=0.02, batch_reduction='none'),
        # PatchSWDLoss(patch_size=3, n_samples=None, num_proj=5000),
        # PatchSWDLoss(patch_size=11, n_samples=None, num_proj=1024),
        # PatchMMDLoss(patch_size=21, n_samples=1024),
        # WindowLoss(PatchSWDLoss(patch_size=11, n_samples=128), window_size=pool, stride=stride, batch_reduction='none'),
        # WindowLoss(PatchMMDLoss(patch_size=11, n_samples=512, sigmas=[0.02]), window_size=pool, stride=stride, batch_reduction='none'),
        # MMDApproximate(patch_size=11, pool_size=128, pool_strides=1, sigma=0.015, r=1024, batch_reduction='none', name='MMDApprox'),
        # VGGPerceptualLoss(pretrained=True, name='VGG-PT-Gram(convs-1-2-3)', features_metric_name='gram',
        #                   layers_and_weights=[('relu1_2', 1.0), ('relu2_2', 1.0), ('relu3_3', 1.0),
        #                                       ('relu4_3', 1.0), ('relu5_3', 1.0)]
        #                   )
        # PatchSWDLoss(patch_size=11, n_samples=None, num_proj=1024),
        # PatchSWDLoss(patch_size=11, stride=11, n_samples=None, num_proj=2048),
        # PatchSWDLoss(patch_size=11, stride=5, n_samples=None, num_proj=2048),
        # PatchMMDLoss(patch_size=11, stride=11, n_samples=None, sigmas=[0.02]),
        # WindowLoss(PatchSWDLoss(patch_size=11, n_samples=128), window_size=pool, stride=stride, batch_reduction='none'),
        # WindowLoss(PatchMMDLoss(patch_size=11, n_samples=512, sigmas=[0.02]), window_size=pool, stride=stride, batch_reduction='none'),
        PatchMMDLoss(patch_size=11, stride=11, n_samples=None, sigmas=[0.02], normalize_patch='channel_mean'),
        # MMDApproximate(patch_size=11, strides=1, pool_size=128, pool_strides=1, sigma=0.02, r=1024, batch_reduction='none', name='MMDApprox'),
        LossesList([
            PatchMMDLoss(patch_size=11, stride=11, n_samples=None, sigmas=[0.02]),
            GradLoss3Channels(batch_reduction='none')],
            weights=[1,0.1], name='MMD(p=11:11)+gradLoss'
                   ),
        PatchMMDLoss(patch_size=11, stride=5, n_samples=None, sigmas=[0.02]),
        LossesList([
            PatchMMDLoss(patch_size=11, stride=5, n_samples=None, sigmas=[0.02]),
            GradLoss3Channels(batch_reduction='none')],
            weights=[1, 0.1], name='MMD(p=11:11)+gradLoss'
        ),
        # WindowLoss(PatchSWDLoss(patch_size=11, n_samples=None, num_proj=256), window_size=pool, stride=stride, batch_reduction='none'),
        # WindowLoss(PatchMMDLoss(patch_size=11, n_samples=2048, sigmas=[0.02]), window_size=pool, stride=stride, batch_reduction='none'),
        # MMDApproximate(patch_size=11, pool_size=128, pool_strides=1, sigma=0.02, r=2048, batch_reduction='none', name='MMDApprox'),
        # VGGPerceptualLoss(pretrained=True),
        # GradLoss3Channels(),
        # PatchRBFLoss(patch_size=11, sigma=0.02),
        # VGGPerceptualLoss(pretrained=True, layers_and_weights=[('relu1_2', 1.0)], name='VGG-relu1_2'),
        # VGGPerceptualLoss(pretrained=True),
    ]

    num_images = 2
    lr = 0.005
    n_steps = 1000
    start_mode = 'mean'
    tag = f"similar"
    root = f'outputs/{num_images}_{start_mode}_{tag}'

    plotter = Plotter(len(losses), len(image_dirs))

    grad_criterion = GradLoss(batch_reduction='none')

    # RUN OPTIMIZATION AND MAIN PLOT
    for i, images_dir in enumerate(image_dirs):
        images_name = tag + "_" + os.path.basename(images_dir)
        images = load_images(images_dir)[:num_images]
        starting_img = get_starting_point(images, start_mode)
        # import torchvision.transforms.functional as F
        # starting_img = F.hflip(starting_img)
        # images, starting_img = split_to_unmatching_squares(images, starting_img)
        plotter.set_result(i, 0, pt2cv(starting_img.detach().cpu().numpy()), 'Starting img' if i == 0 else None)
        for j, loss in enumerate(losses):
            output_dir = os.path.join(root, images_name, loss.name)
            img = optimize_for_mean(starting_img.clone(), images, loss, output_dir, num_steps=n_steps, lr=lr)
            vutils.save_image(img, os.path.join(root, images_name, f"{loss.name}.png"), normalize=True)

            swd_score = compute_lap_swd(img.repeat(len(images), 1, 1, 1), images.to(device), device='cpu', return_by_resolution=True)
            # grad_loss = grad_criterion(img.repeat(len(images), 1, 1, 1), images.to(device))
            # title = f"{loss.name}" \
            #         f"\nSWD:{swd_score.mean():.1f}" \
                    # f"\nGradLoss min: {grad_loss.min():.3f}" \
                    # f"\nGradLoss avg: {grad_loss.mean():.3f}"
            title = loss.name
            plotter.set_result(i, j + 1, pt2cv(img.detach().cpu().numpy()), title  if i == 0 else None)

        plotter.input_images.append(images)

    plotter.save_fig(os.path.join(root, f"{tag}_results.png"))

    # SAVE INPUTS FOR REFERENCE
    vutils.save_image(torch.cat(plotter.input_images), os.path.join(root, f"{tag}_inputs.png"), normalize=True, nrow=num_images)


if __name__ == '__main__':
    batch_run()

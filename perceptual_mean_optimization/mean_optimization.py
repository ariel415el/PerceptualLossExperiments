import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

import torch
import torchvision.utils as vutils
from torchvision import transforms

import losses

from perceptual_mean_optimization.utils import pt2cv, load_images, Plotter, get_pyramid

sys.path.append(os.path.realpath(".."))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


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
        if 'split' in name:
            res = images[0].clone()
            k = res.shape[2] // len(images)
            for i in range(1, len(images)):
                res[:, :, i*k:(i+1)*k] = images[i].clone()[:, :, i*k:(i+1)*k]
                # res[:, i*k:(i+1)*k] = images[i].clone()[:, i*k:(i+1)*k]

    return res


def optimize_patch_distributions(starting_image, target_images, criteria, output_dir, num_steps=400, lr=0.003, pbar=None):
    """
    :param starting_image: tensor of shape (C, H, W)
    :param target_images: tensor of shape (N, C, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)

    target_images = target_images.to(device)
    optimized_variable = starting_image.clone().unsqueeze(0).to(device)
    optimized_variable.requires_grad_(True)
    criteria = criteria.to(device)
    optim = torch.optim.Adam([optimized_variable], lr=lr)

    losses = []
    iterations = range(1, num_steps + 1)
    if not pbar:
        iterations = tqdm(iterations)
    vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-0.png", normalize=True)
    for i in iterations:
        optim.zero_grad()

        loss = criteria(optimized_variable.repeat(len(target_images), 1, 1, 1), target_images)

        loss = loss.mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if i % 1000 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.9
        if i % 100 == 0:
            vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-{i}.png", normalize=True)
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            # ax.plot(np.arange(len(losses)), np.log(losses))
            ax.plot(np.arange(len(losses)), losses)
            # ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)
        if pbar:
            pbar.update(1)
    return torch.clip(optimized_variable.detach()[0], -1, 1)


def optimize_multi_scale_patch_distribution(starting_img, target_images, criteria, output_dir):
    n_scales = 3
    perc = 0.69
    lr = 0.005
    n_steps = 1000
    os.makedirs(output_dir, exist_ok=True)
    pyramids = get_pyramid(target_images, n_scales, perc)

    synth_image = get_pyramid(starting_img.unsqueeze(0), n_scales, perc)[0][0]
    # synth_image = torch.randn((3, pyramids[0].shape[2], pyramids[0].shape[3])) * 0.1

    pbar = tqdm(total=len(pyramids) * n_steps, position=0, leave=True)
    for lvl, lvl_imgs in enumerate(pyramids):
        pbar.set_description(f"{criteria.name}: Lvl-{lvl}")
        if lvl > 0:
            synth_image = transforms.Resize((lvl_imgs.shape[2], lvl_imgs.shape[3]), antialias=True)(synth_image)

        lvl_output_dir = os.path.join(output_dir, str(lvl))
        vutils.save_image(lvl_imgs, os.path.join(output_dir, f"targets-{lvl}.png"), normalize=True)
        vutils.save_image(synth_image, os.path.join(output_dir, f"org-{lvl}.png"), normalize=True)

        synth_image = optimize_patch_distributions(synth_image, lvl_imgs, criteria, lvl_output_dir, n_steps, lr, pbar)
        lr *= 0.9  # n_scales / 10

        vutils.save_image(synth_image, os.path.join(output_dir, f"final-{lvl}.png"), normalize=True)

    return synth_image


def optimize_mean(image_dirs, losses, n_images, start_mode, output_dir_root, multi_scale):
    lr = 0.001
    n_steps = 15000
    plotter = Plotter(len(losses), len(image_dirs))

    # RUN OPTIMIZATION AND MAIN PLOT
    for i, images_dir in enumerate(image_dirs):
        images_name = os.path.basename(images_dir)
        images = load_images(images_dir)[:n_images]
        # starting_img = get_starting_point(images, start_mode)
        starting_img = get_starting_point(load_images('clusters/z_samples_new/16/data_neighbors16')[:10], start_mode)

        plotter.set_result(i, 0, pt2cv(starting_img.detach().cpu()), 'Starting img' if i == 0 else None)

        for j, loss in enumerate(losses):
            output_dir = os.path.join(output_dir_root, images_name, loss.name)

            if multi_scale:
                img = optimize_multi_scale_patch_distribution(starting_img.clone(), images, loss, output_dir)
            else:
                img = optimize_patch_distributions(starting_img.clone(), images, loss, output_dir, num_steps=n_steps, lr=lr)
            img = img.cpu()
            loss = loss.cpu()

            vutils.save_image(img, os.path.join(output_dir_root, images_name, f"{loss.name}.png"), normalize=True)

            title = f"Loss opt:{np.mean([loss(x.unsqueeze(0), img.unsqueeze(0)).item() for x in images]):.3f}" \
                    f"\nLoss start:{np.mean([loss(x.unsqueeze(0),starting_img.unsqueeze(0)).item() for x in images]):.3f}" \
            # title = loss.name
            if i == 0:
                title = f"{loss.name}\n{title}"
            plotter.set_result(i, j + 1, pt2cv(img), title)

        plotter.input_images.append(images)

    plotter.save_fig(os.path.join(output_dir_root, f"results.png"))

    # SAVE INPUTS FOR REFERENCE
    vutils.save_image(torch.cat(plotter.input_images), os.path.join(output_dir_root, f"inputs.png"), normalize=True, nrow=len(image_dirs))


if __name__ == '__main__':
    image_dirs = [
        # 'clusters/view/balloons',
        # 'clusters/view/birds',
        # 'clusters/view/birds_dusk',
        # 'clusters/view/colos',
        # 'clusters/view/grass',

        # Similar faces:
        # 'clusters/z_samples_new/1/data_neighbors1',
        # 'clusters/z_samples_new/4/data_neighbors4',
        'clusters/z_samples_new/6/data_neighbors6',
        # 'clusters/z_samples_new/10/data_neighbors10',
        # 'clusters/z_samples_new/16/data_neighbors16',
        # 'clusters/z_samples_new/51/data_neighbors51',
        # 'clusters/z_samples_new/55/data_neighbors55',
        # 'clusters/z_samples_new/all',
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
        # 'clusters/textures/mixed/dry+longgrass',
        # 'clusters/textures/mixed/ferrs',
        # 'clusters/textures/mixed/wood_mix',

    ]
    losses = [
        losses.MMDApproximate(patch_size=11, strides=1, sigma=0.01, r=128, pool_size=128, pool_strides=1),
        losses.PatchSWDLoss(patch_size=11, stride=3, num_proj=512),
        losses.PatchMMD_RBF(patch_size=11, stride=3),#, sigmas=[0.01]),
        # losses.PatchMMD_DotProd(patch_size=11, stride=6),
        # losses.PatchMMD_SSIM(patch_size=11, stride=6),
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv2_2', 1)]),
        # losses.SSIM(patch_size=11,sigma=1.5),
        # losses.SSIM(patch_size=5,sigma=1.5),
    ]


    n_images = 10
    start_mode = 'mean'
    # start_mode = 'mean'
    multi_scale = False
    tag = f""
    output_dir = f'outputs/{n_images}_{start_mode}_{tag}' + ('_MS' if multi_scale else '')

    optimize_mean(image_dirs, losses, n_images, start_mode, output_dir, multi_scale)

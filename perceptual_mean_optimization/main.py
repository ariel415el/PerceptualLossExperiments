import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

import sys

from losses.composite_losses.laplacian_losses import LaplacyanLoss
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.patch_mmd_loss import MMDApproximate, MMDExact
from losses.patch_mmd_pp import MMD_PP
from losses.swd.lap_swd_loss import compute_lap_swd
from losses.experimental_patch_losses import MMD_PPP, SWD_PPP, GradLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.window_loss import WindowLoss
from losses.swd.swd import PatchSWDLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

sys.path.append(os.path.realpath(".."))
from losses.patch_loss import PatchRBFLoss
from losses.l2 import L2

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
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_images(dir_path, sort=True):
    images = []
    paths = os.listdir(dir_path) if not sort else sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
    for fn in paths:
        # for fn in [f"{i}.png" for i in range(6)]:
        img = cv2.imread(os.path.join(dir_path, fn))
        # img = img[125 - 90:125 + 80, 125 - 75:125 + 75]
        img = cv2pt(img)
        images.append(img)

    return torch.stack(images)

def split_to_lines(txt, n):
    return "\n".join([txt[i:i + n] for i in range(0, len(txt), n)])

def optimize_for_mean(images, criteria, output_dir=None, weights=None, batch_size=None, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    images = images.to(device)
    criteria = criteria.to(device)

    optimized_variable = images.mean(0)
    # optimized_variable = torch.ones((images.shape[1:])).to(device) * torch.mean(images, dim=(0, 2, 3), keepdim=True)[0]
    # optimized_variable += torch.randn(optimized_variable.shape).to(device).clamp(-1, 1) * 0.1

    optimized_variable.requires_grad_(True)
    # optim = torch.optim.SGD([optimized_variable], lr=lr)
    optim = torch.optim.Adam([optimized_variable], lr=lr)
    losses = []
    for i in tqdm(range(num_steps + 1)):
        if i % 500 == 0 and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            optimized_image = optimized_variable
            vutils.save_image(optimized_image, f"{output_dir}/output-{i}.png", normalize=True)

        optim.zero_grad()
        if batch_size is None:
            batch_size = len(images)
            batch_images = images
        else:
            batch_images = images[np.random.choice(range(len(images)), batch_size, replace=False)]
        batch_input = optimized_variable.repeat(batch_size, 1, 1, 1)
        # loss = criteria(batch_input, batch_images) + (optimized_variable**2).mean()
        loss = criteria(batch_input, batch_images)# + calc_TV_Loss(optimized_variable.unsqueeze(0))
        if weights is not None:
            loss *= torch.tensor(weights).to(device)
        loss = loss.mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if i % 200 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.5
        if i % 100 == 0 and output_dir is not None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.set_title(f"last-Loss: {losses[-1]}")
            ax.plot(np.arange(len(losses)), losses)
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    optimized_image = torch.clip(optimized_variable, -1, 1)

    return optimized_image


def run_sigle():
    root = 'single_outputs'
    # images_dir = 'clusters/vid_clusters/woman_speech_128/images'
    # images_dir = 'clusters/stylegan/stylegan_128/images'
    # images_dir = 'clusters/z_samples_new/6/data_neighbors6'
    # images = load_images(images_dir, sort=False)[:1]

    images_dir = 'green_waves'
    images = cv2.imread('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/green_waves.jpg')
    images = cv2.resize(images, (128,128))
    # images = cv2.resize(images, (32,32))
    images = cv2pt(images).unsqueeze(0)

    # loss = VGGPerceptualLoss(pretrained=True)
    # loss = L2()
    # loss = MMDApproximate(patch_size=7, pool_size=32, pool_strides=8, r=64, sigma=0.06,
    #                       normalize_patch='channel_mean')
    # loss = LaplacyanLoss(PatchSWDLoss(batch_reduction='none', num_proj=512, n_samples=None), weightening_mode=3, max_levels=2)
    # loss = MMDApproximate(pool_size=128, strides=1, patch_size=3, r=1024)
    # loss = LapSWDLoss(batch_reduction='none')
    loss = PatchSWDLoss(patch_size=7, batch_reduction='none', num_proj=512, n_samples=None)
    # loss = LaplacyanLoss(PatchSWDLoss(patch_size=7, batch_reduction='none', num_proj=512, n_samples=128), weightening_mode=3, max_levels=2)
    # loss = WindowLoss(PatchSWDLoss(patch_size=3, num_proj=512, n_samples=None), batch_reduction='none')
    # loss = WindowLoss(L1())
    output_dir = os.path.join(root, os.path.basename(images_dir), loss.name)
    os.makedirs(output_dir, exist_ok=True)

    img = optimize_for_mean(images, loss, output_dir=output_dir, num_steps=1000, lr=0.1, batch_size=1)
    vutils.save_image(img, os.path.join(output_dir, f"{loss.name}.png"), normalize=True)


def batch_run():
    image_dirs = [
        'clusters/z_samples_new/1/data_neighbors1',
        'clusters/z_samples_new/4/data_neighbors4',
        # 'clusters/z_samples_new/6/data_neighbors6',
        # 'clusters/z_samples_new/10/data_neighbors10',
        # 'clusters/z_samples_new/16/data_neighbors16',
        # 'clusters/z_samples_new/51/data_neighbors51',
        # 'clusters/z_samples_new/55/data_neighbors55',
        # 'clusters/stylegan/stylegan_128/images',
        # 'clusters/ffhq_jitters/00068_128',
        # 'clusters/ffhq_jitters/00083_128',
        # 'clusters/increasing_variation/36126_s128_c_64/images',
        # 'clusters/increasing_variation/36096_s128_c_32',
        # 'clusters/increasing_variation/36096_s128_c_64',
        # 'clusters/increasing_variation/36096_s128_c_128',
        # 'clusters/increasing_variation/36126_s128_c_32',
        # 'clusters/increasing_variation/36126_s128_c_64',
        # 'clusters/increasing_variation/36126_s128_c_128',
        # 'clusters/increasing_variation/36096_s64_c_64/images'
        # 'clusters/z_samples/latent_neighbors_direction10'
        # 'clusters/00068_128/images',
    ]

    losses = [
        # MMDApproximate(pool_size=128, strides=1, patch_size=3, r=1024),
        L2(batch_reduction='none'),
        # GradLoss(batch_reduction='none')
        PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, batch_reduction='none'),
        # WindowLoss(PatchSWDLoss(patch_size=3, num_proj=64, n_samples=64), batch_reduction='none', window_size=32,stride=16),
        MMDApproximate(patch_size=3, pool_size=32, pool_strides=16, r=256, batch_reduction='none'),
        MMD_PP(),
        MMD_PPP(),
        # MMDApproximate(patch_size=7, strides=1, sigma=0.1, r=64, pool_size=32, pool_strides=16,
        #                batch_reduction='none', normalize_patch='channel_mean', pad_image=True),
        # MMDApproximate(patch_size=7, strides=1, sigma=0.3, r=64, pool_size=32, pool_strides=16,
        #                batch_reduction='none', normalize_patch='channel_mean', pad_image=True),
        # MMDApproximate(patch_size=7, strides=1, sigma=0.5, r=64, pool_size=32, pool_strides=16,
        #                batch_reduction='none', normalize_patch='channel_mean', pad_image=True),
        # LossesList([
        #     PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True,
        #                  batch_reduction='none'),
        #     MMDApproximate(patch_size=7, strides=1, sigma=0.03, r=64, pool_size=32, pool_strides=16,
        #                    batch_reduction='none',
        #                    normalize_patch='channel_mean', pad_image=True)
        # ], weights=[0.0001, 1.0], name='MMD++(p=11)'),
        # WindowLoss(PatchSWDLoss(patch_size=11, num_proj=64, n_samples=None), batch_reduction='none', window_size=32, stride=16),
        # WindowLoss(PatchSWDLoss(patch_size=15, num_proj=64, n_samples=None), batch_reduction='none', window_size=64, stride=16),
        VGGPerceptualLoss(pretrained=True, batch_reduction='none'),
        # L2(batch_reduction='none'),
        # MMD_PP(),
        # LossesList([
        #     L2(batch_reduction='none'),
        #     PatchRBFLoss(patch_size=3, sigma=0.06, pad_image=True,
        #                  batch_reduction='none'),
        #     WindowLoss(PatchSWDLoss(patch_size=3, num_proj=128, n_samples=256), batch_reduction='none')
        # ], weights=[0.002, 0.1, 1.0], name='SWD++'),
        # SWD_PPP(weights=[0, 3, 0.5]),
        # PyramidLoss(PatchSWDLoss(patch_size=7, batch_reduction='none', num_proj=128, n_samples=128), weightening_mode=3, max_levels=2),
        # LaplacyanLoss(PatchSWDLoss(patch_size=7, batch_reduction='none', num_proj=128, n_samples=128), weightening_mode=3, max_levels=2),
        # PyramidLoss(WindowLoss(PatchSWDLoss(patch_size=3, num_proj=128, n_samples=256), batch_reduction='none'), weightening_mode=3, max_levels=2)

        # LapSWDLoss(batch_reduction='none'),

    ]

    # for s in [0.005, 0.01, 0.02, 0.04]:
    #     losses.append(
    #         MMDApproximate(patch_size=3, strides=1, sigma=s, r=64, pool_size=32, pool_strides=16,
    #                        batch_reduction='none',
    #                        normalize_patch='channel_mean', pad_image=True)
    #     )

    num_images = 2
    # weights = [1] * num_images
    weights = None
    root = 'outputs/test'
    tag = f"_{num_images}-{weights}"

    all_inputs = []
    swd_scores = dict()
    fig, axs = plt.subplots(len(image_dirs), len(losses))  # , gridspec_kw={'wspace':0, 'hspace':0.25})

    grad_criterion = GradLoss()

    # RUN OPTIMIZATION AND MAIN PLOT
    for i, images_dir in enumerate(image_dirs):
        images = load_images(images_dir)[:num_images]
        images_name = tag + "_" + os.path.basename(images_dir)
        os.makedirs(os.path.join(root, images_name), exist_ok=True)
        for j, loss in enumerate(losses):
            output_dir = os.path.join(root, images_name, loss.name)
            img = optimize_for_mean(images, loss, output_dir, weights=weights, num_steps=200, lr=0.1)
            vutils.save_image(img, os.path.join(root, images_name, f"{loss.name}.png"), normalize=True)

            swd_score = compute_lap_swd(img.repeat(len(images), 1, 1, 1), images.to(device), device='cpu',
                                    return_by_resolution=True)
            # swd_scores[loss.name] = ",".join([f"{s.item():.1f}" for s in swd_score])
            if loss.name in swd_scores:
                swd_scores[loss.name].append(swd_score.numpy())
            else:
                swd_scores[loss.name] = [swd_score.numpy()]

            grad_loss = grad_criterion(img.repeat(len(images), 1, 1, 1), images.to(device))
            axs[i, j].imshow(pt2cv(img.detach().cpu().numpy()))
            # name = '\n'.join(loss.name.split("="))
            axs[i, j].set_title(f"{loss.name.split('(')[0]}\nSWD:{swd_score.mean():.1f}\nGradLoss: {grad_loss:.2f}", size=7)
            axs[i, j].axis('off')
            axs[i, j].set_aspect('equal')

        all_inputs.append(images)

    plt.tight_layout()
    fig.savefig(os.path.join(root, f"{tag}_results.png"))
    plt.clf()

    # SAVE INPUTS FOR REFERENCE
    vutils.save_image(torch.cat(all_inputs), os.path.join(root, f"{tag}_inputs.png"), normalize=True, nrow=num_images)

    # Plot SWD statistics over different layers
    x_middles = np.arange(len(losses))*2
    w = 0.15
    n_boxs_per_batch = len(swd_score)
    # plt.figure(figsize=(10, 10))
    plt.figure()
    colors = ['r', 'g', 'b', 'c', 'y']
    for level in range(len(swd_score)):
        values = []
        for loss in losses:
            values.append(np.array(swd_scores[loss.name])[:, level])
        pos = x_middles + (w * (1 - n_boxs_per_batch) / 2) + level * w
        plt.boxplot(values, positions=pos, vert=True, patch_artist=True, widths=w,
                    boxprops=dict(facecolor=colors[level], color=colors[level]))
        plt.plot([], [], c=colors[level], label=f"Level-{level}")
    plt.xticks(x_middles, [l.name.split('(')[0] for l in losses], rotation=70)
    plt.legend()

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fancybox=True, shadow=True)
    plt.legend()
    plt.title(f"Per level SWD scores of mean optimization outptut:\nAverraged over {len(image_dirs)} image sets")
    plt.grid()
    plt.ylabel("SWD")
    plt.tight_layout()
    plt.savefig(os.path.join(root, f"{tag}-swds.png"))
    plt.clf()


if __name__ == '__main__':
    batch_run()
    # run_sigle()

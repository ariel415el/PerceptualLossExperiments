import cv2
import numpy as np
import torch
import os

from matplotlib import pyplot as plt

from Experiments.create_mean_optimization_sets import center_crop_image_to_square
import losses
from perceptual_mean_optimization.utils import cv2pt
from style_transfer.utils import imload, calc_TV_Loss
from tqdm import tqdm
import torchvision.utils as vutils

from torchvision import transforms


def imload(path, imsize, square_crop=True):
    img = cv2.imread(path)
    if square_crop:
        img = center_crop_image_to_square(img)
    img = cv2pt(img)
    img = transforms.Resize((imsize,imsize), antialias=True)(img)
    return img

def style_mix_optimization(content_image, style_image, content_criteria, style_criteria, lr, max_iter, save_path, start_from='noise'):
    """
    Optimize an input image that mix style and content of specific two other images
    """
    losses = []
    os.makedirs(save_path, exist_ok=True)
    vutils.save_image(torch.clip(content_image, -1, 1), os.path.join(save_path, f"content_image.png"), normalize=True)
    vutils.save_image(torch.clip(style_image, -1, 1), os.path.join(save_path, f"style_image.png"), normalize=True)

    if start_from == 'noise':
        target_img = torch.randn(content_image.shape).to(device).float() * 0.1
    elif start_from == '+noise':
        target_img = content_image.clone() + torch.randn(content_image.shape).to(device).float() * 0.1
    else:
        target_img = content_image.clone()
    target_img.requires_grad_(True)

    optimizer = torch.optim.Adam([target_img], lr=lr)
    pbar = tqdm(range(max_iter + 1))
    # target_img.data.clamp_(-1, 1)
    for iteration in pbar:
        if iteration % 100 == 0:
            vutils.save_image(torch.clip(target_img, -1, 1), os.path.join(save_path, f"iter-{iteration}.png"), normalize=True)
            # plt.plot(range(len(losses)), losses)
            plt.plot(range(len(losses)), np.log(losses))
            plt.savefig(os.path.join(save_path, f"train_loss.png"))
            plt.clf()
        style_loss = style_criteria(target_img, style_image)
        content_loss = content_criteria(target_img, content_image)

        total_loss = style_loss * style_weight + content_loss + 1e-3*calc_TV_Loss(target_img)
        # total_loss = style_loss * style_weight + content_loss #+ 15 *calc_TV_Loss(target_img)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # print loss logs
        losses.append(total_loss.item())
        if iteration % 100 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.95
        pbar.set_description(f"total_loss: {total_loss}")

    res = torch.clip(target_img, -1, 1)
    return res


def load_and_run(content_img_path, style_img_path, style_loss, content_loss):
    style_img_name = os.path.splitext(os.path.basename(style_img_path))[0]

    content_img_name = os.path.splitext(os.path.basename(content_img_path))[0]

    train_dir = f"{outputs_dir}/{content_img_name}-to-{style_img_name}/{style_weight}xS-({style_loss.name})+C({content_loss.name})_from-{start_from}"

    content_img = imload(content_img_path, imsize).unsqueeze(0).to(device)
    style_img = imload(style_img_path, imsize).unsqueeze(0).to(device)

    mix = style_mix_optimization(content_img, style_img, content_loss.to(device), style_loss.to(device), lr, max_iter, train_dir, start_from)

    return content_img, style_img, mix

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    outputs_dir = 'outputs/outputs'
    max_iter = 3000
    lr = 0.05
    batch_size = 1
    imsize = 256
    style_weight = 30
    start_from = 'content'
    DIR='/home/ariel/university/GPDM/GPDM/images/style_transfer/'
    all_content_images = [
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/lenna.jpg'
        # f'{DIR}/content/home_alone.jpg',
        # f'{DIR}/content/trump.jpg',
        # f'{DIR}/content/cat1.jpg',
        # f'{DIR}/content/bin2.jpg',
        # f'{DIR}/content/golden_gate.jpg',
        # f'{DIR}/content/hillary1.jpg',
        # 'imgs/content/chicago.jpg',
        # 'imgs/content/golden_gate.jpg',
        # 'imgs/cx_images/cat1.jpg',

    ]

    all_style_images = [
        '/home/ariel/university/GPDM/GPDM/images/analogies/duck_mosaic.jpg'
        # f'{DIR}/style/scream.jpg',
        # f'{DIR}/style/mondrian.jpg',
        # f'{DIR}/style/starry_night.jpg',
        # f'{DIR}/style/brick.jpg',
        # f'{DIR}/style/scream.jpg',
        # f'{DIR}/style/thick_oil.jpg',
        # 'imgs/style/yellow_sunset.jpg',
        # 'imgs/style/starry_night.jpg',
        # 'imgs/style/scream.jpg',
    ]

    all_content_losses = [
                losses.NoLoss(),
                # losses.PyramidLoss(losses.GradLoss3Channels(), max_levels=3, weightening_mode=[1, 0, 0, 0]),
                # losses.VGGPerceptualLoss(pretrained=True, features_metric_name='l2', layers_and_weights=[('relu3_3', 1)])
                # losses.VGGPerceptualLoss(pretrained=True, features_metric_name='cx', layers_and_weights=[('relu4_2', 1)])
                # GradLoss()
                # losses.L2()
                # losses.VGGPerceptualLoss(pretrained=True, features_metric_name='cx',layers_and_weights=[('relu4_2', 1.0)])
    ]

    all_style_losses = [
            losses.VGGPerceptualLoss(pretrained=True, features_metric_name='gram',
                 layers_and_weights=[('relu1_2', 1.0), ('relu2_2', 1.0), ('relu3_3', 10.0), ('relu4_3', 5.0), ('relu5_3', 1.0)]),
            # losses.VGGPerceptualLoss(pretrained=True, features_metric_name='cx',
            #       layers_and_weights=[('relu1_2', 1.0), ('relu2_2', 1.0), ('relu3_2', 1.0), ('relu4_2', 1.0)]),
            # losses.PatchMMD_RBF(patch_size=7, stride=3),
            # losses.PatchSWDLoss(patch_size=11, stride=1, num_proj=200, normalize_patch='none'),
            # losses.PatchCoherentLoss(patch_size=11, stride=9),
            # losses.MMDApproximate(patch_size=7, strides=1, pool_size=-1, sigma=0.03, normalize_patch='mean')
    ]

    content_loss = all_content_losses[0]
    style_loss = all_style_losses[0]
    all_rows = []
    first_row = []
    for i, content_img_path in enumerate(all_content_images):
        row = []
        for style_img_path in all_style_images:
            content_img, style_img, mix = load_and_run(content_img_path, style_img_path, style_loss, content_loss)
            row.append(mix.cpu())
            if i == 0:
                first_row.append(style_img.cpu())
        row = [content_img.cpu()] + row
        all_rows.append(torch.cat(row, dim=0))
    first_row = torch.cat([torch.ones_like(first_row[0])] + first_row, axis=0)
    all_rows = [first_row] + all_rows
    vutils.save_image(torch.cat(all_rows, axis=0), f"{outputs_dir}/{style_loss.name}_output.png", normalize=True, nrow=1 + len(all_style_images))

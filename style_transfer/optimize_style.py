import cv2
import numpy as np
import torch
import os

from matplotlib import pyplot as plt

from Experiments.create_mean_optimization_sets import center_crop_image_to_square
from losses.classic_losses.grad_loss import GradLoss3Channels, GradLoss
from losses.classic_losses.l2 import L2
from losses.composite_losses.laplacian_losses import LaplacyanLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.empty_loss import NoLoss
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.swd.patch_swd import PatchSWDLoss
from perceptual_mean_optimization.utils import cv2pt
from style_transfer.utils import imload, calc_TV_Loss
from tqdm import tqdm
import torchvision.utils as vutils

from losses.vgg_loss.vgg_loss import VGGFeatures, get_features_metric, layer_names_to_indices, VGGPerceptualLoss

from torchvision import transforms


def imload(path, imsize, square_crop=True):
    img = cv2.imread(path)
    img = center_crop_image_to_square(img)
    img = cv2pt(img)
    img = transforms.Resize((imsize,imsize), antialias=True)(img)
    return img

def style_mix_optimization(content_image, style_image, content_criteria, style_criteria, lr, max_iter, save_path, device):
    """
    Optimize an input image that mix style and content of specific two other images
    """
    losses = []
    os.makedirs(save_path, exist_ok=True)
    vutils.save_image(torch.clip(content_image, -1, 1), os.path.join(save_path, f"content_image.png"), normalize=True)
    vutils.save_image(torch.clip(style_image, -1, 1), os.path.join(save_path, f"style_image.png"), normalize=True)

    # target_img = torch.randn(content_image.shape).to(device).float() * 0.5
    target_img = content_image.clone()
    # target_img = torch.ones(style_image.shape).to(device) * torch.mean(style_image.clone(), dim=(2,3), keepdim=True) + torch.randn(content_image.shape).to(device).float() * 0.2
    # target_img = torch.zeros(style_image.shape).to(device)
    # target_img = torch.mean(content_image.clone(), dim=(2,3), keepdim=True)
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

        # total_loss = style_loss * style_weight# + content_loss + calc_TV_Loss(target_img)
        total_loss = style_loss * style_weight + content_loss# + calc_TV_Loss(target_img)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # print loss logs
        losses.append(total_loss.item())
        if iteration % 50 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.9
        pbar.set_description(f"total_loss: {total_loss}")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    max_iter = 1000
    lr = 0.05
    batch_size = 1
    imsize = 256
    style_weight = 30

    content_loss = NoLoss()
    # content_loss = VGGPerceptualLoss(pretrained=True, features_metric_name='l2', layers_and_weights=[('conv3_3', 1)])
    # content_loss = GradLoss()
    # content_loss = L2()


    style_loss = VGGPerceptualLoss(pretrained=True, features_metric_name='gram', layers_and_weights=[('relu1_2', 1), ('relu2_2', 1), ('relu3_3', 1), ('relu4_3', 1), ('relu5_3', 1)])
    # style_loss = PatchSWDLoss(patch_size=3, stride=1, num_proj=1024, normalize_patch='none')
    # style_loss = PatchSWDLoss(patch_size=11, stride=3, normalize_patch='none')
    # style_loss = LossesList([
    #     PatchSWDLoss(patch_size=3, stride=1, num_proj=256),
    #     PatchSWDLoss(patch_size=11, stride=1, num_proj=256),
    # ], weights=[1,1])
    # style_loss = PatchMMDLoss(patch_size=11, stride=3)
    # style_loss = MMDApproximate(patch_size=11, strides=3, pool_size=128, pool_strides=128, sigma=0.1)

    tag = ''
    style_img_path = 'imgs/style/starry_night.jpg'
    # style_img_path = 'imgs/style/scream.jpg'
    # style_img_path = 'imgs/style/abstraction.jpg'
    # style_img_path = 'imgs/content/chicago.jpg'
    style_img_name = os.path.splitext(os.path.basename(style_img_path))[0]

    content_img_path = 'imgs/content/chicago.jpg'
    # content_img_path = 'imgs/content/home_alone.jpg'
    # content_img_path = 'imgs/content/cornell.jpg'
    content_img_name = os.path.splitext(os.path.basename(content_img_path))[0]

    train_dir = f"outputs/optimize_output_new/{content_img_name}_{style_img_name}/S({style_loss.name})_C({content_loss.name})-{tag}"


    content_img = imload(content_img_path, imsize).unsqueeze(0).to(device)
    style_img = imload(style_img_path, imsize).unsqueeze(0).to(device)

    style_mix_optimization(content_img, style_img, content_loss.to(device), style_loss.to(device), lr, max_iter, train_dir, device)

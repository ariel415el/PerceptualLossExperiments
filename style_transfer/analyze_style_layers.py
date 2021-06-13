import os

import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from losses.vgg_loss.contextual_loss import contextual_loss
from losses.vgg_loss.gram_loss import gram
from losses.vgg_loss.vgg_loss import VGGFeatures, get_features_metric
from style_transfer.utils import imload, imsave, calc_loss, calc_TV_Loss
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_gram_similarity(image, loss_network, layer_indices, features_metric, save_path, max_iter=100):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    output_image = torch.randn(image.shape).cuda().float() * 0.225
    # output_image = image.clone()
    output_image.requires_grad_(True)
    optimizer = torch.optim.Adam([output_image], lr=0.01)

    image_activations = loss_network.get_activations(image)
    image_activations = [image_activations[i] for i in layer_indices]

    pbar = tqdm(range(max_iter + 1))
    for iteration in pbar:
        target_activations = loss_network.get_activations(output_image)
        target_activations = [target_activations[i] for i in layer_indices]

        style_loss = calc_loss(target_activations, image_activations, features_metric)
        # tv_loss = calc_TV_Loss(output_image)

        total_loss = style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # print loss logs
        if iteration % 250 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.95

            pbar.set_description(f"total_loss: {total_loss.item()}")
    imsave(output_image.cpu(), save_path)

def plot_gram_matrix(image, layers_name, loss_network, save_path):
    with torch.no_grad():
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_activations = loss_network.get_activations(image)
        mat = gram(image_activations[layers_name]).cpu().numpy()
        plt.matshow(mat[0])
        plt.savefig(save_path)
        plt.clf()

def run_batch_experiments():
    train_dir = 'outputs/layer_analysis_new'
    images = [
        'imgs/faces/00001.png',
        'imgs/content/chicago.jpg',
        'imgs/style/Muse.jpg',
        'imgs/style/starry_night.jpg',
        'imgs/style/candy.jpg',
        'imgs/style/yellow_sunset.jpg',
        'imgs/style/abstraction.jpg'
    ]
    losses = [
        # VGGFeatures(pretrained=True).to(device),
        VGGFeatures(pretrained=False).to(device),
        VGGFeatures(pretrained=False, norm_first_conv=True).to(device)
    ]

    features_metric = get_features_metric('gram')

    for image_path in images:
        image_name = os.path.basename(image_path).split('.')[0]
        image = imload(image_path, imsize=128).to(device)
        for loss_network in losses:
            for layers_set in [['relu1_2', 'relu2_2'], ['relu1_2', 'relu2_2', 'relu3_3']]:
                save_path = f"{train_dir}/gram_minimization/{image_name}-{loss_network.name}-{layers_set}.png"
                optimize_gram_similarity(image, loss_network, layers_set, features_metric, save_path, max_iter=5000)

            # save_path = f"{train_dir}/gram_denoising/{image_name}-{loss_network.name}-{layers_set}.png"
            # denoise_with_gram_loss(image, loss_network, layers_set, features_metric, save_path, max_iter=1000)

            for layer_name in layers_set:
                save_path = f"{train_dir}/gram_mats/{image_name}-{loss_network.name}-{layer_name}.png"
                plot_gram_matrix(image, layer_name, loss_network, save_path)


def improve_quality_with_gram_loss(max_iter=1000):
    save_dir = 'outputs/optimize_corrupt'
    os.makedirs(save_dir, exist_ok=True)

    loss_network = VGGFeatures(pretrained=False, norm_first_conv=True).to(device)
    # loss_network = VGGFeatures(pretrained=True).to(device)
    layer_names = ['relu1_1', 'relu1_2']
    save_path = f"{save_dir}/{loss_network.name}_{layer_names}.png"

    # features_metric = get_features_metric('gram')
    features_metric = get_features_metric('l2')

    image = imload('imgs/faces/FFHQ-0.png', imsize=128).to(device)
    # low_quality_image = imload('imgs/faces/FFHQ-0_MMD++-generated.png', imsize=128).to(device)

    low_quality_image = torch.nn.UpsamplingBilinear2d(scale_factor=4)(torch.nn.AvgPool2d(kernel_size=4)(image))
    # low_quality_image = image.clone() + torch.randn(image.shape).cuda().float()
    # low_quality_image = transforms.RandomCrop(int(max(image.shape)*0.75))(low_quality_image)

    debug_low_quality_image_copy = low_quality_image.clone()

    low_quality_image.requires_grad_(True)
    optimizer = torch.optim.Adam([low_quality_image], lr=0.1)

    image_activations = loss_network.get_activations(image)
    image_activations = [image_activations[i] for i in layer_names]

    pbar = tqdm(range(max_iter + 1))
    for iteration in pbar:
        target_activations = loss_network.get_activations(low_quality_image)
        target_activations = [target_activations[i] for i in layer_names]

        style_loss = calc_loss(target_activations, image_activations, features_metric)

        total_loss = style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        if iteration % 500 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.75

            pbar.set_description(f"total_loss: {total_loss.item()}")

    imsave(torch.cat([image.cpu(), debug_low_quality_image_copy.detach().cpu(), low_quality_image.cpu()], dim=0), save_path)


if __name__ == '__main__':
    # improve_quality_with_gram_loss()
    run_batch_experiments()
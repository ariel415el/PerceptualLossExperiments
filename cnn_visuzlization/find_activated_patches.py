import os

import numpy as np
import torch

from cnn_visuzlization.common import find_most_activating_patches, save_scaled_images
from cnn_visuzlization.create_saliency_maps import VanillaBackprop
from style_transfer.utils import calc_TV_Loss


def optimize_patch(net, c_idx, n_patches, n_steps, device):
    # optimized_patch = torch.randn((n_patches, 3, net.m_receptive_field, net.m_receptive_field)).to(device)
    optimized_patch = torch.randn((n_patches, 3, 128, 128)).to(device)
    optimized_patch.requires_grad_(True)
    optim = torch.optim.Adam([optimized_patch], lr=0.01)
    losses = []

    for i in range(n_steps):
        optim.zero_grad()

        loss = -net(optimized_patch)[:, c_idx].mean()
        loss += calc_TV_Loss(optimized_patch)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    return optimized_patch


def show_most_activating_patches(net, dataloader, resize_patch, output_dir, device, n_best, n_channels=10):
    os.makedirs(output_dir, exist_ok=True)

    all_patch_activations_heatmap = []

    for c_idx in np.random.choice(np.arange(0, net.m_n_maps), n_channels, replace=False):
        # create one hot vector to find activating single neurons

        with torch.no_grad():
            c_vec = np.zeros(net.m_n_maps)
            c_vec[c_idx] = 1
            patches, imgs, patch_activations_heatmap = find_most_activating_patches(net, dataloader, c_vec, n_best, device, similarity_mode='cosine')
            all_patch_activations_heatmap.append(patch_activations_heatmap)

        from copy import deepcopy
        VBP = VanillaBackprop(deepcopy(net), guided=True)
        patch_grads = VBP.generate_gradients(patches.to(device).float(), c_idx)

        optimized_patch = optimize_patch(net, c_idx, n_patches=1, n_steps=1000, device=device)

        save_scaled_images(patches, resize_patch,  f"{output_dir}/c-{c_idx}_patches.png")
        save_scaled_images(imgs, 1, f"{output_dir}/c-{c_idx}_imgs.png")
        save_scaled_images(patch_grads, resize_patch, f"{output_dir}/c-{c_idx}_patch_grads.png")
        save_scaled_images(optimized_patch, resize_patch, f"{output_dir}/c-{c_idx}_optimized_patches.png")
        save_scaled_images(torch.from_numpy(patch_activations_heatmap).unsqueeze(0), resize_patch, f"{output_dir}/c-{c_idx}_patch_activations_heatmap.png")

    save_scaled_images(torch.from_numpy(np.stack(all_patch_activations_heatmap).mean(0)), resize_patch, f"{output_dir}/all_patch_activations_heatmap.png")

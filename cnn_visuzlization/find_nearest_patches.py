import os

import numpy as np
import torch
import torch.nn.functional as F

from cnn_visuzlization.common import find_most_activating_patches, save_scaled_images



def get_random_patch_with_activation(net, dataloader, device):
    image = torch.from_numpy(dataloader.dataset[np.random.randint(0, len(dataloader.dataset))][1]).to(device).float().unsqueeze(0)
    # image = next(iter(dataloader))[1][:1].to(device).float()
    all_patches = F.unfold(image, kernel_size=net.m_receptive_field, padding=0, stride=net.m_stride)

    feature_maps = net(image).reshape(1, net.m_n_maps, -1)

    assert (all_patches.shape[-1] == feature_maps.shape[-1])

    patch_idx = np.random.choice(feature_maps.shape[-1])

    patch = all_patches[0, :, patch_idx].reshape(3, net.m_receptive_field, net.m_receptive_field)
    activation = feature_maps[0, :, patch_idx]

    return patch, activation, image

def find_euclidean_nearest_neighbors(patch, dataloader, receptive_field, stride, n_best):
    all_patches = F.unfold(torch.from_numpy(dataloader.dataset.images), kernel_size=receptive_field, padding=0, stride=stride)
    flatt_patch = patch.reshape(1, -1, 1)

    all_patches = (all_patches - flatt_patch)**2
    all_patches = all_patches.mean(-1)

    indices = np.array(np.unravel_index(np.argsort(all_patches, axis=None)[::-1][:n_best], all_patches.shape)).transpose()


    indices = np.array(np.unravel_index(np.argsort(all_maps, axis=None)[::-1][:n_best], all_maps.shape)).transpose()



def show_patch_nearest_neighbors(net, dataloader, resize_patch, n_patches, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    n_best = 9
    for i in range(n_patches):
        patch, activation, image = get_random_patch_with_activation(net, dataloader, device)

        patches, imgs = find_most_activating_patches(net, dataloader, activation.cpu().numpy(), n_best, device, similarity_mode='l2')

        save_scaled_images(patch, resize_patch, f"{output_dir}/{i}_1-patch.png")
        save_scaled_images(image, 1, f"{output_dir}/{i}_2-image.png")
        save_scaled_images(patches, resize_patch, f"{output_dir}/{i}_3-similar-patches.png")
        save_scaled_images(imgs, 1, f"{output_dir}/{i}_4-images.png")

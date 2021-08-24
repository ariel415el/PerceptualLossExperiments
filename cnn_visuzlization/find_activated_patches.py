import os

import numpy as np

from cnn_visuzlization.common import find_most_activating_patches, save_scaled_images


def show_most_activating_patches(net, dataloader, resize_patch, n_best, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for c_idx in range(0, net.m_n_maps, 10):

        # create one hot vector to find activating single neurons
        c_vec = np.zeros(net.m_n_maps)
        c_vec[c_idx] = 1

        patches, imgs = find_most_activating_patches(net, dataloader, c_vec, n_best, device, similarity_mode='cosine')

        save_scaled_images(patches, 1,  f"{output_dir}/c-{c_idx}_patches.png")
        save_scaled_images(imgs, resize_patch, f"{output_dir}/c-{c_idx}_imgs.png")

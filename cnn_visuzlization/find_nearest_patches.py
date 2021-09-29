import os

import numpy as np
import torch
import torch.nn.functional as F

from cnn_visuzlization.common import find_most_activating_patches, save_scaled_images


def get_random_patch_with_activation(net, dataloader, device, sample_from_face_only=False):
    image = torch.from_numpy(dataloader.dataset[np.random.randint(0, len(dataloader.dataset))][1]).to(
        device).float().unsqueeze(0)
    # image = next(iter(dataloader))[1][:1].to(device).float()
    all_patches = F.unfold(image, kernel_size=net.m_receptive_field, padding=0,
                           stride=net.m_stride)  # (1, patch_size, n_patches)

    feature_maps = net(image).reshape(net.m_n_maps, -1)

    assert (all_patches.shape[-1] == feature_maps.shape[-1])

    if sample_from_face_only:
        dim = int(np.sqrt(feature_maps.shape[-1]))
        # eyes
        # w_idx = np.random.randint(low=int(dim // 2 - dim // 4), high=int(dim / 2 + dim / 4), size=1)
        # h_idx = np.random.randint(low=int(dim *0.425), high=int(dim *0.475), size=1)
        # face
        h_idx, w_idx = np.random.randint(low=int(dim // 2 - dim // 4), high=int(dim / 2 + dim / 4), size=2)
        patch_idx = dim * h_idx + w_idx
    else:
        patch_idx = np.random.choice(feature_maps.shape[-1])

    patch = all_patches[:, :, patch_idx].reshape(1, 3, net.m_receptive_field, net.m_receptive_field)
    activation = feature_maps[:, patch_idx]

    return patch, activation, image


def patch_l2_dist(query, patches):
    return ((patches - query) ** 2).mean(1)


def patch_loss_dist(query, patches, loss):
    d = int(np.sqrt(patches.shape[1] / 3))
    reshaped_patches = patches.transpose(2, 1).reshape(-1, 3, d, d)
    reshaped_query = query.transpose(2, 1).reshape(-1, 3, d, d).repeat(reshaped_patches.shape[0], 1, 1, 1)
    dists = loss(reshaped_query.float(), reshaped_patches.float())
    return dists.reshape(patches.shape[0], patches.shape[2])


def find_euclidean_nearest_neighbors(patch, dataloader, receptive_field, stride, n_best, loss):
    flatt_patch = patch.reshape(1, -1, 1)

    all_patch_dists = []
    for (_, images) in dataloader:
        patches = F.unfold(images.to(patch.device), kernel_size=receptive_field, padding=0,
                           stride=stride)  # (b, patch_size, n_patches)

        dists = patch_loss_dist(flatt_patch.clone(), patches.clone(), loss)
        all_patch_dists.append(dists.cpu().numpy())
    all_patch_dists = np.concatenate(all_patch_dists)

    indices = np.array(
        np.unravel_index(np.argsort(all_patch_dists, axis=None)[:n_best], all_patch_dists.shape)).transpose()

    similar_patches = []
    imgs = []
    similarieties = []
    for (im_idx, p_idx) in indices:
        img = torch.from_numpy(dataloader.dataset[im_idx][1])
        all_patches = F.unfold(img.unsqueeze(0), kernel_size=receptive_field, padding=0,
                               stride=stride)  # (b, patch_size, n_patches)
        all_patches = all_patches[0].transpose(1, 0).reshape(-1, 3, receptive_field, receptive_field)
        patch = all_patches[p_idx]
        similar_patches.append(patch)
        imgs.append(img)
        similarieties.append(all_patch_dists[im_idx, p_idx])

    return torch.stack(similar_patches), torch.stack(imgs)


def show_patch_nearest_neighbors(net, dataloader, resize_patch, output_dir, device, n_best, n_patches):
    """
    Select few random patches and show their nearest neighbor in the dataset with perceptual distance and other different metricss
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_patches):
        with torch.no_grad():
            patch, activation, image = get_random_patch_with_activation(net, dataloader, device, sample_from_face_only=True)

            patches, imgs, _ = find_most_activating_patches(net, dataloader, activation.cpu().numpy(), n_best, device, similarity_mode='l2')
        save_scaled_images(patches, resize_patch, f"{output_dir}/{i}_1-similar-vgg-patches.png")

        import losses
        for j, loss in enumerate([
            losses.L2(batch_reduction='none'),
            losses.MMDApproximate(patch_size=5, strides=1, sigma=0.05, r=32, pool_size=net.m_receptive_field, pool_strides=1, batch_reduction='none', normalize_patch='mean'),
            # losses.GradLoss(batch_reduction='none'),
            # losses.PyramidLoss(losses.L2(batch_reduction='none'), max_levels=2, weightening_mode=[0, 0.1, 1]),
            losses.PyramidLoss(losses.GradLoss(batch_reduction='none'), max_levels=2, weightening_mode=[0.1, 0.3, 1]),
            losses.SSIM(patch_size=net.m_receptive_field, batch_reduction='none')
        ]):
            patches, images = find_euclidean_nearest_neighbors(patch, dataloader, net.m_receptive_field, net.m_stride,
                                                               n_best, loss=loss)
            save_scaled_images(patches, resize_patch, f"{output_dir}/{i}_{j + 2}-similar-{loss.name}_patches.png")
        # save_scaled_images(image, 1, f"{output_dir}/{i}_4-image.png")
        # save_scaled_images(imgs, 1, f"{output_dir}/{i}_5-images.png")
        # save_scaled_images(l2_images, 1, f"{output_dir}/{i}_6-l2_images.png")

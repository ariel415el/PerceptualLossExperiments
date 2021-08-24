import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision import utils as vutils

from GenerativeModels.utils.data_utils import get_dataset, MemoryDataset, get_dataloader


def calculate_receptive_field(cfg):
    """
    for [64, 64, 'M', 128, 128] we need to solve (r-2-2)/2 - 2-2 = 1
    i.e  r = 2 + 2*2 + 4*2 => r=14
    for [64, 64, 'M', 128, 128, 'M', 256, 256, 256] we need to solve ((r-2-2)/2 - 2-2)/2 - 3*2 = 1
    i.e  r = 4 + 2*2 + 4*2 + 8*3 => r=40
    """
    conv_batches = []
    tmp = []
    n_max_pools = 0
    for v in cfg:
        if v == "M":
            conv_batches += [tmp]
            tmp = []
            n_max_pools += 1
        else:
            tmp += [v]
    conv_batches += [tmp]

    stride = 2 ** n_max_pools

    receptive_field = 2**n_max_pools
    for i, batch in enumerate(conv_batches):
        receptive_field += len(batch) * 2**(i+1)

    return receptive_field, stride


def get_single_conv_vgg(cfg, drop_last_relu=False, load_weights=True):
    in_channels = 3
    features = []
    for v in cfg:
        if v == 'M':
            features += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=0)
            features += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if drop_last_relu:
        features = features[:-1]

    net = nn.Sequential(*features)

    if load_weights:
        state_dict = torch.load('../losses/vgg_loss/vgg16_head.pth')
        state_dict = {k.replace('features.', ''):v for k,v in state_dict.items() if k.replace('features.', '') in net.state_dict()}
        net.load_state_dict(state_dict)

    net.m_receptive_field, net.m_stride = calculate_receptive_field(cfg)
    net.m_n_maps = cfg[-1]

    print(f"Receptive_field: {net.m_receptive_field}")
    print(f"stride: {net.m_stride}")
    print(f"n_maps: {net.m_n_maps}")

    return net


def create_dataloader(dataset_name, batch_size, resize, device):
    if dataset_name == 'ffhq':
        dataset = get_dataset('ffhq', split='test', resize=128, val_percent=0.02)
    else:
        data_path = '/home/ariel/university/data/imagenette2-320/all'
        dataset = MemoryDataset(paths=[os.path.join(data_path, x) for x in os.listdir(data_path)][:1000], resize=resize)

    return get_dataloader(dataset, batch_size, device, drop_last=False, shuffle=False)


def collect_feature_map_similarities(net, dataloader, feature_vec, device, mode='cosine'):
    """
    Collect the cosine similarities of all feature maps with c_vec
    it turns the full activations of samples (n_samples, d, h, w) into (n_samples, h, w)
    :return: (n_samples, h, w) array
    """
    """"""
    assert (len(feature_vec) == net.m_n_maps)
    all_maps = []
    for (_, images) in dataloader:
        feature_map = net(images.to(device).float()).cpu().numpy()
        if mode == 'cosine':
            feature_map *= feature_vec.reshape(1, -1, 1, 1)
            feature_map = feature_map.sum(1)
        else:
            feature_map -= feature_vec.reshape(1, -1, 1, 1)
            feature_map = (feature_map**2).sum(1)
            feature_map = 1 - feature_map / feature_map.max()
        all_maps.append(feature_map)

    return np.concatenate(all_maps)


def find_most_activating_patches(net, dataloader, feature_vec, n_best, device, similarity_mode='l2'):
    imgs = []
    activated_patches = []
    activations = []

    all_maps = collect_feature_map_similarities(net, dataloader, feature_vec, device, mode=similarity_mode)

    indices = np.array(np.unravel_index(np.argsort(all_maps, axis=None)[::-1][:n_best], all_maps.shape)).transpose()

    for (im_idx, h_idx, w_idx) in indices:
        img = torch.from_numpy(dataloader.dataset[im_idx][1])
        all_patches = F.unfold(img.unsqueeze(0), kernel_size=net.m_receptive_field, padding=0, stride=net.m_stride)
        all_patches = all_patches[0].transpose(1,0).reshape(-1, 3, net.m_receptive_field, net.m_receptive_field)
        dim = int(np.sqrt(all_patches.shape[0]))
        assert dim == all_maps.shape[1]
        patch = all_patches[h_idx * dim + w_idx]
        activated_patches.append(patch)
        imgs.append(img)
        activations.append(all_maps[im_idx, h_idx, w_idx])

    print(activations)
    return torch.stack(activated_patches), torch.stack(imgs)


def save_scaled_images(images, scale, path):
    images = transforms.Resize(images.shape[-1] * scale, interpolation=InterpolationMode.NEAREST)(images)
    vutils.save_image(images, path, normalize=True)

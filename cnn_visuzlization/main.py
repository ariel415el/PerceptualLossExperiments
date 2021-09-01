import torch

from cnn_visuzlization.common import create_dataloader, get_single_conv_vgg, get_single_conv_alexnet
from cnn_visuzlization.create_saliency_maps import show_saliency_maps
from cnn_visuzlization.find_activated_patches import show_most_activating_patches
from cnn_visuzlization.find_nearest_patches import show_patch_nearest_neighbors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
if __name__ == '__main__':
    # Set CNN
    architecture = 'vgg'
    load_weights = True

    if architecture == 'vgg':
        cfg = [64, 64, 'M', 128, 128, 'M', 256]#, 256, 256]#, 'M', 512, 512, 512]
        net = get_single_conv_vgg(cfg, load_weights=load_weights, inplace_relus=False)
    else:
        cfg = 'conv2'
        net = get_single_conv_alexnet(cfg, load_weights=load_weights, inplace_relus=False)

    net = net.to(device)

    # Set Dataset
    # dataset_name = 'imagenet'
    dataset_name = 'ffhq'
    resize = 128
    batch_size = 16
    dataloader = create_dataloader(dataset_name, batch_size=batch_size, resize=resize, device=device)

    tag = f"vgg-face_{dataset_name}_{net.m_receptive_field}" + ("_pt" if load_weights else '_rand')
    # Run experiments
    output_dir = f"Outputs/{architecture}-{cfg}_{tag}_highly_activated_patches"
    show_most_activating_patches(net, dataloader, resize_patch=3, output_dir=output_dir, device=device, n_best=25, n_channels=10)
    output_dir = f"Outputs/{architecture}-{cfg}_{tag}_nearest_neighbor_patches"
    show_patch_nearest_neighbors(net, dataloader, resize_patch=3, output_dir=output_dir, device=device, n_best=25, n_patches=10)
    # output_dir = f"Outputs/{architecture}-{cfg}_{tag}_saliency_maps"
    # show_saliency_maps(net, dataloader, resize_patch=3, output_dir=output_dir, device=device, n_images=10, n_channels=10)


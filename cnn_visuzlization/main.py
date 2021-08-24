import torch

from cnn_visuzlization.common import create_dataloader, get_single_conv_vgg
from cnn_visuzlization.create_saliency_maps import show_saliency_maps
from cnn_visuzlization.find_activated_patches import show_most_activating_patches
from cnn_visuzlization.find_nearest_patches import show_patch_nearest_neighbors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
if __name__ == '__main__':
    with torch.no_grad():
        # Set CNN
        cfg = [64, 64, 'M', 128, 128]#, 'M', 256, 256, 256]
        load_weights = True
        net = get_single_conv_vgg(cfg, load_weights=load_weights)
        net = net.to(device)

        # Set Dataset
        dataset_name = 'imagenet'
        dataset_name = 'ffhq'
        resize = 128
        batch_size = 128
        dataloader = create_dataloader(dataset_name, batch_size=batch_size, resize=resize, device=device)

        tag = f"{dataset_name}_{net.m_receptive_field}" + (f"_pt" if load_weights else '')
        # Run experiments

        # output_dir = f"Outputs/highly_activated_patches_{tag}"
        # show_most_activating_patches(net, dataloader, resize_patch=3, n_best=16, output_dir=output_dir, device=device)
        # output_dir = f"Outputs/nearest_neighbor_patches_{tag}"
        # show_patch_nearest_neighbors(net, dataloader, resize_patch=3, n_patches=16, output_dir=output_dir, device=device)
        output_dir = f"Outputs/saliency_maps{tag}"
        show_saliency_maps(net, dataloader, resize_patch=3, output_dir=output_dir, device=device)


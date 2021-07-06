import os

import torch.nn.functional
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils as vutils

from GenerativeModels import models
from GenerativeModels.utils.data_utils import get_dataset
from losses.composite_losses.laplacian_losses import laplacian_pyramid, get_kernel_gauss, LaplacyanLoss
from losses.experimental_patch_losses import MMD_PPP
from losses.l2 import L2, L1
from losses.lap1_loss import LapLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.patch_mmd_pp import MMD_PP
from losses.swd.lap_swd_loss import LapSWDLoss
from losses.swd.lap_swd_loss_new import LapSWDLoss_new
from losses.swd.swd import PatchSWDLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss
from perceptual_mean_optimization.main import pt2cv

from GenerativeModels.GLO.config import faces_config as params

device = torch.device("cuda")


def get_reconstructions(root, test_images):
    results = dict()
    params.img_dim = 128
    for model_name in os.listdir(root):
        if model_name == 'outputs':
            continue
        # LOAD MODELS
        train_dir = os.path.join(root, model_name)

        generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim).to(device)
        generator.load_state_dict(torch.load(f"{train_dir}/generator.pth", map_location=device))

        encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)
        encoder.load_state_dict(torch.load(f"{train_dir}/encoder.pth", map_location=device))

        encoder.eval()
        generator.eval()

        # RECONSTRUCT DATA
        test_reconstructions = generator(encoder(test_images))
        results[model_name] = test_reconstructions
        # test_reconstructions_cropped = test_reconstructions[:, :, l:r, t:b]
        # vutils.save_image(test_reconstructions, os.path.join(outputs_dir, f"{model_name}_test-reconstructions.png"), normalize=True, nrow=np.int(np.sqrt(n)))

    return results


def plot_loss_matrix(criterions, ref_images, recs_dict, save_path=None):
    mat_data = []
    for recs in recs_dict.values():
        row_data = []
        for criterion in criterions:
            loss_values = criterion.to(device)(ref_images, recs)
            # loss_strings.append(f"{loss_values.mean():.4f}/{loss_values.std():.4f})")
            row_data.append(loss_values.item())

        mat_data.append(row_data)

    # PLOT SIMILARITY MATRIX
    fig, ax = plt.subplots(figsize=(15, 15))
    mat_data = np.array(mat_data)
    mat_data = mat_data / mat_data.max(axis=0)
    mat = ax.matshow(mat_data, cmap='YlGnBu')
    for (i, j), z in np.ndenumerate(mat_data):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    ax.set_xticks(np.arange(len(criterions)))
    ax.set_yticks(np.arange(len(recs_dict)))
    ax.set_xticklabels([l.name for l in criterions], rotation=20)
    ax.set_yticklabels(recs_dict.keys(), rotation=70)
    fig.colorbar(mat)
    fig.savefig(save_path)
    plt.clf()


def plot_lap_pyr(criterions, ref_images, recs_dict, outputs_dir):
    n_layers = 2
    os.makedirs(outputs_dir, exist_ok=True)
    b, c, h, w = ref_images.shape
    for i in range(b):
        ref_pyr = laplacian_pyramid(ref_images[i].unsqueeze(0),
                                    get_kernel_gauss(size=5, sigma=3, n_channels=3).to(device), n_layers)

        fig, axs = plt.subplots(len(ref_pyr) + 1, len(recs_dict) + 1)
        axs[0, 0].imshow(pt2cv(ref_images[i].detach().cpu().numpy()), interpolation='none', aspect='equal')
        axs[0, 0].set_axis_off()
        axs[0, 0].set_title(f'Reference ({h}x{w})', size=6)
        for j, p in enumerate(ref_pyr):
            p_np = torch.nn.functional.interpolate(p, ref_images[i].shape[1:]).detach().cpu().numpy()[0]
            axs[j + 1, 0].imshow(pt2cv(p_np), interpolation='none', aspect='equal')
            axs[j + 1, 0].set_axis_off()

        for k, (model_name, recs) in enumerate(recs_dict.items()):
            axs[0, k + 1].imshow(pt2cv(recs[i].detach().cpu().numpy()), interpolation='none', aspect='equal')
            axs[0, k + 1].set_axis_off()
            axs[0, k + 1].set_title(model_name, size=8)

            pyr = laplacian_pyramid(recs[i].unsqueeze(0), get_kernel_gauss(size=5, sigma=2, n_channels=3).to(device),
                                    n_layers)
            for j, p in enumerate(pyr):
                p_np = torch.nn.functional.interpolate(p, ref_images[i].shape[1:]).detach().cpu().numpy()[0]
                axs[j + 1, k + 1].imshow(pt2cv(p_np), interpolation='none', aspect='equal')
                axs[j + 1, k + 1].set_axis_off()
                title = []
                for loss in criterions:
                    # title += [f"{loss.name.split('(')[0]}: {loss(p, ref_pyr[j]).item():.3f}"]
                    title += [f"{loss.name}: {loss(p, ref_pyr[j]).item():.2f}"]
                axs[j + 1, k + 1].set_title('\n'.join(title), size=5)

        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f"pyr-{i}.png"), dpi=1000)
        plt.clf()


def plot_losses_images(criterions, ref_images, recs_dict, outputs_dir):
    # PLOT DATA SAMPLE LOSSES
    os.makedirs(outputs_dir, exist_ok=True)
    for i in range(len(ref_images)):
        fig, axs = plt.subplots(1, len(recs_dict) + 1, figsize=(6, 2))

        axs[0].imshow(pt2cv(ref_images[i].detach().cpu().numpy()), interpolation='none', aspect='equal', )
        axs[0].set_title('Reference', size=9)
        axs[0].axis('off')

        for k, (model_name, recs) in enumerate(recs_dict.items()):
            rec_img = pt2cv(recs[i].detach().cpu().numpy())
            axs[k + 1].imshow(rec_img, interpolation='none', aspect='equal')
            title = f"Train loss: {model_name}\n"
            for criterion in criterions:
                title += f"{criterion.name:25}: {criterion.to(device)(ref_images[i].unsqueeze(0), recs[i].unsqueeze(0)).item():.2f}\n"
            axs[k + 1].set_title(title, size=4)
            axs[k + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f"losses-{i}.png"), dpi=1000)
        plt.clf()


def crop(refs, recs_dict, c):
    return refs[:, :, c[0]:c[1], c[2]:c[3]], {k: v[:, :, c[0]:c[1], c[2]:c[3]] for k, v in recs_dict.items()}


def compare_trained_generators_losses(models_dir):
    n = 9
    criterions = [
        L2(),
        LapLoss(),
        MMD_PP(r=512),
        VGGPerceptualLoss(pretrained=True),
        VGGPerceptualLoss(pretrained=False, norm_first_conv=True, reinit=True),
        # LapSWDLoss()
        LapSWDLoss_new()
    ]
    lap_criterions = [
        L1(),
        PatchSWDLoss(patch_size=7, num_proj=512, n_samples=512),
        # PatchSWDLoss(patch_size=15, num_proj=512, n_samples=512),
        # MMDApproximate(patch_size=3, pool_size=16, pool_strides=8, r=256)
    ]

    # crop_coords = [16,-16,16,-16]
    # crop_coords = [24,-24,24,-24]
    crop_coords = [10, 74, 10, 74]
    # crop_coords = [0,64,0,64]
    # crop_coords = [0,-1,0,-1]
    # crop_coords = [32, -32, 32, -32]
    # crop_coords = [64, -32, 48, -48]
    # crop_coords = [48, -48, 48, -48]
    # crop_coords = [50,70,50,70]

    outputs_dir = f"Experiments/compare_trained_generators_losses_outputs/{crop_coords}"
    os.makedirs(outputs_dir, exist_ok=True)

    image_indices = [11, 15, 16, 25, 48, 53, 60, 67, 68, 78, 122] # val_percent=0.9
    # image_indices = np.array([0, 10, 20, 30, 40, 50, 60, 80, 90, 100]) + 16

    dataset = get_dataset('ffhq', split='train', resize=params.img_dim, val_percent=0.9)
    ref_images = torch.stack([torch.from_numpy(dataset[(i)][1]) for i in image_indices]).to(device).float()
    vutils.save_image(ref_images, os.path.join(outputs_dir, f"org-imgs.png"), normalize=True, nrow=np.int(np.sqrt(n)))

    recs_dict = get_reconstructions(models_dir, ref_images)
    import collections
    recs_dict = collections.OrderedDict(recs_dict)

    ref_images, recs_dict = crop(ref_images, recs_dict, crop_coords)

    plot_loss_matrix(criterions, ref_images, recs_dict,
                     save_path=os.path.join(outputs_dir, f"lossMat.png"))

    plot_losses_images(criterions, ref_images, recs_dict, os.path.join(outputs_dir, f"images"))

    plot_lap_pyr(lap_criterions, ref_images, recs_dict, os.path.join(outputs_dir, f"images"))


if __name__ == '__main__':
    root = 'Experiments/trained_models'
    compare_trained_generators_losses(root)

import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import losses

from perceptual_mean_optimization.mean_optimization import get_starting_point, optimize_patch_distributions
from perceptual_mean_optimization.utils import pt2cv, load_images



def main():
    """
    generate an image with minimal MMD distance from two images and compare it to a huristic solution
    :return:
    """
    # criteria = MMDApproximate(r=128, pool_size=32, pool_strides=16, normalize_patch='none')
    criteria = losses.PatchMMD_RBF(patch_size=11, stride=11)
    images = load_images('../perceptual_mean_optimization/clusters/textures/mixed/dry+longgrass',)[:2]
    # starting_img = get_starting_point(images, 'zeros+noise')
    starting_img = get_starting_point(images, 'mean+noise')

    img = optimize_patch_distributions(starting_img.clone(), images, criteria, output_dir="2_images_outputs", num_steps=1000, lr=0.005).cpu()

    synthesized_solution = images[0].clone()
    synthesized_solution[:, :synthesized_solution.shape[2]//2] = \
        images[1].clone()[:, :synthesized_solution.shape[2]//2]


    start_loss = np.mean([criteria(starting_img.unsqueeze(0), x.unsqueeze(0)).item() for x in images])
    synth_loss = np.mean([criteria(synthesized_solution.unsqueeze(0), x.unsqueeze(0)).item() for x in images])
    opt_loss = np.mean([criteria(img.unsqueeze(0), x.unsqueeze(0)).item() for x in images])


    fig, axs = plt.subplots(3)
    axs[0].imshow(pt2cv(starting_img.detach().cpu()))
    axs[0].set_title(f"{start_loss:.4f}")
    axs[0].axis('off')

    axs[1].imshow(pt2cv(img.detach().cpu()))
    axs[1].set_title(f"{opt_loss:.4f}")
    axs[1].axis('off')

    axs[2].imshow(pt2cv(synthesized_solution.detach().cpu()))
    axs[2].set_title(f"{synth_loss:.4f}")
    axs[2].axis('off')

    fig.savefig(os.path.join("2_images_outputs", 'res.png'))

if __name__ == '__main__':
    main()
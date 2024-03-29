import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from image_retargeting.utils import aspect_ratio_resize, get_pyramid, quantize_image
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.swd.patch_swd import PatchSWDLoss
from perceptual_mean_optimization.utils import cv2pt
import torchvision.utils as vutils
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def match_patch_distributions(input_img, target_img, criterias, num_steps, lr, output_dir):
    """
    :param images: tensor of shape (H, W, C)
    """
    os.makedirs(output_dir, exist_ok=True)

    optimized_variable = input_img.clone().unsqueeze(0).to(device)
    optimized_variable.requires_grad_(True)
    optim = torch.optim.Adam([optimized_variable], lr=lr)

    target_img_pt = target_img.unsqueeze(0).to(device)
    vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-0.png", normalize=True)

    all_losses = {loss[0].name: [] for loss in criterias}
    all_means = {loss[0].name: [] for loss in criterias}
    for i in tqdm(range(1, num_steps + 1)):
        optim.zero_grad()

        total_loss = 0
        for (criteria, w) in criterias:
            loss = criteria(optimized_variable, target_img_pt)
            all_losses[criteria.name].append(loss.item())
            total_loss += loss

        total_loss.backward()
        optim.step()

        if i % 1000 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.9
        if i % 500 == 0:
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-{i}.png", normalize=True)
        if i % 100 == 0:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            for (criteria, w) in criterias:
                losses = all_losses[criteria.name]
                all_means[criteria.name].append(np.mean(all_losses[criteria.name][-100:]))
                ax.plot(np.arange(len(losses)), np.log(losses),
                        label=f'{criteria.name}: {all_means[criteria.name][-1]:.6f}')
                ax.plot((1 + np.arange(len(all_means[criteria.name]))) * 100, np.log(all_means[criteria.name]), c='y')
            ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    return torch.clip(optimized_variable.detach()[0], -1, 1)


def retarget_image(img_path, criterias, output_dir):
    n_scales = 5
    perc = 0.75
    num_steps = 5000
    lr = 0.005
    img = cv2.imread(img_path)
    # img = aspect_ratio_resize(img, max_dim=256)
    img = cv2pt(img)

    pyramid = get_pyramid(img, n_scales, perc)

    synthesis = torch.randn(pyramid[0].shape) * 0.1

    for lvl, lvl_img in enumerate(pyramid):
        print(f"Starting lvl {lvl}")
        if lvl > 0:
            # synthesis = transforms.Resize((int(lvl_img.shape[1] * 1), int(lvl_img.shape[2] * 0.6)), antialias=True)(synthesis)
            synthesis = transforms.Resize(lvl_img.shape[1:], antialias=True)(synthesis)

        lvl_output_dir = os.path.join(output_dir, str(lvl))
        vutils.save_image(lvl_img, os.path.join(output_dir, f"target-{lvl}.png"), normalize=True)
        vutils.save_image(synthesis, os.path.join(output_dir, f"org-{lvl}.png"), normalize=True)

        synthesis = match_patch_distributions(synthesis, lvl_img, criterias, 6000 if lvl == 0 else 2000, lr,
                                              lvl_output_dir)
        lr *= 0.9  # n_scales / 10
        # num_steps -= 500 # int(1.5* num_steps)

        vutils.save_image(synthesis, os.path.join(output_dir, f"final-{lvl}.png"), normalize=True)


if __name__ == '__main__':
    # img_path = 'images/balloons.png'
    # img_path = 'images/birds.png'
    # img_path = 'images/girafs.png'
    img_path = 'images/cows.png'
    # img_path = 'images/trees3.jpg'
    # img_path = 'images/fruit.png'
    criterias = [
        (PatchMMDLoss(patch_size=11, stride=3), 1, ),
        # (PatchSWDLoss(patch_size=11, stride=3, num_proj=1024), 1, ),
    ]

    # target = cv2pt(cv2.imread('images/balloons.png'))
    # starting = cv2pt(cv2.imread('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/image_retargeting/optimized_images_outputs/balloons/PatchMMD(p-17:1)/final-5.png'))
    # synthesis = match_patch_distributions(starting, target, criterias, 10000, 0.01, "single_outputs")
    # exit()

    img_name = os.path.basename(os.path.splitext(img_path)[0])
    exp_name = "_".join([l[0].name for l in criterias])
    output_dir = f'optimized_images_outputs/{img_name}/{exp_name}'
    os.makedirs(output_dir, exist_ok=True)

    retarget_image(img_path, criterias, output_dir)

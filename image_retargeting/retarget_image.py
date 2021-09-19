import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from image_retargeting.utils import aspect_ratio_resize, get_pyramid, quantize_image
import losses
from perceptual_mean_optimization.utils import cv2pt
import torchvision.utils as vutils
from torchvision import transforms
from losses.composite_losses.laplacian_losses import conv_gauss
from losses.composite_losses.laplacian_losses import get_kernel_gauss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def match_patch_distributions(input_img, target_img, criteria, num_steps, lr, output_dir):
    """
    :param images: tensor of shape (H, W, C)
    """
    os.makedirs(output_dir, exist_ok=True)

    optimized_variable = input_img.clone().unsqueeze(0).to(device)
    optimized_variable.requires_grad_(True)
    optim = torch.optim.Adam([optimized_variable], lr=lr)

    target_img_pt = target_img.unsqueeze(0).to(device)
    vutils.save_image(torch.clip(optimized_variable, -1, 1), f"{output_dir}/output-0.png", normalize=True)

    all_losses = []
    all_means = []
    for i in tqdm(range(1, num_steps + 1)):
        optim.zero_grad()

        loss = criteria(optimized_variable, target_img_pt)
        all_losses.append(loss.item())

        loss.backward()
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

            all_means.append(np.mean(all_losses[-100:]))
            ax.plot(np.arange(len(all_losses)), np.log(all_losses),
                    label=f'{criteria.name}: {all_means[-1]:.6f}')
            ax.plot((1 + np.arange(len(all_means))) * 100, np.log(all_means), c='y')
            ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    return torch.clip(optimized_variable.detach()[0], -1, 1)


def retarget_image(img_path, criteria, output_dir, n_scales, pyr_factor,
                   aspect_ratio=(1,1), resize=256, num_steps=2000, lr=0.001, init='noise'):
    img = cv2.imread(img_path)
    img = aspect_ratio_resize(img, max_dim=resize)
    img = cv2pt(img)

    pyramid = get_pyramid(img, n_scales, pyr_factor)

    for lvl, lvl_img in enumerate(pyramid):
        print(f"Starting lvl {lvl}")
        h, w = int(lvl_img.shape[1] * aspect_ratio[0]), int(lvl_img.shape[2] * aspect_ratio[1])
        if lvl == 0:
            if init == 'noise':
                synthesis = torch.randn((3, h, w)) * 0.1
            else:
                synthesis = transforms.Resize((h, w), antialias=True)(img)
                synthesis = conv_gauss(synthesis.unsqueeze(0), get_kernel_gauss(size=7, sigma=3, n_channels=3))[0]
                synthesis += torch.randn((3, h, w)) * 0.001
        if lvl > 0:
                # synthesis = transforms.Resize((int(lvl_img.shape[1] * 1), int(lvl_img.shape[2] * 0.6)), antialias=True)(synthesis)
                synthesis = transforms.Resize((h, w), antialias=True)(synthesis)

        lvl_output_dir = os.path.join(output_dir, str(lvl))
        vutils.save_image(lvl_img, os.path.join(output_dir, f"target-{lvl}.png"), normalize=True)
        vutils.save_image(synthesis, os.path.join(output_dir, f"org-{lvl}.png"), normalize=True)

        synthesis = match_patch_distributions(synthesis, lvl_img, criteria, 2 * num_steps if lvl == 0 else num_steps, lr,
                                              lvl_output_dir)

        vutils.save_image(synthesis, os.path.join(output_dir, f"final-{lvl}.png"), normalize=True)


if __name__ == '__main__':
    for tag in ['#1', '#2', '#3']:
        # img_path = 'images/balloons.png'
        # img_path = 'images/fruit.png'
        # img_path = 'images/birds.png'
        # img_path = 'images/colusseum.png'
        img_path = 'images/SupremeCourt.jpeg'
        # img_path = 'images/kanyon.jpg'
        # img_path = 'images/soccer1.png'
        # img_path = 'images/soccer2.jpg'
        # img_path = 'images/soccer3.jpg'
        # img_path = 'images/jerusalem1.jpg'
        # img_path = 'images/jerusalem2.jpg'
        # img_path = 'images/balls.jpg'
        # img_path = 'images/mountins2.jpg'
        # img_path = 'images/people_on_the_beach.jpg'
        # img_path = 'images/girafs.png'
        # img_path = 'images/cows.png'
        # img_path = 'images/trees3.jpg'
        # img_path = 'images/fruit.png'

        criteria = losses.MMDApproximate(patch_size=7, strides=1, pool_size=-1, r=1024)

        outputs_dir = 'outputs/test'
        resize = 256
        init = 'mean'
        pyr_factor = 0.65; n_scales = 5; aspect_ratio = (1, 1.5); lr = 0.005; num_steps = 1500
        # pyr_factor = 0.75; n_scales = 5; aspect_ratio = (1, 1); lr = 0.005; num_steps = 1500

        img_name = os.path.basename(os.path.splitext(img_path)[0])

        output_dir = f'{outputs_dir}/{img_name}/{criteria.name}_AR-{aspect_ratio}_R-{resize}_S-{pyr_factor}x{n_scales}_I-{init}_{tag}'
        os.makedirs(output_dir, exist_ok=True)

        retarget_image(img_path, criteria, output_dir, n_scales=n_scales, pyr_factor=pyr_factor,
                       aspect_ratio=aspect_ratio, resize=resize, num_steps=num_steps, lr=lr, init=init)

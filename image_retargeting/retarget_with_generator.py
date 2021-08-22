import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from image_retargeting.generator import GeneratorConcatSkip2CleanAdd, weights_init
from image_retargeting.utils import aspect_ratio_resize, get_pyramid
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.swd.patch_swd import PatchSWDLoss
from perceptual_mean_optimization.utils import cv2pt
import torchvision.utils as vutils
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



def train_generator(target_pyramid, criterias, output_dir):
    z_rec_pyramid = [(torch.randn(target_pyramid[0].shape) * noise_amp).to(device)]
    previous_generators = []

    for lvl in range(len(target_pyramid)):
        train_generator_single_scale(previous_generators, z_rec_pyramid, target_pyramid, criterias, output_dir)

def generate(previous_generators, z_rec_pyramid, noise_amp):
    prev_fake = (torch.zeros(z_rec_pyramid[0].shape) * noise_amp).to(device)
    prev_rec = (torch.zeros(z_rec_pyramid[0].shape) * noise_amp).to(device)
    for i, generator in enumerate(previous_generators):
        noise = (torch.randn(z_rec_pyramid[i].shape) * noise_amp).to(device)
        prev_fake = transforms.Resize(noise.shape[2:], antialias=True)(prev_fake)
        prev_fake = generator(noise + prev_fake)

        prev_rec = transforms.Resize(noise.shape[2:], antialias=True)(prev_rec)
        prev_rec = generator(z_rec_pyramid[i] + prev_rec)

    return prev_fake, prev_rec

def train_generator_single_scale(previous_generators, z_rec_pyramid, target_img, criterias, output_dir):
    """
    :param images: tensor of shape (H, W, C)
    """
    num_steps = 100000
    lr = 0.001
    rec_weight = 0.0001
    noise_amp = 0.1
    os.makedirs(output_dir, exist_ok=True)

    lvl = len(previous_generators) + 1

    generator = GeneratorConcatSkip2CleanAdd().to(device)
    generator.apply(weights_init)

    optim = torch.optim.Adam(generator.parameters(), lr=lr)


    all_gen_losses = {loss[0].name:[] for loss in criterias}
    all_rec_loss = {loss[0].name:[] for loss in criterias}
    for i in tqdm(range(1, num_steps + 1)):
        optim.zero_grad()

        fake_img, rec_img = generate(previous_generators + [generator], z_rec_pyramid, noise_amp)

        total_gen_loss = 0
        for (criteria, w) in criterias:
            loss = criteria(fake_img, target_pyramid[lvl])
            all_gen_losses[criteria.name].append(loss.item())
            total_gen_loss += loss

        total_rec_loss = 0
        for (criteria, w) in criterias:
            # loss = ((rec_img - target_img_pt)**2).mean()
            loss = criteria(rec_img, target_pyramid[lvl])
            all_rec_loss[criteria.name].append(loss.item())
            total_gen_loss += loss

        total_loss = total_gen_loss + rec_weight * total_rec_loss
        total_loss.backward()
        optim.step()

        if i % 1000 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.9
        if i % 500 == 0:
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(torch.clip(fake_img, -1, 1), f"{output_dir}/fake-{i}.png", normalize=True)
            vutils.save_image(torch.clip(rec_img, -1, 1), f"{output_dir}/rec-{i}.png", normalize=True)
        if i % 500 == 0:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            for (criteria, w) in criterias:
                ax.plot(np.arange(len(all_gen_losses[criteria.name])), np.log(all_gen_losses[criteria.name]), label=f'gen-{criteria.name}: {all_gen_losses[criteria.name][-1]:.6f}')
                ax.plot(np.arange(len(all_rec_loss[criteria.name])), np.log(all_rec_loss[criteria.name]), label=f'rec-{criteria.name}: {all_rec_loss[criteria.name][-1]:.6f}')
            ax.legend()
            fig1.savefig(f"{output_dir}/train_loss.png")
            plt.close(fig1)

    return

if __name__ == '__main__':
    img_path = 'images/balloons.png'
    # img_path = 'images/birds.png'
    # img_path = 'images/girafs.png'
    # img_path = 'images/cows.png'

    criterias = [
        # (PatchMMDLoss(patch_size=7, stride=3, n_samples=None, sample_same_locations=False), 1),
        (PatchMMDLoss(patch_size=17, stride=3, n_samples=None), 1),
        # (PatchSWDLoss(patch_size=17, stride=8, n_samples=None, sample_same_locations=False), 1)
    ]

    img_name = os.path.basename(os.path.splitext(img_path)[0])
    exp_name = "_".join([l[0].name for l in criterias])
    output_dir = f'generator_outputs/{img_name}/{exp_name}'
    os.makedirs(output_dir, exist_ok=True)


    ### Train ###
    img = cv2.imread(img_path)
    # img = aspect_ratio_resize(img, max_dim=256)
    img = cv2pt(img).to(device)

    target_pyramid = get_pyramid(img, 8, 0.75)

    train_generator(target_pyramid, criterias, output_dir)

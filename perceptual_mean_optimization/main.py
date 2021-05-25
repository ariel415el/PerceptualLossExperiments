import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.realpath(".."))
from losses.patch_mmd_loss import MMDApproximate
from losses.patch_loss import PatchRBFLaplacianLoss, PatchRBFLoss
from losses.l2 import L2
from losses.mmd_loss import MMD
from losses.lap1_loss import LapLoss
from losses.utils import ListOfLosses
from losses.vgg_loss.vgg_loss import VGGFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1

    # img = np.clip(img, -1 + 1e-9, 1 - 1e-9)
    # img = np.arctanh(img)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def pt2cv(img):
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_images(dir_path):
    images = []
    for fn in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, fn))
        # img = img[125 - 90:125 + 80, 125 - 75:125 + 75]
        img = cv2pt(img)
        images.append(img)

    return torch.stack(images)


def optimize_for_mean(images, criteria, output_dir, batch_size=1, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    criteria = criteria.to(device)

    optimized_variable = images.mean(0)[None, :]
    # optimized_variable *= 0
    # optimized_variable = torch.randn(optimized_variable.shape).to(device).clamp(-1,1)
    optimized_variable.requires_grad_(True)
    # optim = torch.optim.SGD([optimized_variable], lr=lr)
    optim = torch.optim.Adam([optimized_variable], lr=lr)
    losses = []
    for i in tqdm(range(num_steps+1)):
        if i % 500 == 0:
            optimized_image = optimized_variable.detach().cpu().numpy()[0]
            optimized_image = pt2cv(optimized_image)
            cv2.imwrite(f"{output_dir}/output-{i}.png", optimized_image)

        optim.zero_grad()
        batch_images = images[np.random.choice(range(len(images)), batch_size, replace=False)]
        batch_input = optimized_variable.repeat(batch_size, 1, 1, 1)
        loss = criteria(batch_input, batch_images)
        loss = loss.mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if i % 200 == 0:
           for g in optim.param_groups:
                g['lr'] *= 0.5
        if i % 100 == 0:
            plt.title(f"last-Loss: {losses[-1]}")
            plt.plot(np.arange(len(losses)), losses)
            plt.savefig(f"{output_dir}/train_loss.png")
            plt.clf()

    optimized_image = torch.clip(optimized_variable,-1,1).detach().cpu().numpy()[0]
    # optimized_image = optimized_variable.detach().cpu().numpy()[0]
    optimized_image = pt2cv(optimized_image)

    return optimized_image


def batch_run():
    image_dirs = [
                # 'clusters/00083_256/images',
                'clusters/00068_256/images',
                # 'clusters/00083_128/images',
                # 'clusters/00068_128/images',
                # 'clusters/00083_64/images',
                # 'clusters/00068_64/images',
                # 'clusters/stylegan_128/images',
                # 'image_clusters/36126_s128_c_128/images',
                # 'image_clusters/36126_s128_c_64/images',
                # 'image_clusters/36126_s128_c_32/images',
                # 'clusters/00068_128/images',
                # 'vid_clusters/man_speech_128/images', 'vid_clusters/man_speech_256/images', 'vid_clusters/man_speech_64/images',
                # 'vid_clusters/woman_speech_128/images', 'vid_clusters/woman_speech_256/images',  'vid_clusters/woman_speech_64/images'
                  ]
    losses = [
        # MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=1024, pool_size=32, pool_strides=16,
        #                normalize_patch='channel_mean', pad_image=True),
        ListOfLosses([L2(),
                      PatchRBFLoss(patch_size=3, sigma=0.1, pad_image=True, device=device),
                      MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=1024, pool_size=8, pool_strides=4,
                                     normalize_patch='channel_mean', pad_image=True)
                      ],
                     weights=[0.001, 0.05, 1.0], name="MMD++(win=8)"),
        # ListOfLosses([L2(),
        #               PatchRBFLoss(patch_size=3, sigma=0.1, pad_image=True, device=device),
        #               MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=1024, pool_size=16, pool_strides=8,
        #                              normalize_patch='channel_mean', pad_image=True)
        #               ],
        #              weights=[0.001, 0.05, 1.0], name="MMD++(win=16)"),
        ListOfLosses([L2(),
                      PatchRBFLoss(patch_size=3, sigma=0.1, pad_image=True, device=device),
                      MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=1024, pool_size=32, pool_strides=16,
                                     normalize_patch='channel_mean', pad_image=True)
                      ],
                     weights=[0.001, 0.05, 1.0], name="MMD++(win=32)"),
        VGGFeatures(5, pretrained=True, post_relu=True),
        VGGFeatures(5, pretrained=False, post_relu=True),
    ]
    root = 'batch_outputs'
    for images_dir in image_dirs:
        images = load_images(images_dir)
        images_name = os.path.basename(os.path.dirname(images_dir))
        os.makedirs(os.path.join(root,images_name), exist_ok=True)
        cv2.imwrite(os.path.join(root,images_name, "l2_mean.png"), pt2cv(images.mean(0).detach().cpu().numpy()))
        for loss in losses:
            # results_dir = os.path.join("outputs", os.path.basename(os.path.dirname(images_dir)), loss.name)
            output_dir = images_dir + "_training_" + loss.name
            img = optimize_for_mean(images, loss, output_dir, num_steps=1000, lr=0.1, batch_size=4)
            cv2.imwrite(f"{root}/{images_name}/{loss.name}.png", img)


if __name__ == '__main__':
    batch_run()
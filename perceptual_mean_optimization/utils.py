import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import transforms


def pt2cv(img):
    img = (img.numpy() + 1) / 2
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)

    return img


def load_images(dir_path, sort=True):
    images = []
    paths = os.listdir(dir_path) if not sort else sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0]))
    for fn in paths:
        img = cv2.imread(os.path.join(dir_path, fn))
        img = cv2pt(img)
        # from style_transfer.utils import imload
        # img = imload(os.path.join(dir_path, fn))[0]

        images.append(img)

    return torch.stack(images)


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    # img *= 2
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


class Plotter:
    def __init__(self, n_losses, n_dirs):
        if n_dirs == 1:
            self.mode = 'single'
            self.fig, self.axs = plt.subplots(1, n_losses + 1, figsize=(3 + 3 * n_losses, 3))
        else:
            self.mode = 'multi'
            self.fig, self.axs = plt.subplots(n_dirs, n_losses + 1, figsize=(3 + 3 * n_losses, 3 * n_dirs))
        self.input_images = []

    def set_result(self, i, j, img, title=None):
        ax = self.axs[i, j] if self.mode == 'multi' else self.axs[j]
        ax.imshow(img)
        if title:
            ax.set_title(title)
        ax.axis('off')
        ax.set_aspect('equal')

    def save_fig(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)
        plt.clf()


def split_to_unmatching_squares(images, starting_img):
    new_starting_img = torch.zeros(starting_img.shape)
    new_starting_img[:, 48:-16, 48:-16] = starting_img[:, 32:-32, 32:-32].clone()
    # starting_img = new_starting_img
    new_images = torch.zeros(images.shape)
    new_images[0, :, 16:-48, 16:-48] = images[0, :, 32:-32, 32:-32]
    # images = new_images

    return new_images, new_starting_img


def downscale(imgs, perc):
    assert 0 < perc < 1
    b, c, y, x = imgs.shape
    new_x = int(x * perc)
    new_y = int(y * perc)

    return transforms.Resize((new_y, new_x), antialias=True)(imgs)


def get_pyramid(imgs, n_levels, perc):
    res = [imgs]
    for i in range(n_levels):
        imgs = downscale(imgs,perc)
        res = [imgs] + res
    return res
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from torchvision import transforms

def aspect_ratio_resize(img, max_dim=256):
    y, x, c = img.shape
    if x > y:
        return cv2.resize(img, (max_dim, int(y/x*max_dim)))
    else:
        return cv2.resize(img, (int(x/y*max_dim), max_dim))


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    # y, x, c = img.shape
    c, y, x = img.shape
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)

    # return cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA)
    return transforms.Resize((new_y, new_x), antialias=True)(img)


def get_pyramid(img, n_levels, pyr_factor):
    res = [img]
    for i in range(n_levels):
        img = downscale(img, pyr_factor)
        res = [img] + res
    return res


def quantize_image(img, N_colors):
    return np.round_(img*(N_colors/255))*(255/N_colors)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


@dataclass
class SyntesisConfigurations:
    aspect_ratio: Tuple[float, float] = (1.,1.)
    resize: int = 256
    pyr_factor: float = 0.7
    n_scales: int = 5
    lr: float = 0.05
    num_steps: int = 500
    init: str = 'noise'
    blur_loss: float = 0.0
    tv_loss: float = 0.0
    device: str = 'cuda:0'

    def get_conf_tag(self):
        init_name = 'img' if os.path.exists(self.init) else self.init
        if self.blur_loss > 0:
            init_name += f"_BL({self.blur_loss})"
        if self.tv_loss > 0:
            init_name += f"_TV({self.tv_loss})"
        return f'AR-{self.aspect_ratio}_R-{self.resize}_S-{self.pyr_factor}x{self.n_scales}_I-{init_name}'
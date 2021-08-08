import cv2
import numpy as np
from torchvision import transforms

def aspect_ratio_resize(img, max_dim=256):
    y, x, c = img.shape
    if x > y:
        return cv2.resize(img, (max_dim, int(y/x*max_dim)))
    else:
        return cv2.resize(img, (int(x/y*max_dim), max_dim))


def downscale(img, perc):
    assert 0 < perc < 1
    # y, x, c = img.shape
    c, y, x = img.shape
    new_x = int(x * perc)
    new_y = int(y * perc)

    # return cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA)
    return transforms.Resize((new_y, new_x), antialias=True)(img)


def get_pyramid(img, n_levels, perc):
    res = [img]
    for i in range(n_levels):
        img = downscale(img,perc)
        res = [img] + res
    return res


def quantize_image(img, N_colors):
    return np.round_(img*(N_colors/255))*(255/N_colors)
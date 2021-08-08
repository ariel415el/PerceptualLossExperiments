import os
import random

import cv2
import numpy as np
import torch

from Experiments.all import load_models, embedd_data, save_batch
from GenerativeModels.utils.data_utils import get_dataset

device = torch.device("cuda")


def sample_latent_neighbors(outputs_dir, models_dir):
    """Find nearest latent neighbors of data samples and create sets of original/reconstructed similar images """
    # Load models
    n = 32
    train_dataset = get_dataset('ffhq', split='train', resize=128, val_percent=0.15)
    encoder, generator = load_models(device, models_dir)
    embeddings = embedd_data(train_dataset, encoder, 32, device)
    for i in [11, 15, 16, 25, 48, 53, 60, 67, 68, 78, 122]:
        os.makedirs(os.path.join(outputs_dir, os.path.basename(models_dir), f"data_neighbors{i}"), exist_ok=True)

        dists = torch.norm(embeddings - embeddings[i], dim=1)
        neighbor_indices = torch.argsort(dists)[:n]
        neighbors = torch.from_numpy(np.array([train_dataset[x][1] for x in neighbor_indices]))
        save_batch(neighbors, os.path.join(outputs_dir, os.path.basename(models_dir), f"data_neighbors{i}"))


def center_crop_image_to_square(img, edge_perc=0.15):
    h = img.shape[0]
    w = img.shape[1]
    if h > w:
        e = (h - w) // 2
        img = img[e:-e]
    elif h < w:
        e = (w - h) // 2
        img = img[:, e:-e]
    if edge_perc:
        z = int(img.shape[0] * edge_perc)
        img = img[z:-z, z:-z]
    return img


def make_shift_sets(root, edge_size=7, zoom=0.2):
    for path in os.listdir(root):
        img = cv2.imread(os.path.join(root, path))
        img = center_crop_image_to_square(img, zoom)
        img = cv2.resize(img, (128+edge_size, 128 + edge_size))

        dir_name = os.path.join(root, 'jitters', f"{os.path.splitext(path)[0]}_e-{edge_size}_z-{zoom}")
        os.makedirs(dir_name, exist_ok=True)
        for i, (x1, y1) in enumerate([(0, 0), (0, edge_size), (edge_size, 0), (edge_size, edge_size)]):
            # x1 = np.random.randint(0, edge_size)
            # y1 = np.random.randint(0, edge_size)
            img2 = img[y1:img.shape[0] - edge_size + y1]
            img2 = img2[:, x1:img.shape[1] - edge_size + x1]
            img2 = cv2.resize(img2, (128, 128))
            x = cv2.imwrite(os.path.join(dir_name, f"{i}.png"), img2)
            print(x)


def create_shifted_colorfull_box_images():
    im_dim = 128
    n_images = 32
    box_dim = 32
    colors = [[128, 128, 255], [255, 128, 128], [128, 255, 128], [0, 128, 255], [255, 0, 128], [128, 255, 0]]
    os.makedirs('color_box_dataset', exist_ok=True)
    for i in range(n_images):
        x = random.choice(range(0, im_dim - box_dim + 3, 3))
        y = random.choice(range(0, im_dim - box_dim + 3, 3))

        im = np.ones((im_dim, im_dim, 3)) * 127
        im[y:y + box_dim, x:x + box_dim] = colors[i % len(colors)]

        cv2.imwrite(f"color_box_dataset/{i}.png", im)


if __name__ == '__main__':
    # sample_latent_neighbors("latent_neighbors_sets", 'trained_models/VGG-None_PT')
    # sample_latent_neighbors("latent_neighbors_sets", 'trained_models/VGG-random')
    make_shift_sets('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures')
    # create_shifted_colorfull_box_images()

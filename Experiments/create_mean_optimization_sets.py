import os
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


def make_shift_sets(root):
    for edge_size in [64]:
        for size in [128]:
            for path in os.listdir(root):
                img = cv2.imread(os.path.join(root, path))
                dir_name = os.path.join(root, f"{os.path.splitext(path)[0]}_s{size}_c_{edge_size}")
                os.makedirs(dir_name, exist_ok=True)
                for i in range(8):
                    x1 = np.random.randint(0, edge_size)
                    y1 = np.random.randint(0, edge_size)
                    img2 = img[y1:img.shape[0] - edge_size + y1]
                    img2 = img2[:, x1:img.shape[1] - edge_size + x1]
                    img2 = cv2.resize(img2, (size, size))
                    x = cv2.imwrite(os.path.join(dir_name, f"{i}.png"), img2)
                    print(x)


if __name__ == '__main__':
    # sample_latent_neighbors("latent_neighbors_sets", 'trained_models/VGG-None_PT')
    # sample_latent_neighbors("latent_neighbors_sets", 'trained_models/VGG-random')
    make_shift_sets('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/textures')
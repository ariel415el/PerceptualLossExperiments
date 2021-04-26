import os

import numpy as np
import torch
from tqdm import tqdm

import utils.data_utils as data_utils
from utils.fid_scroe.fid_score import calculate_frechet_distance
from utils.fid_scroe.inception import InceptionV3
import torchvision.utils as vutils


def run_FID_tests(train_dir, generator, data_embeddings, train_dataset, test_dataset, samplers, device):
    """Compute The FID score of the train, GLO, IMLE and reconstructed images distribution compared
     to the test distribution"""
    batch_size = 128

    test_dataloader = data_utils.get_dataloader(test_dataset, batch_size, device)
    train_dataloader = data_utils.get_dataloader(train_dataset, batch_size, device)

    inception_model = InceptionV3([3]).to(device).eval()

    data_dict = {"test": [], "train": [], "reconstructions": []}
    data_dict.update({sampler.name: [] for sampler in samplers})

    # Computing Inception activations
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i, (indices, images) in pbar:
            # get activations for real images from test set
            act = inception_model.get_activations(images, device).astype(np.float64)
            data_dict['test'].append(act)

            # get activations for real images from train set
            _, images = train_dataloader.dataset[indices]
            images = torch.from_numpy(images)
            act = inception_model.get_activations(images, device).astype(np.float64)
            data_dict['train'].append(act)

            # get activations for reconstructed images
            images = generator(data_embeddings[indices.long()])
            act = inception_model.get_activations(images, device).astype(np.float64)
            data_dict['reconstructions'].append(act)

            # get activations for generated images with various samplers
            for sampler in samplers:
                act = inception_model.get_activations(generator(sampler.sample(batch_size)), device).astype(np.float64)
                data_dict[sampler.name].append(act)

            pbar.set_description(f"Computing Inception activations: {i * test_dataloader.batch_size} done")

    print(f"Computing activations mean and covariances")
    for k in data_dict:
        activations = np.concatenate(data_dict[k], axis=0)
        mu, cov = np.mean(activations, axis=0), np.cov(activations, rowvar=False)
        data_dict[k] = (mu, cov)

    print("Computing FID scores")
    f = open(os.path.join(train_dir, "FID-scores.txt"), 'w')
    test_mu, test_cov = data_dict['test']
    for k, (mu, cov) in data_dict.items():
        if k != 'test':
            fid_score = calculate_frechet_distance(test_mu, test_cov, mu, cov)
            f.write(f"{k} vs test data FID: {fid_score:.2f}\n")
    f.close()


def find_nearest_neighbor(generator, sampler, dataset, train_dir):
    n = 8
    generated_imgs = generator(sampler.sample(n)).cpu()
    dataset_images = torch.from_numpy(dataset.images)
    dists = torch.sum((generated_imgs.unsqueeze(1) - dataset_images)**2, dim=[2, 3, 4]) # n x len(dataset) matrix of dist images
    nearest_neighbor_idxs = torch.argmin(dists, dim=1)
    nearest_neighbors = dataset_images[nearest_neighbor_idxs]
    vutils.save_image(torch.cat([generated_imgs, nearest_neighbors]), f"{train_dir}/imgs/{sampler.name}_Nearest_neighbors.png", normalize=True, nrow=n)


def plot_interpolations(generator, latent_sampler, data_embeddings, train_dir, z_interpolation=False):
    num_steps = 10
    num_sets = 4

    # Interpolate
    start_code = latent_sampler.sample_input_noise(num_sets)
    end_code = latent_sampler.sample_input_noise(num_sets)
    if z_interpolation:
        # start_code = latent_sampler.mapping(start_code)
        # end_code = latent_sampler.mapping(end_code)
        start_code = data_embeddings[torch.randint(len(data_embeddings), (num_sets,))]
        end_code = data_embeddings[torch.randint(len(data_embeddings), (num_sets,))]
    a = 1/float(num_steps)
    sets = []
    for i in range(num_steps):
        latent_code = (num_steps-i) * a * start_code + i * a * end_code
        if not z_interpolation:
            latent_code = latent_sampler.mapping(latent_code)
        sets.append(generator(latent_code))

    # rearange images to show changes horizontly
    rearanged_images = []
    imgs = torch.cat(sets)
    for j in range(num_sets):
        for i in range(num_steps):
            rearanged_images.append(imgs[i*num_sets + j])
    imgs = torch.stack(rearanged_images)

    vutils.save_image(imgs, f"{train_dir}/imgs/{latent_sampler.name}_{'z' if z_interpolation else 'e'}-Interpolations.png", normalize=True, nrow=num_steps)


class NormalSampler:
    def __init__(self, data_embeddings, device):
        self.device = device
        self.z_mu, self.z_cov = data_utils.get_mu_sigma(data_embeddings.clone().cpu())
        self.name = "NormalSampler"

    def sample(self, num_samples):
        output_vec = data_utils.sample_mv(num_samples, self.z_mu, self.z_cov).to(self.device)
        return output_vec


class MappingSampler:
    def __init__(self, mapping, name, input_dist, device):
        self.input_dist = input_dist
        self.mapping = mapping
        self.device = device
        self.name = f"{name}-Sampler"

    def sample_input_noise(self, num_samples):
        if self.input_dist == "uniform":
            input_noise = torch.FloatTensor(num_samples, self.mapping.input_dim).uniform_(-1.0, 1.0)
        else:
            input_noise = torch.randn(num_samples, self.mapping.input_dim)
        return input_noise.to(self.device)

    def sample(self, num_samples):
        input_noise = self.sample_input_noise(num_samples)
        output_vec = self.mapping(input_noise.to(self.device))
        return output_vec


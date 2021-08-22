import os

import numpy as np
import torch

import torchvision.utils as vutils
from tqdm import tqdm


def find_nearest_neighbor(generator, sampler, dataset, train_dir):
    n = 8
    generated_imgs = generator(sampler.sample(n)).cpu()
    dataset_images = torch.from_numpy(dataset.images)
    dists = torch.sum((generated_imgs.unsqueeze(1) - dataset_images)**2, dim=[2, 3, 4]) # n x len(dataset) matrix of dist images
    nearest_neighbor_idxs = torch.argmin(dists, dim=1)
    nearest_neighbors = dataset_images[nearest_neighbor_idxs]
    vutils.save_image(torch.cat([generated_imgs, nearest_neighbors]), f"{train_dir}/test_imgs/{sampler.name}_Nearest_neighbors.png", normalize=True, nrow=n)

def find_nearest_neighbor_memory_efficient(generator, sampler, dataset, train_dir):
    n = 8
    generated_imgs = generator(sampler.sample(n)).cpu()
    dists = np.zeros((n, dataset.images.shape[0]))
    pbar = tqdm(total=n*dataset.images.shape[0])
    for i in range(n):
        for j in range(dataset.images.shape[0]):
            pbar.set_description(f"Calculating dists: {i},{j}")
            dists[i,j] = np.mean((generated_imgs[i].detach().numpy() - dataset.images[j])**2)
    nearest_neighbor_idxs = np.argmin(dists, axis=1)
    nearest_neighbors = torch.from_numpy(dataset.images[nearest_neighbor_idxs])
    vutils.save_image(torch.cat([generated_imgs, nearest_neighbors]), f"{train_dir}/test_imgs/{sampler.name}_Nearest_neighbors.png", normalize=True, nrow=n)


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
    os.makedirs(f"{train_dir}/imgs", exist_ok=True)
    vutils.save_image(imgs, f"{train_dir}/test_imgs/{latent_sampler.name}_{'z' if z_interpolation else 'e'}-Interpolations.png", normalize=True, nrow=num_steps)


class NormalSampler:
    def __init__(self, data_embeddings, device):
        self.device = device
        self.z_mu, self.z_cov = get_mu_sigma(data_embeddings.clone().cpu())
        self.name = "NormalSampler"

    def sample(self, num_samples):
        output_vec = sample_mv(num_samples, self.z_mu, self.z_cov).to(self.device)
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


def sample_gaussian(x, m, mu=None, cov=None):
    if mu is None:
        mu, cov = get_mu_sigma(x)
    return sample_mv(m, mu, cov)


def get_mu_sigma(x):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov = np.cov(x, rowvar=False)
    return mu, cov


def sample_mv(m, mu, cov, restrict_to_unit_ball=False):
    z = np.random.multivariate_normal(mu, cov, size=m)
    z_t = torch.from_numpy(z).float()
    if restrict_to_unit_ball:
        radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
        z_t = z_t / radius
    return z_t
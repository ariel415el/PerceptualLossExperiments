import os

import numpy as np
import torch
from tqdm import tqdm

from GenerativeModels.utils.data_utils import get_dataloader
from GenerativeModels.utils.fid_scroe.fid_score import calculate_frechet_distance
from GenerativeModels.utils.fid_scroe.inception import InceptionV3
from GenerativeModels.utils.swd import compute_swd


def run_FID_tests(train_dir, generator, data_embeddings, train_dataset, test_dataset, samplers, device):
    """Compute The FID score of the train, GLO, IMLE and reconstructed images distribution compared
     to the test distribution"""
    batch_size = 128

    test_dataloader = get_dataloader(test_dataset, batch_size, device, drop_last=False)
    train_dataloader = get_dataloader(train_dataset, batch_size, device, drop_last=False)

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
            images = np.stack([train_dataloader.dataset[x][1] for x in indices])
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


def run_swd_tests(train_dir, generator, data_embeddings, train_dataset, test_dataset, samplers, device):
    n_samples = len(test_dataset)
    batch_size = 128

    data_dict = {"test": torch.from_numpy([test_dataset[i][1] for i in range(len(test_dataset))]),
                 "train": torch.from_numpy([train_dataset[i][1] for i in range(len(test_dataset))])}

    with torch.no_grad():
        images = []
        for i in range(n_samples // batch_size):
            images.append(generator(data_embeddings[i*batch_size: (i+1)*batch_size]).cpu())
        leftover = n_samples % batch_size
        if leftover > 0:
            images.append(generator(data_embeddings[-leftover:]).cpu())
            data_dict['reconstructions'] = torch.cat(images)

        for sampler in samplers:
            images = []
            for i in range(n_samples // batch_size):
                    images.append(generator(sampler.sample(batch_size)).cpu())
            leftover = n_samples % batch_size
            if leftover > 0:
                images.append(generator(sampler.sample(leftover)).cpu())
            data_dict[sampler.name] = torch.cat(images)

    print("Computing SWD scores")
    f = open(os.path.join(train_dir, "SWD-scores.txt"), 'w')
    for k, data in data_dict.items():
        if k != 'test':
            swd_score = compute_swd(data_dict['test'].float(), data.float(), device='cuda')
            f.write(f"{k} vs test data FID: {swd_score:.2f}\n")
    f.close()
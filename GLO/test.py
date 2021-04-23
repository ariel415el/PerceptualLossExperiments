import os

import numpy as np
import torch
from tqdm import tqdm

from GLO import utils
from utils import MemoryDataset
from IMLE import _netT
from fid_scroe.fid_score import calculate_frechet_distance
from fid_scroe.inception import InceptionV3
from models import DCGANGenerator
from config import faces_config
import torchvision.utils as vutils


def run_FID_tests(train_dir, glo_model, imle_model, gmmn_model, test_dataloader, train_dataloader, device):
    """Compute The FID score of the train, GLO, IMLE and reconstructed images distribution compared
     to the test distribution"""
    inception_model = InceptionV3([3]).to(device).eval()

    test_activations = []
    train_activations = []
    glo_gen_activations = []
    imle_gen_activations = []
    gmmn_gen_activations = []
    rec_activations = []

    print("Computing Inception activations")
    z_mu, z_cov = utils.get_mu_sigma(glo_model.netZ.emb.weight.clone().cpu())
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i, (indices, images) in pbar:
            pbar.set_description(f"Images done: {i * test_dataloader.batch_size}")

            # get activations for real images from test set
            act = inception_model.get_activations(images, device).astype(np.float64)
            test_activations.append(act)

            # get activations for real images from train set
            _, images = train_dataloader.dataset[indices]
            images = torch.from_numpy(images)
            act = inception_model.get_activations(images, device).astype(np.float64)
            train_activations.append(act)

            # get activations for generated images
            Zs = utils.sample_mv(test_dataloader.batch_size, z_mu, z_cov).to(device)
            images = glo_model.netG(Zs)
            act = inception_model.get_activations(images, device).astype(np.float64)
            glo_gen_activations.append(act)

            # get activations for reconstructed images
            images = glo_model.netG(glo_model.netZ(indices.long().to(device)))
            act = inception_model.get_activations(images, device).astype(np.float64)
            rec_activations.append(act)

            # Get activations for IMLE samples
            images = glo_model.netG(imle_model.netT(torch.randn(test_dataloader.batch_size, imle_model.e_dim).to(device)))
            act = inception_model.get_activations(images, device).astype(np.float64)
            imle_gen_activations.append(act)

            images = glo_model.netG(gmmn_model.netT(torch.randn(test_dataloader.batch_size, gmmn_model.e_dim).to(device)))
            act = inception_model.get_activations(images, device).astype(np.float64)
            gmmn_gen_activations.append(act)


    train_activations = np.concatenate(train_activations, axis=0)
    test_activations = np.concatenate(test_activations, axis=0)
    glo_gen_activations = np.concatenate(glo_gen_activations, axis=0)
    rec_activations = np.concatenate(rec_activations, axis=0)
    imle_gen_activations = np.concatenate(imle_gen_activations, axis=0)
    gmmn_gen_activations = np.concatenate(gmmn_gen_activations, axis=0)

    print(f"Computing activations mean and covariances on {test_activations.shape[0]} samples")

    test_mu, test_cov = np.mean(test_activations, axis=0), np.cov(test_activations, rowvar=False)
    train_mu, train_cov = np.mean(train_activations, axis=0), np.cov(train_activations, rowvar=False)
    glo_gen_mu, glo_gen_cov = np.mean(glo_gen_activations, axis=0), np.cov(glo_gen_activations, rowvar=False)
    rec_mu, rec_cov = np.mean(rec_activations, axis=0), np.cov(rec_activations, rowvar=False)
    imle_gen_mu, imle_gen_cov = np.mean(imle_gen_activations, axis=0), np.cov(imle_gen_activations, rowvar=False)
    gmmn_gen_mu, gmmn_gen_cov = np.mean(gmmn_gen_activations, axis=0), np.cov(gmmn_gen_activations, rowvar=False)

    print("Computing FID scores")
    opt_fid = calculate_frechet_distance(test_mu, test_cov, train_mu, train_cov)
    glo_gen_fid = calculate_frechet_distance(test_mu, test_cov, glo_gen_mu, glo_gen_cov)
    rec_fid = calculate_frechet_distance(test_mu, test_cov, rec_mu, rec_cov)
    imle_gen_fid = calculate_frechet_distance(test_mu, test_cov, imle_gen_mu, imle_gen_cov)
    gmmn_gen_fid = calculate_frechet_distance(test_mu, test_cov, gmmn_gen_mu, gmmn_gen_cov)

    f = open(os.path.join(train_dir, "FID-scores.txt"), 'w')
    f.write(f"Optimal scores (Train vs test) FID: {opt_fid:.2f}\n")
    f.write(f"GLO-Generated images FID: {glo_gen_fid:.2f}\n")
    f.write(f"Reconstructed train images FID: {rec_fid:.2f}\n")
    f.write(f"IMLE-Generated images FID: {imle_gen_fid:.2f}\n")
    f.write(f"GMMN-Generated images FID: {gmmn_gen_fid:.2f}\n")
    f.close()


def optimize_nn(generator, latent_encoding, train_dir, n):
    indices = torch.from_numpy(np.random.randint(0, n, 20))
    samples_path = "../../../data/FFHQ/thumbnails128x128"
    img_paths = np.array(os.listdir(samples_path))
    data = MemoryDataset([os.path.join(samples_path, x) for x in img_paths[indices]])
    data_imgs = torch.from_numpy(data[indices][1])

    imgs = generator(latent_encoding(indices))
    vutils.save_image(torch.cat([imgs,data_imgs]), f"{train_dir}/Optimized.png", normalize=True, nrow=20)


def plot_reconstructions(generator, latent_encoding, train_dir, n):
    indices = torch.from_numpy(np.random.randint(0, n, 64))

    imgs = generator(latent_encoding(indices))
    vutils.save_image(imgs, f"{train_dir}/imgs/Reconstructions.png", normalize=True, nrow=8)


def plot_interpolations(generator, latent_sampler, train_dir, z_interpolation=False):
    num_steps = 10
    num_sets = 4

    # Interpolate
    start_code = torch.randn(num_sets, faces_config.e_dim)
    end_code = torch.randn(num_sets, faces_config.e_dim)
    if z_interpolation:
        start_code = latent_sampler(start_code)
        end_code = latent_sampler(end_code)
    a = 1/float(num_steps)
    sets = []
    for i in range(num_steps):
        latent_code = (num_steps-i) * a * start_code + i * a * end_code
        if not z_interpolation:
            latent_code = latent_sampler(latent_code)
        sets.append(generator(latent_code))

    # rearange images to show changes horizontly
    rearanged_images = []
    imgs = torch.cat(sets)
    for j in range(num_sets):
        for i in range(num_steps):
            rearanged_images.append(imgs[i*num_sets + j])
    imgs = torch.stack(rearanged_images)

    vutils.save_image(imgs, f"{train_dir}/imgs/{'z' if z_interpolation else 'e'}-Interpolations.png", normalize=True, nrow=num_steps)


if __name__ == '__main__':
    train_dir = '/home/ariel/universirty/PerceptualLoss/PerceptualLossExperiments/GLO/outputs/global-optimizers/ffhq_VGG_L-5_PR_PT_-No-z-normalization'

    # Load models
    generator = DCGANGenerator(faces_config.z_dim, faces_config.channels, faces_config.img_dim)
    generator.load_state_dict(torch.load(os.path.join(train_dir, 'netG.pth'), map_location=torch.device('cpu')))

    latent_sampler = _netT(faces_config.e_dim, faces_config.z_dim)
    latent_sampler.load_state_dict(torch.load(os.path.join(train_dir, 'netT.pth'), map_location=torch.device('cpu')))

    latent_encoding_weights = torch.load(os.path.join(train_dir, 'netZ.pth'), map_location=torch.device('cpu'))
    n, nz = latent_encoding_weights['emb.weight'].shape
    latent_encodings = torch.nn.Embedding(n, nz)
    latent_encodings.load_state_dict({'weight':latent_encoding_weights['emb.weight']})

    # Test
    # optimize_nn(generator, latent_encodings, train_dir, n=n)
    plot_reconstructions(generator, latent_encodings, train_dir, n=n)
    plot_interpolations(generator, latent_sampler, train_dir)
    plot_interpolations(generator, latent_sampler, train_dir, z_interpolation=True)




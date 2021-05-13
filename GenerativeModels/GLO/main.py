import os

import torch
import torch.utils.data
import torchvision.utils as vutils

from GenerativeModels.GLO.IMLE import IMLE
from config import faces_config
import sys


sys.path.append(os.path.realpath("../.."))
from losses.patch_loss import PatchRBFLoss
from losses.l2 import L2
from losses.lap1_loss import LapLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.vgg_loss.vgg_loss import VGGFeatures
from GenerativeModels.GLO.utils import NormalSampler, MappingSampler, plot_interpolations
from GenerativeModels.utils.data_utils import get_dataset
from GenerativeModels.GMMN.GMMN import GMMN
from GLO import GLOTrainer
from GenerativeModels import models

sys.path.append(os.path.realpath("../.."))
from losses.utils import ListOfLosses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train_GLO(dataset_name, train_dir):
    train_dataset = get_dataset('ffhq', split='train')
    glo_params = faces_config

    # define the generator
    # generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim)
    generator = models.WAE_Decoder(glo_params.z_dim, glo_params.channels, glo_params.img_dim)

    # Define the loss criterion
    criterion = ListOfLosses(
        [
            L2(),
            # VGGFeatures(3 if glo_params.img_dim == 28 else 5, pretrained=True, post_relu=True),
            # LapLoss(max_levels=3 if glo_params.img_dim == 28 else 5, n_channels=glo_params.channels),
            # MMD()
            # MMDApproximate(patch_size=3, sigma=0.06, strides=1, r=1024, pool_size=8, pool_strides=4, normalize_patch='channel_mean', pad_image=True),
            # PatchRBFLoss(patch_size=3, sigma=0.1, pad_image=True, device=device)
            # MMDApproximate(r=1024, pool_size=32, pool_strides=16, normalize_patch='mean'),
            # self.dist = ScnnLoss()
        ]
        # , weights=[0.001, 0.05, 1.0]
    )

    outptus_dir = train_dir
    glo_trainer = GLOTrainer(glo_params, generator, criterion, train_dataset, device)
    glo_trainer.train(outptus_dir)

    # # Save trained model and data embedding
    # torch.save(glo_trainer.latent_codes.state_dict(), f"{outptus_dir}/latent_codes.pth")
    # torch.save(generator.state_dict(), f"{outptus_dir}/generator.pth")


def train_latent_samplers(train_dir):
    latent_codes = torch.load(os.path.join(train_dir, 'latent_codes.pth'), map_location=device)['emb.weight']
    n, z_dim = latent_codes.shape
    e_dim = z_dim

    mapping = models.LatentMapper(e_dim, z_dim).train()
    #
    imle = IMLE(mapping, lr=0.001, batch_size=128, device=device)
    imle.train(latent_codes.cpu().numpy(), train_dir=train_dir, epochs=20)
    torch.save(mapping.state_dict(), f"{train_dir}/IMLE-Mapping.pth")

    # gmmn = GMMN(mapping, lr=0.0001, batch_size=6000, device=device)
    # gmmn.train(latent_codes, train_dir=train_dir, epochs=100)
    # torch.save(mapping.state_dict(), f"{train_dir}/GMMN-Mapping.pth")


def test_models(train_dir):
    glo_params = faces_config
    generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim).to(device)
    imle_mapping = models.LatentMapper(glo_params.e_dim, glo_params.z_dim).to(device)
    # gmmn_mapping = models.LatentMapper(glo_params.e_dim, glo_params.z_dim).to(device)

    data_embeddings = torch.load(os.path.join(train_dir, 'latent_codes.pth'), map_location=device)['emb.weight']
    generator.load_state_dict(torch.load(os.path.join(train_dir, 'generator.pth'), map_location=device))
    imle_mapping.load_state_dict(torch.load(os.path.join(train_dir, 'IMLE-Mapping.pth'), map_location=device))
    # gmmn_mapping.load_state_dict(torch.load(os.path.join(train_dir, 'GMMN-Mapping.pth'), map_location=device))

    # generator.eval()
    # imle_mapping.eval()
    # gmmn_mapping.eval()

    # plot reconstructions
    latent_codes = data_embeddings[torch.arange(64)]
    vutils.save_image(generator(latent_codes).cpu(), f"{train_dir}/imgs/Reconstructions.png", normalize=True)

    samplers = [NormalSampler(data_embeddings, device),
                MappingSampler(imle_mapping, "IMLE", "normal", device),
                # MappingSampler(gmmn_mapping, "GMMN", "uniform", device)
                ]

    # train_dataset = get_dataset('ffhq', split='train')
    test_dataset = get_dataset('ffhq', split='test')

    for sampler in samplers:
        vutils.save_image(generator(sampler.sample(64)), f"{train_dir}/imgs/{sampler.name}_generated_images.png", normalize=True)
        # find_nearest_neighbor(generator, sampler, train_dataset, train_dir)

    for sampler in samplers[1:]:
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=False)
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=True)

    # run_FID_tests(train_dir, generator, data_embeddings, train_dataset, test_dataset, samplers, device)


def plot_GLO_variance():
    os.makedirs('tmps', exist_ok=True)
    import numpy as np
    full_data = '/home/ariel/universirty/PerceptualLoss/PerceptualLossExperiments/GLO/outputs/april_25/dummy'
    partial_data = '/home/ariel/universirty/PerceptualLoss/PerceptualLossExperiments/GLO/outputs/april_25/pretrained-1000_examples_random_unit_normed'
    glo_params = faces_config
    generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim).to(device)
    for dir_name, name in [(full_data, "full"), (partial_data, "partial")]:
        generator.load_state_dict(torch.load(os.path.join(dir_name, 'generator.pth'), map_location=device))
        data_embeddings = torch.load(os.path.join(dir_name, 'latent_codes.pth'), map_location=device)['emb.weight']
        vutils.save_image(generator(data_embeddings[:64]), f"tmps/{name}_recs.png", normalize=True)
        for sigma in [0.00001, 0.001, 0.01, 0.1]:
            noise = torch.from_numpy(np.random.multivariate_normal(data_embeddings[6].cpu().numpy(), torch.eye(glo_params.z_dim)*sigma, size=64)).to(device).float()
            imgs = generator(noise)
            vutils.save_image((imgs+1)/2, f"tmps/{name}_{sigma}-{imgs.min():.3f}_{imgs.max():.3f}.png",normalize=False)


if __name__ == '__main__':
    # plot_GLO_variance()
    train_dir = 'outputs/WAE-arch/l2'
    train_GLO('ffhq', train_dir)
    # train_latent_samplers(train_dir)
    # test_models(train_dir)
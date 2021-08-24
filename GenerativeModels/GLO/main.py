import os

import torch
import torch.utils.data
import torchvision.utils as vutils

import sys

sys.path.append(os.path.realpath("../.."))

from GenerativeModels.GLO.utils import NormalSampler, MappingSampler, plot_interpolations, find_nearest_neighbor_memory_efficient
from GLO import GLOTrainer

import losses

from GenerativeModels.utils.IMLE_sampler import IMLESamplerTrainer
# from GenerativeModels.GLO.GMMN_sampler import GMMNSamplerTrainer
from GenerativeModels.config import default_config
from GenerativeModels.models import weights_init
from GenerativeModels.utils.test_utils import run_FID_tests
from GenerativeModels.utils.data_utils import get_dataset
from GenerativeModels import models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train_GLO(dataset_name, train_name, tag):
    glo_params = default_config
    train_dataset = get_dataset(dataset_name, split='train', resize=default_config.img_dim)

    # define the generator
    generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim)
    generator.apply(weights_init)

    # Define the loss criterion

    # criterion = losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('pixels', 1), ('conv1_2', 1)])
    # criterion.name = "VGG-PT-conv1_2+pix"
    # criterion = losses.MMD_PP(r=128)
    criterion = losses.LossesList([
            losses.L2(),
            # losses.PatchRBFLoss(patch_size=5, strides=1, sigma=0.1, pad_image=True, normalize_patch='none'),
            losses.MMDApproximate(patch_size=5, strides=2, sigma=0.04,pool_size=32, pool_strides=16, r=128, normalize_patch='channel_mean')#, patch_size=11, sigma=0.01)
            # losses.MMDApproximate(patch_size=5, strides=1, sigma=0.03,pool_size=128, pool_strides=1, r=128, normalize_patch='channel_mean')#, patch_size=11, sigma=0.01)
            # losses.PatchSWDLoss(patch_size=3, stride=1, normalize_patch='channel_mean', r=128)  # , patch_size=11, sigma=0.01)

    ], weights=[0.06, 1.0])
    criterion.name = "MMD++no-localtem(5:2)"

    outptus_dir = os.path.join('outputs', train_name, criterion.name + tag)
    glo_trainer = GLOTrainer(glo_params, generator, criterion, train_dataset, device)
    # glo_trainer._load_ckpt(outptus_dir)
    glo_trainer.train(outptus_dir, epochs=glo_params.num_epochs)


def train_latent_samplers(train_dir):
    latent_codes = torch.load(os.path.join(train_dir, 'latent_codes.pth'), map_location=device)['emb.weight']
    n, z_dim = latent_codes.shape
    e_dim = default_config.e_dim

    mapping = models.LatentMapper(e_dim, z_dim).train()

    trainer = IMLESamplerTrainer(mapping, lr=0.001, batch_size=8, device=device)
    trainer.train(latent_codes, train_dir=train_dir, epochs=20)
    torch.save(mapping.state_dict(), f"{train_dir}/IMLE-Mapping.pth")

    # trainer = GMMNSamplerTrainer(mapping, lr=0.001, batch_size=1000, device=device)
    # trainer.train(latent_codes, train_dir=train_dir, epochs=20)
    # torch.save(mapping.state_dict(), f"{train_dir}/GMMN-Mapping.pth")


def test_models(train_dir):
    os.makedirs(f"{train_dir}/test_imgs", exist_ok=True)

    glo_params = default_config
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

    # plot train reconstructions
    latent_codes = data_embeddings[torch.arange(64)]
    vutils.save_image(generator(latent_codes).cpu(), f"{train_dir}/test_imgs/train_reconstructions.png", normalize=True)

    samplers = [NormalSampler(data_embeddings, device),
                MappingSampler(imle_mapping, "IMLE", "normal", device),
                # MappingSampler(gmmn_mapping, "GMMN", "uniform", device)
                ]

    train_dataset = get_dataset('ffhq', split='train', resize=glo_params.img_dim)

    for sampler in samplers:
        vutils.save_image(generator(sampler.sample(64)), f"{train_dir}/test_imgs/{sampler.name}_generated_images.png",
                          normalize=True)
        find_nearest_neighbor_memory_efficient(generator, sampler, train_dataset, train_dir)

    train_dataset = get_dataset('ffhq', split='train', resize=glo_params.img_dim, memory_dataset=False)
    test_dataset = get_dataset('ffhq', split='test', resize=glo_params.img_dim, memory_dataset=False)

    print('as')
    for sampler in samplers[1:]:
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=False)
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=True)

    run_FID_tests(train_dir, generator, data_embeddings, train_dataset, test_dataset, samplers, device)


if __name__ == '__main__':
    # plot_GLO_variance()
    train_GLO('ffhq', "ffhq_128_512-z", '')
    # train_dir = os.path.join('outputs', 'ffhq_128_512-z', 'VGG-PT-conv2_2')
    # train_latent_samplers(train_dir)
    # test_models(train_dir)

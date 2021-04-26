import os
import sys

import torch
import torch.utils.data
import torchvision.utils as vutils

from config import faces_config
from utils.test_utils import NormalSampler, MappingSampler, run_FID_tests, find_nearest_neighbor, plot_interpolations
from utils.data_utils import get_dataset
from GMMN import GMMN
from IMLE import IMLE
from GLO import GLOTrainer
import models

sys.path.append(os.path.realpath(".."))
from losses.utils import ListOfLosses
from losses.vgg_loss.vgg_loss import VGGFeatures


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train_GLO(dataset_name, train_dir):
    train_dataset = get_dataset('ffhq', split='train')
    glo_params = faces_config

    # define the generator
    generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim)

    # Define the loss criterion
    criterion = ListOfLosses([
        # L2(),
        VGGFeatures(3 if glo_params.img_dim == 28 else 5, pretrained=False, post_relu=True),
        # LapLoss(max_levels=3 if glo_params.img_dim == 28 else 5, n_channels=glo_params.channels),
        # MMD()
        # PatchRBFLoss(3, device=self.device),
        # MMDApproximate(r=1024, pool_size=32, pool_strides=16, normalize_patch='mean'),
        # self.dist = ScnnLoss()
    ])

    outptus_dir = train_dir
    glo_trainer = GLOTrainer(glo_params, generator, criterion, train_dataset, device)
    glo_trainer.train(outptus_dir)

    # Save trained model and data embedding
    torch.save(glo_trainer.latent_codes.state_dict(), f"{outptus_dir}/latent_codes.pth")
    torch.save(generator.state_dict(), f"{outptus_dir}/generator.pth")


def train_latent_samplers(train_dir):
    latent_codes = torch.load(os.path.join(train_dir, 'latent_codes.pth'), map_location=device)['emb.weight']
    n, z_dim = latent_codes.shape
    e_dim = z_dim

    mapping = models.LatentMapper(e_dim, z_dim).train()
    #
    imle = IMLE(mapping, lr=0.001, batch_size=128, device=device)
    imle.train(latent_codes.cpu().numpy(), train_dir=train_dir, epochs=50)
    torch.save(mapping.state_dict(), f"{train_dir}/IMLE-Mapping.pth")

    gmmn = GMMN(mapping, lr=0.0001, batch_size=1024, device=device)
    gmmn.train(latent_codes, train_dir=train_dir, epochs=100)
    torch.save(mapping.state_dict(), f"{train_dir}/GMMN-Mapping.pth")


def test_models(train_dir):
    glo_params = faces_config
    generator = models.DCGANGenerator(glo_params.z_dim, glo_params.channels, glo_params.img_dim).to(device)
    imle_mapping = models.LatentMapper(glo_params.e_dim, glo_params.z_dim).to(device)
    gmmn_mapping = models.LatentMapper(glo_params.e_dim, glo_params.z_dim).to(device)

    data_embeddings = torch.load(os.path.join(train_dir, 'latent_codes.pth'), map_location=device)['emb.weight']
    generator.load_state_dict(torch.load(os.path.join(train_dir, 'generator.pth'), map_location=device))
    imle_mapping.load_state_dict(torch.load(os.path.join(train_dir, 'IMLE-Mapping.pth'), map_location=device))
    gmmn_mapping.load_state_dict(torch.load(os.path.join(train_dir, 'GMMN-Mapping.pth'), map_location=device))

    generator.eval()
    imle_mapping.eval()
    gmmn_mapping.eval()

    # plot reconstructions
    latent_codes = data_embeddings[torch.randint(data_embeddings.shape[0], (64,))]
    vutils.save_image(generator(latent_codes).cpu(), f"{train_dir}/imgs/Reconstructions.png", normalize=True)

    samplers = [NormalSampler(data_embeddings, device),
                MappingSampler(imle_mapping, "IMLE", "normal", device),
                MappingSampler(gmmn_mapping, "GMMN", "uniform", device)]

    train_dataset = get_dataset('ffhq', split='train')
    # test_dataset = get_dataset('ffhq', split='test')

    for sampler in samplers:
        vutils.save_image(generator(sampler.sample(64)), f"{train_dir}/imgs/{sampler.name}_generated_images.png", normalize=True)
        find_nearest_neighbor(generator, sampler, train_dataset, train_dir)

    for sampler in samplers[1:]:
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=False)
        plot_interpolations(generator, sampler, data_embeddings, train_dir, z_interpolation=True)

    # run_FID_tests(train_dir, generator, data_embeddings, test_dataset, test_dataset, samplers, device)


if __name__ == '__main__':
    train_dir = 'outputs/april_25/Ininiate_VGG-all-random'
    train_GLO('ffhq', train_dir)
    # train_latent_samplers(train_dir)
    # test_models(train_dir)
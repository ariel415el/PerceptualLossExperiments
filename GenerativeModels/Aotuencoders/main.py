import os
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision.utils as vutils

from GenerativeModels.Aotuencoders.autoencoder import AutoEncoderTraniner
from GenerativeModels.GLO.IMLE import IMLE
from GenerativeModels.GLO.config import faces_config

import sys
from GenerativeModels.GLO.utils import NormalSampler, MappingSampler

sys.path.append(os.path.realpath("../.."))
from losses.patch_loss import PatchRBFLoss
from losses.l2 import L2
from losses.lap1_loss import LapLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.vgg_loss.vgg_loss import VGGFeatures
from losses.utils import ListOfLosses
from GenerativeModels.utils.data_utils import get_dataset, get_dataloader
from GenerativeModels import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train_autoencoder(dataset_name, train_dir):
    train_dataset = get_dataset('ffhq', split='train')
    params = faces_config

    # define the generator
    encoder = models.DCGANEncoder(params.z_dim, params.channels, params.img_dim)
    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim)

    # Define the loss criterion
    criterion = ListOfLosses(
        [
            # L2(),
            VGGFeatures(5, pretrained=False, post_relu=True),
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
    trainer = AutoEncoderTraniner(faces_config, encoder, generator, criterion, train_dataset, device)
    trainer.train(outptus_dir, epochs=200)


def evaluate_generator(outputs_dir):
    params = faces_config
    encoder = models.DCGANEncoder(params.z_dim, params.channels, params.img_dim).to(device)
    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim).to(device)
    generator.load_state_dict(torch.load(f"{outputs_dir}/generator.pth", map_location=device))
    encoder.load_state_dict(torch.load(f"{outputs_dir}/encoder.pth", map_location=device))

    test_dataset = get_dataset('ffhq', split='train')
    dataloader = get_dataloader(test_dataset, params.batch_size, device)

    with torch.no_grad():
        embeddings = []
        for _, imgs in tqdm(dataloader):
            imgs = imgs.to(device).float()
            embeddings.append(encoder(imgs))
        embeddings = torch.cat(embeddings, dim=0)

    sampler = NormalSampler(embeddings, device=device)
    sampled_images = generator(sampler.sample(25))
    vutils.save_image(sampled_images.data, os.path.join(outputs_dir, 'sampled_images.png'), normalize=True, nrow=5)

    mapping = models.LatentMapper(params.z_dim, params.z_dim).train()

    imle = IMLE(mapping, lr=0.001, batch_size=128, device=device)
    imle.train(embeddings.cpu().numpy(), train_dir=train_dir, epochs=50)
    torch.save(mapping.state_dict(), f"{outputs_dir}/IMLE-Mapping.pth")
    sampler = MappingSampler(mapping, "IMLE", "normal", device)
    sampled_images = generator(sampler.sample(25))

    vutils.save_image(sampled_images.data, os.path.join(outputs_dir, 'IMLE_sampled_images.png'), normalize=True, nrow=5)

    test_dataset = get_dataset('ffhq', split='test')
    rec_imgs = generator(encoder(torch.from_numpy(test_dataset.images[:25]).to(device).float()))
    vutils.save_image(rec_imgs, os.path.join(outputs_dir, 'test-reconstructions.png'), normalize=True, nrow=5)

if __name__ == '__main__':
    train_dir = 'outputs/WAE-arch_full_data/VGG-random'
    train_autoencoder('ffhq', train_dir)
    evaluate_generator(train_dir)

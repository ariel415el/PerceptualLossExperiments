import os
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision.utils as vutils

import sys

from GenerativeModels.models import weights_init
from GenerativeModels.utils.test_utils import run_FID_tests, run_swd_tests
from losses.experimental_patch_losses import MMD_PPP
from losses.l2 import L2, L1
from losses.lap1_loss import LapLoss
from losses.patch_loss import PatchRBFLoss
from losses.patch_mmd_pp import MMD_PP
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

sys.path.append(os.path.realpath("../.."))
from GenerativeModels.Aotuencoders.autoencoder import AutoEncoderTraniner
from GenerativeModels.GLO.IMLE import IMLE
from GenerativeModels.GLO.config import faces_config
from GenerativeModels.GLO.utils import NormalSampler, MappingSampler
from GenerativeModels.utils.data_utils import get_dataset, get_dataloader
from GenerativeModels import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def train_autoencoder(dataset_name, train_name, tag):
    params = faces_config
    train_dataset = get_dataset(dataset_name, split='train', resize=params.img_dim, val_percent=0.5)
    print("Dataset size: ", len(train_dataset))

    # define the generator
    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim)
    encoder.apply(weights_init)

    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim)
    generator.apply(weights_init)

    # Define the loss criterion
    # criterion = L1()
    # criterion = LapLoss()
    # criterion = MMD_PPP(device, r=256, weights=[0.001, 0.05, 0.1, 1.0], batch_reduction='none')
    # criterion = VGGPerceptualLoss(pretrained=True)
    criterion = VGGPerceptualLoss(pretrained=False, norm_first_conv=True, reinit=True)
    # criterion = VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv1_2', 0.562), ('conv2_2', 0.098), ('conv3_3', 0.031), ('conv4_3', 0.105), ('conv5_3', 0.904)])
    # criterion = VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True)
    # criterion = MMD_PPP(r=200)
    # criterion = MMDApproximate(r=1024, pool_size=32, pool_strides=16, normalize_patch='channel_mean')
    # criterion = MMD_PPP(r=512, device=device)
    # criterion = MMD_PP(device, r=512)
    # criterion = PatchRBFLoss(patch_size=11, sigma=0.02, pad_image=True, device=device, batch_reduction='none')

    outptus_dir = os.path.join('outputs', train_name, criterion.name + tag)
    trainer = AutoEncoderTraniner(faces_config, encoder, generator, criterion, train_dataset, device)
    # trainer._load_ckpt(outptus_dir)
    trainer.train(outptus_dir, epochs=faces_config.num_epochs)


def train_latent_samplers(train_dir):
    params = faces_config

    train_dataset = get_dataset('ffhq', split='train', resize=params.img_dim)

    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)
    encoder.load_state_dict(torch.load(f"{train_dir}/encoder.pth", map_location=device))

    embeddings = embedd_data(train_dataset, encoder, params.batch_size)

    mapping = models.LatentMapper(params.z_dim, params.z_dim).train()

    imle = IMLE(mapping, lr=0.001, batch_size=128, device=device)
    imle.train(embeddings.cpu().numpy(), train_dir=train_dir, epochs=10)
    torch.save(mapping.state_dict(), f"{train_dir}/IMLE-Mapping.pth")


def embedd_data(dataset, encoder, batch_size):
    dataloader = get_dataloader(dataset, batch_size, device, drop_last=False, shuffle=False)

    with torch.no_grad():
        embeddings = []
        print("Computing embeddings... ", end='')
        for _, imgs in tqdm(dataloader):
            imgs = imgs.to(device).float()
            embeddings.append(encoder(imgs))
        embeddings = torch.cat(embeddings, dim=0)
        print('done')
    return embeddings


def evaluate_generator(outputs_dir):
    params = faces_config

    # Load datasets
    train_dataset = get_dataset('ffhq', split='train', resize=params.img_dim)
    test_dataset = get_dataset('ffhq', split='test', resize=params.img_dim)

    # Load models
    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)
    encoder.load_state_dict(torch.load(f"{outputs_dir}/encoder.pth", map_location=device))

    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim).to(device)
    generator.load_state_dict(torch.load(f"{outputs_dir}/generator.pth", map_location=device))

    mapping = models.LatentMapper(params.z_dim, params.z_dim).to(device)
    mapping.load_state_dict(torch.load(f"{outputs_dir}/IMLE-Mapping.pth", map_location=device))

    embeddings = embedd_data(train_dataset, encoder, params.batch_size)

    # Plot sampled and reconstructions images
    normal_sampler = NormalSampler(embeddings, device=device)
    sampled_images = generator(normal_sampler.sample(25))
    vutils.save_image(sampled_images.data, os.path.join(outputs_dir, 'sampled_images.png'), normalize=True, nrow=5)

    imle_sampler = MappingSampler(mapping, "IMLE", "normal", device)
    sampled_images = generator(imle_sampler.sample(25))
    vutils.save_image(sampled_images.data, os.path.join(outputs_dir, 'IMLE_sampled_images.png'), normalize=True, nrow=5)

    test_images = torch.stack([torch.from_numpy(test_dataset[i][1]) for i in range(25)]).to(device).float()
    rec_imgs = generator(encoder(test_images))
    vutils.save_image(rec_imgs, os.path.join(outputs_dir, 'test-reconstructions.png'), normalize=True, nrow=5)

    # Run statistic test
    run_FID_tests(outputs_dir, generator, embeddings, train_dataset, test_dataset, [normal_sampler, imle_sampler],
                  device)
    run_swd_tests(outputs_dir, generator, embeddings, train_dataset, test_dataset, [normal_sampler, imle_sampler],
                  device)

if __name__ == '__main__':
    train_name = f"ffhq_128-5-epochs"
    train_autoencoder('ffhq', train_name, 'blurPool')
    # train_latent_samplers('outputs/ffhq_128/VGG-None_PT')
    # evaluate_generator('outputs/test/VGG-None_PT')

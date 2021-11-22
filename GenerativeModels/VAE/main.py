import os

import numpy as np
import torch.utils.data

from GenerativeModels.VAE.models import VAEGenerator, VAEEncoder, Encoder, Decoder, DCGANEncoder, DCGANGenerator, \
    weights_init
from GenerativeModels.VAE.vae import VAETrainer
from GenerativeModels.config import default_config
from GenerativeModels.utils.data_utils import get_dataset
from losses.classic_losses.grad_loss import GradLoss3Channels
from losses.classic_losses.l2 import L2
from losses.composite_losses.list_loss import LossesList
from losses.experimental_patch_losses import MMD_PP
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.patch_loss import PatchRBFLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_autoencoder(dataset_name, train_name, tag):
    params = default_config
    params.batch_size = 128
    params.z_dim = 64
    params.lr = 0.001
    train_dataset = get_dataset(dataset_name, split='train', resize=params.img_dim)

    print("Dataset size: ", len(train_dataset))

    # define the generator
    encoder = DCGANEncoder(params.img_dim, params.channels, params.z_dim).train()
    # encoder = VAEEncoder(params.img_dim, params.channels, params.z_dim)
    encoder.apply(weights_init)

    # encoder = Encoder(params.img_dim, params.channels, params.z_dim)

    generator = DCGANGenerator(params.z_dim, params.channels, params.img_dim).train()
    # generator = VAEGenerator(params.z_dim, params.channels, params.img_dim)
    generator.apply(weights_init)
    # generator = Decoder(params.z_dim, params.channels, params.img_dim)

    # criterion = L2()
    # criterion = VGGPerceptualLoss(pretrained=True)
    # criterion = MMD_PP(r=64)
    # criterion = GradLoss3Channels()
    criterion = PatchRBFLoss(patch_size=11, sigma=5, normalize_patch='none'),
    # criterion = LossesList([
    #     L2(),
    #     GradLoss3Channels(),
    #     MMDApproximate(patch_size=11, sigma=0.02, strides=5, r=64, pool_size=32, pool_strides=16, normalize_patch='channell_mean')
    # ], weights=[0.01, 0.09, .9])

    outptus_dir = os.path.join('outputs', train_name, criterion.name + tag)
    trainer = VAETrainer(default_config, encoder, generator, criterion, train_dataset, device)
    trainer.train(outptus_dir, epochs=default_config.num_epochs)

    # trainer._load_ckpt(outptus_dir)

    # with torch.no_grad():
    #     import torchvision.utils as vutils
    #     images = np.stack([train_dataset[x][1] for x in np.arange(25)])
    #     first_imgs = torch.from_numpy(images)
    #     os.makedirs(os.path.join(outptus_dir, "reconstructions"), exist_ok=True)
    #     vutils.save_image(first_imgs.data, os.path.join(outptus_dir, 'reconstructions', 'originals.png'),normalize=True, nrow=5)
    #
    #     mu, log_var = trainer.encoder(first_imgs.to(device).float())
    #     Irec = trainer.generator(mu)
    #     vutils.save_image(Irec, os.path.join(outptus_dir, 'reconstructions', f"recons.png"), normalize=True,nrow=5)


if __name__ == '__main__':
    train_name = f"FFHQ_VAE"
    train_autoencoder('ffhq', train_name, '')
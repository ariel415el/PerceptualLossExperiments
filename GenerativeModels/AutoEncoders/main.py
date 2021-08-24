import os
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision.utils as vutils

import sys

sys.path.append(os.path.realpath(".."))

from GenerativeModels.models import weights_init
from GenerativeModels.utils.test_utils import run_FID_tests, run_swd_tests
from GenerativeModels.AutoEncoders.autoencoder import AutoEncoderTraniner
from GenerativeModels.utils.IMLE_sampler import IMLESamplerTrainer
# from GenerativeModels.GLO.GMMN_sampler import GMMNSamplerTrainer
from GenerativeModels.config import default_config
from GenerativeModels.GLO.utils import NormalSampler, MappingSampler
from GenerativeModels.utils.data_utils import get_dataset, get_dataloader, read_lfw_data
from GenerativeModels import models

import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def copy_files(outputs_dir):
    import glob
    from shutil import copy
    import os
    code_dir = os.path.join(outputs_dir, "code_copy")
    os.makedirs(code_dir, exist_ok=True)
    local_dir = os.path.dirname(os.path.dirname(__file__))
    all_files = glob.glob(local_dir + '/**/*.py', recursive=True)
    all_files_targets = [x.replace(local_dir + "/", '') for x in all_files]
    for py_file, py_target in zip(all_files, all_files_targets):
        py_target = os.path.join(outputs_dir, "code_copy", py_target)
        parent_dir = os.path.dirname(py_target)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        copy(py_file, py_target)

def train_autoencoder(dataset_name, train_name, tag):
    params = default_config
    train_dataset = get_dataset(dataset_name, split='train', resize=params.img_dim)
    # train_dataset, labels = read_lfw_data('../../../../data/LFW', img_size=128)

    print("Dataset size: ", len(train_dataset))

    # define the generator
    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim)
    encoder.apply(weights_init)

    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim)
    generator.apply(weights_init)

    criterion = losses.VGGPerceptualLoss(pretrained=True)

    outptus_dir = os.path.join('outputs', train_name, criterion.name + tag)
    copy_files(outptus_dir)
    trainer = AutoEncoderTraniner(default_config, encoder, generator, criterion, train_dataset, device)
    # trainer._load_ckpt(outptus_dir)
    trainer.train(outptus_dir, epochs=default_config.num_epochs)


def train_latent_samplers(train_dir):
    params = default_config

    train_dataset = get_dataset('ffhq', split='train', resize=params.img_dim)

    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)
    encoder.load_state_dict(torch.load(f"{train_dir}/encoder.pth", map_location=device))

    embeddings = embedd_data(train_dataset, encoder, params.batch_size)

    mapping = models.LatentMapper(params.z_dim, params.z_dim).train()

    imle = IMLESamplerTrainer(mapping, lr=0.001, batch_size=16, device=device)
    imle.train(embeddings.cpu().numpy(), train_dir=train_dir, epochs=20)
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
    params = default_config

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
    train_name = f"ffhq_64_exps"
    train_autoencoder('ffhq', train_name, '')
    # train_latent_samplers('outputs/ffhq_128/VGG-None_PT')
    # evaluate_generator('outputs/test/VGG-None_PT')

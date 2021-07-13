import os
from collections import defaultdict

import torch.nn.functional
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils as vutils
from tqdm import tqdm

from GenerativeModels import models
from GenerativeModels.config import default_config
from GenerativeModels.models import weights_init
from GenerativeModels.utils.data_utils import get_dataset, get_dataloader
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.experimental_patch_losses import MMD_PP
from losses.vgg_loss.vgg_loss import VGGFeatures, VGGPerceptualLoss


def embedd_data(dataset, encoder, batch_size, device):
    """Batch encoding of entire dataset with a given encoder"""
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


def save_batch(batch, dir):
    for i in range(len(batch)):
        vutils.save_image(batch[i], os.path.join(dir, f"{i}.png"), normalize=True)


def load_models(device, ckp_dir=None):
    params = default_config
    encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)

    generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim).to(device)

    if ckp_dir:
        encoder.load_state_dict(torch.load(f"{ckp_dir}/encoder.pth", map_location=device))
        generator.load_state_dict(torch.load(f"{ckp_dir}/generator.pth", map_location=device))
    else:
        encoder.apply(weights_init)
        generator.apply(weights_init)
    encoder.eval()
    generator.eval()
    return encoder, generator


def generate_interpolations_to_nn(train_dataset, indices, outputs_dir, n_steps, n_neighbors=5):
    encoder, generator = load_models('../GenerativeModels/Aotuencoders/outputs/ffhq_128/VGG-None_PT')
    embeddings = embedd_data(train_dataset, encoder, params.batch_size)
    losses = [
        VGGPerceptualLoss(pretrained=True).to(device),
        VGGPerceptualLoss(pretrained=False).to(device),
        MMDApproximate(r=1024, pool_size=32, pool_strides=16, normalize_patch='channel_mean').to(device),
        MMD_PP(device, patch_size=3, pool_size=32, pool_strides=16, r=256, normalize_patch='channel_mean',
               weights=[0.05, 0.1, 1.0])
    ]
    for i in indices:
        latent_code = embeddings[i]

        dists = torch.norm(embeddings - latent_code, dim=1)
        neighbor_indices = torch.argsort(dists)[:25]
        neighbors = torch.tensor([train_dataset[i][1] for i in neighbor_indices])
        neighbor_embeddings = embeddings[neighbor_indices]

        interpolations = {j: [latent_code] for j in range(1, n_neighbors + 1)}
        for j in range(1, n_neighbors + 1):
            losses_first = defaultdict(lambda: list())
            losses_last = defaultdict(lambda: list())
            os.makedirs(os.path.join(outputs_dir, 'interpolations', f"interpolation_to-{j}_{i}"), exist_ok=True)
            for k in range(n_steps):
                latent_interpolation = (n_steps - 1 - k) / (n_steps - 1) * latent_code + (k / (n_steps - 1)) * \
                                       neighbor_embeddings[j]
                image = generator(latent_interpolation.unsqueeze(0))
                vutils.save_image(image,
                                  os.path.join(outputs_dir, 'interpolations', f"interpolation_to-{j}_{i}", f"{k}.png"),
                                  normalize=True)
                for loss in losses:
                    losses_first[loss.name].append(loss(image, neighbors[0].unsqueeze(0).float().to(device)).item())
                    losses_last[loss.name].append(loss(image, neighbors[j].unsqueeze(0).float().to(device)).item())

            save_batch(generator(torch.stack(interpolations[j])),
                       os.path.join(outputs_dir, 'interpolations', f"interpolation_to-{j}_{i}"))
            for loss in losses:
                plt.plot(range(len(losses_first[loss.name])), losses_first[loss.name], label='loss-first', c='blue')
                plt.plot(range(len(losses_last[loss.name])), losses_last[loss.name], label='loss-last', c='red')
                plt.title(f"loss:{loss.name}")
                plt.legend()
                plt.savefig(os.path.join(outputs_dir, 'interpolations', f"interpolation_{i}-to-{j}_{loss.name}.png"))
                plt.clf()


def analyze_perceptual_features_intencities():
    dataset = get_dataset('ffhq', split='test', resize=64)

    with torch.no_grad():
        losses = [
            VGGFeatures(5, pretrained=True).to(device),
            VGGFeatures(5, pretrained=False, reinit=True, norm_first_conv=True).to(device)
        ]
        layer_means = {l.name: {} for l in losses}
        for im in tqdm(dataset.images[:100]):
            im = torch.from_numpy(im).to(device).float().unsqueeze(0)
            for loss in losses:
                z = im
                for i in range(31):
                    z = loss.features[i](z)
                    # if i in [2, 7, 14, 21, 29]:
                    if i in [3, 8, 15, 22, 30]:
                        if i not in layer_means:
                            layer_means[loss.name][i] = torch.zeros_like(z).cpu()
                        layer_means[loss.name][i] += z.cpu()
    for loss_name in layer_means:
        x = np.array([layer_means[loss_name][i].std() for i in layer_means[loss_name]])
        print(x / x.sum())
        for i in layer_means[loss_name]:
            # layer_means[loss.name][i] /= 100
            # print(f"\t{i}: {layer_means[loss_name][i].mean((2,3))[0]}")
            print(
                f"\t{i}: {layer_means[loss_name][i].mean(), layer_means[loss_name][i].min(), layer_means[loss_name][i].max(), layer_means[loss_name][i].std()}")


def invert_perceptual_features(dataset):
    """Optimize noise to match feature maps of vgg and visualize the results"""
    n = 4
    nets = [VGGFeatures(pretrained=True).to(device), VGGFeatures(pretrained=False).to(device),
            VGGFeatures(pretrained=False, norm_first_conv=True).to(device)]
    # nets = [VGGFeatures(pretrained=True).to(device)]
    results = dict()
    for net in nets:
        images = torch.from_numpy(np.stack([dataset[x][1] for x in range(n)])).float().to(device)
        features_org = net.get_activations(images)
        criterion = torch.nn.MSELoss()
        results[net.name] = dict()
        for layer_name in ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']:

            output_images = torch.randn(images.shape).to(device).float() * 0.225
            output_images.requires_grad_(True)
            optimizer = torch.optim.Adam([output_images], lr=0.1)
            pbar = tqdm(range(1000))
            for i in pbar:
                features_opt = net.get_activations(output_images)
                loss = criterion(features_opt[layer_name], features_org[layer_name])

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"Loss: {loss.item()}")

                if (i + 1) % 100 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.75

            # normalize output image
            res = output_images.detach().cpu()
            res -= res.min()
            res /= res.max()
            res = res * 2 - 1

            results[net.name][layer_name] = res
            # results[net.name][layer_name] = output_images.detach().cpu()
        vutils.save_image(torch.cat([images.detach().cpu(), *results[net.name].values()]),
                          os.path.join(outputs_dir, f"{net.name}.png"), normalize=True, nrow=n)


def mix_patches():
    def perm_patch(x):
        x_unfolded = unfold(x[:, :, :32, :32])
        x_unfolded = x_unfolded[:, :, torch.randperm(x_unfolded.shape[-1])]
        refold = fold(x_unfolded) / fold(unfold(torch.ones((1, 3, 32, 32))))
        x_hat = x.clone()
        x_hat[:, :, :32, :32] = refold
        return x_hat

    patch_size = 3
    strides = 1
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=strides, padding=False)
    fold = torch.nn.Fold(kernel_size=patch_size, stride=strides, output_size=(32, 32), padding=False)
    dataset = get_dataset('ffhq', split='test', resize=64)
    x = torch.from_numpy(dataset.images[0]).unsqueeze(0).float()
    x_hat = perm_patch(x)
    vutils.save_image(x, 'x.png', normalize=True)
    vutils.save_image(x_hat, 'x_hat.png', normalize=True)

    loss = MMDApproximate(r=256, pool_size=32, pool_strides=16, normalize_patch='channel_mean')
    print(loss(x, x))
    print(loss(x, x_hat))


if __name__ == '__main__':
    device = torch.device("cuda")

    outputs_dir = 'optimize_model'
    os.makedirs(outputs_dir, exist_ok=True)
    params = default_config
    train_dataset = get_dataset('ffhq', split='test', resize=params.img_dim)


    # mix_patches()
    # analyze_perceptual_features_intencities()

    # test_swd(outputs_dir)
    # invert_perceptual_features(train_dataset)

    # optimize_model(outputs_dir, train_dataset, n=64)
    # generate_interpolations_to_nn(train_dataset, [6, 55, 56, 57, 58], outputs_dir, n_steps=50, n_neighbors=5)
    # sample_latent_neighbors(outputs_dir)

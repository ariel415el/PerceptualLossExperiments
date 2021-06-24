import os
from collections import defaultdict

import cv2
import torch.nn.functional
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils as vutils
from tqdm import tqdm

from GenerativeModels import models
from GenerativeModels.GLO.config import faces_config
from GenerativeModels.models import weights_init
from GenerativeModels.utils.data_utils import get_dataset, get_dataloader
from GenerativeModels.utils.swd import compute_swd, SWDLoss
from losses.experimental_patch_losses import MMD_PPP
from losses.l2 import L2
from losses.lap1_loss import LapLoss
from losses.patch_mmd_loss import MMDApproximate
from losses.patch_mmd_pp import MMD_PP
from losses.vgg_loss.vgg_loss import VGGFeatures, VGGPerceptualLoss
from perceptual_mean_optimization.main import pt2cv


def embedd_data(dataset, encoder, batch_size):
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


def load_models(ckp_dir=None):
    params = faces_config
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
    encoder, generator = load_models('GenerativeModels/Aotuencoders/outputs/ffhq_128/VGG-None_PT')
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


def sample_latent_neighbors(outputs_dir):
    """Find nearest latent neighbors of data samples and create sets of original/reconstructed similar images """
    # Load models
    encoder, generator = load_models('GenerativeModels/Aotuencoders/outputs/ffhq_128/VGG-None_PT')
    embeddings = embedd_data(train_dataset, encoder, params.batch_size)
    for i in [1, 4, 6, 10]:
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"data_neighbors{i}"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"reconstructions{i}"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"latent_neighbors_gauss{i}"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"latent_neighbors_direction{i}"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"mean_latent_neighbors_gauss{i}"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, 'sampling', f"{i}", f"mean_latent_neighbors_direction{i}"), exist_ok=True)

        dists = torch.norm(embeddings - embeddings[i], dim=1)
        neighbor_indices = torch.argsort(dists)[:25]
        neighbors = torch.from_numpy(np.array([train_dataset[x][1] for x in neighbor_indices]))
        save_batch(neighbors, os.path.join(outputs_dir, 'sampling', f"{i}", f"data_neighbors{i}"))

        neighbor_embeddings = embeddings[neighbor_indices]
        reconstructions = generator(neighbor_embeddings)
        save_batch(reconstructions, os.path.join(outputs_dir, 'sampling', f"{i}", f"reconstructions{i}"))

        mean_embedding = embeddings[neighbor_indices].mean(0)
        latent_code = embeddings[i]
        latent_neighbors_gauss = [latent_code]
        latent_neighbors_direction = [latent_code]
        mean_latent_neighbors_gauss = [mean_embedding]
        mean_latent_neighbors_direction = [mean_embedding]

        coeff = 20
        for j in range(1, 25):
            latent_neighbors_gauss.append(latent_code + coeff * torch.randn(latent_code.shape).to(device))
            latent_neighbors_direction.append((latent_code + neighbor_embeddings[j]) / 2)
            mean_latent_neighbors_gauss.append(mean_embedding + coeff * torch.randn(mean_embedding.shape).to(device))
            mean_latent_neighbors_direction.append((mean_embedding + neighbor_embeddings[j]) / 2)
        save_batch(generator(torch.stack(latent_neighbors_gauss)),
                   os.path.join(outputs_dir, 'sampling', f"{i}", f"latent_neighbors_gauss{i}"))
        save_batch(generator(torch.stack(latent_neighbors_direction)),
                   os.path.join(outputs_dir, 'sampling', f"{i}", f"latent_neighbors_direction{i}"))
        save_batch(generator(torch.stack(mean_latent_neighbors_gauss)),
                   os.path.join(outputs_dir, 'sampling', f"{i}", f"mean_latent_neighbors_gauss{i}"))
        save_batch(generator(torch.stack(mean_latent_neighbors_direction)),
                   os.path.join(outputs_dir, 'sampling', f"{i}", f"mean_latent_neighbors_direction{i}"))


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


def optimize_model(outputs_dir, train_dataset, n=25):
    # encoder, generator = load_models('GenerativeModels/Aotuencoders/outputs/ffhq_128/L2/')
    encoder, generator = load_models()
    encoder.train()
    generator.train()

    # criterion = MMD_PP(device, patch_size=7, pool_size=32, pool_strides=16, r=128, normalize_patch='channel_mean',
    #                    weights=[0.001, 0.05, 1.0]).to(device)
    # criterion = L2().to(device)
    criterion = VGGFeatures(pretrained=True).to(device)
    # criterion = MMDApproximate(patch_size=7, pool_size=32, pool_strides=16, r=128, normalize_patch='channel_mean').to(device)
    # criterion = VGGFeatures(pretrained=True).to(device)
    # criterion = VGGFeatures(pretrained=False, norm_first_conv=True, reinit=True, weights=[1,1,1,1,1,1]).to(device)

    images = torch.from_numpy(np.array([train_dataset[i][1] for i in range(n)])).to(device).float()
    # images = torch.from_numpy(np.array([train_dataset[i][1] for i in [1, 56475, 11821, 20768,  6, 52650,  7563]])).to(device).float()
    vutils.save_image(images, os.path.join(outputs_dir, f"orig.png"), normalize=True, nrow=int(np.sqrt(n)))

    # Define optimizers
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizerE = torch.optim.Adam(encoder.parameters(), lr=0.001)
    pbar = tqdm(range(10001))
    for i in pbar:
        if i % 500 == 0:
            recons = generator(encoder(images))
            vutils.save_image(recons, os.path.join(outputs_dir, f"recons{i}.png"), normalize=True, nrow=int(np.sqrt(n)))

        latent_codes = encoder(images)
        fake_images = generator(latent_codes)

        rec_loss = criterion(fake_images, images).mean()
        rec_loss.backward()

        optimizerE.step()
        encoder.zero_grad()
        optimizerG.step()
        generator.zero_grad()
        pbar.set_description(f"loss: {rec_loss}")
        if (i + 1) % 1000 == 0:
            for g, e in zip(optimizerG.param_groups, optimizerE.param_groups):
                e['lr'] *= 0.9
                g['lr'] *= 0.9


def test_swd(output_dir):
    def cv2pt(img):
        img = img.astype(np.float64) / 255.
        img = img * 2 - 1
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img
    os.makedirs(output_dir, exist_ok=True)
    x = '/home/ariel/university/PerceptualLoss/sty  ',
    paths = [
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/bair.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/modern.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/lenna.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/green_eye.jpg',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/FFHQ-0.png'
    ]
    for path in paths:
        img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, (128, 128))

        def noisy(x, i):
            return np.clip(x + np.random.normal(0, 10 * (i+1), x.shape), 0, 255), f"+ N(0,{10 * (i+1)})"

        def blur(x, i):
            return cv2.blur(x, ksize=(2 + i, 2 + i)), f"Blur(k={2 +  i})"

        def compress(x, i):
            return cv2.imdecode(cv2.imencode('.jpg', x, [int(cv2.IMWRITE_JPEG_QUALITY), 90 - i * 17])[1], 1), f"JPG({90 - i * 17})"

        degrade_funcs = [noisy, blur, compress]

        n = 6
        font_size = 12
        fig, axs = plt.subplots(len(degrade_funcs) + 1, n, figsize=(20, 10))#, gridspec_kw={'wspace': 0, 'hspace': 0.25})

        axs[0, 0].imshow(img_np)
        axs[0, 0].set_title(f"orignal", size=font_size)
        axs[0, 0].axis('off')

        img_pt = cv2pt(img_np)[None, :]
        for i in range(n):
            for j, deg_func in enumerate(degrade_funcs):
                degraded_img, name = deg_func(img_np, i)
                swd_score = compute_swd(img_pt, cv2pt(degraded_img)[None, :], device="cuda")
                axs[j + 1, i].imshow(degraded_img.astype(int))
                axs[j + 1, i].set_title(f"{name}\nSWD:{swd_score:.1f}", size=font_size)
                axs[j + 1, i].axis('off')
                axs[j + 1, i].set_aspect('equal')

            if i > 0:
                fig.delaxes(axs[0, i])

        plt.tight_layout()
        img_name = os.path.splitext(os.path.basename(path))
        plt.savefig(f"{output_dir}/SWD-{img_name}.png")


def compare_trained_generators_losses():
    root = '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/GenerativeModels/Aotuencoders/outputs/test'
    from GenerativeModels.GLO.config import faces_config as params
    n = 9
    losses = [
        L2(),
        LapLoss(),
        LapLoss(no_last_layer=True),
        MMD_PP(r=256),
        MMD_PPP(r=256),
        VGGPerceptualLoss(pretrained=True),
        VGGPerceptualLoss(pretrained=False, norm_first_conv=True, reinit=True),
        SWDLoss()

    ]

    all_data = []
    model_names = []
    test_dataset = get_dataset('ffhq', split='test', resize=params.img_dim, val_percent=0.005)

    test_images = torch.stack([torch.from_numpy(test_dataset[i * 2][1]) for i in range(n)]).to(device).float()
    test_images_cropped = test_images[:, :, 0:64, 0:64]
    vutils.save_image(test_images, os.path.join(root, f"test-imgs.png"), normalize=True, nrow=np.int(np.sqrt(n)))

    for model_name in os.listdir(root):
        if model_name.endswith('.png') or model_name.endswith('.txt') or model_name == 'recons':
            continue
        # LOAD MODELS
        train_dir = os.path.join(root, model_name)

        generator = models.DCGANGenerator(params.z_dim, params.channels, params.img_dim).to(device)
        generator.load_state_dict(torch.load(f"{train_dir}/generator.pth", map_location=device))

        encoder = models.DCGANEncoder(params.img_dim, params.channels, params.z_dim).to(device)
        encoder.load_state_dict(torch.load(f"{train_dir}/encoder.pth", map_location=device))

        encoder.eval()
        generator.eval()

        # RECONSTRUCT DATA
        test_reconstructions = generator(encoder(test_images))
        test_reconstructions_cropped = test_reconstructions[:, :, 0:64, 0:64]
        vutils.save_image(test_reconstructions, os.path.join(root, f"{model_name}_test-reconstructions.png"), normalize=True, nrow=np.int(np.sqrt(n)))

        # COMPUTE LOSSES
        data = []
        for loss in losses:
            loss_values = loss.to(device)(test_images_cropped, test_reconstructions_cropped)
            # loss_strings.append(f"{loss_values.mean():.4f}/{loss_values.std():.4f})")
            data.append(loss_values.item())

        all_data.append(data)
        model_names.append(model_name)

        # PLOT DATA SAMPLE LOSSES
        for k in range(len(test_reconstructions)):
            os.makedirs(os.path.join(root, 'recons', str(k)), exist_ok=True)
            plt.imshow(pt2cv(test_images_cropped[k].detach().cpu().numpy()))
            plt.tight_layout()
            plt.savefig(os.path.join(root, 'recons', str(k), f"test-img.png"))
            plt.clf()

            img = pt2cv(test_reconstructions_cropped[k].detach().cpu().numpy())
            plt.imshow(img)
            title = ''
            for loss in losses:
                # title += f"{loss.name.ljust(25)}: {loss.to(device)(test_images_cropped[k].unsqueeze(0), test_reconstructions_cropped[k].unsqueeze(0)).item():.2f}\n"
                title += f"{loss.name:30}: {loss.to(device)(test_images_cropped[k].unsqueeze(0), test_reconstructions_cropped[k].unsqueeze(0)).item():.2f}\n"
            plt.title(title, size=10)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(root, 'recons', str(k), f"{model_name}_recon.png"))
            plt.clf()

    # PLOT SIMILARITY MATRIX
    fig, ax = plt.subplots(figsize=(15,15))
    all_data = np.array(all_data)
    all_data = all_data / all_data.max(axis=0)
    mat = ax.matshow(all_data, cmap='YlGnBu')
    for (i, j), z in np.ndenumerate(all_data):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    ax.set_xticks(np.arange(len(losses)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([l.name for l in losses], rotation=20)
    ax.set_yticklabels(model_names, rotation=70)
    fig.colorbar(mat)
    fig.savefig(os.path.join(root, f"losses.png"))
    plt.clf()



if __name__ == '__main__':
    device = torch.device("cuda")

    compare_trained_generators_losses()
    exit()

    outputs_dir = 'Experiments_outputs'
    # mix_patches()
    # analyze_perceptual_features_intencities()

    test_swd(outputs_dir)

    os.makedirs(outputs_dir, exist_ok=True)

    params = faces_config
    train_dataset = get_dataset('ffhq', split='test', resize=params.img_dim)
    # invert_perceptual_features(train_dataset)

    # optimize_model(outputs_dir, train_dataset, n=64)
    # generate_interpolations_to_nn(train_dataset, [6, 55, 56, 57, 58], outputs_dir, n_steps=50, n_neighbors=5)
    # sample_latent_neighbors(outputs_dir)

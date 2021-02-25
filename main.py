import utils
from fid_scroe.fid_score import calculate_frechet_distance
from fid_scroe.inception import InceptionV3
from glo import GLO
from config import faces_config
import torch
from utils import DiskDataset
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main():
    # train_samples_path = "../data/fashionMnist/train_fashion.npy"
    # train_samples_path = "../data/Mnist/train_Mnist.npy"
    # dataset = MnistDataset(train_samples_path)
    # conf = mnist_configs
    # train_dir = "training_dir/FMNIST-vgg_loss"
    # train_dir = "training_dir/MNIST-vgg_loss"

    train_samples_path = "../data/img_align_celeba"
    dataset = DiskDataset(train_samples_path)
    conf = faces_config
    train_dir = "training_dir/celebA-vgg_loss_2"

    glo = GLO(conf, dataset_size=len(dataset), device=device)
    glo.load_weights(train_dir, device)

    dataloader = utils.get_dataloader(dataset, 128, device)
    # glo.train(dataloader, conf, outptus_dir=train_dir)

    test_trained_model(glo, dataloader, device)


def test_trained_model(model, dataloader, device):
    inception_model = InceptionV3([3]).to(device).eval()

    real_activations = []
    gen_activations = []
    rec_activations = []

    print("Computing Inception activations")
    z_mu, z_cov = utils.get_mu_sigma(model.netZ.emb.weight.clone().cpu())
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for i, (indices, images) in pbar:
            pbar.set_description(f"Images done: {i * dataloader.batch_size}")
            # get activations for real images
            act = inception_model(images.to(device).float())[0].squeeze().cpu().numpy()
            real_activations.append(act)

            # get activations for generated images
            Zs = utils.sample_mv(dataloader.batch_size, z_mu, z_cov).to(device)
            images = model.netG(Zs)
            act = inception_model(images)[0].squeeze().cpu().numpy()
            gen_activations.append(act)

            # get activations for reconstructed images
            images = model.netG(model.netZ(indices.long().to(device)))
            act = inception_model(images)[0].squeeze().cpu().numpy()
            rec_activations.append(act)
            if i > 500:
                break

    real_activations = np.concatenate(real_activations, axis=0)
    gen_activations = np.concatenate(gen_activations, axis=0)
    rec_activations = np.concatenate(rec_activations, axis=0)

    print(f"Computing activations mean and covariances on {real_activations.shape[0]} samples")

    real_mu, real_cov = np.mean(real_activations, axis=0), np.cov(act, rowvar=False)
    gen_mu, gen_cov = np.mean(gen_activations, axis=0), np.cov(act, rowvar=False)
    rec_mu, rec_cov = np.mean(rec_activations, axis=0), np.cov(act, rowvar=False)

    print("Computing FID scores")
    gen_fid = calculate_frechet_distance(real_mu, real_cov, gen_mu, gen_cov)
    rec_fid = calculate_frechet_distance(real_mu, real_cov, rec_mu, rec_cov)

    print(f"Generated images FID: {gen_fid}")
    print(f"Reconstructed train images FID: {rec_fid}")

if __name__ == '__main__':
    main()
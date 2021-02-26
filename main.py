import utils
from fid_scroe.fid_score import calculate_frechet_distance
from fid_scroe.inception import InceptionV3
from glo import GLO
from config import faces_config, mnist_configs
import torch
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main():
    train_dataset = utils.MnistDataset("../data/Mnist/train_Mnist.npy")
    test_dataset = utils.MnistDataset("../data/Mnist/test_Mnist.npy")
    conf = mnist_configs
    train_dir = "training_dir/MNIST-vgg-loss"

    # train_dir = "training_dir/FMNIST-vgg_loss_new"

    # train_samples_path = "../data/img_align_celeba"
    # dataset = utils.DiskDataset(train_samples_path)
    # conf = faces_config
    # train_dir = "training_dir/celebA-vgg_loss_2"

    glo = GLO(conf, dataset_size=len(train_dataset), device=device)
    # glo.load_weights(train_dir, device)

    train_dataloader = utils.get_dataloader(train_dataset, conf.batch_size, device)
    glo.train(train_dataloader, conf, outptus_dir=train_dir)

    test_dataloader = utils.get_dataloader(test_dataset, conf.batch_size, device)
    test_trained_model(glo, test_dataloader, train_dataloader, device)


def test_trained_model(model, test_dataloader, train_dataloader, device):
    inception_model = InceptionV3([3]).to(device).eval()

    test_activations = []
    train_activations = []
    gen_activations = []
    rec_activations = []

    print("Computing Inception activations")
    z_mu, z_cov = utils.get_mu_sigma(model.netZ.emb.weight.clone().cpu())
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i, (indices, images) in pbar:
            pbar.set_description(f"Images done: {i * test_dataloader.batch_size}")

            # get activations for real images from test set
            if images.size(1) == 1:
                images = images.expand((images.size(0), 3, images.size(2), images.size(3)))
            act = inception_model(images.to(device).float())[0].squeeze().cpu().numpy().astype(np.float64)
            test_activations.append(act)

            # get activations for real images from test set
            _, images = train_dataloader.dataset[indices]
            images = torch.from_numpy(images)
            if images.size(1) == 1:
                images = images.expand((images.size(0), 3, images.size(2), images.size(3)))
            act = inception_model(images.to(device).float())[0].squeeze().cpu().numpy().astype(np.float64)
            train_activations.append(act)

            # get activations for generated images
            Zs = utils.sample_mv(test_dataloader.batch_size, z_mu, z_cov).to(device)
            images = model.netG(Zs)
            if images.size(1) == 1:
                images = images.expand((images.size(0), 3, images.size(2), images.size(3)))
            act = inception_model(images)[0].squeeze().cpu().numpy()
            gen_activations.append(act)

            # get activations for reconstructed images
            images = model.netG(model.netZ(indices.long().to(device)))
            if images.size(1) == 1:
                images = images.expand((images.size(0), 3, images.size(2), images.size(3)))
            act = inception_model(images)[0].squeeze().cpu().numpy()
            rec_activations.append(act)


    train_activations = np.concatenate(train_activations, axis=0)
    test_activations = np.concatenate(test_activations, axis=0)
    gen_activations = np.concatenate(gen_activations, axis=0)
    rec_activations = np.concatenate(rec_activations, axis=0)

    print(f"Computing activations mean and covariances on {test_activations.shape[0]} samples")

    test_mu, test_cov = np.mean(test_activations, axis=0), np.cov(test_activations, rowvar=False)
    train_mu, train_cov = np.mean(train_activations, axis=0), np.cov(train_activations, rowvar=False)
    gen_mu, gen_cov = np.mean(gen_activations, axis=0), np.cov(gen_activations, rowvar=False)
    rec_mu, rec_cov = np.mean(rec_activations, axis=0), np.cov(rec_activations, rowvar=False)

    print("Computing FID scores")
    opt_fid = calculate_frechet_distance(test_mu, test_cov, train_mu, train_cov)
    gen_fid = calculate_frechet_distance(test_mu, test_cov, gen_mu, gen_cov)
    rec_fid = calculate_frechet_distance(test_mu, test_cov, rec_mu, rec_cov)

    print(f"Optimal scores (Train vs test) FID: {opt_fid}")
    print(f"Generated images FID: {gen_fid}")
    print(f"Reconstructed train images FID: {rec_fid}")

if __name__ == '__main__':
    main()
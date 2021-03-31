import utils
from IMLE import IMLE
from fid_scroe.fid_score import calculate_frechet_distance
from fid_scroe.inception import InceptionV3
from GLO import GLO
from config import faces_config, mnist_configs
import torch
from tqdm import tqdm
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataloaders(dataset_name):
    if dataset_name == "mnist":
        train_dataset = utils.MnistDataset("../../../data/Mnist/train_Mnist.npy")
        test_dataset = utils.MnistDataset("../../../data/Mnist/test_Mnist.npy")
        conf = mnist_configs
        train_dataloader = utils.get_dataloader(train_dataset, conf.batch_size, device)
        test_dataloader = utils.get_dataloader(test_dataset, conf.batch_size, device)

    elif dataset_name in ['celeba', 'ffhq']:
        if dataset_name == 'celeba':
            train_samples_path = "../../../data/img_align_celeba"
            dataset_type = utils.DiskDataset
        else:
            train_samples_path = "../../../data/FFHQ/thumbnails128x128"
            dataset_type = utils.MemoryDataset

        img_paths = [os.path.join(train_samples_path, x) for x in os.listdir(train_samples_path)]
        np.random.shuffle(img_paths)
        val_size = int(0.15 * len(img_paths))
        train_dataset = dataset_type(img_paths[val_size:])
        test_dataset = dataset_type(img_paths[:val_size])

        conf = faces_config
        train_dataloader = utils.get_dataloader(train_dataset, conf.batch_size, device)
        test_dataloader = utils.get_dataloader(test_dataset, conf.batch_size, device)
    else:
        raise ValueError("No such dataset")

    return train_dataloader, test_dataloader, conf


def main():
    dataset_name = 'ffhq'
    train_dataloader, test_dataloader, conf = get_dataloaders(dataset_name)
    train_dir = f"test-training_dir/{dataset_name}-l2_3"
    # train_dir = f"training_dir/{dataset_name}-mmd-tests_2"

    glo = GLO(conf, dataset_size=len(train_dataloader.dataset), device=device)
    glo.train(train_dataloader, conf, outptus_dir=train_dir, start_epoch=0)
    # glo.load_weights(train_dir, device)

    imle = IMLE(conf.e_dim, conf.z_dim)
    # imle.load_weights(train_dir, device)
    Zs = glo.netZ.emb.weight.data.cpu().numpy()
    imle.train(Zs, train_dir=train_dir, epochs=50)

    z = imle.netT(torch.randn(64, imle.e_dim).cuda())
    ims = glo.netG(z)
    vutils.save_image(ims, f"{train_dir}/imgs/IMLE-sampled.png", normalize=False)

    test_trained_model(train_dir, glo, imle, test_dataloader, train_dataloader, device)


def test_trained_model(train_dir, glo_model, imle_model, test_dataloader, train_dataloader, device):
    inception_model = InceptionV3([3]).to(device).eval()

    test_activations = []
    train_activations = []
    glo_gen_activations = []
    imle_gen_activations = []
    rec_activations = []

    print("Computing Inception activations")
    z_mu, z_cov = utils.get_mu_sigma(glo_model.netZ.emb.weight.clone().cpu())
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i, (indices, images) in pbar:
            pbar.set_description(f"Images done: {i * test_dataloader.batch_size}")

            # get activations for real images from test set
            act = inception_model.get_activations(images, device).astype(np.float64)
            test_activations.append(act)

            # get activations for real images from train set
            _, images = train_dataloader.dataset[indices]
            images = torch.from_numpy(images)
            act = inception_model.get_activations(images, device).astype(np.float64)
            train_activations.append(act)

            # get activations for generated images
            Zs = utils.sample_mv(test_dataloader.batch_size, z_mu, z_cov).to(device)
            images = glo_model.netG(Zs)
            act = inception_model.get_activations(images, device).astype(np.float64)
            glo_gen_activations.append(act)

            # get activations for reconstructed images
            images = glo_model.netG(glo_model.netZ(indices.long().to(device)))
            act = inception_model.get_activations(images, device).astype(np.float64)
            rec_activations.append(act)

            # Get activations for IMLE samples
            images = glo_model.netG(imle_model.netT(torch.randn(test_dataloader.batch_size, imle_model.e_dim).to(device)))
            act = inception_model.get_activations(images, device).astype(np.float64)
            imle_gen_activations.append(act)

    train_activations = np.concatenate(train_activations, axis=0)
    test_activations = np.concatenate(test_activations, axis=0)
    glo_gen_activations = np.concatenate(glo_gen_activations, axis=0)
    rec_activations = np.concatenate(rec_activations, axis=0)
    imle_gen_activations = np.concatenate(imle_gen_activations, axis=0)

    print(f"Computing activations mean and covariances on {test_activations.shape[0]} samples")

    test_mu, test_cov = np.mean(test_activations, axis=0), np.cov(test_activations, rowvar=False)
    train_mu, train_cov = np.mean(train_activations, axis=0), np.cov(train_activations, rowvar=False)
    glo_gen_mu, glo_gen_cov = np.mean(glo_gen_activations, axis=0), np.cov(glo_gen_activations, rowvar=False)
    rec_mu, rec_cov = np.mean(rec_activations, axis=0), np.cov(rec_activations, rowvar=False)
    imle_gen_mu, imle_gen_cov = np.mean(imle_gen_activations, axis=0), np.cov(imle_gen_activations, rowvar=False)

    print("Computing FID scores")
    opt_fid = calculate_frechet_distance(test_mu, test_cov, train_mu, train_cov)
    glo_gen_fid = calculate_frechet_distance(test_mu, test_cov, glo_gen_mu, glo_gen_cov)
    rec_fid = calculate_frechet_distance(test_mu, test_cov, rec_mu, rec_cov)
    imle_gen_fid = calculate_frechet_distance(test_mu, test_cov, imle_gen_mu, imle_gen_cov)

    f = open(os.path.join(train_dir, "FID-scores.txt"), 'w')
    f.write(f"Optimal scores (Train vs test) FID: {opt_fid:.2f}\n")
    f.write(f"GLO-Generated images FID: {glo_gen_fid:.2f}\n")
    f.write(f"Reconstructed train images FID: {rec_fid:.2f}\n")
    f.write(f"IMLE-Generated images FID: {imle_gen_fid:.2f}\n")
    f.close()


if __name__ == '__main__':
    main()
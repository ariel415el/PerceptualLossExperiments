import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from config import mnist_configs, faces_config

def sample_gaussian(x, m, mu=None, cov=None):
    if mu is None:
        mu, cov = get_mu_sigma(x)
    return sample_mv(m, mu, cov)


def get_mu_sigma(x):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov = np.cov(x, rowvar=False)
    return mu, cov


def sample_mv(m, mu, cov):
    z = np.random.multivariate_normal(mu, cov, size=m)
    z_t = torch.from_numpy(z).float()
    radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
    z_t = z_t / radius
    return z_t


def get_dataloader(dataset, batch_size, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True}
    if device == "cuda:0":
        kwargs.update({'num_workers': 12,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


class DiskDataset(Dataset):
    def __init__(self, paths, crop=False):
        self.image_paths = paths
        self.crop = crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.crop:
            img = img[109 - 90:109 + 80, 89 - 85:89 + 85] # CelebA
        img = cv2.resize(img, (64, 64)) / 255.0
        img = img.transpose((2, 0, 1))

        img = 2 * img - 1  # transform to -1 1

        return idx, img


class MemoryDataset(Dataset):
    def __init__(self, paths, crop=False):
        self.images = []
        self.crop = crop
        print("Loading data into memory")
        for path in paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.crop:
                img = img[109 - 90:109 + 80, 89 - 85:89 + 85] # CelebA
            img = cv2.resize(img, (64, 64)) / 255.0
            img = 2 * img - 1  # transform to -1 1

            img = img.transpose((2, 0, 1))

            self.images.append(img)

        self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return idx, self.images[idx]


class MnistDataset(Dataset):
    def __init__(self, mnist_file):
        self.imgs = np.load(mnist_file)
        self.imgs = self.imgs.transpose((0, 3, 1, 2)) / 255.0
        # self.imgs = np.pad(self.imgs,((0,0),(0,0),(2,2),(2,2)), constant)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return idx, self.imgs[idx]


def download_ffhq_thumbnails(data_dir):
    print("Downloadint FFHQ-thumbnails from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('greatgamedota/ffhq-face-data-set', path=data_dir, unzip=True, quiet=False)
    print("Done.")


def get_dataloaders(dataset_name, device):
    if dataset_name == "mnist":
        train_dataset = MnistDataset("../../../data/Mnist/train_Mnist.npy")
        test_dataset = MnistDataset("../../../data/Mnist/test_Mnist.npy")
        conf = mnist_configs
        train_dataloader = get_dataloader(train_dataset, conf.batch_size, device)
        test_dataloader = get_dataloader(test_dataset, conf.batch_size, device)

    elif dataset_name in ['celeba', 'ffhq']:
        if dataset_name == 'celeba':
            train_samples_path = "../../../data/img_align_celeba"
            dataset_type = DiskDataset
        else:
            train_samples_path = "../../../data/FFHQ/thumbnails128x128"
            if not os.path.exists(train_samples_path):
                download_ffhq_thumbnails(os.path.dirname(train_samples_path))
            dataset_type = MemoryDataset

        img_paths = [os.path.join(train_samples_path, x) for x in os.listdir(train_samples_path)]
        np.random.shuffle(img_paths)
        val_size = int(0.15 * len(img_paths))
        train_dataset = dataset_type(img_paths[val_size:])
        test_dataset = dataset_type(img_paths[:val_size])

        conf = faces_config
        train_dataloader = get_dataloader(train_dataset, conf.batch_size, device)
        test_dataloader = get_dataloader(test_dataset, conf.batch_size, device)
    else:
        raise ValueError("No such dataset")

    return train_dataloader, test_dataloader, conf
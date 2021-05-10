import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_dataloader(dataset, batch_size, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True}
    if device == "cuda:0":
        kwargs.update({'num_workers': 8,
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

class DspriteDatset(Dataset):
    def __init__(self, dsprite_imgs_file):
        self.imgs = np.load(dsprite_imgs_file)[:,None]

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


def get_dataset(dataset_name, split='train'):
    if dataset_name == "dsprites":
        dataset = DspriteDatset(f"../../../../data/dsprites/imgs.npy")

    elif dataset_name == "mnist":
        dataset = MnistDataset(f"../../../../data/Mnist/{split}_Mnist.npy")

    elif dataset_name in ['celeba', 'ffhq']:
        if dataset_name == 'celeba':
            train_samples_path = "../../../../data/img_align_celeba"
            dataset_type = DiskDataset
        else:
            train_samples_path = "../../../../data/FFHQ/thumbnails128x128"
            if not os.path.exists(train_samples_path):
                download_ffhq_thumbnails(os.path.dirname(train_samples_path))
            dataset_type = MemoryDataset

        img_paths = [os.path.join(train_samples_path, x) for x in os.listdir(train_samples_path)]
        # np.random.shuffle(img_paths)  # Avoid mixing the test and train for test consistency
        val_size = int(0.15 * len(img_paths))
        if split == "train":
            dataset = dataset_type(img_paths[val_size:])
        else:
            dataset = dataset_type(img_paths[:val_size])

    else:
        raise ValueError("No such dataset")

    return dataset
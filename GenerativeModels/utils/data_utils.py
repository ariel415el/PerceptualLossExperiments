import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_dataloader(dataset, batch_size, device, drop_last=True, shuffle=True):
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last': drop_last}
    if device == "cuda:0":
        kwargs.update({'num_workers': 8,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


class DiskDataset(Dataset):
    def __init__(self, paths, resize=64, crop=False):
        self.image_paths = np.array(paths)
        self.crop = crop
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.crop == 'lfw':
            img = img[64:-64, 64:-64]
        elif self.crop == 'celebA':
            img = img[109 - 90:109 + 80, 89 - 85:89 + 85]  # CelebA
        img = cv2.resize(img, (self.resize, self.resize)) / 255.0
        img = img.transpose((2, 0, 1))

        img = 2 * img - 1  # transform to -1 1

        return idx, img


class MemoryDataset(Dataset):
    def __init__(self, paths, resize=64, crop=False):
        self.images = np.zeros((len(paths), 3, resize, resize))
        self.crop = crop
        print("Loading data into memory... ", end='')
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.crop:
                img = img[109 - 90:109 + 80, 89 - 85:89 + 85]  # CelebA
            img = cv2.resize(img, (resize, resize)) / 255.0
            img = 2 * img - 1  # transform to -1 1

            img = img.transpose((2, 0, 1))

            # self.images.append(img)
            self.images[i] = img

        print("done")
        # self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return idx, self.images[idx]


def download_ffhq_thumbnails(data_dir):
    print("Downloadint FFHQ-thumbnails from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('greatgamedota/ffhq-face-data-set', path=data_dir, unzip=True, quiet=False)
    print("Done.")


def get_dataset(dataset_name, resize, split='train', default_data_root='/home/ariel/university/data', val_percent=0.15, memory_dataset=True):
    kwargs = dict()

    if dataset_name == 'celeba':
        train_samples_path = os.path.join(default_data_root, "img_align_celeba")
        dataset_type = DiskDataset
    elif dataset_name == 'ffhq':
        train_samples_path = os.path.join(default_data_root, "FFHQ/thumbnails128x128")
        if not os.path.exists(train_samples_path):
            download_ffhq_thumbnails(os.path.dirname(train_samples_path))
        kwargs['resize'] = resize
        if memory_dataset:
            dataset_type = MemoryDataset
        else:
            dataset_type = DiskDataset
    else:
        raise ValueError("Dataset not supported")

    img_paths = [os.path.join(train_samples_path, x) for x in os.listdir(train_samples_path)]

    val_size = int(val_percent * len(img_paths))
    if split == "train":
        dataset = dataset_type(img_paths[val_size:], **kwargs)
    else:
        dataset = dataset_type(img_paths[:val_size], **kwargs)

    return dataset


def read_lfw_data(root_dir, img_size):
    paths = []
    labesl = []
    # celeb_names = os.listdir(os.path.join(root_dir, 'lfw-deepfunneled'))
    celeb_names = ['Ariel_Sharon', 'Serena_Williams', 'Andre_Agassi', 'Luiz_Inacio_Lula_da_Silva', 'Jennifer_Lopez',
                   'Nicole_Kidman', 'Halle_Berry', 'Laura_Bush', 'Roh_Moo-hyun', 'Kofi_Annan']
    for celeb_name in celeb_names:
        for fn in os.listdir(os.path.join(root_dir, 'lfw-deepfunneled', celeb_name)):
            paths.append(os.path.join(root_dir, 'lfw-deepfunneled', celeb_name, fn))
            labesl.append(celeb_name)
    random.shuffle(paths)
    return DiskDataset(paths, resize=img_size, crop='lfw'), labesl
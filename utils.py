import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def sample_gaussian(x, m):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov2 = np.cov(x, rowvar=False)
    z = np.random.multivariate_normal(mu, cov2, size=m)
    z_t = torch.from_numpy(z).float()
    radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
    z_t = z_t / radius
    return z_t


def get_dataloader(dataset, batch_size, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    if device == "cuda:0":
        kwargs.update({'num_workers': 12,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


class DiskDataset(Dataset):
    def __init__(self, imgs_dir):
        self.image_paths = [os.path.join(imgs_dir, x) for x in os.listdir(imgs_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64)) / 255.0
        img = img.transpose((2, 0, 1))

        return idx, img


class MnistDataset(Dataset):
    def __init__(self, mnist_file):
        self.imgs = np.load(mnist_file)
        self.imgs = self.imgs.transpose((0, 3, 1, 2)) / 255.0
        # self.imgs = np.pad(self.imgs,((0,0),(0,0),(2,2),(2,2)), constant)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return idx, self.imgs[idx]
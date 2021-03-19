import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from losses.ScnnLoss_ariel import SCNNNetwork
from losses.lap1_loss import LapLoss
# from losses.mmd_exact_loss import MMDExact
from losses.mmd_loss import MMDApproximate
from losses.patch_loss import PatchRBFLoss, PatchRBFLaplacianLoss
from losses.vgg_loss.vgg_loss import VGGFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ListOfLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super(ListOfLosses, self).__init__()
        self.losses = torch.nn.ModuleList(losses)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones((len(losses),))

    def forward(self, x, y):
        return sum([self.losses[i](x,y) * self.weights[i] for i in range(len(self.losses))])

def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1

    img = np.clip(img, -1 + 1e-9, 1 - 1e-9)
    img = np.arctanh(img)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def pt2cv(img):
    img = img
    img = (img + 1) / 2
    img *= 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_images(dir_path):
    images = []
    for fn in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, fn))
        img = cv2pt(img)
        images.append(img)

    return torch.stack(images)


def optimize_for_mean(images, criteria, output_dir, batch_size=1, num_steps=400, lr=0.003):
    """
    :param images: tensor of shape (N, C, H, W)
    """
    N = images.shape[0]
    os.makedirs(output_dir, exist_ok=True)
    images = images.to(device)
    criteria = criteria.to(device)

    optimized_variable = images.mean(0)[None, :]
    optimized_variable.requires_grad_(True)

    losses = []

    for i in tqdm(range(num_steps)):
        optim = torch.optim.Adam([optimized_variable], lr=lr)
        optim.zero_grad()

        for b in range(N // batch_size):
            batch_images = images[b * batch_size: (b+1) * batch_size]
            batch_input = torch.tanh(optimized_variable.repeat(batch_size, 1, 1, 1))
            loss = criteria(batch_input, batch_images)
            loss = loss.mean()
            loss.backward()
            optim.step()
        losses.append(loss.item())

        if i % 50 == 0:
            lr *= 0.5

            plt.plot(np.arange(len(losses)), losses)
            plt.savefig(f"{output_dir}/train_loss.png")
            plt.clf()

            optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
            optimized_image = pt2cv(optimized_image)
            cv2.imwrite(f"{output_dir}/output-{i}.png", optimized_image)

    optimized_image = torch.tanh(optimized_variable).detach().cpu().numpy()[0]
    optimized_image = pt2cv(optimized_image)

    return optimized_image


if __name__ == '__main__':
    images_dir = '/home/ariel/universirty/PerceptualLoss/DansPaper/examples/stylegan1'
    output_dir = "VGG-Random"
    images = load_images(images_dir)

    creiterion = ListOfLosses([
            # LapLoss(n_channels=3, max_level=6),
            VGGFeatures(5),
            # MMDApproximate(batch_reduction='none', normalize_patch='channel_mean', pool_size=32, pool_strides=16),
            # SCNNNetwork(),
            # PatchRBFLaplacianLoss(patch_size=3, batch_reduction='none', normalize_patch='none', ignore_patch_norm=False, device=device)
    ])


    img = optimize_for_mean(images, creiterion, output_dir, num_steps=400, lr=0.001, batch_size=10)


import os

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

from Experiments.all import load_models
from perceptual_mean_optimization.main import cv2pt, pt2cv

device = torch.device("cuda")


def get_edges(img):
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    edges_x = cv2.filter2D(img, cv2.CV_8U, kernelx)
    edges_y = cv2.filter2D(img, cv2.CV_8U, kernely)

    return edges_x, edges_y

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def download_ffhq_thumbnails(data_dir):
    print("Downloadint FFHQ-thumbnails from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('greatgamedota/ffhq-face-data-set', path=data_dir, unzip=True, quiet=False)
    print("Done.")


def show_mean_gradient_map(models_dir, images_dir):
    n = 25

    all_x_edges = []
    all_y_edges = []
    all_images = []
    for i, img_name in enumerate(os.listdir(images_dir)[:n]):
        img = cv2.imread(os.path.join(images_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges_x, edges_y = get_edges(img)

        all_images.append(img)
        all_x_edges.append(edges_x)
        all_y_edges.append(edges_y)

    mean_img = np.mean(all_images, axis=0).astype(img.dtype)
    mean_edges_x, mean_edges_y = get_edges(mean_img)

    fig, axs = plt.subplots(3, 2)  # , gridspec_kw={'wspace':0, 'hspace':0.25})

    # axs[0, i].imshow(img, cmap='gray')
    axs[0, 0].imshow(mean_img, cmap='gray')
    axs[1, 0].imshow(mean_edges_x, cmap='gray')
    axs[2, 0].imshow(mean_edges_y, cmap='gray')

    axs[1, 1].imshow(np.mean(all_x_edges, axis=0), cmap='gray')
    axs[2, 1].imshow(np.mean(all_y_edges, axis=0), cmap='gray')

    plt.show()


def print_mean_diff(x_diffs_per_model, y_diffs_per_model):
    for (SIGN, diffs_per_model) in [("X", x_diffs_per_model), ("Y", y_diffs_per_model)]:
        for m in diffs_per_model:
            diffs_per_model[m] = np.mean(diffs_per_model[m])
        diffs_sum = sum([diffs_per_model[m] for m in diffs_per_model])
        print(f"{SIGN}-diff: " + ','.join([f"{m}: {diffs_per_model[m] / diffs_sum:.4f}" for m in diffs_per_model]))



def show_reconstruction_gradient_map(models_dir, images_dir):
    models = os.listdir(models_dir)
    imgs = os.listdir(images_dir)[:3]
    # crop = [0,-1,0,-1]
    crop = [32,-32,32,-32]
    # crop = [0,64,0,64]
    fig, axs = plt.subplots(len(imgs), 3 * (len(models) + 1))#, gridspec_kw={'wspace':0.1, 'hspace':0.25})

    x_dists = {n: [] for n in models}
    y_dists = {n: [] for n in models}

    for i, img_name in enumerate(imgs):
        img = cv2.imread(os.path.join(images_dir, img_name))

        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = img.mean(-1).astype('uint8')
        # gray_img = crop_center(gray_img, 64, 64)
        gray_img = gray_img[crop[0]:crop[1], crop[2]: crop[3]]
        img_edges_x, img_edges_y = get_edges(gray_img)

        axs[i, 0].imshow(gray_img, cmap='gray')
        axs[i, 1].imshow(img_edges_x, cmap='gray')
        axs[i, 2].imshow(img_edges_y, cmap='gray')

        for j, model_name in enumerate(models):
            encoder, generator = load_models(device, os.path.join(models_dir, model_name))
            recon = pt2cv(generator(encoder(cv2pt(img).to(device).unsqueeze(0))).detach().cpu().numpy()[0])

            # gray_recon = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
            gray_recon = recon.mean(-1).astype('uint8')

            # gray_recon = crop_center(gray_recon, 64, 64)
            gray_recon = gray_recon[crop[0]:crop[1], crop[2]: crop[3]]
            recon_edges_x, recon_edges_y = get_edges(gray_recon)
            x_dist = ((img_edges_x - recon_edges_x) ** 2).mean()
            y_dist = ((img_edges_y - recon_edges_y) ** 2).mean()
            x_dists[model_name].append(x_dist)
            y_dists[model_name].append(y_dist)
            axs[i, 3 * (j + 1), ].imshow(gray_recon, cmap='gray')
            axs[i, 3 * (j + 1), ].set_title(model_name)
            axs[i, 3 * (j + 1) + 1].imshow(recon_edges_x, cmap='gray')
            axs[i, 3 * (j + 1) + 1].set_title(f"{x_dist:.2f}")
            axs[i, 3 * (j + 1) + 2].imshow(recon_edges_y, cmap='gray')
            axs[i, 3 * (j + 1) + 2].set_title(f"{y_dist:.2f}")

    title = [f"{m}: X-L2: {np.mean(x_dists[m]):.3f}, Y-L2: {np.mean(y_dists[m]):.3f}" for m in models]
    print_mean_diff(x_dists, y_dists)
    fig.suptitle('\n'.join(title))
    map(lambda axi: axi.set_axis_off(), axs.ravel())
    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # download_ffhq_thumbnails('../../data')
    images_dir = '../perceptual_mean_optimization/clusters/z_samples_new/1/data_neighbors1'
    show_reconstruction_gradient_map('trained_models', images_dir)

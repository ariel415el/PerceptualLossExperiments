import os

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from Experiments.all import load_models
from perceptual_mean_optimization.main import cv2pt, pt2cv

device = torch.device("cpu")


def get_edges(img):
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    edges_x = cv2.filter2D(img, cv2.CV_8U, kernelx)
    edges_y = cv2.filter2D(img, cv2.CV_8U, kernely)

    return edges_x, edges_y


def print_mean_diff(x_diffs_per_model, y_diffs_per_model):
    for (SIGN, diffs_per_model) in [("X", x_diffs_per_model), ("Y", y_diffs_per_model)]:
        for m in diffs_per_model:
            diffs_per_model[m] = np.mean(diffs_per_model[m])
        diffs_sum = sum([diffs_per_model[m] for m in diffs_per_model])
        print(f"{SIGN}-diff: " + ','.join([f"{m}: {diffs_per_model[m] / diffs_sum:.4f}" for m in diffs_per_model]))




def get_derivatives(img_cv, crop=None):
    gray_img = img_cv.mean(-1).astype('uint8')
    if crop:
        gray_img = gray_img[crop[0]:crop[1], crop[2]: crop[3]]

    img_edges_x, img_edges_y = get_edges(gray_img)
    return gray_img, img_edges_x, img_edges_y


def show_reconstruction_gradients(ref_img, recs_dict, save_path):
    fig, axs = plt.subplots(3, len(recs_dict) + 1, figsize=(10, 10))  # , gridspec_kw={'wspace':0.1, 'hspace':0.25})

    gray_img, img_edges_x, img_edges_y = get_derivatives(ref_img)

    axs[0,0].imshow(gray_img, cmap='gray')
    axs[0, 0].set_title("Reference")
    axs[1, 0].imshow(img_edges_x, cmap='gray')
    axs[1, 0].set_ylabel("d/dx")
    axs[2, 0].imshow(img_edges_y, cmap='gray')
    axs[2, 0].set_ylabel("d/dy")

    for i, (name, rec) in enumerate(recs_dict.items()):
        gray_img, rec_edges_x, rec_edges_y = get_derivatives(rec)

        axs[0, i+1].imshow(gray_img, cmap='gray')
        axs[0, i+1].set_title(name)
        axs[1, i+1].imshow(rec_edges_x, cmap='gray')
        axs[1, i+1].set_title(f"L1: {np.abs(rec_edges_x - img_edges_x).mean():.2f}\nL2: {((rec_edges_x - img_edges_x)**2).mean():.2f}")
        axs[2, i+1].imshow(rec_edges_y, cmap='gray')
        axs[2, i+1].set_title(f"L1: {np.abs(rec_edges_y - img_edges_y).mean():.2f}\nL2: {((rec_edges_y - img_edges_y)**2).mean():.2f}")

    plt.tick_params(axis='both', labelsize=0, length=0)

    [(plt.setp(axi.get_xticklabels(), visible=False), plt.setp(axi.get_yticklabels(), visible=False)) for axi in axs.ravel()]
    plt.tight_layout()
    plt.savefig(save_path)


def show_model_reconstruction(models_dir, images_dir, outputs_dir):
    models = ['VGG-None_PT', 'VGG-random', 'MMD++']
    x_dists = {n: [] for n in models}
    y_dists = {n: [] for n in models}
    for i, img_name in enumerate(os.listdir(images_dir)):
        recs_dict = {}
        crop = [0, -1, 0, -1]
        # crop = [32,-32,32,-32]
        # crop = [0,64,0,64]

        img = cv2.imread(os.path.join(images_dir, img_name))
        _, edges_x, edges_y = get_derivatives(img)

        for j, model_name in enumerate(models):
            encoder, generator = load_models(device, os.path.join(models_dir, model_name))
            recon = pt2cv(generator(encoder(cv2pt(img).to(device).unsqueeze(0))).detach().cpu().numpy()[0])
            _, rec_edges_x, rec_edges_y = get_derivatives(recon)

            x_dists[model_name].append(np.abs(rec_edges_x - edges_x).mean())
            y_dists[model_name].append(np.abs(rec_edges_y - edges_y).mean())

            recs_dict[model_name] = recon

        show_reconstruction_gradients(img, recs_dict, os.path.join(outputs_dir, f'reconstruction-{i}.png'))
    print_mean_diff(x_dists, y_dists)


def show_mean_gradient_map(images_dir):
    n = 3

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


if __name__ == '__main__':
    # download_ffhq_thumbnails('../../data')
    images_dir = '../perceptual_mean_optimization/clusters/z_samples_new/6/data_neighbors6'
    outputs_dir = 'gradient_comparisons/reconstrucitons'
    os.makedirs('gradient_comparisons/reconstrucitons', exist_ok=True)
    show_model_reconstruction('trained_models', images_dir, outputs_dir)
    # show_mean_gradient_map(images_dir)

import os

from tqdm import tqdm

import utils
from KMeansFast import KMeansFast
import torch
import numpy as np
import cv2

from metrics.vgg_metric import VGGSpace
from metrics.l2_metric import L2Metric
from metrics.mmd_approximate_metric import MMDpace
# device = torch.device("cuda")
device = torch.device("cpu")

def main():
    # data, labels, img_size, dataset_name = utils.read_cifar_10_data('../../../data/cifar-10-batches-py', limit_samples=2000)
    data, labels, img_size, dataset_name = utils.read_lfw_data('../../../data/LFW', num_celebs=10)
    num_trials = 1
    # num_clusters = len(np.unique(labels))
    num_clusters = 20

    kmeans = KMeansFast(n_clusters=num_clusters, max_iter=15)

    space_metric = VGGSpace(img_size, levels=5, mode='weighted-flatten', pretrained=True)
    # space_metric = VGGSpace(img_size, levels=5, mode='last', pretrained=False)
    # space_metric = L2Metric()
    # space_metric = MMDpace(num_features=2**17, spatial_mode='mean', patch_mode='mean')
    train_features = utils.extrac_features(space_metric.feature_extractor, data, batch_size=1, device=device)

    # utils.plot_imgs(kmeans.centroids, len(kmeans.centroids))

    f1s = []
    recals = []
    precisions = []
    outputs_dir = os.path.join("pmo_output_dir", f"{dataset_name}_{space_metric.name}")
    for t in range(num_trials):
        print(f"#### Iteration : {t} ####", flush=True)
        os.makedirs(f"{outputs_dir}/{t}", exist_ok=True)
        kmeans.fit(train_features, metric=space_metric.metric, plot_path=f"{outputs_dir}/{t}/train_loss.png")

        print("Writing cluster debug_images...")
        for i in range(kmeans.n_clusters):
            cluster_data = data[kmeans.cluster_assignments == i] * 255
            utils.plot_imgs(cluster_data[:25], img_size, path=f"{outputs_dir}/{t}/C-{i}.png")
            os.makedirs(f"{outputs_dir}/{t}/C-{i}-images", exist_ok=True)
            for j, img in enumerate(cluster_data):
                img = img.transpose(1,2,0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{outputs_dir}/{t}/C-{i}-images/{j}.png", img)

        print("Computing metrics...", end='')
        f1, recal, precision = utils.compute_metrics(kmeans, labels, report_path=f"{outputs_dir}/{t}/report.txt")
        f1s.append(f1)
        recals.append(recal)
        precisions.append(precision)
        print(f"F1 score:, {f1:.2f}, Recal: {recal:.2f}, Precision: {precision:.2f}")

    with open(f"{outputs_dir}/results.txt", 'w') as f:
        f.write(f"F1: mean: {np.mean(f1s):.2f}, +- {np.std(f1s):.2f}\n")
        f.write(f"Recall: mean: {np.mean(recals):.2f}, +- {np.std(recals):.2f}\n")
        f.write(f"Precision: mean: {np.mean(precisions):.2f}, +- {np.std(precisions):.2f}\n")


if __name__ == '__main__':
    main()
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils import slow_pairwise_distances


class KMeansFast:
    def __init__(self, n_clusters=10, max_iter=500):
        print(f"Kmeans initiated with {n_clusters} clusters")
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, fit_data, metric, plot_path):
        N, d = fit_data.shape[0], fit_data.shape[1:]
        self.cluster_assignments = np.random.randint(0, self.n_clusters, size=N)

        old_centroids = np.zeros((self.n_clusters, *d))
        self.centroids = fit_data[np.random.choice(N, self.n_clusters, replace=False)]

        self.iterations = 0
        losses = []
        pbar = tqdm(total=self.max_iter, position=0, leave=True)
        pbar.set_description_str(f"Running Kmeans on {N} samples of dimension {d}")
        while not self._is_converged(self.iterations, old_centroids, self.centroids):
            old_centroids = copy.deepcopy(self.centroids)

            self._assign_to_clusters(fit_data, metric)
            self._compute_centroids(fit_data)

            loss = self.calculate_loss(fit_data)
            self.iterations += 1
            losses.append(loss)
            pbar.update(1)
            pbar.set_description(f"Iteration: {self.iterations}, Loss: {loss:.2f}", refresh=False)
        plt.plot(np.arange(len(losses)), losses)
        pbar.close()
        plt.savefig(plot_path)

    def _assign_to_clusters(self, fit_data, metric):
        if metric is not None:
            dists = slow_pairwise_distances(fit_data, self.centroids, metric=metric)
        else:
            dists = pairwise_distances(fit_data, self.centroids, metric='l2')
        self.cluster_assignments = dists.argmin(axis=1)

    def _compute_centroids(self, fit_data):
        for i in range(self.n_clusters):
            cluster = fit_data[self.cluster_assignments == i]
            self.centroids[i] = np.mean(np.vstack((self.centroids[i][None,:], cluster)), axis=0)

    def _is_converged(self, iterations, centroids, updated_centroids):
        if iterations > self.max_iter:
            return True
        centroids_dist = np.linalg.norm(np.array(updated_centroids) - np.array(centroids))
        if centroids_dist <= 1e-10:
            print("Converged! With distance:", centroids_dist)
            return True
        return False

    def calculate_loss(self, fit_data):
        loss = 0
        for i in range(self.n_clusters):
            cluster = fit_data[self.cluster_assignments == i]
            if cluster is not None:
                loss += np.sum(np.linalg.norm(cluster - self.centroids[i], axis=1))

        return loss
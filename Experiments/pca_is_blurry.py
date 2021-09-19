from time import time

import numpy as np
from scipy.linalg import eigh


class AnalyticalPCA:
    def __init__(self, data_dim, latent_dim):
        self.restoration_matrix = None
        self.projection_matrix = None
        self.name = "AnalyticalPCA"
        self.train_mean = None

    def learn_encoder_decoder(self, train_samples, plot_dir=None):
        """
        Perform PCA by triming the result orthonormal transformation of SVD
        Assumes X is zero centered
        """
        start = time()
        print("\tLearning encoder decoder... ",end="")
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        CovMat = np.dot(data.transpose(), data)

        # vals_np, vecs_np = np.linalg.eigh(CovMat)
        # # Take rows corresponding to highest eiegenvalues
        # order = np.argsort(vals_np)[::-1][:self.latent_dim]
        # self.projection_matrix = vecs_np[order].transpose()
        # self.restoration_matrix = vecs_np[order]

        vals, vecs = eigh(CovMat, subset_by_index=[self.data_dim - self.latent_dim, self.data_dim - 1])
        self.projection_matrix = vecs
        self.restoration_matrix = vecs.transpose()

        print(f"Finished in {time() - start:.2f} sec")

    def encode(self, data):
        zero_mean_data = data - self.train_mean
        return np.dot(zero_mean_data, self.projection_matrix)

    def decode(self, features):
        return np.dot(features, self.restoration_matrix) + self.train_mean

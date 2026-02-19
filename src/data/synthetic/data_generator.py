#!/usr/bin/env python
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, iid,
                       num_user=30,
                       num_class=10,
                       input_dim=60,
                       num_samples_per_user=1000,
                       cluster_seed=0,
                       data_seed=0):
    dimension = input_dim
    samples_per_user = np.full(num_user, num_samples_per_user)
    X_split = [[] for _ in range(num_user)]
    y_split = [[] for _ in range(num_user)]
    # Separate RNGs
    rng_cluster = np.random.default_rng(cluster_seed)
    rng_data = np.random.default_rng(data_seed)
    # ----- Distribution-level randomness -----
    mean_W = rng_cluster.normal(0, alpha, num_user)
    mean_b = mean_W
    B = rng_cluster.normal(0, beta, num_user)
    mean_x = np.zeros((num_user, dimension))
    diagonal = np.array([(j+1)**(-1.2) for j in range(dimension)])
    cov_x = np.diag(diagonal)

    for i in range(num_user):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]
        else:
            mean_x[i] = rng_cluster.normal(B[i], 1, dimension)
            # mean_x[i] = rng_cluster.normal(B[i], 0, dimension)

    if iid == 1:
        W_global = rng_cluster.normal(0, 1, (dimension, num_class))
        b_global = rng_cluster.normal(0, 1, num_class)

    # ----- Data-level randomness -----
    for i in range(num_user):

        W = rng_cluster.normal(mean_W[i], 1, (dimension, num_class))
        b = rng_cluster.normal(mean_b[i], 1, num_class)

        if iid == 1:
            W = W_global
            b = b_global

        xx = rng_data.multivariate_normal(
            mean_x[i], cov_x, samples_per_user[i]
        )

        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()
    print(f"Generated synthetic data with {num_user} users, {num_class} classes, input dimension {input_dim}, and {num_samples_per_user} samples per user).")
    return X_split, y_split

class SyntheticDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: A tuple or list containing (X_data, y_data)
        """
        self.X = dataset[0]
        self.y = dataset[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return a dictionary here
        return {
            'x': self.X[idx], 
            'y': self.y[idx]
        }

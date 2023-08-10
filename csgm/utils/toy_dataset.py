"""Inspired by from https://github.com/tanelp/tiny-diffusion."""

import os
import random

import numpy as np
import torch
import h5py

from torch.utils.data import TensorDataset

from .project_path import datadir
from .normalizer import Normalizer


def get_seismic_dataset():

    # Define data directory
    data_path = os.path.join(datadir("training-data"), "training-pairs.h5")

    # Download the dataset into the data directory if it does not exist
    if not os.path.isfile(data_path):
        os.system("wget https://www.dropbox.com/s/53u8ckb9aje8xv4/"
                  "training-pairs.h5 --no-check-certificate -O" + data_path)

    # Load seismic images and create training and testing data
    file = h5py.File(data_path, 'r')
    x = torch.from_numpy(file['dm'][...])
    y = torch.from_numpy(file['rtm'][...])
    file.close()

    # Zero out water layer.
    y[..., :10] = 0.0

    # Normalize the seismic images in the training data.
    x_normalizer = Normalizer(x)
    x = x_normalizer.normalize(x)

    # Normalize the seismic images in the training data.
    y_normalizer = Normalizer(y)
    y = y_normalizer.normalize(y)

    nsamples = x.shape[0]
    perm_idxs = torch.randperm(nsamples)
    x = x[perm_idxs, ...].permute(0, 1, 3, 2).unsqueeze(-1)
    y = y[perm_idxs, ...].permute(0, 1, 3, 2).unsqueeze(-1)
    data = torch.cat((x, y), dim=1)

    ntrain = nsamples // 10 * 9

    return (TensorDataset(data[:ntrain, ...]),
            TensorDataset(data[ntrain:, ...]), x_normalizer, y_normalizer)


def find_replace_closest_number(some_set, k):
    closest_num = min(some_set, key=lambda x: abs(x - k))
    some_set.remove(closest_num)
    some_set.add(k)
    return some_set


def optimal_jittered_sampling(interval, num_samples):
    # Calculate the subinterval size
    subinterval_size = (interval[1] - interval[0]) / num_samples

    samples = []

    for i in range(num_samples):
        # Calculate the center of the current subinterval
        center = interval[0] + (i + 0.5) * subinterval_size

        # Add a random jitter within the subinterval
        jitter = random.uniform(-subinterval_size / 2, subinterval_size / 2)
        sample = center + jitter

        # Ensure the sample is within the interval bounds
        sample = max(interval[0], min(interval[1], sample))

        samples.append(sample)

    return np.array(samples).astype(np.float32)


def quadratic(n=200,
              s=15,
              x_range=(-3, 3),
              eval_pattern='jitter',
              phase='train',
              device='cpu'):
    """Creat quadratic toy dataset of pairs of coordinates and function values.

    This toy dataset is obtained from: https://arxiv.org/pdf/2209.14125.pdf.

    Args:
        n: Number of data points.
        s: Maximum number of points at which functions is evaluated.
        d: Dimension of the input space.
        x_range: Range of the input space. eval_pattern: Whether to evaluate
        the function on the the same coordinates ('same')  or random
        coordinates with the same size ('same_size'). Default is 'same'.

    Returns:
        A list of n data points. Each data point is a tuple of two arrays with
        the first array being the coordinates and the second array being the
        function values.
    """
    noise_dist = torch.distributions.gamma.Gamma(1.0, 2.5)
    a_choices = torch.tensor([-1.0, 1.0], device=device)
    if eval_pattern == 'same':
        x = np.linspace(*x_range, s)
    elif eval_pattern == 'jitter':
        x = optimal_jittered_sampling(x_range, s)

    if phase != 'train':
        x = set(x)
        for val in [-1.0, 0.0, 0.5]:
            find_replace_closest_number(x, val)
        x = list(x)
        x.sort()
        x = np.array(x)

    x = torch.from_numpy(np.array(x).astype(np.float32)).reshape(
        1, 1, -1).repeat(n, 1, 1).to(device)
    a = a_choices[torch.randint(0, a_choices.size(0), (n, ),
                                device=device)].reshape(-1, 1, 1)
    eps = noise_dist.sample((n, 1)).reshape(-1, 1, 1).to(device)
    y = a * x**2 + eps

    return torch.cat((y, x), dim=1)

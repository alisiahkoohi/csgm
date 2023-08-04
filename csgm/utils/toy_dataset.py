"""Inspired by from https://github.com/tanelp/tiny-diffusion."""

import numpy as np
import torch

from torch.utils.data import TensorDataset
import os
import h5py

from .project_path import datadir
from .normalizer import Normalizer


def get_conditional_dataset(name,
                            n=8000,
                            n_val=1024,
                            input_size=2,
                            device="cpu"):

    if name == 'quadratic':
        samples = torch.zeros((n + n_val, 2, 1), dtype=torch.float)

        with torch.no_grad():
            data = np.array(quadratic(n=n + n_val, s=input_size))[..., 0]
            data[:, [0, 1], :] = data[:, [1, 0], :]
            data = torch.from_numpy(data.astype(np.float32)).to(device)
            # from IPython import embed; embed()
        return (TensorDataset(data[:n, ...]), TensorDataset(data[-n_val:,
                                                                 ...]))

    elif name == "seismic":
        # Define data directory
        data_path = os.path.join(datadir("training-data"), "training-pairs.h5")

        # Download the dataset into the data directory if it does not exist
        if not os.path.isfile(data_path):
            os.system("wget https://www.dropbox.com/s/53u8ckb9aje8xv4/"
                      "training-pairs.h5 --no-check-certificate -O" +
                      data_path)

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
    else:
        raise ValueError(f"Unknown dataset: {name}")


def find_replace_closest_number(some_set, k):
    closest_num = min(some_set, key=lambda x: abs(x - k))
    some_set.remove(closest_num)
    some_set.add(k)
    return some_set


def quadratic(n=200, s=15, d=1, x_range=(-3, 3), eval_pattern='same'):
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
    data = []
    noise_dist = torch.distributions.gamma.Gamma(1.0, 2.5)
    if eval_pattern == 'same':
        x = np.linspace(*x_range, s)
        x = set(x)
        for val in [-1.0, 1.0, 0.0]:
            find_replace_closest_number(x, val)
        x = list(x)
        x.sort()
        x = np.array(x).repeat(d).reshape(s, d).astype(np.float32)
        for i in range(n):
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.array(noise_dist.sample())
            y = a * x**2 + eps
            data.append((x, y))

    if eval_pattern == 'same_size':
        for i in range(n):
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.random.randn()
            x = np.random.uniform(*x_range, size=(s, d)).astype(np.float32)
            y = a * x**2 + eps
            data.append((x, y))

    return data

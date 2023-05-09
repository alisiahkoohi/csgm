"""Obtained from https://github.com/tanelp/tiny-diffusion."""

import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def multivariate_normal_dataset(n=8000, input_size=2, device="cpu"):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, input_size))

    return torch.from_numpy(X.astype(np.float32)).to(device)


def moons_dataset(n=8000, device="cpu"):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return torch.from_numpy(X.astype(np.float32)).to(device)


def line_dataset(n=8000, device="cpu"):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return torch.from_numpy(X.astype(np.float32)).to(device)


def circle_dataset(n=8000, device="cpu"):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return torch.from_numpy(X.astype(np.float32)).to(device)


def dino_dataset(n=8000, device="cpu"):
    df = pd.read_csv("assets/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return torch.from_numpy(X.astype(np.float32)).to(device)


def get_dataset(name, n=8000, n_val=1024, input_size=2, device="cpu"):
    if name == "moons":
        data = moons_dataset(n + n_val, device=device)
        return (TensorDataset(data[:n]), TensorDataset(data[-n_val:]))
    elif name == "dino":
        data = dino_dataset(n + n_val, device=device)
        return (TensorDataset(data[:n]), TensorDataset(data[-n_val:]))
    elif name == "line":
        data = line_dataset(n + n_val, device=device)
        return (TensorDataset(data[:n]), TensorDataset(data[-n_val:]))
    elif name == "circle":
        data = circle_dataset(n + n_val, device=device)
        return (TensorDataset(data[:n]), TensorDataset(data[-n_val:]))
    elif name == "mvn":
        data = multivariate_normal_dataset(n + n_val,
                                           input_size=input_size,
                                           device=device)
        return (TensorDataset(data[:n]), TensorDataset(data[-n_val:]))
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_conditional_dataset(name,
                            n=8000,
                            n_val=1024,
                            input_size=[2, 2],
                            device="cpu"):
    if name in ['mgan_4', 'mgan_5', 'mgan_6']:
        data = mgan_paper_toy_examples(name, n_pairs=n + n_val, device=device)
        return (TensorDataset(data[:n, ...]), TensorDataset(data[-n_val:,
                                                                 ...]))


    elif name == 'quadratic':
        samples = torch.zeros((n + n_val, 2, 1), dtype=torch.float)

        with torch.no_grad():
            data = np.array(quadratic(n=n + n_val, s = input_size[0]))[..., 0]
            data[:, [0, 1], :] = data[:, [1, 0], :]
            data = torch.from_numpy(data.astype(np.float32)).to(device)
        return (TensorDataset(data[:n, ...]), TensorDataset(data[-n_val:,
                                                                 ...]))

    else:
        raise ValueError(f"Unknown dataset: {name}")


def mgan_paper_toy_examples(name, n_pairs=8000, device="cpu"):
    """
    Generates training pairs
    """
    if name in ['mgan_4', 'mgan_5', 'mgan_6']:
        fwd_op = ExamplesMGAN(name=name)
        samples = torch.zeros((n_pairs, 2, 1), dtype=torch.float)

        with torch.no_grad():
            for j in range(n_pairs):
                x = 6.0 * torch.rand(1, dtype=torch.float) - 3.0
                samples[j, 1, :] = x[0]
                y = fwd_op(x)
                samples[j, 0, :] = y[0]

        return samples.to(device)

    else:
        raise AssertionError()


class ExamplesMGAN(torch.nn.Module):
    """
    Forward operator
    """

    def __init__(self, name='mgan_4'):
        super(ExamplesMGAN, self).__init__()
        self.name = name
        if name == 'mgan_4' or name == 'mgan_6':
            self.noise_dist = torch.distributions.gamma.Gamma(1.0, 1.0 / 0.3)
        elif name == 'mgan_5':
            self.noise_dist = torch.distributions.normal.Normal(0., 0.05)

    def forward(self, x):
        if self.name == 'mgan_4':
            return torch.tanh(x) + self.noise_dist.sample(x.shape)
        elif self.name == 'mgan_5':
            return torch.tanh(x + self.noise_dist.sample(x.shape))
        elif self.name == 'mgan_6':
            return torch.tanh(x) * self.noise_dist.sample(x.shape)



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

    if eval_pattern == 'same':
        # x = np.sort(np.random.uniform(*x_range, size=(s, d)).astype(np.float32))
        x = np.linspace(*x_range, s).repeat(d).reshape(s, d).astype(np.float32)
        for i in range(n):
            # a = np.random.uniform(-1, 1, size=(d, )).astype(np.float32)
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.random.randn()
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


def sort_coordinates(data):
    """Sort coordinates in a toy dataset.

    Args:
        data: A list of data points created by toy_dataset.

    Returns:
        A list of data points with sorted coordinates.
    """
    if data[0][0].shape[1] != 1:
        raise ValueError(
            'Coordinates are multi-dimensional. Sorting not possible.')
    sorted_data = []
    for pair in data:
        x, y = pair
        str_idx = np.argsort(x[: ,0])
        x = x[str_idx, :]
        y = y[str_idx, :]
        sorted_data.append((x, y))
    return sorted_data
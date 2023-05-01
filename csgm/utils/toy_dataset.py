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

# def mgan_eq_4(n=8000, device="cpu"):

#     noise_dist = torch.distributions.gamma.Gamma(1.0, 1.0/0.3)

#     for j in range(n):
#         x = 6.0*torch.rand(1) - 3.0
#         x_samples[j, :] = x.cpu().numpy()[0]
#         y = fwd_op(x)
#         y_samples[j, :] = y.cpu().numpy()[0]

#     torch.tanh(x) + self.noise_dist.sample(x.shape)

#     rng = np.random.default_rng(42)
#     ix = rng.integers(0, len(df), n)
#     x = df["x"].iloc[ix].tolist()
#     x = np.array(x) + rng.normal(size=len(x)) * 0.15
#     y = df["y"].iloc[ix].tolist()
#     y = np.array(y) + rng.normal(size=len(x)) * 0.15
#     x = (x / 54 - 1) * 4
#     y = (y / 48 - 1) * 4
#     X = np.stack((x, y), axis=1)
#     return torch.from_numpy(X.astype(np.float32)).to(device)


# class ForwardOperator(torch.nn.Module):
#     """
#     Forward operator
#     """
#     def __init__(self, equation=4):
#         super(ForwardOperator, self).__init__()
#         self.equation = equation
#         if equation == 4 or equation == 6:
#             self.noise_dist = torch.distributions.gamma.Gamma(1.0, 1.0/0.3)
#         elif equation == 5:
#             self.noise_dist = torch.distributions.normal.Normal(0., 0.05)
#     def forward(self, x):
#         if self.equation == 4:
#             return torch.tanh(x) + self.noise_dist.sample(x.shape)
#         elif self.equation == 5:
#             return torch.tanh(x + self.noise_dist.sample(x.shape))
#         elif self.equation == 6:
#             return torch.tanh(x) * self.noise_dist.sample(x.shape)


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

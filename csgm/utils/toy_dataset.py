"""Inspired by from https://github.com/tanelp/tiny-diffusion."""

import numpy as np
import torch

from torch.utils.data import TensorDataset


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
        return (TensorDataset(data[:n, ...]), TensorDataset(data[-n_val:,
                                                                    ...]))

    else:
        raise ValueError(f"Unknown dataset: {name}")


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
        x = np.linspace(*x_range, s).repeat(d).reshape(s, d).astype(np.float32)
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

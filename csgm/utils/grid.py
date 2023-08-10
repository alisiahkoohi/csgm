from typing import Sequence

import numpy as np
import torch


def make_grid(spatial_dim: Sequence[int]) -> torch.Tensor:
    """Make the grid of coordinates for the Fourier neural operator input.

    Args:
        spatial_dim: A sequence of spatial deimensions `(height, width)`.

    Returns:
        A torch.Tensor with the grid of coordinates of size `(1, height, width,
            2)`.
    """
    grids = []
    grids.append(np.linspace(0, 1, spatial_dim[0]) * 2.0 - 1.0)
    grids.append(np.linspace(0, 1, spatial_dim[1]) * 2.0 - 1.0)
    grid = np.vstack([u.ravel() for u in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, spatial_dim[0], spatial_dim[1], 2)
    grid = grid.astype(np.float32)
    return torch.tensor(grid)
# https://github.com/tanelp/tiny-diffusion

import torch
from torch import nn
from typing import Optional

from .fourier_neural_operator import FourierNeuralOperator


class Block(nn.Module):

    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.bn(self.ff(x)))


class ConditionalScoreGenerativeModel(nn.Module):

    def __init__(self,
                 modes: int = 5,
                 hidden_dim: int = 128,
                 nlayers: int = 5,
                 nt: int = 500):
        super().__init__()

        self.nt = nt
        self.network = FourierNeuralOperator(modes, hidden_dim, 3, 1, nlayers)

    def forward(self, x, y, t):

        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        t = t.reshape(-1, 1, 1).repeat(1, x.shape[1], 1) / self.nt * 2.0 - 1.0
        z = torch.cat((x, y, t), dim=-1)
        z = self.network(z)
        return z

# https://github.com/tanelp/tiny-diffusion

import torch
from torch import nn
from typing import Optional

from .embeddings import Embedding
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
                 input_size: int = 25,
                 hidden_dim: int = 128,
                 nlayers: int = 5,
                 time_emb: str = "sinusoidal"):
        super().__init__()

        self.time_emb = Embedding(input_size, time_emb)
        self.network = FourierNeuralOperator(10, hidden_dim, 3, 1, nlayers)

    def forward(self, x, y, t):

        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        t_emb = self.time_emb(t).reshape(t.shape[0], x.shape[1], -1)
        z = torch.cat((x, y, t_emb), dim=-1)
        z = self.network(z)
        return z

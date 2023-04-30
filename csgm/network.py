"""Partially obtained from https://github.com/tanelp/tiny-diffusion."""

import torch
from torch import nn

from .embeddings import Embedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class ScoreGenerativeModel(nn.Module):
    def __init__(self,
                 input_size: int = 2,
                 hidden_dim: int = 128,
                 nlayers: int = 3,
                 emb_size: int = 128,
                 time_emb: str = "sinusoidal",
                 input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = Embedding(emb_size, time_emb)
        self.input_mlp = Embedding(emb_size, input_emb, scale=25.0)

        concat_size = (input_size * len(self.input_mlp.layer) +
                       len(self.time_mlp.layer))
        layers = [nn.Linear(concat_size, hidden_dim), nn.GELU()]
        for _ in range(nlayers):
            layers.append(Block(hidden_dim))
        layers.append(nn.Linear(hidden_dim, input_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x_emb = self.input_mlp(x).reshape(x.shape[0], -1)
        t_emb = self.time_mlp(t)
        x = torch.cat((x_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

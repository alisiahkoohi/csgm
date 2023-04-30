"""Partially obtained from https://github.com/tanelp/tiny-diffusion."""

import torch
from torch import nn

from .embeddings import Embedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x))


class ScoreGenerativeModel(nn.Module):
    def __init__(self, 
                 input_size: int = 2,
                 hidden_dim: int = 128,
                 nlayers: int = 3,
                 emb_size: int = 32,
                 time_emb: str = "sinusoidal",
                 input_emb: str = "identity"):
        super().__init__()

        self.input_mlp = Embedding(emb_size, input_emb, scale=25.0)

        layers = [nn.Linear(input_size * len(self.input_mlp.layer), hidden_dim), nn.LeakyReLU()]
        for _ in range(nlayers):
            layers.append(Block(hidden_dim))
        layers.append(nn.Linear(hidden_dim, input_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, sigma):
        x_emb = self.input_mlp(x).reshape(x.shape[0], -1)
        x = self.joint_mlp(x_emb)
        return x / sigma.reshape(-1, 1)**2

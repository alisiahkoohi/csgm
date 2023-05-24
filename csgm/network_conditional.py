# Author: Ali Siahkoohi, alisk@rice.edu
# Partially based on https://github.com/zongyi-li/fourier_neural_operator and
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
                 input_size: int = [2, 2],
                 hidden_dim: int = 128,
                 nlayers: int = 3,
                 emb_size: int = [128, 128],
                 time_emb: str = "sinusoidal",
                 input_emb: str = "sinusoidal",
                 model: str = "mlp"):
        super().__init__()

        x_emb_size, y_emb_size, t_emb_size = emb_size
        self.model = model
        self.time_mlp = Embedding(t_emb_size, time_emb)
        self.x_input_mlp = Embedding(x_emb_size, input_emb, scale=25.0)
        self.y_input_mlp = Embedding(y_emb_size, input_emb, scale=25.0)

        concat_size = (input_size[0] * len(self.x_input_mlp.layer) +
                       input_size[1] * len(self.y_input_mlp.layer) +
                       len(self.time_mlp.layer))
        if model == "mlp":
            layers = [nn.Linear(concat_size, hidden_dim), nn.GELU()]
            for _ in range(nlayers):
                layers.append(Block(hidden_dim))
            layers.append(nn.Linear(hidden_dim, input_size[0]))
            self.network = nn.Sequential(*layers)
        elif model == "fno":
            self.network = FourierNeuralOperator(10, hidden_dim, 3,
                                                 input_size[0], nlayers)

    def forward(self, x, y, t):
        # from IPython import embed; embed()

        if self.model == "mlp":
            x_emb = self.x_input_mlp(x).reshape(x.shape[0], -1)
            y_emb = self.y_input_mlp(y).reshape(y.shape[0], -1)
            t_emb = self.time_mlp(t)
            z = torch.cat((x_emb, y_emb, t_emb), dim=-1)
            z = self.network(z)
        elif self.model == "fno":
            x_emb = self.x_input_mlp(x).reshape(x.shape[0], x.shape[1], -1)
            y_emb = self.y_input_mlp(y).reshape(y.shape[0], y.shape[1], -1)
            t_emb = self.time_mlp(t).reshape(t.shape[0], x.shape[1], -1)
            # print(x.shape, y.shape, t.shape, x_emb.shape, y_emb.shape, t_emb.shape)
            z = torch.cat((x_emb, y_emb, t_emb), dim=-1)
            # z = z.unsqueeze(1)
            z = self.network(z)
        return z

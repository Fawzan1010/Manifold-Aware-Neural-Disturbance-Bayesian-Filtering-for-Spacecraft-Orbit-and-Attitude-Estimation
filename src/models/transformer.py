from __future__ import annotations

import torch
from torch import nn


class ResidualTransformer(nn.Module):
    def __init__(self, feature_dim: int, model_dim: int = 96, num_heads: int = 4, num_layers: int = 2, output_dim: int = 6) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, model_dim)
        enc = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dim_feedforward=2 * model_dim, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, output_dim))
        self.state_head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 25))

    def forward(self, x: torch.Tensor):
        z = self.in_proj(x)
        h = self.encoder(z)[:, -1]
        return self.head(h), self.state_head(h)

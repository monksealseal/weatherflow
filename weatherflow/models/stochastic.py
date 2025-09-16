"""Stochastic flow model implementations used in the tests."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseWeatherModel


class TimeEmbedding(nn.Module):
    """Small MLP that embeds a scalar timestep."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return self.mlp(t.view(-1, 1))


class ResidualFlowBlock(nn.Module):
    """Simple residual block with depthwise separable convolutions."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = F.gelu(self.depthwise(h))
        h = self.pointwise(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.out_proj(F.gelu(h))
        return residual + self.dropout(h)


class StochasticFlowModel(BaseWeatherModel):
    """Deterministic approximation of a stochastic flow used by the tests."""

    def __init__(
        self,
        channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualFlowBlock(hidden_dim, dropout) for _ in range(max(num_layers, 1))
        )
        self.decoder = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        self.flow_scale = nn.Parameter(torch.tensor(0.25))

    def _prepare_time(self, t: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if t is None:
            t = torch.zeros(batch_size, device=device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != batch_size:
            raise ValueError(
                f"Expected a timestep for each batch element, got shape {t.shape} instead."
            )
        return t.to(device=device)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor of shape [B, C, H, W].")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels but received {x.shape[1]}."
            )

        batch_size = x.shape[0]
        device = x.device
        time_emb = self.time_embedding(self._prepare_time(t, batch_size, device))
        time_emb = time_emb.view(batch_size, self.hidden_dim, 1, 1)

        h = self.encoder(x)
        h = h + time_emb
        for block in self.blocks:
            h = block(h)

        drift = self.decoder(h)
        output = x + self.flow_scale * drift
        return output

    def mass_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:
        grad_y, grad_x = torch.gradient(x, dim=(-2, -1))
        divergence = grad_x + grad_y
        return divergence.pow(2).mean()

    def energy_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).sum(dim=(1, 2, 3), keepdim=True)
        mean_energy = energy.mean(dim=0, keepdim=True)
        return (energy - mean_energy).pow(2).mean()

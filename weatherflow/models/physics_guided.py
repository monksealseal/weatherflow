"""Physics guided attention modules used in the unit tests."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Create sinusoidal embeddings for a scalar timestep."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("The embedding dimension must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        device = t.device
        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=t.dtype)
            * -(np.log(10000.0) / max(half_dim - 1, 1))
        )
        args = t.view(-1, 1) * frequencies.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualConvBlock(nn.Module):
    """A lightweight residual block mixing local spatial information."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = F.gelu(self.conv1(h))
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.conv2(F.gelu(h))
        h = self.dropout(h)
        return residual + h


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(hidden_dim // 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, reduced),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.mlp(weights).view(b, c, 1, 1)
        return x * weights + x


class PhysicsBlock(nn.Module):
    """Block combining local mixing and global channel attention."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.residual = ResidualConvBlock(hidden_dim, dropout)
        self.channel_attn = ChannelAttention(hidden_dim, dropout)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.residual(x)
        h = self.channel_attn(h)
        return self.norm(h)


class PhysicsGuidedAttention(nn.Module):
    """Compact physics guided attention model used in the tests."""

    def __init__(
        self,
        channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        self.input_proj = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            PhysicsBlock(hidden_dim, dropout) for _ in range(max(num_layers, 1))
        )
        self.output_proj = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        self.final_norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("_energy_eps", torch.tensor(1e-6), persistent=False)

    def _apply_time_embedding(
        self, features: torch.Tensor, timestep: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if timestep is None:
            return features
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        emb = self.time_embedding(timestep.to(features.dtype))
        emb = self.time_mlp(emb)
        emb = emb.view(features.size(0), -1, 1, 1)
        return features + emb

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor of shape [B, C, H, W].")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels but received {x.shape[1]}."
            )

        h = self.input_proj(x)
        h = self._apply_time_embedding(h, timestep)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        output = self.output_proj(h)
        output = output + self.residual_scale * x

        input_energy = x.pow(2).sum(dim=(1, 2, 3), keepdim=True) + self._energy_eps
        output_energy = output.pow(2).sum(dim=(1, 2, 3), keepdim=True) + self._energy_eps
        scale = torch.sqrt(input_energy / output_energy)
        return output * scale

    def compute_physics_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        grad_y, grad_x = torch.gradient(pred, dim=(-2, -1))
        divergence = grad_x + grad_y
        mass_loss = divergence.pow(2).mean()

        pred_energy = pred.pow(2).sum(dim=(1, 2, 3))
        target_energy = target.pow(2).sum(dim=(1, 2, 3))
        energy_loss = (pred_energy - target_energy).pow(2).mean()

        return mass_loss + 0.1 * energy_loss

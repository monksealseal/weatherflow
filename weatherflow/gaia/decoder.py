"""GAIA decoders: conditional diffusion or ensemble heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    """Configuration for GAIA decoder selection."""

    mode: str = "diffusion"
    diffusion_steps: int = 4
    ensemble_members: int = 4


class ConditionalDiffusionDecoder(nn.Module):
    """Lightweight conditional diffusion-style decoder on mesh features."""

    def __init__(self, hidden_dim: int, output_channels: int, steps: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.steps = steps
        self.denoise_mlp = nn.Sequential(
            nn.Linear(hidden_dim + output_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        output = torch.zeros(batch, nodes, self.output_channels, device=x.device, dtype=x.dtype)
        for _ in range(self.steps):
            update = self.denoise_mlp(torch.cat([x, output], dim=-1))
            output = output + update / self.steps
        return output


class EnsembleDecoder(nn.Module):
    """Ensemble head that emits multiple mesh predictions."""

    def __init__(self, hidden_dim: int, output_channels: int, members: int) -> None:
        super().__init__()
        self.members = members
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_channels))
             for _ in range(members)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=1)


class GaiaDecoder(nn.Module):
    """Select between diffusion and ensemble decoder heads."""

    def __init__(
        self,
        hidden_dim: int,
        output_channels: int,
        config: Optional[DecoderConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        if self.config.mode == "diffusion":
            self.decoder = ConditionalDiffusionDecoder(hidden_dim, output_channels, self.config.diffusion_steps)
        elif self.config.mode == "ensemble":
            self.decoder = EnsembleDecoder(hidden_dim, output_channels, self.config.ensemble_members)
        else:
            raise ValueError(f"Unsupported decoder mode: {self.config.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

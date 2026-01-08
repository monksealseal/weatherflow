"""GAIA processor: alternating GNN message passing and spectral/global mixing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProcessorConfig:
    """Configuration for GAIA processor blocks."""

    num_blocks: int = 4
    knn_k: int = 8
    dropout: float = 0.1


class LatAwareBias(nn.Module):
    """Latitude-aware bias projection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, lat: torch.Tensor) -> torch.Tensor:
        return self.proj(lat.unsqueeze(-1))


class GNNBlock(nn.Module):
    """Simple message passing block with KNN adjacency."""

    def __init__(self, hidden_dim: int, knn_idx: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("knn_idx", knn_idx, persistent=False)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lat_bias: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = x.shape
        knn_idx = self.knn_idx.to(x.device)
        neighbors = x[:, knn_idx]
        center = x.unsqueeze(2).expand_as(neighbors)
        messages = self.message_mlp(torch.cat([center, neighbors], dim=-1))
        messages = messages.mean(dim=2)
        updated = self.update_mlp(torch.cat([x, messages], dim=-1))
        updated = self.dropout(updated + lat_bias)
        return self.norm(x + updated)


class SpectralMixingBlock(nn.Module):
    """Spectral/global mixing with latitude-aware bias."""

    def __init__(self, hidden_dim: int, order_idx: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("order_idx", order_idx, persistent=False)
        self.register_buffer("inv_order_idx", torch.argsort(order_idx), persistent=False)
        self.freq_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lat_bias: torch.Tensor) -> torch.Tensor:
        order_idx = self.order_idx.to(x.device)
        inv_idx = self.inv_order_idx.to(x.device)
        x_sorted = x[:, order_idx]
        fft = torch.fft.rfft(x_sorted, dim=1)
        real = self.freq_mlp(fft.real)
        imag = self.freq_mlp(fft.imag)
        fft_mixed = torch.complex(real, imag)
        x_time = torch.fft.irfft(fft_mixed, n=x_sorted.shape[1], dim=1)
        x_time = x_time[:, inv_idx]
        global_context = self.global_mlp(x.mean(dim=1, keepdim=True))
        out = x + x_time + global_context + lat_bias
        out = self.dropout(out)
        return self.norm(out)


class GaiaProcessor(nn.Module):
    """Alternate GNN and spectral mixing blocks on mesh features."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_vertices: torch.Tensor,
        config: ProcessorConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or ProcessorConfig()
        knn_idx, order_idx, lat = self._build_graph(mesh_vertices, self.config.knn_k)
        self.register_buffer("lat", lat, persistent=False)
        self.lat_bias = LatAwareBias(hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(self.config.num_blocks):
            self.blocks.append(GNNBlock(hidden_dim, knn_idx, self.config.dropout))
            self.blocks.append(SpectralMixingBlock(hidden_dim, order_idx, self.config.dropout))

    @staticmethod
    def _build_graph(mesh_vertices: torch.Tensor, knn_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.cdist(mesh_vertices.unsqueeze(0), mesh_vertices.unsqueeze(0)).squeeze(0)
        knn_idx = torch.topk(dist, k=min(knn_k + 1, mesh_vertices.shape[0]), largest=False).indices[:, 1:]
        lat = torch.asin(mesh_vertices[:, 2]).unsqueeze(0)
        order_idx = torch.argsort(lat.squeeze(0))
        return knn_idx, order_idx, lat

    def forward(self, x: torch.Tensor, mesh_vertices: torch.Tensor) -> torch.Tensor:
        lat_bias = self.lat_bias(self.lat.to(x.device)).expand_as(x)
        out = x
        for block in self.blocks:
            out = block(out, lat_bias)
        return out

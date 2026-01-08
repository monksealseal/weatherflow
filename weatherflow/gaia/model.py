"""End-to-end GAIA model assembling encoder, processor, and decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from weatherflow.gaia.decoder import DecoderConfig, GaiaDecoder
from weatherflow.gaia.encoder import GaiaGridEncoder, TokenizationConfig
from weatherflow.gaia.processor import GaiaProcessor, ProcessorConfig


@dataclass
class GaiaConfig:
    """Configuration for the GAIA model components."""

    input_channels: int
    output_channels: int
    hidden_dim: int = 256
    mesh_subdivisions: int = 1
    encoder_knn_k: int = 8
    processor: Optional[ProcessorConfig] = None
    decoder: Optional[DecoderConfig] = None
    tokenization: Optional[TokenizationConfig] = None
    output_knn_k: int = 8


class GaiaModel(nn.Module):
    """GAIA end-to-end model with consistent grid input/output shapes."""

    def __init__(self, config: GaiaConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = GaiaGridEncoder(
            input_channels=config.input_channels,
            hidden_dim=config.hidden_dim,
            subdivisions=config.mesh_subdivisions,
            knn_k=config.encoder_knn_k,
            tokenization=config.tokenization,
        )
        self.processor = GaiaProcessor(
            hidden_dim=config.hidden_dim,
            mesh_vertices=self.encoder.mesh_vertices,
            config=config.processor,
        )
        self.decoder = GaiaDecoder(
            hidden_dim=config.hidden_dim,
            output_channels=config.output_channels,
            config=config.decoder,
        )
        self.output_knn_k = config.output_knn_k
        self.output_proj = nn.Linear(config.output_channels, config.output_channels)

    @staticmethod
    def _grid_positions(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        x = torch.cos(lat_grid) * torch.cos(lon_grid)
        y = torch.cos(lat_grid) * torch.sin(lon_grid)
        z = torch.sin(lat_grid)
        return torch.stack([x, y, z], dim=-1)

    def _mesh_to_grid(
        self,
        mesh_output: torch.Tensor,
        mesh_vertices: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        lat = lat.to(mesh_output.device)
        lon = lon.to(mesh_output.device)
        grid_pos = self._grid_positions(lat, lon).reshape(-1, 3)
        mesh_pos = mesh_vertices.to(mesh_output.device)
        dist = torch.cdist(grid_pos.unsqueeze(0), mesh_pos.unsqueeze(0)).squeeze(0)
        knn_dist, knn_idx = torch.topk(dist, k=min(self.output_knn_k, mesh_pos.shape[0]), largest=False)
        weights = F.softmax(-knn_dist, dim=-1)

        if mesh_output.dim() == 3:
            batch, nodes, channels = mesh_output.shape
            gathered = mesh_output[:, knn_idx]
            output = (weights.unsqueeze(0).unsqueeze(-1) * gathered).sum(dim=2)
            output = output.view(batch, lat.shape[0], lon.shape[0], channels).permute(0, 3, 1, 2)
            return self.output_proj(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch, members, nodes, channels = mesh_output.shape
        gathered = mesh_output[:, :, knn_idx]
        output = (weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * gathered).sum(dim=3)
        output = output.view(batch, members, lat.shape[0], lon.shape[0], channels)
        output = output.permute(0, 1, 4, 2, 3)
        output = self.output_proj(output.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        return output

    def forward(self, x: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        mesh_features, mesh_vertices = self.encoder(x, lat, lon)
        processed = self.processor(mesh_features, mesh_vertices)
        mesh_output = self.decoder(processed)
        return self._mesh_to_grid(mesh_output, mesh_vertices, lat, lon)

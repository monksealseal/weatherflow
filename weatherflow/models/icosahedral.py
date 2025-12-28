"""Icosahedral mesh-based flow matching model."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _icosahedron() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return vertices and faces for a unit icosahedron."""
    phi = (1 + 5 ** 0.5) / 2
    verts = torch.tensor(
        [
            (-1,  phi,  0),
            ( 1,  phi,  0),
            (-1, -phi,  0),
            ( 1, -phi,  0),
            ( 0, -1,  phi),
            ( 0,  1,  phi),
            ( 0, -1, -phi),
            ( 0,  1, -phi),
            ( phi,  0, -1),
            ( phi,  0,  1),
            (-phi, 0, -1),
            (-phi, 0,  1),
        ],
        dtype=torch.float32,
    )
    verts = verts / verts.norm(dim=1, keepdim=True)
    faces = torch.tensor(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=torch.long,
    )
    return verts, faces


def _faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = set()
    for f in faces.tolist():
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edges.add(tuple(sorted((a, b))))
    return torch.tensor(sorted(list(edges)), dtype=torch.long)


class IcosahedralFlowMatch(nn.Module):
    """Graph-based flow model on an icosahedral mesh."""

    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        verts, faces = _icosahedron()
        edges = _faces_to_edges(faces)
        self.register_buffer("verts", verts, persistent=False)
        self.register_buffer("faces", faces, persistent=False)
        self.register_buffer("edges", edges, persistent=False)

        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, input_channels)

    def _grid_mapping(self, lat: int, lon: int, device: torch.device):
        """Precompute nearest grid cell for each vertex."""
        lat_centers = torch.linspace(-math.pi / 2, math.pi / 2, steps=lat, device=device)
        lon_centers = torch.linspace(-math.pi, math.pi, steps=lon, device=device)
        lon_grid, lat_grid = torch.meshgrid(lon_centers, lat_centers, indexing="xy")
        latlon_grid = torch.stack([lat_grid, lon_grid], dim=-1)  # [lon, lat, 2]

        verts = self.verts.to(device)
        vert_lat = torch.asin(verts[:, 2])
        vert_lon = torch.atan2(verts[:, 1], verts[:, 0])
        vert_latlon = torch.stack([vert_lat, vert_lon], dim=1)  # [V,2]

        flat_grid = latlon_grid.view(-1, 2)  # [lon*lat,2]
        dlat = flat_grid[:, 0].unsqueeze(0) - vert_latlon[:, 0].unsqueeze(1)
        dlon = torch.remainder(
            flat_grid[:, 1].unsqueeze(0) - vert_latlon[:, 1].unsqueeze(1) + math.pi,
            2 * math.pi,
        ) - math.pi
        dist2 = dlat**2 + dlon**2
        nearest = dist2.argmin(dim=1)  # [V]

        # grid -> vertex mapping (nearest)
        grid_to_vert = dist2.argmin(dim=0)  # [grid_size]
        return nearest, grid_to_vert.view(lon, lat).permute(1, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] grid data.
            t: [B] time (unused; placeholder for parity with WeatherFlowMatch).
        Returns:
            [B, C, H, W] grid velocities.
        """
        b, c, h, w = x.shape
        device = x.device
        vert_to_grid, grid_to_vert = self._grid_mapping(h, w, device)

        # Grid -> vertex gather
        nodes = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        node_feats = nodes[:, grid_to_vert.reshape(-1), :]  # [B, V, C]

        h_nodes = self.input_proj(node_feats)
        edges = self.edges.to(device)
        for layer in self.layers:
            src, dst = edges[:, 0], edges[:, 1]
            agg = torch.zeros_like(h_nodes)
            agg.index_add_(1, dst, h_nodes[:, src])
            agg.index_add_(1, src, h_nodes[:, dst])
            deg = torch.zeros(h_nodes.shape[1], device=device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float, device=device))
            deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float, device=device))
            deg = deg.clamp(min=1.0).view(1, -1, 1)
            agg = agg / deg
            h_nodes = h_nodes + F.relu(layer(agg))

        out_nodes = self.output_proj(h_nodes)  # [B, V, C]

        # Vertex -> grid scatter (nearest)
        out_grid = out_nodes[:, vert_to_grid, :].view(b, h, w, c).permute(0, 3, 1, 2)
        return out_grid

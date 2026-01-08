"""GAIA encoder: grid-to-mesh projection with variable tokenization and KNN attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _icosahedron() -> Tuple[torch.Tensor, torch.Tensor]:
    phi = (1 + 5 ** 0.5) / 2
    verts = torch.tensor(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
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


def _subdivide(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    vert_cache = {}
    verts_list = verts.tolist()
    new_faces = []

    def midpoint(a: int, b: int) -> int:
        key = tuple(sorted((a, b)))
        if key in vert_cache:
            return vert_cache[key]
        va = torch.tensor(verts_list[a])
        vb = torch.tensor(verts_list[b])
        vm = (va + vb) / 2
        vm = (vm / vm.norm()).tolist()
        verts_list.append(vm)
        idx = len(verts_list) - 1
        vert_cache[key] = idx
        return idx

    for (a, b, c) in faces.tolist():
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend(
            [
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ]
        )

    new_verts = torch.tensor(verts_list, dtype=torch.float32)
    new_verts = new_verts / new_verts.norm(dim=1, keepdim=True)
    new_faces_t = torch.tensor(new_faces, dtype=torch.long)
    return new_verts, new_faces_t


@dataclass
class TokenizationConfig:
    """Configuration for variable grid tokenization."""

    token_grid: Optional[Tuple[int, int]] = None


class GaiaGridEncoder(nn.Module):
    """Project grid inputs onto an icosahedral mesh with KNN attention."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        subdivisions: int = 1,
        knn_k: int = 8,
        tokenization: Optional[TokenizationConfig] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.knn_k = knn_k
        self.tokenization = tokenization or TokenizationConfig()

        verts, faces = _icosahedron()
        for _ in range(max(0, subdivisions)):
            verts, faces = _subdivide(verts, faces)
        self.register_buffer("mesh_vertices", verts, persistent=False)
        self.register_buffer("mesh_faces", faces, persistent=False)

        self.token_proj = nn.Linear(input_channels, hidden_dim)
        self.pos_proj = nn.Linear(3, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _grid_positions(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        x = torch.cos(lat_grid) * torch.cos(lon_grid)
        y = torch.cos(lat_grid) * torch.sin(lon_grid)
        z = torch.sin(lat_grid)
        return torch.stack([x, y, z], dim=-1)

    def _tokenize(
        self, x: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        token_grid = self.tokenization.token_grid
        if token_grid is None or token_grid == (height, width):
            tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            positions = self._grid_positions(lat, lon).reshape(height * width, 3)
            return tokens, positions

        target_h = max(1, min(height, token_grid[0]))
        target_w = max(1, min(width, token_grid[1]))
        stride_h = max(1, height // target_h)
        stride_w = max(1, width // target_w)
        kernel = (stride_h, stride_w)
        x_tokens = F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        lat_tokens = F.avg_pool2d(lat_grid[None, None], kernel_size=kernel, stride=kernel)
        lon_tokens = F.avg_pool2d(lon_grid[None, None], kernel_size=kernel, stride=kernel)
        positions = self._grid_positions(lat_tokens.squeeze(0).squeeze(0), lon_tokens.squeeze(0).squeeze(0))
        tokens = x_tokens.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        return tokens, positions.reshape(-1, 3)

    def forward(self, x: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        lat = lat.to(x.device)
        lon = lon.to(x.device)
        tokens, token_pos = self._tokenize(x, lat, lon)
        batch, token_count, _ = tokens.shape

        token_embed = self.token_proj(tokens) + self.pos_proj(token_pos).unsqueeze(0)
        mesh_pos = self.mesh_vertices.to(token_embed.device)
        query = self.query_proj(self.pos_proj(mesh_pos)).unsqueeze(0)

        dist = torch.cdist(mesh_pos.unsqueeze(0), token_pos.unsqueeze(0)).squeeze(0)
        knn_dist, knn_idx = torch.topk(dist, k=min(self.knn_k, token_count), largest=False)

        token_embed_b = token_embed[:, knn_idx]
        key = self.key_proj(token_embed_b)
        value = self.value_proj(token_embed_b)
        query = query.unsqueeze(2)
        attn_logits = (query * key).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn_logits = attn_logits + (-knn_dist).unsqueeze(0)
        attn = F.softmax(attn_logits, dim=-1)
        mesh_features = (attn.unsqueeze(-1) * value).sum(dim=-2)
        mesh_features = self.norm(mesh_features)
        return mesh_features, mesh_pos

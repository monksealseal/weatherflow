"""Icosahedral mesh-based flow matching model with subdivision and attention MP."""
from __future__ import annotations

import math
from functools import lru_cache
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


def _subdivide(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loop-like subdivision: split each triangle into 4, project to sphere."""
    vert_cache = {}
    new_faces = []
    verts_list = verts.tolist()

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


def _faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = set()
    for f in faces.tolist():
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edges.add(tuple(sorted((a, b))))
    return torch.tensor(sorted(list(edges)), dtype=torch.long)


class IcosahedralFlowMatch(nn.Module):
    """Graph-based flow model on a subdivided icosahedral mesh."""

    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        subdivisions: int = 1,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads

        verts, faces = _icosahedron()
        for _ in range(max(0, subdivisions)):
            verts, faces = _subdivide(verts, faces)
        edges = _faces_to_edges(faces)
        self.register_buffer("verts", verts, persistent=False)
        self.register_buffer("faces", faces, persistent=False)
        self.register_buffer("edges", edges, persistent=False)

        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_channels)
        self.norm = nn.LayerNorm(hidden_dim)

    @lru_cache(maxsize=32)
    def _grid_mapping(self, lat: int, lon: int, device: torch.device):
        """Precompute nearest grid cell for each vertex and vertex for each grid cell."""
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

        grid_to_vert = dist2.argmin(dim=0)  # [grid_size]
        vert_to_grid = dist2.argmin(dim=1)  # [V]
        return vert_to_grid.view(-1), grid_to_vert.view(lon, lat).permute(1, 0)

    def _edge_geometry(self, device: torch.device) -> torch.Tensor:
        """Compute edge direction (2 angles) and length on the sphere."""
        verts = self.verts.to(device)
        edges = self.edges.to(device)
        v0 = verts[edges[:, 0]]
        v1 = verts[edges[:, 1]]
        dot = (v0 * v1).sum(dim=1).clamp(-1.0, 1.0)
        length = torch.acos(dot).unsqueeze(-1)  # geodesic distance
        # direction as delta lat/lon
        lat0 = torch.asin(v0[:, 2])
        lon0 = torch.atan2(v0[:, 1], v0[:, 0])
        lat1 = torch.asin(v1[:, 2])
        lon1 = torch.atan2(v1[:, 1], v1[:, 0])
        dlat = (lat1 - lat0).unsqueeze(-1)
        dlon = torch.remainder((lon1 - lon0).unsqueeze(-1) + math.pi, 2 * math.pi) - math.pi
        return torch.cat([dlat, dlon, length], dim=1)  # [E,3]

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
        edge_feat = self._edge_geometry(device)  # [E,3]
        src, dst = edges[:, 0], edges[:, 1]
        for layer in self.layers:
            q = self.attn_proj_q(h_nodes)
            k = self.attn_proj_k(h_nodes)
            v = self.attn_proj_v(h_nodes)

            q_src = q[:, src]
            k_dst = k[:, dst]
            v_src = v[:, src]

            # edge-conditioned attention
            e_emb = self.edge_mlp(edge_feat).unsqueeze(0)  # [1,E,H]
            logits = (q_src + e_emb) * k_dst
            logits = logits.view(b, logits.shape[1], self.heads, -1).sum(dim=-1) / math.sqrt(
                logits.shape[-1]
            )
            alpha = torch.zeros_like(logits)
            alpha.index_add_(1, dst, logits)
            alpha = F.softmax(alpha, dim=1)

            msg = v_src + e_emb
            msg = msg.view(b, msg.shape[1], self.heads, -1)
            alpha = alpha.unsqueeze(-1)
            agg = torch.zeros_like(msg)
            agg.index_add_(1, dst, alpha * msg)
            agg = agg.view(b, agg.shape[1], -1)
            h_nodes = h_nodes + self.norm(layer(h_nodes) + agg)

        out_nodes = self.output_proj(h_nodes)  # [B, V, C]

        # Vertex -> grid scatter (nearest)
        out_grid = out_nodes[:, vert_to_grid, :].view(b, h, w, c).permute(0, 3, 1, 2)
        return out_grid

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
        self.interp_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid_to_cartesian(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)

    def _barycentric_weights(
        self, point: torch.Tensor, tri: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute barycentric weights of point wrt triangle (on sphere approximated in 3D)."""
        v0, v1, v2 = tri
        mat = torch.stack([v0 - v2, v1 - v2], dim=1)  # [2,3]
        try:
            inv = torch.linalg.pinv(mat)
            coords = inv @ (point - v2)
            w0, w1 = coords[0], coords[1]
            w2 = 1 - w0 - w1
            return torch.stack([w0, w1, w2]), torch.tensor(1.0)
        except Exception:
            return torch.tensor([1.0, 0.0, 0.0], device=point.device), torch.tensor(0.0)

    @lru_cache(maxsize=32)
    def _grid_mapping(self, lat: int, lon: int, device: torch.device):
        """Precompute grid->mesh barycentric weights and mesh->grid scatter indices."""
        lat_centers = torch.linspace(-math.pi / 2, math.pi / 2, steps=lat, device=device)
        lon_centers = torch.linspace(-math.pi, math.pi, steps=lon, device=device)
        lon_grid, lat_grid = torch.meshgrid(lon_centers, lat_centers, indexing="xy")
        grid_xyz = self._grid_to_cartesian(lat_grid, lon_grid).view(-1, 3)  # [G,3]

        verts = self.verts.to(device)
        faces = self.faces.to(device)

        # For each grid point, find nearest face centroid
        face_centers = verts[faces].mean(dim=1)  # [F,3]
        face_norm = face_centers / face_centers.norm(dim=1, keepdim=True)

        grid_face = []
        weights = []
        for p in grid_xyz:
            # nearest face
            dist = (face_norm - p).pow(2).sum(dim=1)
            face_idx = dist.argmin()
            tri = verts[faces[face_idx]]
            bary, _ = self._barycentric_weights(p, tri)
            grid_face.append(face_idx)
            weights.append(bary)
        grid_face = torch.tensor(grid_face, device=device, dtype=torch.long)  # [G]
        weights = torch.stack(weights, dim=0)  # [G,3]

        vert_to_grid = torch.zeros(verts.shape[0], dtype=torch.long, device=device)
        # scatter nearest centroid for reverse mapping (approximate)
        _, closest_face_per_vert = torch.cdist(verts, face_centers).min(dim=1)
        vert_to_grid = grid_face[closest_face_per_vert]
        return (grid_face, weights), vert_to_grid

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
        (grid_face, grid_weights), vert_to_grid = self._grid_mapping(h, w, device)

        # Grid -> vertex gather
        nodes = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        face_indices = grid_face  # [G]
        tri_verts = self.faces.to(device)[face_indices]  # [G,3]
        weights = grid_weights  # [G,3]
        # Aggregate grid cells to vertices via barycentric weights
        node_feats = torch.zeros(b, self.verts.shape[0], c, device=device)
        node_feats.index_add_(1, tri_verts[:, 0], nodes * weights[:, 0].view(1, -1, 1))
        node_feats.index_add_(1, tri_verts[:, 1], nodes * weights[:, 1].view(1, -1, 1))
        node_feats.index_add_(1, tri_verts[:, 2], nodes * weights[:, 2].view(1, -1, 1))

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

        # Vertex -> grid scatter using face weights
        out_grid = torch.zeros(b, h * w, c, device=device)
        out_grid += out_nodes[:, tri_verts[:, 0], :] * weights[:, 0].view(1, -1, 1)
        out_grid += out_nodes[:, tri_verts[:, 1], :] * weights[:, 1].view(1, -1, 1)
        out_grid += out_nodes[:, tri_verts[:, 2], :] * weights[:, 2].view(1, -1, 1)
        out_grid = out_grid.view(b, h, w, c).permute(0, 3, 1, 2)
        return out_grid

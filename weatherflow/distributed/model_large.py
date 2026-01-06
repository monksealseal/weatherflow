"""Large-scale WeatherFlow foundation model with axial ViT backbone.

This module implements a scalable architecture suitable for global 0.25°
resolution training while retaining compatibility with the existing
``WeatherFlowMatch`` interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..manifolds.sphere import Sphere
from ..models.flow_matching import TimeEncoder, WeatherFlowMatch


def _split_levels(x: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Reshape channel-first input into level-major format.

    Args:
        x: Tensor of shape [B, C, H, W]
        n_levels: Number of vertical levels to split channels into.

    Returns:
        Tensor of shape [B, L, C_per_level, H, W]
    """
    if x.shape[1] % n_levels != 0:
        raise ValueError(
            f"Input channels ({x.shape[1]}) must be divisible by n_levels ({n_levels})."
        )
    c_per_level = x.shape[1] // n_levels
    return x.view(x.shape[0], n_levels, c_per_level, x.shape[2], x.shape[3])


def rotary_embedding(lat: torch.Tensor, lon: torch.Tensor, dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary embeddings aware of spherical geometry.

    The phase combines wrapped longitude (scaled by cos(lat)) and clipped
    latitude to mitigate pole singularities while preserving periodicity in
    longitude.
    """
    lon_wrapped = torch.remainder(lon + math.pi, 2 * math.pi) - math.pi
    lat_clamped = torch.clamp(lat, -math.pi / 2 + 1e-6, math.pi / 2 - 1e-6)
    scaled_lon = lon_wrapped * torch.cos(lat_clamped)

    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=lat.device, dtype=lat.dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    phase = (lat_clamped + scaled_lon)[..., None] * inv_freq
    return phase.sin(), phase.cos()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query/key tensors."""
    def _reshape(x: torch.Tensor) -> torch.Tensor:
        b, heads, seq, dim = x.shape
        x = x.view(b, heads, seq, dim // 2, 2)
        x1, x2 = x.unbind(-1)
        return x1, x2

    q1, q2 = _reshape(q)
    k1, k2 = _reshape(k)

    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot.view_as(q), k_rot.view_as(k)


class PatchEmbed(nn.Module):
    """Simple convolutional patch embedding for lat/lon grids."""

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AxialAttention3D(nn.Module):
    """Axial attention over (lat, lon) followed by vertical attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.lat_qkv = nn.Linear(dim, dim * 3)
        self.lon_qkv = nn.Linear(dim, dim * 3)
        self.level_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm_lat = nn.LayerNorm(dim)
        self.norm_lon = nn.LayerNorm(dim)
        self.norm_level = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, L, H, W, D]
            lat_grid: Tensor [H]
            lon_grid: Tensor [W]
        """
        b, levels, h, w, dim = x.shape

        # Latitude attention
        lat_tokens = x.permute(0, 1, 3, 2, 4).reshape(b * levels * w, h, dim)
        lat_positions = lat_grid.view(1, h).expand(b * levels * w, -1)
        head_dim = dim // self.num_heads
        lat_sin, lat_cos = rotary_embedding(lat_positions, torch.zeros_like(lat_positions), head_dim)
        qkv = self.lat_qkv(self.norm_lat(lat_tokens)).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(-1, h, self.num_heads, dim // self.num_heads).transpose(1, 2)
        k = k.view(-1, h, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(-1, h, self.num_heads, dim // self.num_heads).transpose(1, 2)
        q_rot, k_rot = apply_rotary(q, k, lat_sin, lat_cos)
        attn_lat = F.scaled_dot_product_attention(q_rot, k_rot, v)
        attn_lat = attn_lat.transpose(1, 2).reshape(-1, h, dim)
        lat_out = lat_tokens + self.dropout(self.proj(attn_lat))
        lat_out = lat_out.view(b, levels, w, h, dim).permute(0, 1, 3, 2, 4)

        # Longitude attention
        lon_tokens = lat_out.permute(0, 1, 2, 3, 4).reshape(b * levels * h, w, dim)
        lon_range = torch.linspace(-math.pi, math.pi, steps=w, device=lon_tokens.device, dtype=lon_tokens.dtype)
        lon_positions = lon_range.view(1, w).expand(b * levels * h, -1)
        lon_sin, lon_cos = rotary_embedding(torch.zeros_like(lon_positions), lon_positions, head_dim)
        qkv = self.lon_qkv(self.norm_lon(lon_tokens)).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(-1, w, self.num_heads, dim // self.num_heads).transpose(1, 2)
        k = k.view(-1, w, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(-1, w, self.num_heads, dim // self.num_heads).transpose(1, 2)
        q_rot, k_rot = apply_rotary(q, k, lon_sin, lon_cos)
        attn_lon = F.scaled_dot_product_attention(q_rot, k_rot, v)
        attn_lon = attn_lon.transpose(1, 2).reshape(-1, w, dim)
        lon_out = lon_tokens + self.dropout(self.proj(attn_lon))
        lon_out = lon_out.view(b, levels, h, w, dim)

        # Vertical attention
        level_tokens = lon_out.permute(0, 2, 3, 1, 4).reshape(b * h * w, levels, dim)
        qkv = self.level_qkv(self.norm_level(level_tokens)).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(-1, levels, self.num_heads, dim // self.num_heads).transpose(1, 2)
        k = k.view(-1, levels, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(-1, levels, self.num_heads, dim // self.num_heads).transpose(1, 2)
        attn_level = F.scaled_dot_product_attention(q, k, v)
        attn_level = attn_level.transpose(1, 2).reshape(-1, levels, dim)
        level_out = level_tokens + self.dropout(self.proj(attn_level))
        level_out = level_out.view(b, h, w, levels, dim).permute(0, 3, 1, 2, 4)
        return level_out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


@dataclass
class WeatherFlowFoundationConfig:
    input_channels: int
    levels: int
    embed_dim: int = 1536
    depth: int = 24
    num_heads: int = 16
    patch_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_grad_checkpointing: bool = True
    static_channels: int = 0
    forcing_dim: int = 0
    sphere_regularization: bool = True


class TransformerBlock(nn.Module):
    """Transformer block with axial attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AxialAttention3D(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, lat_grid: torch.Tensor, lon_grid: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), lat_grid, lon_grid)
        x = x + self.mlp(self.norm2(x))
        return x


class WeatherFlowFoundation(nn.Module):
    """Large-scale axial-ViT backbone for WeatherFlow."""

    def __init__(self, config: WeatherFlowFoundationConfig):
        super().__init__()
        self.config = config
        if config.input_channels % config.levels != 0:
            raise ValueError("input_channels must be divisible by levels.")
        self.patch_embed = PatchEmbed(config.input_channels, config.embed_dim, config.patch_size)
        self.static_proj = (
            PatchEmbed(config.static_channels, config.embed_dim, config.patch_size)
            if config.static_channels > 0
            else None
        )
        self.forcing_proj = nn.Linear(config.forcing_dim, config.embed_dim) if config.forcing_dim > 0 else None
        self.time_encoder = TimeEncoder(config.embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.input_channels // config.levels)
        self.sphere = Sphere() if config.sphere_regularization else None

    def _get_grids(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = torch.linspace(-math.pi / 2, math.pi / 2, steps=h, device=device, dtype=dtype)
        lon = torch.linspace(-math.pi, math.pi, steps=w, device=device, dtype=dtype)
        return lat, lon

    def _apply_blocks(self, x: torch.Tensor, lat_grid: torch.Tensor, lon_grid: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.config.use_grad_checkpointing:
                x = checkpoint(block, x, lat_grid, lon_grid, use_reentrant=False)
            else:
                x = block(x, lat_grid, lon_grid)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        levels = self.config.levels
        tokens = _split_levels(x, levels)  # [B, L, C_per_level, H, W]
        tokens = tokens.reshape(b * levels, c // levels, h, w)
        patch_tokens = self.patch_embed(tokens)  # [B*L, D, H', W']

        if static is not None and self.static_proj is not None:
            static_tokens = _split_levels(static, levels).reshape(b * levels, static.shape[1] // levels, h, w)
            patch_tokens = patch_tokens + self.static_proj(static_tokens)

        _, _, h_patch, w_patch = patch_tokens.shape
        patch_tokens = patch_tokens.view(b, levels, self.config.embed_dim, h_patch, w_patch)
        patch_tokens = patch_tokens.permute(0, 1, 3, 4, 2)  # [B, L, H', W', D]

        time_embed = self.time_encoder(t).view(b, 1, 1, 1, -1)
        patch_tokens = patch_tokens + time_embed
        if forcing is not None and self.forcing_proj is not None:
            patch_tokens = patch_tokens + self.forcing_proj(forcing).view(b, 1, 1, 1, -1)

        lat_grid, lon_grid = self._get_grids(h_patch, w_patch, x.device, x.dtype)
        hidden = self._apply_blocks(patch_tokens, lat_grid, lon_grid)
        hidden = self.norm(hidden)
        channels_per_level = c // levels

        token_grid = hidden.view(b, levels, h_patch, w_patch, self.config.embed_dim)
        token_grid = token_grid.reshape(b * levels, h_patch * w_patch, self.config.embed_dim)
        token_grid = self.head(token_grid)
        token_grid = token_grid.view(b * levels, h_patch, w_patch, channels_per_level).permute(0, 3, 1, 2)
        logits = F.interpolate(token_grid, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits.view(b, levels * channels_per_level, h, w)

        if self.sphere is not None and logits.shape[1] >= 2:
            u = logits[:, 0:1]
            v_comp = logits[:, 1:2]
            div = self._spherical_divergence(u, v_comp)
            logits = logits.clone()
            logits[:, 0:1] = u - 0.1 * torch.gradient(div, dim=3)[0]
            logits[:, 1:2] = v_comp - 0.1 * torch.gradient(div, dim=2)[0]
        return logits

    def _spherical_divergence(self, u: torch.Tensor, v_comp: torch.Tensor) -> torch.Tensor:
        if self.sphere is None:
            raise RuntimeError("Sphere regularization is disabled.")
        _, _, lat_size, lon_size = u.shape
        lat_grid = torch.linspace(-math.pi / 2, math.pi / 2, steps=lat_size, device=u.device, dtype=u.dtype)
        eps = self.sphere._get_eps(u.dtype)
        cos_lat = torch.cos(lat_grid).clamp(min=eps).view(1, 1, lat_size, 1)
        dlon = (2 * math.pi) / max(lon_size, 1)
        dphi = math.pi / max(lat_size - 1, 1)
        du_dlambda = torch.gradient(u, spacing=(dlon,), dim=(3,))[0]
        dvcos_dphi = torch.gradient(v_comp * cos_lat, spacing=(dphi,), dim=(2,))[0]
        radius = torch.tensor(self.sphere.radius, device=u.device, dtype=u.dtype)
        return (du_dlambda / (radius * cos_lat)) + (dvcos_dphi / radius)

    @classmethod
    def from_small_model(cls, small_model: WeatherFlowMatch, levels: Optional[int] = None, **overrides: object) -> "WeatherFlowFoundation":
        """Initialize a foundation model using weights from a small WeatherFlowMatch."""
        config = WeatherFlowFoundationConfig(
            input_channels=small_model.input_channels,
            levels=levels or small_model.grid_size[0] if hasattr(small_model, "grid_size") else overrides.get("levels", 1),  # type: ignore[arg-type]
            embed_dim=overrides.get("embed_dim", 768),  # type: ignore[arg-type]
            depth=overrides.get("depth", 12),  # type: ignore[arg-type]
            num_heads=overrides.get("num_heads", 12),  # type: ignore[arg-type]
            patch_size=overrides.get("patch_size", 4),  # type: ignore[arg-type]
            mlp_ratio=overrides.get("mlp_ratio", 4.0),  # type: ignore[arg-type]
            static_channels=getattr(small_model, "static_channels", 0),
            forcing_dim=getattr(small_model, "forcing_dim", 0),
            use_grad_checkpointing=overrides.get("use_grad_checkpointing", True),  # type: ignore[arg-type]
            sphere_regularization=getattr(small_model, "physics_informed", False),
        )
        foundation = cls(config)
        small_state = small_model.state_dict()
        foundation_state = foundation.state_dict()
        compatible = {
            k: v for k, v in small_state.items() if k in foundation_state and foundation_state[k].shape == v.shape
        }
        foundation_state.update(compatible)
        foundation.load_state_dict(foundation_state)
        return foundation


__all__ = [
    "WeatherFlowFoundation",
    "WeatherFlowFoundationConfig",
]

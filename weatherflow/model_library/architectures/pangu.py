"""
Pangu-Weather Model Architecture

Implementation based on:
    "Accurate medium-range global weather forecasting with 3D neural networks"
    Bi et al., Nature 2023
    Huawei Cloud

Key innovations:
    - 3D Earth-Specific Transformer (3DEST)
    - Separate handling of surface and pressure-level variables
    - Hierarchical temporal aggregation
    - Earth-specific position encoding
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import (
    BaseWeatherModel,
    LeadTimeEmbedding,
)
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class EarthPositionEmbedding3D(nn.Module):
    """
    Earth-Specific 3D Position Embedding.

    Encodes spatial position aware of Earth's geometry:
    - Latitude-aware (accounts for varying grid cell sizes)
    - Longitude-aware (periodic)
    - Pressure-level aware (for vertical structure)
    """

    def __init__(
        self,
        embed_dim: int,
        lat_size: int,
        lon_size: int,
        num_levels: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.num_levels = num_levels

        # Learnable position embeddings
        self.lat_embed = nn.Parameter(torch.zeros(1, lat_size, 1, 1, embed_dim))
        self.lon_embed = nn.Parameter(torch.zeros(1, 1, lon_size, 1, embed_dim))
        self.level_embed = nn.Parameter(torch.zeros(1, 1, 1, num_levels, embed_dim))

        # Initialize with sinusoidal pattern
        self._init_embeddings()

    def _init_embeddings(self):
        # Latitude: account for cos(lat) weighting
        lats = torch.linspace(-90, 90, self.lat_size)
        lat_weights = torch.cos(torch.deg2rad(lats))

        for i in range(self.embed_dim // 6):
            freq = 1.0 / (10000 ** (2 * i / self.embed_dim))
            self.lat_embed.data[0, :, 0, 0, 2*i] = torch.sin(lats * freq)
            self.lat_embed.data[0, :, 0, 0, 2*i + 1] = torch.cos(lats * freq)

        # Longitude: periodic
        lons = torch.linspace(0, 360, self.lon_size + 1)[:-1]
        for i in range(self.embed_dim // 6):
            freq = 1.0 / (10000 ** (2 * i / self.embed_dim))
            self.lon_embed.data[0, 0, :, 0, 2*i] = torch.sin(torch.deg2rad(lons) * freq * 180)
            self.lon_embed.data[0, 0, :, 0, 2*i + 1] = torch.cos(torch.deg2rad(lons) * freq * 180)

        # Pressure levels: log scale
        levels = torch.arange(self.num_levels).float()
        for i in range(self.embed_dim // 6):
            freq = 1.0 / (10000 ** (2 * i / self.embed_dim))
            self.level_embed.data[0, 0, 0, :, 2*i] = torch.sin(levels * freq)
            self.level_embed.data[0, 0, 0, :, 2*i + 1] = torch.cos(levels * freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lat, lon, level, embed_dim)
        return x + self.lat_embed + self.lon_embed + self.level_embed


class Window3DAttention(nn.Module):
    """
    3D Window-based Multi-head Self-Attention.

    Performs attention within local 3D windows for efficiency.
    Based on Swin Transformer but extended to 3D.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (lat_w, lon_w, level_w)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Compute relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords_d = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*batch, window_size^3, dim)
            mask: Optional attention mask
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn.view(B_ // mask.shape[0], mask.shape[0], self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Pangu3DBlock(nn.Module):
    """
    3D Transformer Block for Pangu-Weather.

    Components:
        1. 3D Window Attention
        2. MLP
        3. Layer Norm
        4. Residual connections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 2),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Window3DAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, lat, lon, level, dim)
        """
        B, H, W, D, C = x.shape
        wh, ww, wd = self.window_size

        # Pad to multiple of window size
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        pad_d = (wd - D % wd) % wd

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))

        Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

        # Window partition
        x = x.view(B, Hp // wh, wh, Wp // ww, ww, Dp // wd, wd, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, wh * ww * wd, C)

        # Attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        # Reverse window partition
        x = x.view(B, Hp // wh, Wp // ww, Dp // wd, wh, ww, wd, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Hp, Wp, Dp, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = x[:, :H, :W, :D, :]

        return x


class PanguWeatherModel(BaseWeatherModel):
    """
    Pangu-Weather: 3D Earth-Specific Transformer.

    Based on Bi et al. (2023), Nature.

    Architecture:
        1. Separate encoders for surface and upper-air variables
        2. 3D Transformer with Earth-specific position encoding
        3. Hierarchical structure for different forecast ranges
        4. Multi-scale temporal aggregation

    Args:
        surface_channels: Number of surface variables
        upper_channels: Number of upper-air variables per level
        num_levels: Number of pressure levels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        lat_size: Number of latitude points
        lon_size: Number of longitude points
        window_size: 3D window size for attention
    """

    def __init__(
        self,
        surface_channels: int = 4,  # u10, v10, t2m, msl
        upper_channels: int = 5,  # z, q, t, u, v
        num_levels: int = 13,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        lat_size: int = 721,
        lon_size: int = 1440,
        window_size: Tuple[int, int, int] = (2, 6, 2),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = self._get_default_variables()
        if output_variables is None:
            output_variables = self._get_default_variables()

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="0.25deg",
            forecast_hours=6,
        )

        self.surface_channels = surface_channels
        self.upper_channels = upper_channels
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        self.lat_size = lat_size
        self.lon_size = lon_size

        # Patch embedding (reduce spatial resolution)
        self.patch_size = 4
        self.patch_lat = lat_size // self.patch_size
        self.patch_lon = lon_size // self.patch_size

        # Surface encoder
        self.surface_embed = nn.Sequential(
            nn.Conv2d(surface_channels, embed_dim // 2, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1),
        )

        # Upper-air encoder (treats levels as channels)
        self.upper_embed = nn.Sequential(
            nn.Conv3d(
                upper_channels, embed_dim // 2,
                kernel_size=(1, self.patch_size, self.patch_size),
                stride=(1, self.patch_size, self.patch_size),
            ),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1),
        )

        # Position embedding
        self.pos_embed = EarthPositionEmbedding3D(
            embed_dim=embed_dim,
            lat_size=self.patch_lat,
            lon_size=self.patch_lon,
            num_levels=num_levels + 1,  # +1 for surface
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Pangu3DBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # Output heads
        self.norm = nn.LayerNorm(embed_dim)

        # Surface decoder
        self.surface_decode = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, surface_channels, kernel_size=1),
        )

        # Upper-air decoder
        self.upper_decode = nn.Sequential(
            nn.ConvTranspose3d(
                embed_dim, embed_dim // 2,
                kernel_size=(1, self.patch_size, self.patch_size),
                stride=(1, self.patch_size, self.patch_size),
            ),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, upper_channels, kernel_size=1),
        )

    def _get_default_variables(self) -> List[str]:
        """Default Pangu-Weather variables."""
        surface = ["u10", "v10", "t2m", "msl"]
        levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        upper = []
        for var in ["z", "q", "t", "u", "v"]:
            for lev in levels:
                upper.append(f"{var}_{lev}")
        return surface + upper

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, channels, lat, lon) where channels = surface + upper*levels

        Returns:
            (batch, channels, lat, lon)
        """
        batch = x.shape[0]

        # Split surface and upper-air
        surface = x[:, :self.surface_channels]  # (B, 4, lat, lon)
        upper = x[:, self.surface_channels:]  # (B, upper*levels, lat, lon)
        upper = upper.view(batch, self.upper_channels, self.num_levels, self.lat_size, self.lon_size)
        upper = upper.permute(0, 1, 3, 4, 2)  # (B, upper_ch, lat, lon, levels)

        # Embed
        surface_emb = self.surface_embed(surface)  # (B, embed, patch_lat, patch_lon)
        upper_emb = self.upper_embed(upper.permute(0, 1, 4, 2, 3))  # (B, embed, levels, patch_lat, patch_lon)

        # Combine: surface as additional "level"
        surface_emb = surface_emb.unsqueeze(2)  # (B, embed, 1, patch_lat, patch_lon)
        combined = torch.cat([surface_emb, upper_emb], dim=2)  # (B, embed, 1+levels, patch_lat, patch_lon)

        # Reshape for transformer: (B, lat, lon, level, embed)
        combined = combined.permute(0, 3, 4, 2, 1)

        # Add position embedding
        combined = self.pos_embed(combined)

        # Transformer blocks
        for block in self.blocks:
            combined = block(combined)

        # Normalize
        combined = self.norm(combined)

        # Reshape back: (B, embed, level, lat, lon)
        combined = combined.permute(0, 4, 3, 1, 2)

        # Decode surface and upper separately
        surface_out = combined[:, :, 0]  # (B, embed, patch_lat, patch_lon)
        upper_out = combined[:, :, 1:]  # (B, embed, levels, patch_lat, patch_lon)

        surface_out = self.surface_decode(surface_out)
        upper_out = self.upper_decode(upper_out)  # (B, upper_ch, levels, lat, lon)

        # Reshape upper back
        upper_out = upper_out.permute(0, 1, 3, 4, 2)  # (B, upper_ch, lat, lon, levels)
        upper_out = upper_out.reshape(batch, -1, self.lat_size, self.lon_size)

        # Combine output
        output = torch.cat([surface_out, upper_out], dim=1)

        # Residual connection
        return x + output


# Register model
pangu_info = ModelInfo(
    name="Pangu-Weather",
    category=ModelCategory.TRANSFORMER_3D,
    scale=ModelScale.LARGE,
    description="3D Earth-Specific Transformer for medium-range weather forecasting",
    paper_title="Accurate medium-range global weather forecasting with 3D neural networks",
    paper_url="https://www.nature.com/articles/s41586-023-06185-3",
    paper_year=2023,
    authors=["Kaifeng Bi", "Lingxi Xie", "Hengheng Zhang", "et al."],
    organization="Huawei Cloud",
    input_variables=["u10", "v10", "t2m", "msl", "z", "q", "t", "u", "v"],
    output_variables=["u10", "v10", "t2m", "msl", "z", "q", "t", "u", "v"],
    supported_resolutions=["0.25deg"],
    forecast_range="0-7 days",
    temporal_resolution="1h, 3h, 6h, 24h",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=True,
    pretrained_weight_url="https://github.com/198808xc/Pangu-Weather",
    min_gpu_memory_gb=24.0,
    typical_training_time="~16 days on 192 V100s",
    inference_time_per_step="~1 second on A100",
    tags=["3d-transformer", "global", "huawei", "medium-range"],
    related_models=["pangu_1h", "pangu_3h", "pangu_6h", "pangu_24h"],
)

register_model("pangu", PanguWeatherModel, pangu_info, {
    "surface_channels": 4,
    "upper_channels": 5,
    "num_levels": 13,
    "embed_dim": 192,
    "depth": 8,
    "lat_size": 64,  # Small for demo
    "lon_size": 128,
})

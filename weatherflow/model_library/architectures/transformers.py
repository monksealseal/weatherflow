"""
Transformer Architectures for Weather Prediction

Collection of transformer-based architectures adapted for weather:
    - Vision Transformer (ViT)
    - Swin Transformer 3D
    - Earth-Specific Transformer
    - Axial Attention Transformer
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: int = 4,
        in_channels: int = 20,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias, attn_drop, drop
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) adapted for weather prediction.

    Based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    adapted for weather data.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: int = 4,
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels * patch_size * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = self.img_size
        P = self.patch_size

        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Decode to output
        x = self.decoder(x)

        # Reshape to image
        x = x.reshape(B, H // P, W // P, self.out_channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, self.out_channels, H, W)

        return x


class WindowAttention3D(nn.Module):
    """3D window attention for Swin Transformer."""

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
        self.window_size = window_size
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer block for 3D data."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Pad to multiple of window size
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )

        # Window partition
        x = x.view(
            B,
            Dp // self.window_size[0], self.window_size[0],
            Hp // self.window_size[1], self.window_size[1],
            Wp // self.window_size[2], self.window_size[2],
            C
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # Window attention
        x = self.attn(x)

        # Reverse window partition
        x = x.view(
            B,
            Dp // self.window_size[0],
            Hp // self.window_size[1],
            Wp // self.window_size[2],
            self.window_size[0],
            self.window_size[1],
            self.window_size[2],
            C
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Dp, Hp, Wp, C)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )

        # Remove padding
        x = x[:, :D, :H, :W, :]

        # Residual and MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinTransformer3D(nn.Module):
    """
    Swin Transformer for 3D weather data.

    Adapts the Swin Transformer architecture for
    atmospheric data with vertical levels.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        num_levels: int = 13,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (2, 7, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding for 3D
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=(0, 0, 0) if j % 2 == 0 else tuple(w // 2 for w in window_size),
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                )
                for j in range(depths[i_layer])
            ])
            self.layers.append(layer)

            # Downsampling (except last layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(
                    nn.Conv3d(
                        int(embed_dim * 2 ** i_layer),
                        int(embed_dim * 2 ** (i_layer + 1)),
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2),
                    )
                )

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                int(embed_dim * 2 ** (self.num_layers - 1)),
                embed_dim,
                kernel_size=(1, 2**(self.num_layers-1), 2**(self.num_layers-1)),
                stride=(1, 2**(self.num_layers-1), 2**(self.num_layers-1)),
            ),
            nn.Conv3d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, embed_dim, D, H', W')
        x = self.pos_drop(x)

        # Transpose to (B, D, H, W, C) for attention
        x = x.permute(0, 2, 3, 4, 1)

        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    x = block(x)
            else:
                # Downsampling
                x = x.permute(0, 4, 1, 2, 3)
                x = layer(x)
                x = x.permute(0, 2, 3, 4, 1)

        x = self.norm(x)

        # Decode
        x = x.permute(0, 4, 1, 2, 3)
        x = self.decoder(x)

        return x


class EarthSpecificTransformer(nn.Module):
    """
    Earth-Specific Transformer with geographical awareness.

    Incorporates latitude-dependent position encodings and
    spherical geometry considerations.
    """

    def __init__(
        self,
        in_channels: int = 20,
        out_channels: int = 20,
        img_size: Tuple[int, int] = (64, 128),
        patch_size: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_h = img_size[0] // patch_size
        grid_w = img_size[1] // patch_size

        # Earth-specific positional encoding
        # Latitude-dependent (accounts for cos(lat) weighting)
        lats = torch.linspace(-90, 90, grid_h)
        lons = torch.linspace(0, 360, grid_w + 1)[:-1]

        lat_emb = torch.zeros(grid_h, embed_dim // 2)
        lon_emb = torch.zeros(grid_w, embed_dim // 2)

        for i in range(embed_dim // 4):
            freq = 1.0 / (10000 ** (2 * i / embed_dim))
            lat_emb[:, 2*i] = torch.sin(lats * freq)
            lat_emb[:, 2*i + 1] = torch.cos(lats * freq)
            lon_emb[:, 2*i] = torch.sin(torch.deg2rad(lons) * freq * 180)
            lon_emb[:, 2*i + 1] = torch.cos(torch.deg2rad(lons) * freq * 180)

        # Combine lat/lon embeddings
        pos_embed = torch.zeros(1, num_patches, embed_dim)
        for i in range(grid_h):
            for j in range(grid_w):
                idx = i * grid_w + j
                pos_embed[0, idx, :embed_dim//2] = lat_emb[i]
                pos_embed[0, idx, embed_dim//2:] = lon_emb[j]

        self.register_buffer("pos_embed", pos_embed)

        # Latitude weighting for attention
        lat_weights = torch.cos(torch.deg2rad(lats)).unsqueeze(1).repeat(1, grid_w).flatten()
        self.register_buffer("lat_weights", lat_weights)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels * patch_size * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = self.img_size
        P = self.patch_size

        # Patch embedding with earth-specific position
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # Apply latitude weighting
        x = x * self.lat_weights.unsqueeze(0).unsqueeze(-1)

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Decode
        x = self.decoder(x)

        # Reshape to image
        x = x.reshape(B, H // P, W // P, self.out_channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, self.out_channels, H, W)

        return x

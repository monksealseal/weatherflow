"""
Encoder Components for Weather AI Models

Encoders transform input weather data into latent representations.
Each encoder handles different input formats and spatial structures.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, L, H, W] for 3D

        Returns:
            Encoded tensor of shape [B, embed_dim, ...] or [B, N, embed_dim]
        """
        pass

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return output shape given input shape."""
        raise NotImplementedError


class CNNEncoder(BaseEncoder):
    """
    Convolutional encoder with configurable depth and downsampling.

    Good for: Flow matching, simple transformers, UNet-style models
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depths: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        downsample: bool = True,
        norm: str = "batch",
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.depths = depths
        self.downsample = downsample

        layers = []
        in_ch = in_channels

        for i, out_ch in enumerate(depths):
            # Convolution
            stride = 2 if (downsample and i > 0) else 1
            padding = kernel_size // 2
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))

            # Normalization
            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == "instance":
                layers.append(nn.InstanceNorm2d(out_ch))
            elif norm == "group":
                layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
            elif norm == "layer":
                layers.append(nn.GroupNorm(1, out_ch))  # LayerNorm equivalent

            # Activation
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_ch = out_ch

        # Final projection to embed_dim
        layers.append(nn.Conv2d(depths[-1], embed_dim, 1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViTEncoder(BaseEncoder):
    """
    Vision Transformer encoder with patch embedding.

    Good for: FourCastNet, ClimaX, general ViT-based models
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        img_size: Tuple[int, int] = (721, 1440),
        patch_size: int = 4,
        norm: str = "layer",
        flatten: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten

        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding via conv
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Normalization
        if norm == "layer":
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.Identity()

        # CLS token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding [B, C, H, W] -> [B, embed_dim, H', W']
        x = self.proj(x)

        if self.flatten:
            # Flatten patches [B, embed_dim, H', W'] -> [B, N, embed_dim]
            x = x.flatten(2).transpose(1, 2)

            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add position embedding
            x = x + self.pos_embed[:, :x.size(1), :]

            # Normalize
            x = self.norm(x)

        return x

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.flatten:
            return (self.num_patches + 1, self.embed_dim)
        return (self.embed_dim, self.num_patches_h, self.num_patches_w)


class GraphEncoder(BaseEncoder):
    """
    Graph Neural Network encoder for mesh-based representations.

    Good for: GraphCast, mesh-based weather models
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_nodes: int = 40962,  # Icosahedral mesh resolution
        hidden_dim: int = 256,
        num_layers: int = 2,
        edge_dim: int = 3,  # lat, lon, distance
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Initial message passing layers
        self.mp_layers = nn.ModuleList([
            GraphConvLayer(embed_dim, embed_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, C] or grid [B, C, H, W]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
        """
        # Handle grid input - flatten to nodes
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]

        # Encode nodes
        x = self.node_encoder(x)

        # Message passing (simplified without edge_index for basic use)
        for layer in self.mp_layers:
            x = layer(x)

        return x


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: just transform and normalize
        # In practice, would use message passing with edges
        return self.norm(self.linear(x) + x)


class IcosahedralEncoder(BaseEncoder):
    """
    Icosahedral mesh encoder (like GAIA).

    Projects lat-lon grid to icosahedral mesh using KNN attention.
    Good for: GAIA, spherical weather models
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        mesh_resolution: int = 5,  # Icosahedral subdivision level
        knn_k: int = 8,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.mesh_resolution = mesh_resolution
        self.knn_k = knn_k

        # Compute number of mesh vertices
        self.num_vertices = 10 * (4 ** mesh_resolution) + 2

        # Input projection
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # KNN attention for grid-to-mesh projection
        self.knn_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Mesh vertex positions (precomputed icosahedral grid)
        self.register_buffer(
            "mesh_vertices",
            self._generate_icosahedral_mesh(mesh_resolution)
        )

        # Learnable mesh embeddings
        self.mesh_embed = nn.Parameter(torch.randn(1, self.num_vertices, embed_dim) * 0.02)

    def _generate_icosahedral_mesh(self, resolution: int) -> torch.Tensor:
        """Generate icosahedral mesh vertex positions."""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2

        # Base icosahedron vertices
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        vertices = torch.tensor(vertices, dtype=torch.float32)

        # Normalize to unit sphere
        vertices = F.normalize(vertices, dim=1)

        # For simplicity, just return base vertices
        # In practice, would subdivide for higher resolution
        # Pad to expected size
        if vertices.size(0) < self.num_vertices:
            # Generate additional points by subdivision (simplified)
            additional = torch.randn(self.num_vertices - vertices.size(0), 3)
            additional = F.normalize(additional, dim=1)
            vertices = torch.cat([vertices, additional], dim=0)

        return vertices[:self.num_vertices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project grid to icosahedral mesh.

        Args:
            x: Grid input [B, C, H, W]

        Returns:
            Mesh features [B, num_vertices, embed_dim]
        """
        B, C, H, W = x.shape

        # Flatten grid to sequence [B, H*W, C]
        x_flat = x.view(B, C, -1).transpose(1, 2)

        # Project to embed_dim
        x_flat = self.input_proj(x_flat)

        # KNN attention: mesh queries, grid keys/values
        mesh_queries = self.mesh_embed.expand(B, -1, -1)
        x_mesh, _ = self.knn_attention(mesh_queries, x_flat, x_flat)

        return x_mesh


class FourierEncoder(BaseEncoder):
    """
    Fourier/Spectral encoder using FFT.

    Good for: FourCastNet (AFNO), spectral methods
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        img_size: Tuple[int, int] = (721, 1440),
        num_blocks: int = 4,
        sparsity_threshold: float = 0.01,
        hard_threshold: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.img_size = img_size
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_threshold = hard_threshold

        h, w = img_size
        self.num_freq_h = h // 2 + 1
        self.num_freq_w = w // 2 + 1

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 1)

        # Learnable Fourier weights (complex-valued)
        self.fourier_weight_real = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim, self.num_freq_h, self.num_freq_w) * 0.02)
            for _ in range(num_blocks)
        ])
        self.fourier_weight_imag = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim, self.num_freq_h, self.num_freq_w) * 0.02)
            for _ in range(num_blocks)
        ])

        # MLP for mixing
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
        )

        self.norm = nn.LayerNorm([embed_dim, img_size[0], img_size[1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]

        Returns:
            Fourier-encoded features [B, embed_dim, H, W]
        """
        # Project to embed_dim
        x = self.input_proj(x)

        # Fourier blocks
        for i in range(self.num_blocks):
            residual = x

            # FFT
            x_fft = torch.fft.rfft2(x, norm="ortho")

            # Complex multiplication with learnable weights
            weight = torch.complex(
                self.fourier_weight_real[i],
                self.fourier_weight_imag[i]
            )

            # Truncate to match
            h_freq = min(x_fft.size(2), weight.size(2))
            w_freq = min(x_fft.size(3), weight.size(3))

            x_fft_trunc = x_fft[:, :, :h_freq, :w_freq]
            weight_trunc = weight[:, :, :h_freq, :w_freq]

            # Apply weights via einsum
            out_fft = torch.einsum("bcij,dcij->bdij", x_fft_trunc, weight_trunc)

            # Inverse FFT
            x_spatial = torch.fft.irfft2(out_fft, s=(x.size(2), x.size(3)), norm="ortho")

            # Add residual and normalize
            x = residual + x_spatial

        # Final MLP mixing
        x = x + self.mlp(x)

        return x


class SwinEncoder(BaseEncoder):
    """
    Swin Transformer encoder with shifted window attention.

    Good for: Pangu-Weather style 3D models, hierarchical transformers
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        img_size: Tuple[int, int] = (721, 1440),
        patch_size: int = 4,
        window_size: int = 7,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        **kwargs,
    ):
        super().__init__(in_channels, embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.num_patches_h, self.num_patches_w)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Swin blocks (simplified)
        self.blocks = nn.ModuleList()
        dim = embed_dim
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            for j in range(depth):
                shift = (j % 2) * (window_size // 2)
                self.blocks.append(
                    SwinBlock(dim, heads, window_size, shift)
                )

            # Downsample between stages (except last)
            if i < len(depths) - 1:
                self.blocks.append(PatchMerging(dim))
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.final_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # Apply blocks
        for block in self.blocks:
            x = block(x)

        return x


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Reshape for attention [B, H*W, C]
        x = x.view(B, C, -1).transpose(1, 2)

        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        # Reshape back [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class PatchMerging(nn.Module):
    """Patch merging for downsampling."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Reshape to merge 2x2 patches
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)

        # Reduce
        x = self.norm(x)
        x = self.reduction(x)

        # Reshape back
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

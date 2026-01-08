"""
Decoder Components for Weather AI Models

Decoders transform latent representations back to weather fields.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for all decoders."""

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation.

        Args:
            x: Latent tensor

        Returns:
            Decoded output [B, out_channels, H, W]
        """
        pass


class GridDecoder(BaseDecoder):
    """
    Decode to regular lat-lon grid.

    Good for: Most weather models that output on regular grids
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        hidden_dims: list = [512, 256, 128],
        output_size: Tuple[int, int] = (721, 1440),
        upsample_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)
        self.output_size = output_size
        self.upsample_mode = upsample_mode

        layers = []
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Conv2d(in_dim, out_channels, 1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle sequence input [B, N, C]
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, H, W)

        # Decode
        x = self.decoder(x)

        # Upsample to output size if needed
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(
                x, size=self.output_size,
                mode=self.upsample_mode, align_corners=False
            )

        return x


class PatchDecoder(BaseDecoder):
    """
    Reverse patch embedding decoder (for ViT models).

    Good for: FourCastNet, ClimaX, ViT-based models
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        img_size: Tuple[int, int] = (721, 1440),
        patch_size: int = 4,
        has_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)
        self.img_size = img_size
        self.patch_size = patch_size
        self.has_cls_token = has_cls_token

        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size

        # Project to patch output
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # Optional refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] where N = num_patches (+1 for cls token)
        B = x.shape[0]

        # Remove CLS token if present
        if self.has_cls_token and x.shape[1] > self.num_patches_h * self.num_patches_w:
            x = x[:, 1:]  # Remove first token

        # Project to patch pixels
        x = self.proj(x)  # [B, N, patch_size^2 * out_channels]

        # Reshape to image
        x = x.view(
            B, self.num_patches_h, self.num_patches_w,
            self.patch_size, self.patch_size, self.out_channels
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.img_size[0], self.img_size[1])

        # Refine
        x = x + self.refine(x)

        return x


class IcosahedralDecoder(BaseDecoder):
    """
    Decode from icosahedral mesh to lat-lon grid.

    Good for: GAIA, spherical mesh models
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        mesh_resolution: int = 5,
        output_size: Tuple[int, int] = (721, 1440),
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)
        self.mesh_resolution = mesh_resolution
        self.output_size = output_size

        self.num_vertices = 10 * (4 ** mesh_resolution) + 2
        self.num_output_points = output_size[0] * output_size[1]

        # Cross-attention from grid queries to mesh keys
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Grid position queries
        self.grid_queries = nn.Parameter(
            torch.randn(1, self.num_output_points, embed_dim) * 0.02
        )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mesh features [B, num_vertices, embed_dim]

        Returns:
            Grid output [B, out_channels, H, W]
        """
        B = x.shape[0]

        # Expand grid queries
        queries = self.grid_queries.expand(B, -1, -1)

        # Cross-attention: grid queries attend to mesh
        output, _ = self.cross_attn(queries, x, x)

        # Project to output channels
        output = self.output_proj(output)

        # Reshape to grid
        output = output.view(B, self.output_size[0], self.output_size[1], self.out_channels)
        output = output.permute(0, 3, 1, 2)

        return output


class ConvDecoder(BaseDecoder):
    """
    Convolutional decoder with progressive upsampling.

    Good for: UNet-style models, GANs
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        depths: list = [512, 256, 128, 64],
        output_size: Optional[Tuple[int, int]] = None,
        norm: str = "batch",
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)
        self.output_size = output_size

        layers = []
        in_ch = embed_dim

        for i, out_ch in enumerate(depths):
            # Upsample
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

            # Conv block
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))

            # Normalization
            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == "instance":
                layers.append(nn.InstanceNorm2d(out_ch))
            elif norm == "group":
                layers.append(nn.GroupNorm(min(32, out_ch), out_ch))

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

        # Final projection
        layers.append(nn.Conv2d(depths[-1], out_channels, 1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)

        if self.output_size is not None and x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        return x


class DiffusionDecoder(BaseDecoder):
    """
    Decoder for diffusion models with noise prediction head.

    Good for: GenCast, diffusion-based weather models
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)

        self.norm = nn.GroupNorm(32, embed_dim)
        self.act = nn.SiLU()

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle sequence input
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, H, W)

        x = self.norm(x)
        x = self.act(x)
        x = self.head(x)

        return x


class HybridDecoder(BaseDecoder):
    """
    Hybrid decoder combining multiple strategies.

    Good for: Complex models requiring multi-scale outputs
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        output_size: Tuple[int, int] = (721, 1440),
        use_attention: bool = True,
        use_conv: bool = True,
        **kwargs,
    ):
        super().__init__(embed_dim, out_channels)
        self.output_size = output_size

        # Attention-based path
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
            self.attn_norm = nn.LayerNorm(embed_dim)

        # Conv-based path
        self.use_conv = use_conv
        if use_conv:
            self.conv_path = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.GroupNorm(32, embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            )

        # Fusion and output
        self.fusion = nn.Conv2d(embed_dim * (use_attention + use_conv), embed_dim, 1)
        self.output_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle sequence input
        is_sequence = x.dim() == 3
        if is_sequence:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x_spatial = x.transpose(1, 2).view(B, C, H, W)
        else:
            B, C, H, W = x.shape
            x_spatial = x

        outputs = []

        # Attention path
        if self.use_attention:
            x_flat = x_spatial.view(B, C, -1).transpose(1, 2)
            x_attn = self.attn_norm(x_flat)
            x_attn, _ = self.attn(x_attn, x_attn, x_attn)
            x_attn = x_attn.transpose(1, 2).view(B, C, H, W)
            outputs.append(x_attn)

        # Conv path
        if self.use_conv:
            x_conv = self.conv_path(x_spatial)
            outputs.append(x_conv)

        # Fuse
        if len(outputs) > 1:
            x = torch.cat(outputs, dim=1)
            x = self.fusion(x)
        else:
            x = outputs[0]

        # Output
        x = self.output_head(x)

        # Resize if needed
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        return x

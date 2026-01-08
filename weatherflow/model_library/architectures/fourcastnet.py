"""
FourCastNet Model Architecture

Implementation based on:
    "FourCastNet: A Global Data-driven High-resolution Weather Model
     using Adaptive Fourier Neural Operators"
    Pathak et al., 2022
    NVIDIA

Key innovations:
    - Adaptive Fourier Neural Operator (AFNO) blocks
    - Spectral convolutions in Fourier space
    - Efficient global attention via FFT
    - Vision Transformer backbone with AFNO replacing attention
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import (
    BaseWeatherModel,
    SphericalPadding,
    PositionalEncoding2D,
)
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution via FFT.

    Performs convolution in Fourier space for efficient global operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (height)
        self.modes2 = modes2  # Number of Fourier modes to keep (width)

        # Learnable weights for spectral convolution
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication in 2D."""
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h, w = x.shape[2], x.shape[3]

        # Transform to Fourier space
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Lower modes
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )

        # Upper modes (negative frequencies)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # Transform back to physical space
        return torch.fft.irfft2(out_ft, s=(h, w))


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator Block.

    Replaces standard attention in Vision Transformer with
    efficient spectral attention via FFT.

    Components:
        1. Layer Norm
        2. AFNO (token mixing in Fourier space)
        3. Layer Norm
        4. MLP (channel mixing)
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        h: int = 64,
        w: int = 128,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.h = h
        self.w = w
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # AFNO components
        self.block_size = dim // num_blocks
        assert dim % num_blocks == 0

        # Learnable spectral weights (real and imaginary parts)
        self.scale = 0.02
        self.w1 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size)
        )

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Softmax for sparsity
        self.softshrink = nn.Softshrink(self.sparsity_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, h*w, dim)
        batch = x.shape[0]

        # Reshape to spatial
        x_spatial = x.reshape(batch, self.h, self.w, self.dim)

        # AFNO block
        residual = x_spatial
        x_spatial = self.norm1(x_spatial)

        # Move channels to front
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)

        # FFT
        x_fft = torch.fft.rfft2(x_spatial, norm="ortho")
        x_fft = x_fft.reshape(batch, self.num_blocks, self.block_size, x_fft.shape[2], x_fft.shape[3])

        # Apply learned weights in frequency domain (complex multiplication)
        o1_real = torch.zeros_like(x_fft.real)
        o1_imag = torch.zeros_like(x_fft.imag)

        for i in range(self.num_blocks):
            o1_real[:, i] = F.relu(
                torch.einsum("bjkl,jm->bmkl", x_fft[:, i].real, self.w1[i]) -
                torch.einsum("bjkl,jm->bmkl", x_fft[:, i].imag, self.w1[i]) +
                self.b1[i].view(1, -1, 1, 1)
            )
            o1_imag[:, i] = F.relu(
                torch.einsum("bjkl,jm->bmkl", x_fft[:, i].imag, self.w1[i]) +
                torch.einsum("bjkl,jm->bmkl", x_fft[:, i].real, self.w1[i]) +
                self.b1[i].view(1, -1, 1, 1)
            )

        o1 = torch.complex(o1_real, o1_imag)

        # Second layer in frequency domain
        o2_real = torch.zeros_like(o1.real)
        o2_imag = torch.zeros_like(o1.imag)

        for i in range(self.num_blocks):
            o2_real[:, i] = (
                torch.einsum("bjkl,jm->bmkl", o1[:, i].real, self.w2[i]) -
                torch.einsum("bjkl,jm->bmkl", o1[:, i].imag, self.w2[i]) +
                self.b2[i].view(1, -1, 1, 1)
            )
            o2_imag[:, i] = (
                torch.einsum("bjkl,jm->bmkl", o1[:, i].imag, self.w2[i]) +
                torch.einsum("bjkl,jm->bmkl", o1[:, i].real, self.w2[i]) +
                self.b2[i].view(1, -1, 1, 1)
            )

        # Apply soft thresholding for sparsity
        o2 = torch.stack([o2_real, o2_imag], dim=-1)
        o2 = self.softshrink(o2)
        o2 = torch.view_as_complex(o2)

        # Reshape and iFFT
        o2 = o2.reshape(batch, self.dim, o2.shape[3], o2.shape[4])
        x_spatial = torch.fft.irfft2(o2, s=(self.h, self.w), norm="ortho")

        # Back to (B, H, W, C)
        x_spatial = x_spatial.permute(0, 2, 3, 1)

        # Residual
        x_spatial = x_spatial + residual

        # MLP block
        residual = x_spatial
        x_spatial = self.norm2(x_spatial)
        x_spatial = self.mlp(x_spatial)
        x_spatial = x_spatial + residual

        # Flatten back
        return x_spatial.reshape(batch, self.h * self.w, self.dim)


class PatchEmbed(nn.Module):
    """Embed patches from image into tokens."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (720, 1440),
        patch_size: Tuple[int, int] = (4, 4),
        in_channels: int = 20,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class FourCastNetModel(BaseWeatherModel):
    """
    FourCastNet: Adaptive Fourier Neural Operator for Weather Forecasting.

    Based on Pathak et al. (2022), NVIDIA.

    Architecture:
        1. Patch Embedding (like ViT)
        2. Stack of AFNO blocks (replacing attention)
        3. Patch Decoding to output grid

    Args:
        img_size: (height, width) of input images
        patch_size: (patch_h, patch_w) for patch embedding
        in_channels: Number of input atmospheric variables
        out_channels: Number of output variables
        embed_dim: Embedding dimension
        depth: Number of AFNO blocks
        num_blocks: Number of blocks for AFNO mixing
        mlp_ratio: MLP expansion ratio
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (720, 1440),
        patch_size: Tuple[int, int] = (4, 4),
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 768,
        depth: int = 12,
        num_blocks: int = 8,
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

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # Grid size after patching
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # AFNO blocks
        self.blocks = nn.ModuleList([
            AFNOBlock(
                dim=embed_dim,
                num_blocks=num_blocks,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                h=self.grid_h,
                w=self.grid_w,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: project back to output channels
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels * patch_size[0] * patch_size[1]),
        )

    def _get_default_variables(self) -> List[str]:
        """Default FourCastNet variables (20 channels)."""
        return [
            "u10", "v10", "t2m", "sp", "msl",  # Surface
            "u_850", "v_850", "z_850", "t_850", "q_850",  # 850 hPa
            "u_500", "v_500", "z_500", "t_500", "q_500",  # 500 hPa
            "u_250", "v_250", "z_250", "t_250", "q_250",  # 250 hPa
        ]

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, in_channels, height, width)
            lead_time: Not used in basic FourCastNet

        Returns:
            (batch, out_channels, height, width)
        """
        batch = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # AFNO blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Decode to output
        x = self.decoder(x)  # (B, num_patches, out_channels * patch_h * patch_w)

        # Reshape to image
        x = x.reshape(
            batch,
            self.grid_h,
            self.grid_w,
            self.out_channels,
            self.patch_size[0],
            self.patch_size[1],
        )
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, grid_h, patch_h, grid_w, patch_w)
        x = x.reshape(batch, self.out_channels, self.img_size[0], self.img_size[1])

        return x


class FourCastNetPrecip(FourCastNetModel):
    """
    FourCastNet variant for precipitation forecasting.

    Uses additional precipitation-specific architecture choices.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (720, 1440),
        patch_size: Tuple[int, int] = (4, 4),
        in_channels: int = 20,
        out_channels: int = 1,  # Just precipitation
        embed_dim: int = 768,
        depth: int = 12,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            **kwargs,
        )

        # Override decoder with precipitation-specific head
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, out_channels * patch_size[0] * patch_size[1]),
            # Softplus to ensure positive precipitation
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output = super().forward(x, **kwargs)
        # Apply softplus to ensure non-negative precipitation
        return F.softplus(output)


# Register models
fourcastnet_info = ModelInfo(
    name="FourCastNet",
    category=ModelCategory.VISION_TRANSFORMER,
    scale=ModelScale.LARGE,
    description="Adaptive Fourier Neural Operator for global weather forecasting",
    paper_title="FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators",
    paper_url="https://arxiv.org/abs/2202.11214",
    paper_year=2022,
    authors=["Jaideep Pathak", "Shashank Subramanian", "Peter Harrington", "et al."],
    organization="NVIDIA",
    input_variables=["u10", "v10", "t2m", "sp", "msl", "z_500", "t_850", "u_500", "v_500"],
    output_variables=["u10", "v10", "t2m", "sp", "msl", "z_500", "t_850", "u_500", "v_500"],
    supported_resolutions=["0.25deg"],
    forecast_range="0-7 days",
    temporal_resolution="6h",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=True,
    pretrained_weight_url="https://github.com/NVlabs/FourCastNet",
    min_gpu_memory_gb=16.0,
    typical_training_time="~16 hours on 64 A100s",
    inference_time_per_step="~2 seconds on A100",
    tags=["afno", "fft", "vit", "global", "nvidia"],
    related_models=["fourcastnet_precip", "sfno"],
)

register_model("fourcastnet", FourCastNetModel, fourcastnet_info, {
    "img_size": (64, 128),  # Small default for demo
    "patch_size": (4, 4),
    "in_channels": 20,
    "out_channels": 20,
    "embed_dim": 256,
    "depth": 8,
})

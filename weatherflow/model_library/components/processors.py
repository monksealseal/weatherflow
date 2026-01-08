"""
Processor Components for Weather AI Models

Processors transform encoded representations through various mechanisms:
- Attention (self-attention, cross-attention)
- Fourier (spectral processing)
- Message Passing (graph-based)
- Convolutional (local processing)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseProcessor(nn.Module, ABC):
    """Abstract base class for all processors."""

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input tensor.

        Args:
            x: Input tensor
            context: Optional context for cross-attention
            time_emb: Optional time embedding

        Returns:
            Processed tensor
        """
        pass


class AttentionProcessor(BaseProcessor):
    """
    Standard multi-head self-attention processor.

    Good for: ViT-based models, ClimaX, general transformers
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        norm: str = "layer",
        **kwargs,
    ):
        super().__init__(dim)
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qkv_bias=qkv_bias,
                norm=norm,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, time_emb=time_emb)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        norm: str = "layer",
    ):
        super().__init__()
        self.dim = dim

        # Normalization
        if norm == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Self-attention
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True, bias=qkv_bias
        )

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        ) if dim > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time conditioning via scale and shift
        if time_emb is not None and self.time_proj is not None:
            time_params = self.time_proj(time_emb)
            if time_params.dim() == 2:
                time_params = time_params.unsqueeze(1)
            scale, shift = time_params.chunk(2, dim=-1)
        else:
            scale, shift = 1.0, 0.0

        # Self-attention with residual
        x_norm = self.norm1(x) * (1 + scale) + shift
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x_norm = self.norm2(x) * (1 + scale) + shift
        x = x + self.mlp(x_norm)

        return x


class AFNOProcessor(BaseProcessor):
    """
    Adaptive Fourier Neural Operator processor.

    From FourCastNet: spectral processing via FFT with learnable filters.
    Good for: FourCastNet, spectral weather models
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_threshold: bool = True,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(dim)
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold

        self.blocks = nn.ModuleList([
            AFNOBlock(dim, mlp_ratio, sparsity_threshold, hard_threshold)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, C, H, W] or [B, N, C]
        is_sequence = x.dim() == 3

        if is_sequence:
            # Assume square image
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, H, W)

        for block in self.blocks:
            x = block(x)

        if is_sequence:
            x = x.view(B, C, -1).transpose(1, 2)

        return x


class AFNOBlock(nn.Module):
    """Single AFNO block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        sparsity_threshold: float = 0.01,
        hard_threshold: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        self.hard_threshold = hard_threshold

        self.norm1 = nn.LayerNorm([dim])
        self.norm2 = nn.LayerNorm([dim])

        # Fourier weights - will be set dynamically based on input size
        self.scale = 0.02

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # Learnable complex weights
        self.w1 = nn.Parameter(torch.randn(2, dim, dim) * self.scale)
        self.w2 = nn.Parameter(torch.randn(2, dim, dim) * self.scale)
        self.b1 = nn.Parameter(torch.zeros(2, 1, 1, dim))
        self.b2 = nn.Parameter(torch.zeros(2, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        residual = x

        # Normalize (reshape for LayerNorm)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm1(x)

        # FFT
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # Complex MLP in frequency domain
        x_real = x_fft.real
        x_imag = x_fft.imag

        # Layer 1
        o1_real = torch.einsum("bhwc,cd->bhwd", x_real, self.w1[0]) - \
                  torch.einsum("bhwc,cd->bhwd", x_imag, self.w1[1])
        o1_imag = torch.einsum("bhwc,cd->bhwd", x_real, self.w1[1]) + \
                  torch.einsum("bhwc,cd->bhwd", x_imag, self.w1[0])
        o1_real = o1_real + self.b1[0]
        o1_imag = o1_imag + self.b1[1]

        # ReLU in complex domain
        o1_real = F.relu(o1_real)
        o1_imag = F.relu(o1_imag)

        # Layer 2
        o2_real = torch.einsum("bhwc,cd->bhwd", o1_real, self.w2[0]) - \
                  torch.einsum("bhwc,cd->bhwd", o1_imag, self.w2[1])
        o2_imag = torch.einsum("bhwc,cd->bhwd", o1_real, self.w2[1]) + \
                  torch.einsum("bhwc,cd->bhwd", o1_imag, self.w2[0])
        o2_real = o2_real + self.b2[0]
        o2_imag = o2_imag + self.b2[1]

        # Sparsity thresholding
        if self.hard_threshold:
            mask = (o2_real.abs() > self.sparsity_threshold) | \
                   (o2_imag.abs() > self.sparsity_threshold)
            o2_real = o2_real * mask
            o2_imag = o2_imag * mask

        # Reconstruct complex
        x_fft_out = torch.complex(o2_real, o2_imag)

        # Inverse FFT
        x = torch.fft.irfft2(x_fft_out, s=(H, W), dim=(1, 2), norm="ortho")

        # Reshape back
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Residual
        x = residual + x

        # MLP block
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_perm = self.norm2(x_perm)
        x_perm = self.mlp(x_perm)
        x = x + x_perm.permute(0, 3, 1, 2)

        return x


class MessagePassingProcessor(BaseProcessor):
    """
    Graph message passing processor.

    Good for: GraphCast, mesh-based models, multi-mesh architectures
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 16,
        edge_dim: int = 3,
        aggregation: str = "mean",
        **kwargs,
    ):
        super().__init__(dim)
        self.num_layers = num_layers
        self.aggregation = aggregation

        self.layers = nn.ModuleList([
            MessagePassingLayer(dim, edge_dim, aggregation)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class MessagePassingLayer(nn.Module):
    """Single message passing layer."""

    def __init__(
        self,
        dim: int,
        edge_dim: int = 3,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.dim = dim
        self.aggregation = aggregation

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, C]
            edge_index: Edge indices [2, E] (optional)
            edge_attr: Edge features [E, edge_dim] (optional)
        """
        residual = x

        # Simplified: self-attention as message passing
        # In practice, would use actual edge structure
        x_norm = self.norm(x)

        # Compute pairwise interactions (simplified)
        B, N, C = x.shape

        # Self-attention-like aggregation
        attn = torch.matmul(x_norm, x_norm.transpose(-1, -2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        messages = torch.matmul(attn, x_norm)

        # Update
        x_cat = torch.cat([x_norm, messages], dim=-1)
        x_update = self.node_mlp(x_cat)

        return residual + x_update


class ConvNextProcessor(BaseProcessor):
    """
    ConvNext-style processor with modern conv blocks.

    Good for: Flow matching, CNN-based weather models
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 6,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(dim)
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            ConvNextBlock(dim, kernel_size, mlp_ratio, drop_path)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle sequence input
        is_sequence = x.dim() == 3
        if is_sequence:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, H, W)

        for block in self.blocks:
            x = block(x, time_emb)

        if is_sequence:
            x = x.view(B, C, -1).transpose(1, 2)

        return x


class ConvNextBlock(nn.Module):
    """ConvNext block."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Depthwise conv
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size,
            padding=kernel_size // 2, groups=dim
        )

        self.norm = nn.LayerNorm(dim)

        # Pointwise MLP
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Layer scale
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Reshape for layer norm
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)

        # Add time embedding
        if time_emb is not None:
            time_cond = self.time_mlp(time_emb)
            if time_cond.dim() == 2:
                time_cond = time_cond.unsqueeze(1).unsqueeze(1)
            x = x + time_cond

        # Pointwise MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        x = self.gamma * x

        # Reshape back
        x = x.permute(0, 3, 1, 2)

        return residual + x


class UNetProcessor(BaseProcessor):
    """
    UNet-style encoder-decoder processor.

    Good for: Diffusion models, image-to-image translation
    """

    def __init__(
        self,
        dim: int,
        dim_mults: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(dim)
        self.dim = dim
        self.dim_mults = dim_mults

        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        for i, (in_ch, out_ch) in enumerate(in_out):
            blocks = nn.ModuleList([
                ResBlock(in_ch if j == 0 else out_ch, out_ch, dropout)
                for j in range(num_res_blocks)
            ])
            self.encoder_blocks.append(blocks)

            if i < len(in_out) - 1:
                self.encoder_downsamples.append(nn.Conv2d(out_ch, out_ch, 3, 2, 1))
            else:
                self.encoder_downsamples.append(nn.Identity())

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = nn.MultiheadAttention(mid_dim, 8, batch_first=True)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, dropout)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for i, (in_ch, out_ch) in enumerate(reversed(in_out)):
            blocks = nn.ModuleList([
                ResBlock(out_ch * 2 if j == 0 else out_ch, out_ch, dropout)
                for j in range(num_res_blocks)
            ])
            self.decoder_blocks.append(blocks)

            if i < len(in_out) - 1:
                self.decoder_upsamples.append(
                    nn.ConvTranspose2d(out_ch, in_ch, 4, 2, 1)
                )
            else:
                self.decoder_upsamples.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder
        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.encoder_downsamples):
            for block in blocks:
                x = block(x, time_emb)
            skips.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, time_emb)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        x_flat = x_flat + self.mid_attn(x_flat, x_flat, x_flat)[0]
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        x = self.mid_block2(x, time_emb)

        # Decoder
        for blocks, upsample in zip(self.decoder_blocks, self.decoder_upsamples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, time_emb)
            x = upsample(x)

        return x


class ResBlock(nn.Module):
    """Residual block for UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if time_emb is not None:
            time_cond = self.time_mlp(time_emb)
            h = h + time_cond.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)

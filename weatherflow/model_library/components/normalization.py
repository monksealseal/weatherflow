"""
Normalization Layers for Weather AI Models

Various normalization strategies with unified interface.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard normalizations
LayerNorm = nn.LayerNorm
BatchNorm = nn.BatchNorm2d
GroupNorm = nn.GroupNorm
InstanceNorm = nn.InstanceNorm2d


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm, used in many modern transformers.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms

        if self.weight is not None:
            x = x * self.weight

        return x


class LayerNorm2D(nn.Module):
    """
    Layer normalization for 2D feature maps (channels last then permute).

    Normalizes over channels for each spatial position.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class ConditionalBatchNorm(nn.Module):
    """
    Conditional Batch Normalization.

    Uses external conditioning (e.g., time embedding) to modulate scale/shift.
    Good for: Diffusion models, conditional generation
    """

    def __init__(
        self,
        num_features: int,
        cond_dim: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum, eps=eps)

        # Condition to scale and shift
        self.cond_proj = nn.Linear(cond_dim, num_features * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
            cond: Conditioning [B, cond_dim]
        """
        # Standard batch norm
        x = self.bn(x)

        # Get scale and shift from conditioning
        params = self.cond_proj(cond)  # [B, C*2]
        scale, shift = params.chunk(2, dim=-1)

        # Apply modulation
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + scale) + shift


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).

    Used for style transfer and conditional generation.
    Good for: GANs, style transfer models
    """

    def __init__(
        self,
        num_features: int,
        style_dim: int,
    ):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        # Style to parameters
        self.style_proj = nn.Linear(style_dim, num_features * 2)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Content features [B, C, H, W]
            style: Style features [B, style_dim]
        """
        # Normalize content
        x = self.norm(x)

        # Get style parameters
        params = self.style_proj(style)
        gamma, beta = params.chunk(2, dim=-1)

        # Apply style
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * gamma + beta


class SpectralNorm(nn.Module):
    """
    Wrapper for spectral normalization.

    Constrains the Lipschitz constant of layers for stable training.
    Good for: GANs, discriminators
    """

    def __init__(self, module: nn.Module, name: str = "weight", n_power_iterations: int = 1):
        super().__init__()
        self.module = nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class WeightStandardization(nn.Module):
    """
    Weight Standardization wrapper.

    Standardizes the weights of convolutions for better training.
    Good for: Small batch training, BatchNorm alternatives
    """

    def __init__(self, module: nn.Conv2d, eps: float = 1e-5):
        super().__init__()
        self.module = module
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.module.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        std = weight.std(dim=(1, 2, 3), keepdim=True)
        weight = (weight - mean) / (std + self.eps)

        return F.conv2d(
            x, weight, self.module.bias,
            self.module.stride, self.module.padding,
            self.module.dilation, self.module.groups
        )


class SwitchNorm(nn.Module):
    """
    Switch Normalization.

    Learns to combine different normalization strategies.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Individual norms
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum, eps=eps)
        self.ln = nn.GroupNorm(1, num_features, affine=False, eps=eps)
        self.in_norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)

        # Learnable weights
        self.switch_weight = nn.Parameter(torch.ones(3) / 3)

        # Affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute all normalizations
        bn_out = self.bn(x)
        ln_out = self.ln(x)
        in_out = self.in_norm(x)

        # Combine with learned weights
        w = F.softmax(self.switch_weight, dim=0)
        out = w[0] * bn_out + w[1] * ln_out + w[2] * in_out

        # Apply affine
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return out


# Normalization registry
_NORMALIZATIONS = {
    "batch": lambda channels: nn.BatchNorm2d(channels),
    "layer": lambda channels: LayerNorm2D(channels),
    "instance": lambda channels: nn.InstanceNorm2d(channels, affine=True),
    "group": lambda channels: nn.GroupNorm(min(32, channels), channels),
    "rms": lambda channels: RMSNorm(channels),
    "switch": lambda channels: SwitchNorm(channels),
    "none": lambda channels: nn.Identity(),
    "identity": lambda channels: nn.Identity(),
}


def get_normalization(
    name: str,
    num_features: int,
    **kwargs,
) -> nn.Module:
    """
    Get normalization layer by name.

    Args:
        name: Normalization name (batch, layer, instance, group, rms, switch, none)
        num_features: Number of features/channels
        **kwargs: Additional arguments

    Returns:
        Normalization module

    Example:
        >>> norm = get_normalization("batch", 256)
        >>> norm = get_normalization("group", 256, num_groups=8)
    """
    name = name.lower()

    if name not in _NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization: {name}. "
            f"Available: {list(_NORMALIZATIONS.keys())}"
        )

    if name == "group" and "num_groups" in kwargs:
        return nn.GroupNorm(kwargs["num_groups"], num_features)

    return _NORMALIZATIONS[name](num_features)


def list_normalizations() -> list:
    """List all available normalization types."""
    return list(_NORMALIZATIONS.keys())

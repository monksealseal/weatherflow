"""
Activation Functions for Weather AI Models

Standard and custom activation functions with unified interface.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard activations
GELU = nn.GELU
ReLU = nn.ReLU
SiLU = nn.SiLU
LeakyReLU = nn.LeakyReLU
PReLU = nn.PReLU


class Swish(nn.Module):
    """Swish activation (x * sigmoid(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation (x * tanh(softplus(x)))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class QuickGELU(nn.Module):
    """Quick approximation of GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SquaredReLU(nn.Module):
    """Squared ReLU: max(0, x)^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU/Swish activation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


class StarReLU(nn.Module):
    """StarReLU from MetaFormer paper: s * relu(x)^2 + b."""

    def __init__(
        self,
        scale: float = 0.8944,
        bias: float = -0.4472,
        learnable: bool = False,
    ):
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale))
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer("scale", torch.tensor(scale))
            self.register_buffer("bias", torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x) ** 2 + self.bias


class HardSwish(nn.Module):
    """Hard Swish activation (efficient approximation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3) / 6


# Activation registry
_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "swish": Swish,
    "mish": Mish,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "prelu": nn.PReLU,
    "quick_gelu": QuickGELU,
    "squared_relu": SquaredReLU,
    "star_relu": StarReLU,
    "hard_swish": HardSwish,
    "hard_sigmoid": HardSigmoid,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "none": nn.Identity,
    "identity": nn.Identity,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation name (gelu, relu, silu, swish, mish, leaky_relu, etc.)
        **kwargs: Additional arguments for the activation

    Returns:
        Activation module

    Example:
        >>> act = get_activation("gelu")
        >>> act = get_activation("leaky_relu", negative_slope=0.1)
    """
    name = name.lower()

    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(_ACTIVATIONS.keys())}"
        )

    act_cls = _ACTIVATIONS[name]

    # Handle special cases with kwargs
    if name == "leaky_relu" and "negative_slope" in kwargs:
        return nn.LeakyReLU(kwargs["negative_slope"])

    if callable(act_cls):
        try:
            return act_cls(**kwargs)
        except TypeError:
            return act_cls()

    return act_cls


def list_activations() -> list:
    """List all available activation functions."""
    return list(_ACTIVATIONS.keys())


class AdaptiveActivation(nn.Module):
    """
    Learnable combination of multiple activations.

    Learns weights to combine different activations.
    """

    def __init__(
        self,
        activations: list = ["relu", "gelu", "silu"],
        learnable: bool = True,
    ):
        super().__init__()

        self.acts = nn.ModuleList([get_activation(a) for a in activations])

        if learnable:
            self.weights = nn.Parameter(torch.ones(len(activations)) / len(activations))
        else:
            self.register_buffer(
                "weights",
                torch.ones(len(activations)) / len(activations)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.weights, dim=0)
        out = sum(w * act(x) for w, act in zip(weights, self.acts))
        return out

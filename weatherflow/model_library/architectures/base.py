"""
Base classes for all weather AI models.

Provides common interfaces and utilities for weather prediction models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


class BaseWeatherModel(nn.Module, ABC):
    """
    Abstract base class for all weather prediction models.

    All weather models must implement:
        - forward(): Single-step prediction
        - rollout(): Multi-step autoregressive prediction

    Attributes:
        input_variables: List of input variable names
        output_variables: List of output variable names
        resolution: Spatial resolution (e.g., "0.25deg")
        forecast_hours: Forecast time step in hours
    """

    def __init__(
        self,
        input_variables: List[str],
        output_variables: List[str],
        resolution: str = "1deg",
        forecast_hours: int = 6,
    ):
        super().__init__()
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.resolution = resolution
        self.forecast_hours = forecast_hours

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Single forward prediction step.

        Args:
            x: Input tensor of shape (batch, channels, height, width) or
               (batch, channels, levels, height, width) for 3D models
            lead_time: Optional lead time encoding
            **kwargs: Model-specific arguments

        Returns:
            Predicted tensor of same spatial dimensions as input
        """
        pass

    def rollout(
        self,
        x: torch.Tensor,
        steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Multi-step autoregressive prediction.

        Args:
            x: Initial state tensor
            steps: Number of forecast steps
            **kwargs: Additional arguments for forward()

        Returns:
            Tensor of shape (batch, steps, channels, height, width)
        """
        predictions = []
        current = x

        for step in range(steps):
            # Compute lead time if needed
            lead_time = torch.full(
                (x.shape[0],),
                (step + 1) * self.forecast_hours,
                device=x.device,
                dtype=x.dtype,
            )

            # Single step prediction
            pred = self.forward(current, lead_time=lead_time, **kwargs)
            predictions.append(pred)

            # Autoregressive update
            current = pred

        return torch.stack(predictions, dim=1)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_mb(self) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024


class ProbabilisticWeatherModel(BaseWeatherModel):
    """
    Base class for probabilistic weather models.

    Extends BaseWeatherModel with methods for sampling and uncertainty quantification.
    """

    def __init__(
        self,
        input_variables: List[str],
        output_variables: List[str],
        resolution: str = "1deg",
        forecast_hours: int = 6,
        num_ensemble_members: int = 50,
    ):
        super().__init__(input_variables, output_variables, resolution, forecast_hours)
        self.num_ensemble_members = num_ensemble_members

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate multiple stochastic samples.

        Args:
            x: Input state tensor
            num_samples: Number of ensemble members to generate
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch, num_samples, channels, height, width)
        """
        pass

    def get_mean(
        self,
        x: torch.Tensor,
        num_samples: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute ensemble mean prediction."""
        samples = self.sample(x, num_samples, **kwargs)
        return samples.mean(dim=1)

    def get_std(
        self,
        x: torch.Tensor,
        num_samples: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute ensemble standard deviation (uncertainty)."""
        samples = self.sample(x, num_samples, **kwargs)
        return samples.std(dim=1)

    def get_quantiles(
        self,
        x: torch.Tensor,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 50,
        **kwargs,
    ) -> Dict[float, torch.Tensor]:
        """Compute specified quantiles from ensemble."""
        samples = self.sample(x, num_samples, **kwargs)
        results = {}
        for q in quantiles:
            results[q] = torch.quantile(samples, q, dim=1)
        return results


class EnsembleWeatherModel(nn.Module):
    """
    Wrapper for creating ensembles from deterministic models.

    Can combine multiple different models or multiple instances
    of the same model with different initializations.
    """

    def __init__(
        self,
        models: List[BaseWeatherModel],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_members = len(models)

        if weights is not None:
            assert len(weights) == len(models)
            self.register_buffer(
                "weights",
                torch.tensor(weights) / sum(weights)
            )
        else:
            self.register_buffer(
                "weights",
                torch.ones(len(models)) / len(models)
            )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Weighted ensemble mean prediction."""
        predictions = []
        for model in self.models:
            pred = model(x, **kwargs)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=1)  # (batch, members, ...)
        weighted = stacked * self.weights.view(1, -1, *([1] * (stacked.dim() - 2)))
        return weighted.sum(dim=1)

    def get_all_predictions(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Get predictions from all ensemble members."""
        predictions = []
        for model in self.models:
            pred = model(x, **kwargs)
            predictions.append(pred)
        return torch.stack(predictions, dim=1)

    def get_spread(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute ensemble spread (std across members)."""
        all_preds = self.get_all_predictions(x, **kwargs)
        return all_preds.std(dim=1)


class ConvBlock(nn.Module):
    """Standard convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "batch",
        activation: str = "gelu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "group":
            self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = nn.Identity()

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if isinstance(self.norm, nn.LayerNorm):
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "gelu",
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            norm=norm,
            activation=activation,
        )
        self.conv2 = ConvBlock(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            norm=norm,
            activation="none",
        )

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class SphericalPadding(nn.Module):
    """
    Spherical padding for weather data on lat-lon grids.

    Handles periodic boundary in longitude and reflection at poles.
    """

    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, lat, lon)
        p = self.padding

        # Periodic padding in longitude (wrap around)
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)

        # Reflection padding in latitude (poles)
        x = torch.cat([
            x[..., p:2*p, :].flip(dims=[-2]),
            x,
            x[..., -2*p:-p, :].flip(dims=[-2])
        ], dim=-2)

        return x


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for lat-lon grids."""

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels

        pe = torch.zeros(channels, height, width)

        # Latitude encoding (y)
        y_pos = torch.arange(height).unsqueeze(1).float()
        y_div = torch.exp(
            torch.arange(0, channels // 2, 2).float() *
            (-np.log(10000.0) / (channels // 2))
        )
        pe[0::4, :, :] = torch.sin(y_pos * y_div).unsqueeze(2).repeat(1, 1, width)
        pe[1::4, :, :] = torch.cos(y_pos * y_div).unsqueeze(2).repeat(1, 1, width)

        # Longitude encoding (x)
        x_pos = torch.arange(width).unsqueeze(0).float()
        x_div = torch.exp(
            torch.arange(0, channels // 2, 2).float() *
            (-np.log(10000.0) / (channels // 2))
        )
        pe[2::4, :, :] = torch.sin(x_pos * x_div.unsqueeze(1)).unsqueeze(1).repeat(1, height, 1)
        pe[3::4, :, :] = torch.cos(x_pos * x_div.unsqueeze(1)).unsqueeze(1).repeat(1, height, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.unsqueeze(0)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models and lead time encoding."""

    def __init__(self, dim: int, max_time: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_time = max_time

        half_dim = dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float() / self.max_time
        t = t.unsqueeze(-1) * self.emb.unsqueeze(0) * 1000
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        return self.mlp(t)


class LeadTimeEmbedding(nn.Module):
    """Embedding for forecast lead time (hours)."""

    def __init__(self, dim: int, max_lead_hours: int = 240):
        super().__init__()
        self.time_emb = TimeEmbedding(dim, max_time=max_lead_hours)

    def forward(self, lead_hours: torch.Tensor) -> torch.Tensor:
        return self.time_emb(lead_hours)

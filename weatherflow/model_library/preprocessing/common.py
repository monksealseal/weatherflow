"""
Common Preprocessing Utilities

Shared normalization and preprocessing components used
across multiple weather AI models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""
    mean: torch.Tensor
    std: torch.Tensor
    variable_names: List[str]


class Normalizer:
    """
    Base normalizer for weather data.

    Supports per-variable normalization with mean/std statistics.
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Dict[str, float]],
        std: Union[torch.Tensor, Dict[str, float]],
        variable_names: Optional[List[str]] = None,
    ):
        if isinstance(mean, dict):
            variable_names = list(mean.keys())
            mean = torch.tensor([mean[v] for v in variable_names])
            std = torch.tensor([std[v] for v in variable_names])

        self.mean = mean
        self.std = std
        self.variable_names = variable_names

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor.

        Args:
            x: (batch, channels, ...) or (channels, ...)

        Returns:
            Normalized tensor
        """
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        # Reshape for broadcasting
        if x.dim() == 4:  # (batch, channels, height, width)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.dim() == 3:  # (channels, height, width)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif x.dim() == 5:  # (batch, channels, level, height, width)
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        if x.dim() == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.dim() == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif x.dim() == 5:
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)

        return x * std + mean

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class ERA5Normalizer(Normalizer):
    """
    ERA5 Normalizer with pre-computed statistics.

    Statistics computed from ERA5 1979-2018 climatology.
    Used by GraphCast, FourCastNet, and other models trained on ERA5.
    """

    # Pre-computed ERA5 normalization statistics
    # Based on WeatherBench2 and model paper specifications
    ERA5_STATS = {
        # Surface variables
        "t2m": {"mean": 278.0, "std": 22.0},  # 2m temperature (K)
        "u10": {"mean": 0.0, "std": 6.0},  # 10m u-wind (m/s)
        "v10": {"mean": 0.0, "std": 6.0},  # 10m v-wind (m/s)
        "msl": {"mean": 101325.0, "std": 1500.0},  # Mean sea level pressure (Pa)
        "sp": {"mean": 96500.0, "std": 8000.0},  # Surface pressure (Pa)
        "tp": {"mean": 0.0, "std": 0.001},  # Total precipitation (m)
        "tcwv": {"mean": 20.0, "std": 15.0},  # Total column water vapor (kg/m2)

        # Pressure level variables (geopotential in m2/s2)
        "z_50": {"mean": 200000.0, "std": 5000.0},
        "z_100": {"mean": 160000.0, "std": 4000.0},
        "z_150": {"mean": 140000.0, "std": 3500.0},
        "z_200": {"mean": 120000.0, "std": 3000.0},
        "z_250": {"mean": 105000.0, "std": 2800.0},
        "z_300": {"mean": 92000.0, "std": 2500.0},
        "z_400": {"mean": 72000.0, "std": 2200.0},
        "z_500": {"mean": 55000.0, "std": 2000.0},
        "z_600": {"mean": 42000.0, "std": 1800.0},
        "z_700": {"mean": 30000.0, "std": 1500.0},
        "z_850": {"mean": 14000.0, "std": 1200.0},
        "z_925": {"mean": 7500.0, "std": 1000.0},
        "z_1000": {"mean": 500.0, "std": 800.0},

        # Temperature (K)
        "t_50": {"mean": 210.0, "std": 12.0},
        "t_100": {"mean": 205.0, "std": 10.0},
        "t_150": {"mean": 210.0, "std": 10.0},
        "t_200": {"mean": 218.0, "std": 10.0},
        "t_250": {"mean": 225.0, "std": 12.0},
        "t_300": {"mean": 235.0, "std": 14.0},
        "t_400": {"mean": 250.0, "std": 16.0},
        "t_500": {"mean": 260.0, "std": 18.0},
        "t_600": {"mean": 268.0, "std": 18.0},
        "t_700": {"mean": 275.0, "std": 18.0},
        "t_850": {"mean": 282.0, "std": 18.0},
        "t_925": {"mean": 286.0, "std": 18.0},
        "t_1000": {"mean": 290.0, "std": 18.0},

        # U-wind (m/s)
        "u_50": {"mean": 0.0, "std": 25.0},
        "u_100": {"mean": 0.0, "std": 20.0},
        "u_150": {"mean": 0.0, "std": 18.0},
        "u_200": {"mean": 5.0, "std": 20.0},
        "u_250": {"mean": 8.0, "std": 22.0},
        "u_300": {"mean": 8.0, "std": 20.0},
        "u_400": {"mean": 5.0, "std": 16.0},
        "u_500": {"mean": 3.0, "std": 14.0},
        "u_600": {"mean": 2.0, "std": 12.0},
        "u_700": {"mean": 1.0, "std": 10.0},
        "u_850": {"mean": 0.5, "std": 8.0},
        "u_925": {"mean": 0.0, "std": 7.0},
        "u_1000": {"mean": 0.0, "std": 6.0},

        # V-wind (m/s)
        "v_50": {"mean": 0.0, "std": 12.0},
        "v_100": {"mean": 0.0, "std": 10.0},
        "v_150": {"mean": 0.0, "std": 10.0},
        "v_200": {"mean": 0.0, "std": 12.0},
        "v_250": {"mean": 0.0, "std": 14.0},
        "v_300": {"mean": 0.0, "std": 14.0},
        "v_400": {"mean": 0.0, "std": 12.0},
        "v_500": {"mean": 0.0, "std": 10.0},
        "v_600": {"mean": 0.0, "std": 9.0},
        "v_700": {"mean": 0.0, "std": 8.0},
        "v_850": {"mean": 0.0, "std": 7.0},
        "v_925": {"mean": 0.0, "std": 6.0},
        "v_1000": {"mean": 0.0, "std": 5.0},

        # Specific humidity (kg/kg)
        "q_50": {"mean": 3e-6, "std": 2e-6},
        "q_100": {"mean": 3e-6, "std": 2e-6},
        "q_150": {"mean": 5e-6, "std": 5e-6},
        "q_200": {"mean": 2e-5, "std": 2e-5},
        "q_250": {"mean": 5e-5, "std": 5e-5},
        "q_300": {"mean": 1e-4, "std": 1e-4},
        "q_400": {"mean": 5e-4, "std": 5e-4},
        "q_500": {"mean": 0.001, "std": 0.001},
        "q_600": {"mean": 0.002, "std": 0.002},
        "q_700": {"mean": 0.004, "std": 0.003},
        "q_850": {"mean": 0.007, "std": 0.004},
        "q_925": {"mean": 0.009, "std": 0.005},
        "q_1000": {"mean": 0.010, "std": 0.006},

        # Relative humidity (0-1)
        "r_500": {"mean": 0.4, "std": 0.3},
        "r_700": {"mean": 0.5, "std": 0.3},
        "r_850": {"mean": 0.6, "std": 0.3},
        "r_1000": {"mean": 0.7, "std": 0.25},

        # Vertical velocity (Pa/s)
        "w_500": {"mean": 0.0, "std": 0.3},
        "w_700": {"mean": 0.0, "std": 0.3},
        "w_850": {"mean": 0.0, "std": 0.2},
    }

    def __init__(self, variables: List[str]):
        """
        Initialize ERA5 normalizer for specific variables.

        Args:
            variables: List of ERA5 variable names
        """
        mean_dict = {}
        std_dict = {}

        for var in variables:
            if var in self.ERA5_STATS:
                mean_dict[var] = self.ERA5_STATS[var]["mean"]
                std_dict[var] = self.ERA5_STATS[var]["std"]
            else:
                # Default normalization for unknown variables
                print(f"Warning: Unknown variable {var}, using default normalization")
                mean_dict[var] = 0.0
                std_dict[var] = 1.0

        super().__init__(mean_dict, std_dict, variables)


class WeatherBenchNormalizer(Normalizer):
    """
    WeatherBench2 Normalizer.

    Statistics from the WeatherBench2 benchmark dataset.
    Useful for standardized model evaluation.
    """

    # WeatherBench2 standard variables and stats
    WEATHERBENCH2_STATS = {
        "geopotential_500": {"mean": 54000.0, "std": 3000.0},
        "temperature_850": {"mean": 280.0, "std": 15.0},
        "2m_temperature": {"mean": 280.0, "std": 20.0},
        "10m_u_component_of_wind": {"mean": 0.0, "std": 5.0},
        "10m_v_component_of_wind": {"mean": 0.0, "std": 5.0},
        "mean_sea_level_pressure": {"mean": 101000.0, "std": 1500.0},
        "total_precipitation_6hr": {"mean": 0.0002, "std": 0.001},
    }

    def __init__(self, variables: Optional[List[str]] = None):
        if variables is None:
            variables = list(self.WEATHERBENCH2_STATS.keys())

        mean_dict = {}
        std_dict = {}

        for var in variables:
            if var in self.WEATHERBENCH2_STATS:
                mean_dict[var] = self.WEATHERBENCH2_STATS[var]["mean"]
                std_dict[var] = self.WEATHERBENCH2_STATS[var]["std"]
            else:
                mean_dict[var] = 0.0
                std_dict[var] = 1.0

        super().__init__(mean_dict, std_dict, variables)


def compute_normalization_stats(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (0, 2, 3),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std for normalization.

    Args:
        data: Data tensor
        dim: Dimensions to reduce over

    Returns:
        mean, std tensors
    """
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    std = torch.clamp(std, min=1e-8)  # Avoid division by zero
    return mean, std


def lat_weighted_mean(
    data: torch.Tensor,
    lat: torch.Tensor,
) -> torch.Tensor:
    """
    Compute latitude-weighted global mean.

    Args:
        data: (batch, channels, lat, lon)
        lat: Latitude values in degrees

    Returns:
        Weighted mean (batch, channels)
    """
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.sum()
    weights = weights.view(1, 1, -1, 1)

    weighted = data * weights
    return weighted.sum(dim=(2, 3))

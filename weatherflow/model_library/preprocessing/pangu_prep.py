"""
Pangu-Weather Preprocessing Pipeline

Preprocessing steps matching the Huawei Pangu-Weather implementation:
    1. Surface/upper-air variable separation
    2. 3D tensor organization (lat, lon, level)
    3. Variable-specific normalization
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


# Pangu-Weather variable specification (from Bi et al. 2023)
PANGU_SURFACE_VARS = ["u10", "v10", "t2m", "msl"]
PANGU_UPPER_VARS = ["z", "q", "t", "u", "v"]
PANGU_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
]

# Pangu normalization statistics
PANGU_SURFACE_STATS = {
    "u10": {"mean": 0.0, "std": 5.5},
    "v10": {"mean": 0.0, "std": 4.5},
    "t2m": {"mean": 278.5, "std": 21.5},
    "msl": {"mean": 101100.0, "std": 1300.0},
}

PANGU_UPPER_STATS = {
    "z": {"mean": 50000.0, "std": 50000.0},  # Varies significantly by level
    "q": {"mean": 0.003, "std": 0.004},
    "t": {"mean": 260.0, "std": 30.0},
    "u": {"mean": 3.0, "std": 12.0},
    "v": {"mean": 0.0, "std": 8.0},
}


def split_surface_upper(
    data: torch.Tensor,
    surface_vars: List[str] = PANGU_SURFACE_VARS,
    upper_vars: List[str] = PANGU_UPPER_VARS,
    num_levels: int = 13,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split combined data into surface and upper-air tensors.

    Args:
        data: (batch, channels, lat, lon) tensor
        surface_vars: List of surface variable names
        upper_vars: List of upper-air variable names
        num_levels: Number of pressure levels

    Returns:
        surface: (batch, num_surface, lat, lon)
        upper: (batch, num_upper, num_levels, lat, lon)
    """
    num_surface = len(surface_vars)
    num_upper = len(upper_vars)

    surface = data[:, :num_surface]

    # Upper air is organized as (var1_lev1, var1_lev2, ..., var2_lev1, ...)
    upper_flat = data[:, num_surface:]
    batch, _, lat, lon = upper_flat.shape

    upper = upper_flat.view(batch, num_upper, num_levels, lat, lon)

    return surface, upper


def combine_surface_upper(
    surface: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    Combine surface and upper-air tensors back to flat format.

    Args:
        surface: (batch, num_surface, lat, lon)
        upper: (batch, num_upper, num_levels, lat, lon)

    Returns:
        Combined tensor (batch, channels, lat, lon)
    """
    batch = surface.shape[0]

    upper_flat = upper.view(batch, -1, upper.shape[3], upper.shape[4])

    return torch.cat([surface, upper_flat], dim=1)


class PanguPreprocessor:
    """
    Complete preprocessing pipeline for Pangu-Weather.

    Handles:
        - Surface/upper-air separation
        - Variable-specific normalization
        - 3D tensor organization
        - Multi-step forecast preparation
    """

    def __init__(
        self,
        surface_vars: List[str] = PANGU_SURFACE_VARS,
        upper_vars: List[str] = PANGU_UPPER_VARS,
        pressure_levels: List[int] = PANGU_PRESSURE_LEVELS,
        img_size: Tuple[int, int] = (721, 1440),
    ):
        self.surface_vars = surface_vars
        self.upper_vars = upper_vars
        self.pressure_levels = pressure_levels
        self.num_levels = len(pressure_levels)
        self.img_size = img_size

        # Build normalization tensors
        self._build_normalization()

    def _build_normalization(self):
        """Build normalization statistics tensors."""
        # Surface normalization
        surface_mean = []
        surface_std = []
        for var in self.surface_vars:
            if var in PANGU_SURFACE_STATS:
                surface_mean.append(PANGU_SURFACE_STATS[var]["mean"])
                surface_std.append(PANGU_SURFACE_STATS[var]["std"])
            else:
                surface_mean.append(0.0)
                surface_std.append(1.0)

        self.surface_mean = torch.tensor(surface_mean).view(1, -1, 1, 1)
        self.surface_std = torch.tensor(surface_std).view(1, -1, 1, 1)

        # Upper-air normalization
        upper_mean = []
        upper_std = []
        for var in self.upper_vars:
            if var in PANGU_UPPER_STATS:
                upper_mean.append(PANGU_UPPER_STATS[var]["mean"])
                upper_std.append(PANGU_UPPER_STATS[var]["std"])
            else:
                upper_mean.append(0.0)
                upper_std.append(1.0)

        self.upper_mean = torch.tensor(upper_mean).view(1, -1, 1, 1, 1)
        self.upper_std = torch.tensor(upper_std).view(1, -1, 1, 1, 1)

    def normalize_surface(self, surface: torch.Tensor) -> torch.Tensor:
        """Normalize surface variables."""
        mean = self.surface_mean.to(surface.device, surface.dtype)
        std = self.surface_std.to(surface.device, surface.dtype)
        return (surface - mean) / std

    def normalize_upper(self, upper: torch.Tensor) -> torch.Tensor:
        """Normalize upper-air variables."""
        mean = self.upper_mean.to(upper.device, upper.dtype)
        std = self.upper_std.to(upper.device, upper.dtype)
        return (upper - mean) / std

    def denormalize_surface(self, surface: torch.Tensor) -> torch.Tensor:
        """Denormalize surface variables."""
        mean = self.surface_mean.to(surface.device, surface.dtype)
        std = self.surface_std.to(surface.device, surface.dtype)
        return surface * std + mean

    def denormalize_upper(self, upper: torch.Tensor) -> torch.Tensor:
        """Denormalize upper-air variables."""
        mean = self.upper_mean.to(upper.device, upper.dtype)
        std = self.upper_std.to(upper.device, upper.dtype)
        return upper * std + mean

    def __call__(
        self,
        data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess data for Pangu-Weather.

        Args:
            data: Combined tensor (batch, channels, lat, lon)

        Returns:
            surface_norm: Normalized surface tensor
            upper_norm: Normalized upper-air tensor
        """
        # Split
        surface, upper = split_surface_upper(
            data, self.surface_vars, self.upper_vars, self.num_levels
        )

        # Normalize
        surface_norm = self.normalize_surface(surface)
        upper_norm = self.normalize_upper(upper)

        return surface_norm, upper_norm

    def inverse_transform(
        self,
        surface: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model output back to physical units.

        Args:
            surface: Normalized surface prediction
            upper: Normalized upper-air prediction

        Returns:
            Combined tensor in physical units
        """
        surface_phys = self.denormalize_surface(surface)
        upper_phys = self.denormalize_upper(upper)

        return combine_surface_upper(surface_phys, upper_phys)

    def prepare_multi_step(
        self,
        data_sequence: List[torch.Tensor],
        forecast_steps: List[int] = [1, 3, 6, 24],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare data for multi-step Pangu models.

        Pangu has separate models for 1h, 3h, 6h, and 24h forecasts.

        Args:
            data_sequence: List of data tensors at different times
            forecast_steps: Forecast hours to prepare

        Returns:
            Dictionary mapping forecast hours to (input, target) pairs
        """
        prepared = {}
        for step in forecast_steps:
            if step < len(data_sequence):
                input_data = data_sequence[0]
                target_data = data_sequence[step]

                input_surface, input_upper = self(input_data)
                target_surface, target_upper = self(target_data)

                prepared[step] = {
                    "input_surface": input_surface,
                    "input_upper": input_upper,
                    "target_surface": target_surface,
                    "target_upper": target_upper,
                }

        return prepared

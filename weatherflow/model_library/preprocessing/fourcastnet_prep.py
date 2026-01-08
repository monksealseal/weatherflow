"""
FourCastNet Preprocessing Pipeline

Preprocessing steps matching the NVIDIA FourCastNet implementation:
    1. Variable selection (20 channels)
    2. Global mean/std normalization
    3. Patch preparation for AFNO
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from weatherflow.model_library.preprocessing.common import ERA5Normalizer


# FourCastNet variable specification (from Pathak et al. 2022)
FOURCASTNET_VARIABLES = [
    "u10",  # 10m u-component of wind
    "v10",  # 10m v-component of wind
    "t2m",  # 2m temperature
    "sp",   # Surface pressure
    "msl",  # Mean sea level pressure
    "t_850",  # Temperature at 850 hPa
    "u_1000",  # U-wind at 1000 hPa
    "v_1000",  # V-wind at 1000 hPa
    "z_1000",  # Geopotential at 1000 hPa
    "u_850",  # U-wind at 850 hPa
    "v_850",  # V-wind at 850 hPa
    "z_850",  # Geopotential at 850 hPa
    "u_500",  # U-wind at 500 hPa
    "v_500",  # V-wind at 500 hPa
    "z_500",  # Geopotential at 500 hPa
    "t_500",  # Temperature at 500 hPa
    "z_50",   # Geopotential at 50 hPa
    "r_500",  # Relative humidity at 500 hPa
    "r_850",  # Relative humidity at 850 hPa
    "tcwv",   # Total column water vapor
]

# FourCastNet normalization statistics (from official implementation)
FOURCASTNET_STATS = {
    "u10": {"mean": 0.0, "std": 5.2},
    "v10": {"mean": 0.1, "std": 4.4},
    "t2m": {"mean": 278.8, "std": 22.1},
    "sp": {"mean": 96467.0, "std": 8299.0},
    "msl": {"mean": 101164.0, "std": 1247.0},
    "t_850": {"mean": 275.8, "std": 14.3},
    "u_1000": {"mean": 0.3, "std": 6.1},
    "v_1000": {"mean": 0.0, "std": 5.0},
    "z_1000": {"mean": 564.0, "std": 863.0},
    "u_850": {"mean": 0.6, "std": 8.5},
    "v_850": {"mean": 0.1, "std": 6.3},
    "z_850": {"mean": 13716.0, "std": 1009.0},
    "u_500": {"mean": 4.6, "std": 13.7},
    "v_500": {"mean": 0.1, "std": 10.1},
    "z_500": {"mean": 54684.0, "std": 1890.0},
    "t_500": {"mean": 252.8, "std": 10.7},
    "z_50": {"mean": 200779.0, "std": 5076.0},
    "r_500": {"mean": 42.1, "std": 30.1},
    "r_850": {"mean": 69.5, "std": 26.4},
    "tcwv": {"mean": 24.1, "std": 16.4},
}


def normalize_fourcastnet(
    data: torch.Tensor,
    variables: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Normalize data using FourCastNet statistics.

    Args:
        data: (batch, channels, lat, lon) tensor
        variables: List of variable names

    Returns:
        Normalized tensor
    """
    if variables is None:
        variables = FOURCASTNET_VARIABLES

    mean = []
    std = []

    for var in variables:
        if var in FOURCASTNET_STATS:
            mean.append(FOURCASTNET_STATS[var]["mean"])
            std.append(FOURCASTNET_STATS[var]["std"])
        else:
            mean.append(0.0)
            std.append(1.0)

    mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
    std = torch.tensor(std, dtype=data.dtype, device=data.device)

    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    return (data - mean) / std


def denormalize_fourcastnet(
    data: torch.Tensor,
    variables: Optional[List[str]] = None,
) -> torch.Tensor:
    """Reverse FourCastNet normalization."""
    if variables is None:
        variables = FOURCASTNET_VARIABLES

    mean = []
    std = []

    for var in variables:
        if var in FOURCASTNET_STATS:
            mean.append(FOURCASTNET_STATS[var]["mean"])
            std.append(FOURCASTNET_STATS[var]["std"])
        else:
            mean.append(0.0)
            std.append(1.0)

    mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
    std = torch.tensor(std, dtype=data.dtype, device=data.device)

    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    return data * std + mean


class FourCastNetPreprocessor:
    """
    Complete preprocessing pipeline for FourCastNet.

    Handles:
        - Variable selection (20 channels)
        - Normalization with NVIDIA statistics
        - Resolution adjustment
        - Patch alignment for AFNO
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (720, 1440),
        patch_size: int = 4,
    ):
        if variables is None:
            variables = FOURCASTNET_VARIABLES

        self.variables = variables
        self.img_size = img_size
        self.patch_size = patch_size

        # Ensure image size is divisible by patch size
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preprocess data for FourCastNet.

        Args:
            data: (batch, channels, lat, lon) tensor

        Returns:
            Preprocessed tensor
        """
        # Normalize
        data_norm = normalize_fourcastnet(data, self.variables)

        # Ensure correct spatial dimensions
        if data_norm.shape[2:] != self.img_size:
            data_norm = torch.nn.functional.interpolate(
                data_norm,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )

        return data_norm

    def inverse_transform(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Convert model output back to physical units."""
        return denormalize_fourcastnet(prediction, self.variables)

    def get_variable_indices(
        self,
        variable_names: List[str],
    ) -> List[int]:
        """Get indices of specific variables."""
        return [self.variables.index(v) for v in variable_names if v in self.variables]


class FourCastNetPrecipPreprocessor(FourCastNetPreprocessor):
    """
    Preprocessing for FourCastNet precipitation model.

    Uses same preprocessing as base but with different output handling.
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (720, 1440),
        precip_transform: str = "log",
    ):
        super().__init__(variables, img_size)
        self.precip_transform = precip_transform

    def transform_precipitation(
        self,
        precip: torch.Tensor,
    ) -> torch.Tensor:
        """Transform precipitation for training."""
        if self.precip_transform == "log":
            # Log transform for heavy-tailed distribution
            return torch.log1p(precip * 1000)  # Convert to mm and log
        elif self.precip_transform == "sqrt":
            return torch.sqrt(precip * 1000)
        else:
            return precip

    def inverse_transform_precipitation(
        self,
        precip: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse transform precipitation to physical units."""
        if self.precip_transform == "log":
            return (torch.expm1(precip)) / 1000
        elif self.precip_transform == "sqrt":
            return (precip ** 2) / 1000
        else:
            return precip

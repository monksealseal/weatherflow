"""
GraphCast Preprocessing Pipeline

Preprocessing steps matching the original GraphCast implementation:
    1. Variable selection and ordering
    2. Normalization with ERA5 statistics
    3. Mesh feature creation
    4. Time difference computation for inputs
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from weatherflow.model_library.preprocessing.common import ERA5Normalizer


# GraphCast variable specification (from Lam et al. 2023)
GRAPHCAST_SURFACE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
]

GRAPHCAST_PRESSURE_LEVELS = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
]

GRAPHCAST_PRESSURE_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

# Static features
GRAPHCAST_STATIC_VARS = [
    "land_sea_mask",
    "geopotential_at_surface",
    "latitude",
    "longitude",
]


def get_graphcast_variable_list() -> List[str]:
    """Get the full list of GraphCast variables in correct order."""
    variables = GRAPHCAST_SURFACE_VARS.copy()

    for var in GRAPHCAST_PRESSURE_VARS:
        for level in GRAPHCAST_PRESSURE_LEVELS:
            variables.append(f"{var}_{level}")

    return variables


def normalize_graphcast(
    data: torch.Tensor,
    variables: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Normalize data using GraphCast statistics.

    Args:
        data: (batch, channels, lat, lon) tensor
        variables: List of variable names

    Returns:
        Normalized tensor
    """
    if variables is None:
        variables = get_graphcast_variable_list()

    # Map variable names to ERA5 normalizer format
    var_mapping = {
        "2m_temperature": "t2m",
        "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "total_precipitation_6hr": "tp",
        "geopotential": "z",
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "specific_humidity": "q",
        "vertical_velocity": "w",
    }

    # Create normalizer with mapped variables
    mapped_vars = []
    for var in variables:
        if "_" in var and var.split("_")[-1].isdigit():
            # Pressure level variable
            base_var = "_".join(var.split("_")[:-1])
            level = var.split("_")[-1]
            if base_var in var_mapping:
                mapped_vars.append(f"{var_mapping[base_var]}_{level}")
            else:
                mapped_vars.append(var)
        elif var in var_mapping:
            mapped_vars.append(var_mapping[var])
        else:
            mapped_vars.append(var)

    normalizer = ERA5Normalizer(mapped_vars)
    return normalizer.normalize(data)


def create_mesh_features(
    lat_size: int,
    lon_size: int,
    include_time: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Create mesh features for GraphCast.

    Args:
        lat_size: Number of latitude points
        lon_size: Number of longitude points
        include_time: Whether to include time-of-year features

    Returns:
        Dictionary of mesh feature tensors
    """
    # Create lat/lon grids
    lats = torch.linspace(-90, 90, lat_size)
    lons = torch.linspace(0, 360, lon_size + 1)[:-1]

    lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing="ij")

    features = {
        "latitude": lat_grid,
        "longitude": lon_grid,
        "cos_latitude": torch.cos(torch.deg2rad(lat_grid)),
        "sin_latitude": torch.sin(torch.deg2rad(lat_grid)),
        "cos_longitude": torch.cos(torch.deg2rad(lon_grid)),
        "sin_longitude": torch.sin(torch.deg2rad(lon_grid)),
    }

    return features


def compute_temporal_difference(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """
    Compute temporal difference for GraphCast input.

    GraphCast uses the difference between current and previous
    timestep as additional input features.

    Args:
        current: Current state (batch, channels, lat, lon)
        previous: Previous state (batch, channels, lat, lon)

    Returns:
        Temporal difference tensor
    """
    return current - previous


class GraphCastPreprocessor:
    """
    Complete preprocessing pipeline for GraphCast.

    Handles:
        - Variable selection and ordering
        - Normalization
        - Temporal difference computation
        - Mesh feature preparation
        - Static feature integration
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        lat_size: int = 721,
        lon_size: int = 1440,
        include_static: bool = True,
    ):
        if variables is None:
            variables = get_graphcast_variable_list()

        self.variables = variables
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.include_static = include_static

        # Create normalizer
        self.normalizer = None  # Lazy initialization

        # Create static features
        if include_static:
            self.mesh_features = create_mesh_features(lat_size, lon_size)

    def __call__(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        static_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Preprocess data for GraphCast.

        Args:
            current: Current atmospheric state
            previous: Previous atmospheric state (for temporal difference)
            static_features: Optional static features (land-sea mask, etc.)

        Returns:
            Preprocessed tensor ready for GraphCast
        """
        # Normalize
        current_norm = normalize_graphcast(current, self.variables)

        features = [current_norm]

        # Add temporal difference if previous state provided
        if previous is not None:
            previous_norm = normalize_graphcast(previous, self.variables)
            diff = compute_temporal_difference(current_norm, previous_norm)
            features.append(diff)

        # Add static features
        if self.include_static and static_features is not None:
            for name, feature in static_features.items():
                if feature.dim() == 2:
                    feature = feature.unsqueeze(0).unsqueeze(0)
                features.append(feature.expand(current.shape[0], -1, -1, -1))

        return torch.cat(features, dim=1)

    def get_target(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare target for training (residual prediction).

        GraphCast predicts the residual (target - current).

        Args:
            target: Target state
            current: Current state

        Returns:
            Normalized residual target
        """
        residual = target - current
        return normalize_graphcast(residual, self.variables)

    def inverse_transform(
        self,
        prediction: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model output back to physical units.

        Args:
            prediction: Model prediction (normalized residual)
            current: Current state (physical units)

        Returns:
            Prediction in physical units
        """
        # Denormalize residual
        if self.normalizer is None:
            # Create a simple denormalizer
            mapped_vars = []
            for var in self.variables:
                if "_" in var and var.split("_")[-1].isdigit():
                    base_var = "_".join(var.split("_")[:-1])
                    level = var.split("_")[-1]
                    mapped_vars.append(f"{base_var[0]}_{level}")
                else:
                    mapped_vars.append(var[:3] if len(var) > 3 else var)
            self.normalizer = ERA5Normalizer(mapped_vars[:prediction.shape[1]])

        residual = self.normalizer.denormalize(prediction)

        # Add to current state
        return current + residual

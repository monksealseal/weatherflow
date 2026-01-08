"""
GenCast Preprocessing Pipeline

Preprocessing for diffusion-based weather prediction:
    1. Normalization to [-1, 1] range for diffusion
    2. Condition preparation (initial state)
    3. Noise scheduling compatible preprocessing
"""

from typing import List, Optional, Tuple

import torch
import numpy as np

from weatherflow.model_library.preprocessing.common import ERA5Normalizer


class GenCastPreprocessor:
    """
    Preprocessing pipeline for GenCast diffusion model.

    Diffusion models work best with data normalized to [-1, 1].
    This preprocessor handles the transformation and provides
    utilities for noise scheduling.
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        normalization_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        if variables is None:
            variables = [
                "z_500", "z_850", "z_1000",
                "t_500", "t_850", "t_1000",
                "u_500", "u_850", "u_1000",
                "v_500", "v_850", "v_1000",
                "q_500", "q_850", "q_1000",
                "t2m", "u10", "v10", "msl", "tp",
            ]

        self.variables = variables
        self.norm_min, self.norm_max = normalization_range

        # ERA5 normalizer for initial transform
        self.era5_normalizer = ERA5Normalizer(variables)

        # Track data statistics for inverse transform
        self.data_min = None
        self.data_max = None

    def fit(self, data: torch.Tensor) -> None:
        """
        Fit preprocessor to data to learn min/max ranges.

        Args:
            data: (N, channels, lat, lon) training data
        """
        # First normalize with ERA5 stats
        normalized = self.era5_normalizer.normalize(data)

        # Learn per-channel min/max
        self.data_min = normalized.amin(dim=(0, 2, 3), keepdim=True)
        self.data_max = normalized.amax(dim=(0, 2, 3), keepdim=True)

    def __call__(
        self,
        data: torch.Tensor,
        is_condition: bool = False,
    ) -> torch.Tensor:
        """
        Preprocess data for GenCast.

        Args:
            data: (batch, channels, lat, lon) tensor
            is_condition: Whether this is conditioning data

        Returns:
            Preprocessed tensor in [-1, 1] range
        """
        # Apply ERA5 normalization
        normalized = self.era5_normalizer.normalize(data)

        # Scale to [-1, 1]
        if self.data_min is not None and self.data_max is not None:
            data_range = self.data_max - self.data_min
            data_range = torch.clamp(data_range, min=1e-8)

            scaled = (normalized - self.data_min) / data_range
            scaled = scaled * (self.norm_max - self.norm_min) + self.norm_min
        else:
            # Fallback: assume roughly normal distribution
            scaled = torch.tanh(normalized / 3.0)

        return scaled

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convert model output back to physical units.

        Args:
            data: Tensor in [-1, 1] range

        Returns:
            Tensor in physical units
        """
        # Unscale from [-1, 1]
        if self.data_min is not None and self.data_max is not None:
            data_range = self.data_max - self.data_min
            unscaled = (data - self.norm_min) / (self.norm_max - self.norm_min)
            unscaled = unscaled * data_range + self.data_min
        else:
            unscaled = torch.atanh(torch.clamp(data, -0.999, 0.999)) * 3.0

        # Denormalize
        return self.era5_normalizer.denormalize(unscaled)

    def prepare_diffusion_target(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare target for diffusion training.

        For conditional diffusion, we predict the next state
        conditioned on the current state.

        Args:
            current: Current atmospheric state
            target: Target atmospheric state

        Returns:
            Preprocessed target tensor
        """
        return self(target)

    def prepare_condition(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare conditioning information.

        Args:
            condition: Conditioning data (e.g., initial state)

        Returns:
            Preprocessed condition tensor
        """
        return self(condition, is_condition=True)

"""
ClimaX Preprocessing Pipeline

Preprocessing for the ClimaX foundation model:
    1. Variable tokenization
    2. Lead time encoding
    3. Multi-resolution handling
    4. Variable embedding preparation
"""

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from weatherflow.model_library.preprocessing.common import ERA5Normalizer


# ClimaX variable specification
CLIMAX_VARIABLES = [
    "z_500", "z_850", "z_1000",
    "t_500", "t_850", "t_1000",
    "u_500", "u_850", "u_1000",
    "v_500", "v_850", "v_1000",
    "q_500", "q_850", "q_1000",
    "t2m", "u10", "v10",
]

# ClimaX normalization (from Microsoft implementation)
CLIMAX_STATS = {
    "z_500": {"mean": 54684.0, "std": 1890.0},
    "z_850": {"mean": 13716.0, "std": 1009.0},
    "z_1000": {"mean": 564.0, "std": 863.0},
    "t_500": {"mean": 252.8, "std": 10.7},
    "t_850": {"mean": 275.8, "std": 14.3},
    "t_1000": {"mean": 288.0, "std": 14.0},
    "u_500": {"mean": 4.6, "std": 13.7},
    "u_850": {"mean": 0.6, "std": 8.5},
    "u_1000": {"mean": 0.3, "std": 6.1},
    "v_500": {"mean": 0.1, "std": 10.1},
    "v_850": {"mean": 0.1, "std": 6.3},
    "v_1000": {"mean": 0.0, "std": 5.0},
    "q_500": {"mean": 0.001, "std": 0.001},
    "q_850": {"mean": 0.007, "std": 0.004},
    "q_1000": {"mean": 0.010, "std": 0.006},
    "t2m": {"mean": 278.8, "std": 22.1},
    "u10": {"mean": 0.0, "std": 5.2},
    "v10": {"mean": 0.1, "std": 4.4},
}


class ClimaXPreprocessor:
    """
    Preprocessing pipeline for ClimaX foundation model.

    Handles:
        - Variable-agnostic normalization
        - Multi-resolution support
        - Lead time encoding
        - Flexible variable subset selection
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (32, 64),
        patch_size: int = 2,
        max_lead_hours: int = 240,
    ):
        if variables is None:
            variables = CLIMAX_VARIABLES

        self.variables = variables
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_lead_hours = max_lead_hours

        # Variable to index mapping
        self.var_to_idx = {var: i for i, var in enumerate(variables)}

        # Build normalization stats
        self._build_normalization()

    def _build_normalization(self):
        """Build normalization statistics tensors."""
        mean = []
        std = []

        for var in self.variables:
            if var in CLIMAX_STATS:
                mean.append(CLIMAX_STATS[var]["mean"])
                std.append(CLIMAX_STATS[var]["std"])
            else:
                mean.append(0.0)
                std.append(1.0)

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data."""
        mean = self.mean.to(data.device, data.dtype)
        std = self.std.to(data.device, data.dtype)
        return (data - mean) / std

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data."""
        mean = self.mean.to(data.device, data.dtype)
        std = self.std.to(data.device, data.dtype)
        return data * std + mean

    def encode_lead_time(
        self,
        lead_hours: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode lead time for ClimaX.

        Args:
            lead_hours: Lead time in hours (batch,)

        Returns:
            Lead time embedding
        """
        # Normalize to [0, 1]
        normalized = lead_hours.float() / self.max_lead_hours

        # Sinusoidal encoding
        dim = 64
        half_dim = dim // 2
        emb = torch.exp(
            -np.log(10000) * torch.arange(half_dim, device=lead_hours.device) / half_dim
        )
        emb = normalized.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

    def select_variables(
        self,
        data: torch.Tensor,
        variable_subset: List[str],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Select a subset of variables.

        Args:
            data: (batch, channels, lat, lon) tensor
            variable_subset: List of variable names to select

        Returns:
            Selected data and variable indices
        """
        indices = [self.var_to_idx[v] for v in variable_subset if v in self.var_to_idx]
        return data[:, indices], indices

    def __call__(
        self,
        data: torch.Tensor,
        lead_hours: Optional[torch.Tensor] = None,
        variable_subset: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess data for ClimaX.

        Args:
            data: (batch, channels, lat, lon) tensor
            lead_hours: Optional lead time in hours
            variable_subset: Optional subset of variables

        Returns:
            Dictionary with preprocessed tensors
        """
        # Select variables if subset specified
        if variable_subset is not None:
            data, var_indices = self.select_variables(data, variable_subset)
            # Re-normalize for subset
            mean = self.mean[:, var_indices]
            std = self.std[:, var_indices]
            data_norm = (data - mean.to(data.device)) / std.to(data.device)
        else:
            data_norm = self.normalize(data)
            var_indices = list(range(len(self.variables)))

        result = {
            "data": data_norm,
            "var_indices": torch.tensor(var_indices),
        }

        # Add lead time encoding
        if lead_hours is not None:
            result["lead_time_emb"] = self.encode_lead_time(lead_hours)

        return result

    def inverse_transform(
        self,
        prediction: torch.Tensor,
        variable_subset: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Convert model output back to physical units.

        Args:
            prediction: Model prediction
            variable_subset: Variables that were predicted

        Returns:
            Prediction in physical units
        """
        if variable_subset is not None:
            indices = [self.var_to_idx[v] for v in variable_subset if v in self.var_to_idx]
            mean = self.mean[:, indices].to(prediction.device)
            std = self.std[:, indices].to(prediction.device)
            return prediction * std + mean
        else:
            return self.denormalize(prediction)

    def prepare_multitask(
        self,
        data: torch.Tensor,
        task: str,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for different ClimaX tasks.

        ClimaX supports multiple tasks:
            - forecasting: Standard weather prediction
            - downscaling: Increase spatial resolution
            - climate_projection: Long-term climate prediction

        Args:
            data: Input data
            task: Task name
            **kwargs: Task-specific arguments

        Returns:
            Task-specific preprocessed data
        """
        if task == "forecasting":
            lead_hours = kwargs.get("lead_hours", torch.tensor([6]))
            return self(data, lead_hours=lead_hours)

        elif task == "downscaling":
            # For downscaling, we also need the low-resolution target
            scale_factor = kwargs.get("scale_factor", 4)

            # Downsample input
            low_res = torch.nn.functional.interpolate(
                data,
                scale_factor=1/scale_factor,
                mode="bilinear",
                align_corners=False,
            )

            return {
                "low_res": self.normalize(low_res),
                "high_res": self.normalize(data),
                "scale_factor": scale_factor,
            }

        elif task == "climate_projection":
            # Climate projection uses longer lead times
            lead_years = kwargs.get("lead_years", 1)
            lead_hours = torch.tensor([lead_years * 365 * 24])

            return self(data, lead_hours=lead_hours)

        else:
            return self(data)

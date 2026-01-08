"""
Classical Weather Prediction Methods

Baseline models for comparison:
    - Persistence: Tomorrow's weather = today's weather
    - Climatology: Predict historical average
    - Linear Regression: Simple linear model
    - Analog Method: Find similar historical patterns
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import BaseWeatherModel
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class PersistenceModel(BaseWeatherModel):
    """
    Persistence Forecast Model.

    The simplest baseline: predict that tomorrow's weather
    will be the same as today's.

    Surprisingly competitive for very short-term forecasts
    and useful as a baseline for forecast skill.
    """

    def __init__(
        self,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = ["z_500", "t_850", "t2m"]
        if output_variables is None:
            output_variables = input_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="any",
            forecast_hours=6,
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return input as prediction (persistence)."""
        return x.clone()

    def rollout(
        self,
        x: torch.Tensor,
        steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """Multi-step persistence: all steps are the same."""
        return x.unsqueeze(1).expand(-1, steps, -1, -1, -1)


class ClimatologyModel(BaseWeatherModel):
    """
    Climatology Forecast Model.

    Predicts the historical average for each location and time of year.
    A strong baseline for longer-range forecasts.
    """

    def __init__(
        self,
        climatology_mean: Optional[torch.Tensor] = None,
        climatology_std: Optional[torch.Tensor] = None,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = ["z_500", "t_850", "t2m"]
        if output_variables is None:
            output_variables = input_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="any",
            forecast_hours=6,
        )

        # Climatology statistics (can be set from data)
        if climatology_mean is not None:
            self.register_buffer("climatology_mean", climatology_mean)
        else:
            self.climatology_mean = None

        if climatology_std is not None:
            self.register_buffer("climatology_std", climatology_std)
        else:
            self.climatology_std = None

    def set_climatology(
        self,
        mean: torch.Tensor,
        std: Optional[torch.Tensor] = None,
    ):
        """Set climatology statistics from data."""
        self.register_buffer("climatology_mean", mean)
        if std is not None:
            self.register_buffer("climatology_std", std)

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return climatological mean."""
        if self.climatology_mean is not None:
            # Broadcast to batch size
            return self.climatology_mean.expand(x.shape[0], -1, -1, -1)
        else:
            # Fall back to input mean if climatology not set
            return x.mean(dim=0, keepdim=True).expand(x.shape[0], -1, -1, -1)


class LinearRegressionModel(BaseWeatherModel):
    """
    Linear Regression Weather Model.

    Simple linear model that learns weights to predict
    the next state from the current state.

    y = Wx + b

    Can capture basic patterns like advection.
    """

    def __init__(
        self,
        in_channels: int = 20,
        out_channels: Optional[int] = None,
        img_size: Tuple[int, int] = (64, 128),
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if out_channels is None:
            out_channels = in_channels
        if input_variables is None:
            input_variables = [f"var_{i}" for i in range(in_channels)]
        if output_variables is None:
            output_variables = input_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="any",
            forecast_hours=6,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

        # Linear transformation per grid point
        self.weight = nn.Parameter(torch.eye(in_channels).unsqueeze(-1).unsqueeze(-1))
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Linear prediction."""
        # x: (batch, channels, height, width)
        # Apply per-channel linear transformation
        out = torch.einsum("bcij,cd->bdij", x, self.weight.squeeze(-1).squeeze(-1))
        out = out + self.bias
        return out


class AnalogModel(BaseWeatherModel):
    """
    Analog Ensemble Method.

    Finds similar historical patterns and uses their
    evolution as forecasts.

    Based on Delle Monache et al. (2013)
    "Probabilistic Weather Prediction with an Analog Ensemble"
    """

    def __init__(
        self,
        historical_states: Optional[torch.Tensor] = None,
        historical_targets: Optional[torch.Tensor] = None,
        num_analogs: int = 20,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = ["z_500", "t_850"]
        if output_variables is None:
            output_variables = input_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="any",
            forecast_hours=6,
        )

        self.num_analogs = num_analogs

        # Historical database (can be set later)
        if historical_states is not None:
            self.register_buffer("historical_states", historical_states)
        else:
            self.historical_states = None

        if historical_targets is not None:
            self.register_buffer("historical_targets", historical_targets)
        else:
            self.historical_targets = None

    def set_historical_data(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Set historical database."""
        self.register_buffer("historical_states", states)
        self.register_buffer("historical_targets", targets)

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Find analogs and predict based on their evolution."""
        if self.historical_states is None:
            # No historical data - fall back to persistence
            return x.clone()

        batch = x.shape[0]
        predictions = []

        for i in range(batch):
            # Compute distances to all historical states
            query = x[i].flatten()
            historical_flat = self.historical_states.flatten(1)
            distances = torch.norm(historical_flat - query.unsqueeze(0), dim=1)

            # Find k nearest analogs
            _, indices = torch.topk(distances, self.num_analogs, largest=False)

            # Average their targets
            analog_targets = self.historical_targets[indices]
            pred = analog_targets.mean(dim=0)
            predictions.append(pred)

        return torch.stack(predictions, dim=0)


class WeightedEnsembleModel(BaseWeatherModel):
    """
    Weighted Ensemble of Multiple Models.

    Combines predictions from multiple models with
    learned or fixed weights.
    """

    def __init__(
        self,
        models: List[BaseWeatherModel],
        weights: Optional[List[float]] = None,
        learn_weights: bool = False,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        # Get variables from first model
        if input_variables is None:
            input_variables = models[0].input_variables
        if output_variables is None:
            output_variables = models[0].output_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="any",
            forecast_hours=models[0].forecast_hours,
        )

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models

        if learn_weights:
            self.weights = nn.Parameter(torch.tensor(weights))
        else:
            self.register_buffer("weights", torch.tensor(weights))

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Weighted average of model predictions."""
        predictions = []
        for model in self.models:
            pred = model(x, lead_time, **kwargs)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)
        normalized_weights = F.softmax(self.weights, dim=0)

        weighted_sum = torch.einsum("m,mbchw->bchw", normalized_weights, stacked)
        return weighted_sum


# Register models
persistence_info = ModelInfo(
    name="Persistence",
    category=ModelCategory.CLASSICAL,
    scale=ModelScale.TINY,
    description="Baseline model: predict that weather stays the same",
    paper_title="N/A (classical baseline)",
    paper_url="",
    paper_year=1900,
    authors=["Traditional"],
    organization="Meteorology",
    input_variables=["any"],
    output_variables=["any"],
    supported_resolutions=["any"],
    forecast_range="0-24h (competitive)",
    temporal_resolution="any",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=0.0,
    typical_training_time="N/A",
    inference_time_per_step="<1ms",
    tags=["baseline", "classical", "persistence"],
    related_models=["climatology"],
)

climatology_info = ModelInfo(
    name="Climatology",
    category=ModelCategory.CLASSICAL,
    scale=ModelScale.TINY,
    description="Baseline model: predict historical average for each location/time",
    paper_title="N/A (classical baseline)",
    paper_url="",
    paper_year=1900,
    authors=["Traditional"],
    organization="Meteorology",
    input_variables=["any"],
    output_variables=["any"],
    supported_resolutions=["any"],
    forecast_range="7+ days (competitive for long-range)",
    temporal_resolution="any",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=0.0,
    typical_training_time="N/A (precompute statistics)",
    inference_time_per_step="<1ms",
    tags=["baseline", "classical", "climatology"],
    related_models=["persistence"],
)

linear_info = ModelInfo(
    name="LinearRegression",
    category=ModelCategory.CLASSICAL,
    scale=ModelScale.TINY,
    description="Simple linear regression weather model",
    paper_title="N/A (classical method)",
    paper_url="",
    paper_year=1900,
    authors=["Traditional"],
    organization="Statistics",
    input_variables=["any"],
    output_variables=["any"],
    supported_resolutions=["any"],
    forecast_range="0-48h",
    temporal_resolution="any",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=0.0,
    typical_training_time="<1 minute",
    inference_time_per_step="<1ms",
    tags=["baseline", "classical", "linear"],
    related_models=["persistence", "climatology"],
)

register_model("persistence", PersistenceModel, persistence_info, {})
register_model("climatology", ClimatologyModel, climatology_info, {})
register_model("linear_regression", LinearRegressionModel, linear_info, {
    "in_channels": 20,
    "out_channels": 20,
    "img_size": (64, 128),
})

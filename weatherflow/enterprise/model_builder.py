"""
Enterprise Model Builder

The core engine that creates AI models combining enterprise business data
with weather and climate intelligence.

Key Insight: Why choose between a weather forecasting model OR a business
forecasting model when you can build ONE unified model that leverages both?

This builder creates models that understand:
1. Your business data (sales, inventory, policies, claims, etc.)
2. Weather patterns (temperature, precipitation, storms, etc.)
3. How they INTERACT (weather-sensitive demand, climate risk, etc.)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from .data_models import (
    EnterpriseDataset,
    CombinedDataset,
    WeatherDataType,
    RetailDataSchema,
    InsuranceDataSchema,
    DataField,
)


class IndustryVertical(Enum):
    """Supported industry verticals."""
    RETAIL = "retail"
    INSURANCE = "insurance"


class ModelPurpose(Enum):
    """What the model is designed to predict/optimize."""

    # Retail purposes
    DEMAND_FORECAST = "demand_forecast"
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    MARGIN_PREDICTION = "margin_prediction"
    PURCHASING_BEHAVIOR = "purchasing_behavior"

    # Insurance purposes
    RISK_ASSESSMENT = "risk_assessment"
    PRICING_OPTIMIZATION = "pricing_optimization"
    CLAIMS_PREDICTION = "claims_prediction"
    PORTFOLIO_RISK = "portfolio_risk"


@dataclass
class EnterpriseModelConfig:
    """
    Configuration for an enterprise custom model.

    This defines exactly what the model does, what data it uses,
    and how it combines enterprise + weather features.
    """

    # Identity
    name: str
    description: str
    version: str = "1.0"

    # Industry and purpose
    industry: IndustryVertical = IndustryVertical.RETAIL
    purpose: ModelPurpose = ModelPurpose.DEMAND_FORECAST

    # Data configuration
    enterprise_features: List[str] = field(default_factory=list)
    weather_features: List[WeatherDataType] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)

    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    use_residual: bool = True

    # Weather integration strategy
    weather_integration: str = "concatenate"  # concatenate, attention, gating, cross_attention
    weather_context_window: int = 7  # Days of weather history to consider
    weather_forecast_horizon: int = 14  # Days of weather forecast to use

    # Temporal modeling
    use_temporal_encoding: bool = True
    temporal_encoding_dim: int = 32
    use_seasonality: bool = True

    # Spatial modeling
    use_spatial_encoding: bool = True
    spatial_encoding_dim: int = 32

    # Embedding dimensions for categorical variables
    categorical_embedding_dim: int = 16

    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Loss configuration
    loss_type: str = "mse"  # mse, mae, huber, quantile
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    company: Optional[str] = None
    decision_context: Optional[str] = None  # What decision is this model supporting?


class TemporalEncoding(nn.Module):
    """Encode temporal features (day of week, month, season, etc.)."""

    def __init__(self, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim

        # Learnable embeddings for temporal features
        self.day_of_week_embed = nn.Embedding(7, encoding_dim // 4)
        self.month_embed = nn.Embedding(12, encoding_dim // 4)
        self.day_of_month_embed = nn.Embedding(31, encoding_dim // 4)
        self.quarter_embed = nn.Embedding(4, encoding_dim // 4)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Convert timestamps to temporal encodings.

        Args:
            timestamps: [B] tensor of unix timestamps or [B, 4] tensor of
                       (day_of_week, month, day_of_month, quarter)

        Returns:
            [B, encoding_dim] temporal encoding
        """
        if timestamps.dim() == 1:
            # Convert unix timestamps to components
            # This is a simplified version - in production would use proper date parsing
            day_of_week = (timestamps // 86400) % 7
            day_of_year = (timestamps // 86400) % 365
            month = (day_of_year // 30) % 12
            day_of_month = day_of_year % 31
            quarter = month // 3
        else:
            day_of_week = timestamps[:, 0]
            month = timestamps[:, 1]
            day_of_month = timestamps[:, 2]
            quarter = timestamps[:, 3]

        # Ensure indices are valid
        day_of_week = day_of_week.long().clamp(0, 6)
        month = month.long().clamp(0, 11)
        day_of_month = day_of_month.long().clamp(0, 30)
        quarter = quarter.long().clamp(0, 3)

        # Get embeddings
        dow_emb = self.day_of_week_embed(day_of_week)
        month_emb = self.month_embed(month)
        dom_emb = self.day_of_month_embed(day_of_month)
        quarter_emb = self.quarter_embed(quarter)

        # Concatenate
        return torch.cat([dow_emb, month_emb, dom_emb, quarter_emb], dim=-1)


class SpatialEncoding(nn.Module):
    """Encode spatial features (latitude, longitude)."""

    def __init__(self, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim

        # MLP to encode lat/lon
        self.encoder = nn.Sequential(
            nn.Linear(2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim)
        )

        # Add sinusoidal position encoding for better spatial representation
        self.register_buffer(
            "freq_bands",
            2 ** torch.linspace(0, 4, encoding_dim // 4)
        )

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Convert lat/lon to spatial encodings.

        Args:
            locations: [B, 2] tensor of (latitude, longitude)

        Returns:
            [B, encoding_dim] spatial encoding
        """
        # Normalize lat/lon to [-1, 1]
        lat_norm = locations[:, 0:1] / 90.0
        lon_norm = locations[:, 1:2] / 180.0

        # MLP encoding
        mlp_enc = self.encoder(torch.cat([lat_norm, lon_norm], dim=-1))

        return mlp_enc


class WeatherIntegrationLayer(nn.Module):
    """
    Layer that integrates weather features with enterprise features.

    Supports multiple integration strategies:
    - concatenate: Simple concatenation
    - attention: Weather-attended enterprise features
    - gating: Gated combination
    - cross_attention: Full cross-attention between domains
    """

    def __init__(
        self,
        enterprise_dim: int,
        weather_dim: int,
        output_dim: int,
        integration_type: str = "concatenate"
    ):
        super().__init__()
        self.integration_type = integration_type

        if integration_type == "concatenate":
            self.output_layer = nn.Linear(enterprise_dim + weather_dim, output_dim)

        elif integration_type == "attention":
            self.query = nn.Linear(enterprise_dim, output_dim)
            self.key = nn.Linear(weather_dim, output_dim)
            self.value = nn.Linear(weather_dim, output_dim)
            self.output_layer = nn.Linear(output_dim + enterprise_dim, output_dim)

        elif integration_type == "gating":
            self.gate = nn.Sequential(
                nn.Linear(enterprise_dim + weather_dim, output_dim),
                nn.Sigmoid()
            )
            self.enterprise_proj = nn.Linear(enterprise_dim, output_dim)
            self.weather_proj = nn.Linear(weather_dim, output_dim)

        elif integration_type == "cross_attention":
            self.enterprise_to_weather = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )
            self.weather_to_enterprise = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )
            self.enterprise_proj = nn.Linear(enterprise_dim, output_dim)
            self.weather_proj = nn.Linear(weather_dim, output_dim)
            self.output_layer = nn.Linear(output_dim * 2, output_dim)

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate enterprise and weather features.

        Args:
            enterprise_features: [B, enterprise_dim]
            weather_features: [B, weather_dim]

        Returns:
            [B, output_dim] integrated features
        """
        if self.integration_type == "concatenate":
            combined = torch.cat([enterprise_features, weather_features], dim=-1)
            return self.output_layer(combined)

        elif self.integration_type == "attention":
            q = self.query(enterprise_features).unsqueeze(1)  # [B, 1, D]
            k = self.key(weather_features).unsqueeze(1)  # [B, 1, D]
            v = self.value(weather_features).unsqueeze(1)  # [B, 1, D]

            # Attention
            attn_weights = F.softmax(
                torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.size(-1)),
                dim=-1
            )
            attended = torch.bmm(attn_weights, v).squeeze(1)  # [B, D]

            combined = torch.cat([enterprise_features, attended], dim=-1)
            return self.output_layer(combined)

        elif self.integration_type == "gating":
            gate = self.gate(torch.cat([enterprise_features, weather_features], dim=-1))
            ent_proj = self.enterprise_proj(enterprise_features)
            wea_proj = self.weather_proj(weather_features)
            return gate * ent_proj + (1 - gate) * wea_proj

        elif self.integration_type == "cross_attention":
            ent_proj = self.enterprise_proj(enterprise_features).unsqueeze(1)
            wea_proj = self.weather_proj(weather_features).unsqueeze(1)

            # Cross attention both directions
            ent_attended, _ = self.enterprise_to_weather(ent_proj, wea_proj, wea_proj)
            wea_attended, _ = self.weather_to_enterprise(wea_proj, ent_proj, ent_proj)

            combined = torch.cat([ent_attended.squeeze(1), wea_attended.squeeze(1)], dim=-1)
            return self.output_layer(combined)


class EnterpriseWeatherModel(nn.Module):
    """
    The unified model that combines enterprise data with weather intelligence.

    This is THE model that enterprises build - it understands both their
    business domain and weather/climate patterns, and how they interact.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__()
        self.config = config

        # Calculate input dimensions
        self.n_enterprise_features = len(config.enterprise_features)
        self.n_weather_features = len(config.weather_features) * (
            config.weather_context_window + config.weather_forecast_horizon
        )
        self.n_targets = len(config.target_variables)

        # Temporal and spatial encodings
        self.temporal_encoding = None
        self.spatial_encoding = None
        extra_dims = 0

        if config.use_temporal_encoding:
            self.temporal_encoding = TemporalEncoding(config.temporal_encoding_dim)
            extra_dims += config.temporal_encoding_dim

        if config.use_spatial_encoding:
            self.spatial_encoding = SpatialEncoding(config.spatial_encoding_dim)
            extra_dims += config.spatial_encoding_dim

        # Enterprise feature encoder
        enterprise_input_dim = self.n_enterprise_features + extra_dims
        self.enterprise_encoder = nn.Sequential(
            nn.Linear(enterprise_input_dim, config.hidden_dims[0]),
            nn.BatchNorm1d(config.hidden_dims[0]) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

        # Weather feature encoder
        self.weather_encoder = nn.Sequential(
            nn.Linear(self.n_weather_features, config.hidden_dims[0]),
            nn.BatchNorm1d(config.hidden_dims[0]) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

        # Weather integration layer
        self.weather_integration = WeatherIntegrationLayer(
            enterprise_dim=config.hidden_dims[0],
            weather_dim=config.hidden_dims[0],
            output_dim=config.hidden_dims[0],
            integration_type=config.weather_integration
        )

        # Main processing layers
        layers = []
        for i in range(len(config.hidden_dims) - 1):
            in_dim = config.hidden_dims[i]
            out_dim = config.hidden_dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))

            # Residual connection if dimensions match
            if config.use_residual and in_dim == out_dim:
                layers.append(ResidualWrapper(layers[-4:]))
                layers = layers[:-4]

        self.processor = nn.Sequential(*layers)

        # Output head
        if config.loss_type == "quantile":
            # Output multiple quantiles
            self.output_head = nn.Linear(
                config.hidden_dims[-1],
                self.n_targets * len(config.quantiles)
            )
        else:
            self.output_head = nn.Linear(config.hidden_dims[-1], self.n_targets)

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining enterprise and weather data.

        Args:
            enterprise_features: [B, n_enterprise_features] business data
            weather_features: [B, n_weather_features] weather data
            timestamps: [B] or [B, 4] temporal info (optional)
            locations: [B, 2] lat/lon (optional)

        Returns:
            [B, n_targets] predictions
        """
        # Add temporal encoding
        if self.temporal_encoding is not None and timestamps is not None:
            temporal_enc = self.temporal_encoding(timestamps)
            enterprise_features = torch.cat([enterprise_features, temporal_enc], dim=-1)

        # Add spatial encoding
        if self.spatial_encoding is not None and locations is not None:
            spatial_enc = self.spatial_encoding(locations)
            enterprise_features = torch.cat([enterprise_features, spatial_enc], dim=-1)

        # Encode enterprise features
        enterprise_encoded = self.enterprise_encoder(enterprise_features)

        # Encode weather features
        weather_encoded = self.weather_encoder(weather_features)

        # Integrate weather with enterprise features
        integrated = self.weather_integration(enterprise_encoded, weather_encoded)

        # Process through main network
        processed = self.processor(integrated)

        # Output
        output = self.output_head(processed)

        return output

    def predict_with_uncertainty(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using MC Dropout.

        Returns:
            mean: [B, n_targets] mean predictions
            std: [B, n_targets] standard deviation of predictions
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(
                    enterprise_features, weather_features, timestamps, locations
                )
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [n_samples, B, n_targets]

        self.eval()

        return predictions.mean(dim=0), predictions.std(dim=0)


class ResidualWrapper(nn.Module):
    """Wrap a sequence of layers with a residual connection."""

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EnterpriseModelBuilder:
    """
    The main builder class for creating enterprise custom models.

    This is the entry point for enterprises to create their models.
    """

    def __init__(self):
        self.registered_models: Dict[str, EnterpriseModelConfig] = {}

    def create_model(self, config: EnterpriseModelConfig) -> EnterpriseWeatherModel:
        """Create a model from configuration."""
        model = EnterpriseWeatherModel(config)
        self.registered_models[config.name] = config
        return model

    def create_from_template(
        self,
        template_name: str,
        company: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnterpriseModelConfig, EnterpriseWeatherModel]:
        """
        Create a model from a pre-defined template.

        Templates are optimized starting points for common use cases.
        """
        from .retail import RetailModelTemplates
        from .insurance import InsuranceModelTemplates

        # Get template
        if template_name in RetailModelTemplates.list_templates():
            config = RetailModelTemplates.get_template(template_name)
        elif template_name in InsuranceModelTemplates.list_templates():
            config = InsuranceModelTemplates.get_template(template_name)
        else:
            raise ValueError(f"Unknown template: {template_name}")

        # Apply customizations
        config.company = company
        if customizations:
            for key, value in customizations.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create model
        model = self.create_model(config)

        return config, model

    def combine_datasets(
        self,
        enterprise_data: EnterpriseDataset,
        weather_data: Dict[WeatherDataType, np.ndarray],
        config: EnterpriseModelConfig
    ) -> CombinedDataset:
        """
        Combine enterprise data with weather data into a unified dataset.

        This handles the alignment of enterprise records with corresponding
        weather observations based on timestamp and location.
        """
        # This is a simplified implementation - production would handle
        # more complex temporal and spatial alignment

        combined = CombinedDataset(
            name=f"{enterprise_data.name}_with_weather",
            description=f"Combined dataset: {enterprise_data.description}",
            enterprise_data=enterprise_data,
            weather_features=config.weather_features
        )

        # Build feature names
        for feat in config.enterprise_features:
            combined.feature_names.append(f"ent_{feat}")

        for wtype in config.weather_features:
            for day in range(-config.weather_context_window, config.weather_forecast_horizon):
                combined.feature_names.append(f"wx_{wtype.value}_day{day}")

        combined.target_names = config.target_variables

        return combined

    def get_loss_function(self, config: EnterpriseModelConfig) -> nn.Module:
        """Get the appropriate loss function for the model."""
        if config.loss_type == "mse":
            return nn.MSELoss()
        elif config.loss_type == "mae":
            return nn.L1Loss()
        elif config.loss_type == "huber":
            return nn.HuberLoss()
        elif config.loss_type == "quantile":
            return QuantileLoss(config.quantiles)
        else:
            return nn.MSELoss()

    def save_model(
        self,
        model: EnterpriseWeatherModel,
        config: EnterpriseModelConfig,
        path: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Save model checkpoint with config and metrics."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
            "version": config.version
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> Tuple[EnterpriseWeatherModel, EnterpriseModelConfig]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        # Reconstruct config
        config = EnterpriseModelConfig(**checkpoint["config"])

        # Create and load model
        model = EnterpriseWeatherModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, config


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic predictions."""

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: [B, n_targets * n_quantiles]
            targets: [B, n_targets]
        """
        n_quantiles = len(self.quantiles)
        n_targets = targets.size(-1)

        predictions = predictions.view(-1, n_targets, n_quantiles)
        targets = targets.unsqueeze(-1).expand_as(predictions)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets[..., i] - predictions[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors))

        return torch.stack(losses, dim=-1).mean()


# Convenience function for quick model creation
def build_enterprise_model(
    name: str,
    industry: str,
    purpose: str,
    enterprise_features: List[str],
    weather_features: List[str],
    targets: List[str],
    **kwargs
) -> Tuple[EnterpriseModelConfig, EnterpriseWeatherModel]:
    """
    Quick helper to build an enterprise model.

    Example:
        config, model = build_enterprise_model(
            name="winter_jacket_demand",
            industry="retail",
            purpose="demand_forecast",
            enterprise_features=["inventory_level", "unit_price", "is_promoted"],
            weather_features=["temperature", "precipitation", "wind_speed"],
            targets=["units_sold"]
        )
    """
    config = EnterpriseModelConfig(
        name=name,
        industry=IndustryVertical(industry),
        purpose=ModelPurpose(purpose),
        enterprise_features=enterprise_features,
        weather_features=[WeatherDataType(w) for w in weather_features],
        target_variables=targets,
        **kwargs
    )

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    return config, model

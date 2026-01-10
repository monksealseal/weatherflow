"""
Enterprise Data Models

Define the data schemas and structures that enterprises use to combine
their proprietary business data with weather and climate intelligence.

The goal: Make it dead simple for enterprises to describe their data
and automatically get a model architecture that can learn from both
their business data AND weather patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import torch
import torch.nn as nn
import numpy as np


class EnterpriseDataType(Enum):
    """Types of enterprise data that can be combined with weather."""

    # Retail Data Types
    SALES_HISTORY = "sales_history"
    INVENTORY_LEVELS = "inventory_levels"
    PRODUCT_CATALOG = "product_catalog"
    PRICING_DATA = "pricing_data"
    PROMOTIONAL_CALENDAR = "promotional_calendar"
    STORE_LOCATIONS = "store_locations"
    CUSTOMER_SEGMENTS = "customer_segments"
    GROSS_MARGIN = "gross_margin"
    PURCHASING_ORDERS = "purchasing_orders"

    # Insurance Data Types
    POLICY_DATA = "policy_data"
    CLAIMS_HISTORY = "claims_history"
    PROPERTY_LOCATIONS = "property_locations"
    COVERAGE_DETAILS = "coverage_details"
    RISK_SCORES = "risk_scores"
    PREMIUM_HISTORY = "premium_history"
    LOSS_RATIOS = "loss_ratios"
    UNDERWRITING_DATA = "underwriting_data"
    PORTFOLIO_COMPOSITION = "portfolio_composition"

    # Common Types
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


class WeatherDataType(Enum):
    """Weather and climate data types available for integration."""

    # Current/Forecast Weather
    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    HUMIDITY = "humidity"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    PRESSURE = "pressure"
    CLOUD_COVER = "cloud_cover"
    UV_INDEX = "uv_index"
    VISIBILITY = "visibility"
    DEW_POINT = "dew_point"

    # Climate Patterns
    CLIMATE_NORMALS = "climate_normals"
    SEASONAL_PATTERNS = "seasonal_patterns"
    HISTORICAL_EXTREMES = "historical_extremes"
    TREND_ANALYSIS = "trend_analysis"

    # Extreme Events
    STORM_EVENTS = "storm_events"
    FLOOD_RISK = "flood_risk"
    WILDFIRE_RISK = "wildfire_risk"
    HURRICANE_TRACKS = "hurricane_tracks"
    TORNADO_RISK = "tornado_risk"
    HAIL_RISK = "hail_risk"
    WINTER_STORM_RISK = "winter_storm_risk"

    # Derived Metrics
    HEATING_DEGREE_DAYS = "heating_degree_days"
    COOLING_DEGREE_DAYS = "cooling_degree_days"
    GROWING_DEGREE_DAYS = "growing_degree_days"
    WEATHER_VOLATILITY = "weather_volatility"
    COMFORT_INDEX = "comfort_index"


@dataclass
class DataField:
    """Definition of a single data field in an enterprise schema."""

    name: str
    data_type: str  # float, int, str, bool, datetime, list
    description: str
    required: bool = True
    is_target: bool = False  # True if this is what we're predicting
    is_temporal: bool = False  # True if this varies over time
    is_spatial: bool = False  # True if this varies by location
    units: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None  # For categorical fields
    embedding_dim: Optional[int] = None  # Dimension for learned embeddings


@dataclass
class RetailDataSchema:
    """
    Schema for retail enterprise data.

    Retailers care about:
    - Inventory optimization (not too much, not too little)
    - Gross margin protection
    - Understanding purchasing behavior
    - Demand forecasting
    """

    # Core identifiers
    store_id: DataField = field(default_factory=lambda: DataField(
        name="store_id",
        data_type="str",
        description="Unique identifier for each store location",
        is_spatial=True,
        embedding_dim=32
    ))

    product_id: DataField = field(default_factory=lambda: DataField(
        name="product_id",
        data_type="str",
        description="SKU or product identifier",
        embedding_dim=64
    ))

    category: DataField = field(default_factory=lambda: DataField(
        name="category",
        data_type="str",
        description="Product category (e.g., winter_jackets, umbrellas, sunscreen)",
        categories=["winter_apparel", "rain_gear", "summer_apparel",
                   "outdoor_equipment", "seasonal_decor", "beverages",
                   "frozen_foods", "fresh_produce", "home_heating",
                   "air_conditioning", "gardening", "automotive"],
        embedding_dim=16
    ))

    # Temporal data
    date: DataField = field(default_factory=lambda: DataField(
        name="date",
        data_type="datetime",
        description="Date of the observation",
        is_temporal=True
    ))

    # Sales and inventory
    units_sold: DataField = field(default_factory=lambda: DataField(
        name="units_sold",
        data_type="int",
        description="Number of units sold",
        min_value=0,
        is_target=True
    ))

    inventory_level: DataField = field(default_factory=lambda: DataField(
        name="inventory_level",
        data_type="int",
        description="Current inventory count",
        min_value=0
    ))

    reorder_point: DataField = field(default_factory=lambda: DataField(
        name="reorder_point",
        data_type="int",
        description="Inventory level that triggers reorder",
        min_value=0
    ))

    # Financial metrics
    unit_price: DataField = field(default_factory=lambda: DataField(
        name="unit_price",
        data_type="float",
        description="Selling price per unit",
        units="USD",
        min_value=0
    ))

    unit_cost: DataField = field(default_factory=lambda: DataField(
        name="unit_cost",
        data_type="float",
        description="Cost per unit",
        units="USD",
        min_value=0
    ))

    gross_margin: DataField = field(default_factory=lambda: DataField(
        name="gross_margin",
        data_type="float",
        description="Gross margin percentage",
        units="percent",
        min_value=0,
        max_value=100
    ))

    # Location data
    latitude: DataField = field(default_factory=lambda: DataField(
        name="latitude",
        data_type="float",
        description="Store latitude",
        is_spatial=True,
        min_value=-90,
        max_value=90
    ))

    longitude: DataField = field(default_factory=lambda: DataField(
        name="longitude",
        data_type="float",
        description="Store longitude",
        is_spatial=True,
        min_value=-180,
        max_value=180
    ))

    # Promotional data
    is_promoted: DataField = field(default_factory=lambda: DataField(
        name="is_promoted",
        data_type="bool",
        description="Whether product is on promotion"
    ))

    discount_percent: DataField = field(default_factory=lambda: DataField(
        name="discount_percent",
        data_type="float",
        description="Discount percentage if promoted",
        units="percent",
        min_value=0,
        max_value=100
    ))

    def get_all_fields(self) -> List[DataField]:
        """Return all defined fields."""
        return [getattr(self, f) for f in self.__dataclass_fields__]

    def get_target_fields(self) -> List[DataField]:
        """Return fields that can be prediction targets."""
        return [f for f in self.get_all_fields() if f.is_target]

    def get_feature_fields(self) -> List[DataField]:
        """Return fields that can be used as features."""
        return [f for f in self.get_all_fields() if not f.is_target]


@dataclass
class InsuranceDataSchema:
    """
    Schema for insurance broker enterprise data.

    Insurance brokers care about:
    - Getting the best price for their clients
    - Properly pricing risk based on location
    - Understanding weather/climate exposure
    - Portfolio risk management across locations
    """

    # Policy identifiers
    policy_id: DataField = field(default_factory=lambda: DataField(
        name="policy_id",
        data_type="str",
        description="Unique policy identifier",
        embedding_dim=32
    ))

    client_id: DataField = field(default_factory=lambda: DataField(
        name="client_id",
        data_type="str",
        description="Client or account identifier",
        embedding_dim=32
    ))

    # Property/Location data
    property_type: DataField = field(default_factory=lambda: DataField(
        name="property_type",
        data_type="str",
        description="Type of property (residential, commercial, industrial)",
        categories=["residential_single", "residential_multi", "commercial_retail",
                   "commercial_office", "industrial", "warehouse", "mixed_use",
                   "agricultural", "hospitality", "healthcare"],
        embedding_dim=16
    ))

    construction_type: DataField = field(default_factory=lambda: DataField(
        name="construction_type",
        data_type="str",
        description="Building construction type",
        categories=["frame", "masonry", "steel", "concrete", "mixed"],
        embedding_dim=8
    ))

    year_built: DataField = field(default_factory=lambda: DataField(
        name="year_built",
        data_type="int",
        description="Year the property was built",
        min_value=1800,
        max_value=2030
    ))

    square_footage: DataField = field(default_factory=lambda: DataField(
        name="square_footage",
        data_type="float",
        description="Property size in square feet",
        units="sq_ft",
        min_value=0
    ))

    # Location data
    latitude: DataField = field(default_factory=lambda: DataField(
        name="latitude",
        data_type="float",
        description="Property latitude",
        is_spatial=True,
        min_value=-90,
        max_value=90
    ))

    longitude: DataField = field(default_factory=lambda: DataField(
        name="longitude",
        data_type="float",
        description="Property longitude",
        is_spatial=True,
        min_value=-180,
        max_value=180
    ))

    elevation: DataField = field(default_factory=lambda: DataField(
        name="elevation",
        data_type="float",
        description="Property elevation above sea level",
        units="meters"
    ))

    distance_to_coast: DataField = field(default_factory=lambda: DataField(
        name="distance_to_coast",
        data_type="float",
        description="Distance to nearest coastline",
        units="km",
        min_value=0
    ))

    flood_zone: DataField = field(default_factory=lambda: DataField(
        name="flood_zone",
        data_type="str",
        description="FEMA flood zone designation",
        categories=["A", "AE", "AH", "AO", "AR", "A99", "V", "VE",
                   "X", "X500", "D", "UNKNOWN"],
        embedding_dim=8
    ))

    # Coverage and pricing
    coverage_amount: DataField = field(default_factory=lambda: DataField(
        name="coverage_amount",
        data_type="float",
        description="Total coverage amount",
        units="USD",
        min_value=0
    ))

    deductible: DataField = field(default_factory=lambda: DataField(
        name="deductible",
        data_type="float",
        description="Policy deductible amount",
        units="USD",
        min_value=0
    ))

    premium: DataField = field(default_factory=lambda: DataField(
        name="premium",
        data_type="float",
        description="Annual premium amount",
        units="USD",
        min_value=0,
        is_target=True
    ))

    # Risk metrics
    risk_score: DataField = field(default_factory=lambda: DataField(
        name="risk_score",
        data_type="float",
        description="Calculated risk score (0-100)",
        min_value=0,
        max_value=100,
        is_target=True
    ))

    # Claims history
    claims_count: DataField = field(default_factory=lambda: DataField(
        name="claims_count",
        data_type="int",
        description="Number of historical claims",
        min_value=0
    ))

    claims_total: DataField = field(default_factory=lambda: DataField(
        name="claims_total",
        data_type="float",
        description="Total historical claims amount",
        units="USD",
        min_value=0
    ))

    loss_ratio: DataField = field(default_factory=lambda: DataField(
        name="loss_ratio",
        data_type="float",
        description="Claims paid / Premium collected",
        min_value=0,
        is_target=True
    ))

    # Weather-specific risk factors
    wind_risk_factor: DataField = field(default_factory=lambda: DataField(
        name="wind_risk_factor",
        data_type="float",
        description="Wind damage risk multiplier",
        min_value=0,
        max_value=10
    ))

    hail_risk_factor: DataField = field(default_factory=lambda: DataField(
        name="hail_risk_factor",
        data_type="float",
        description="Hail damage risk multiplier",
        min_value=0,
        max_value=10
    ))

    flood_risk_factor: DataField = field(default_factory=lambda: DataField(
        name="flood_risk_factor",
        data_type="float",
        description="Flood damage risk multiplier",
        min_value=0,
        max_value=10
    ))

    wildfire_risk_factor: DataField = field(default_factory=lambda: DataField(
        name="wildfire_risk_factor",
        data_type="float",
        description="Wildfire damage risk multiplier",
        min_value=0,
        max_value=10
    ))

    def get_all_fields(self) -> List[DataField]:
        """Return all defined fields."""
        return [getattr(self, f) for f in self.__dataclass_fields__]

    def get_target_fields(self) -> List[DataField]:
        """Return fields that can be prediction targets."""
        return [f for f in self.get_all_fields() if f.is_target]

    def get_feature_fields(self) -> List[DataField]:
        """Return fields that can be used as features."""
        return [f for f in self.get_all_fields() if not f.is_target]


@dataclass
class EnterpriseDataset:
    """
    Container for enterprise data that will be combined with weather.

    This is what the enterprise provides - their business data with
    timestamps and locations so we can match it with weather data.
    """

    name: str
    description: str
    industry: str  # "retail" or "insurance"
    schema: Union[RetailDataSchema, InsuranceDataSchema]

    # The actual data
    data: Optional[Dict[str, np.ndarray]] = None

    # Temporal range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Spatial coverage
    locations: Optional[List[Dict[str, float]]] = None  # List of {lat, lon}

    # Metadata
    record_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def validate(self) -> List[str]:
        """Validate the dataset against its schema. Returns list of errors."""
        errors = []

        if self.data is None:
            errors.append("No data provided")
            return errors

        # Check required fields
        for field_def in self.schema.get_all_fields():
            if field_def.required and field_def.name not in self.data:
                errors.append(f"Required field '{field_def.name}' is missing")

        return errors

    def get_location_bounds(self) -> Dict[str, float]:
        """Get the bounding box of all locations in the dataset."""
        if not self.locations:
            return {"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180}

        lats = [loc["lat"] for loc in self.locations]
        lons = [loc["lon"] for loc in self.locations]

        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }


@dataclass
class CombinedDataset:
    """
    The unified dataset that combines enterprise data with weather data.

    This is the magic - we take the enterprise's business data and
    automatically align it with relevant weather and climate features.
    """

    name: str
    description: str

    # Source datasets
    enterprise_data: EnterpriseDataset
    weather_features: List[WeatherDataType]

    # Combined tensors ready for training
    features: Optional[torch.Tensor] = None  # [N, F] where F = enterprise + weather features
    targets: Optional[torch.Tensor] = None   # [N, T] where T = number of targets
    timestamps: Optional[torch.Tensor] = None  # [N] timestamps
    locations: Optional[torch.Tensor] = None  # [N, 2] lat/lon pairs

    # Feature metadata
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)

    # Statistics for normalization
    feature_means: Optional[torch.Tensor] = None
    feature_stds: Optional[torch.Tensor] = None
    target_means: Optional[torch.Tensor] = None
    target_stds: Optional[torch.Tensor] = None

    # Split indices
    train_indices: Optional[torch.Tensor] = None
    val_indices: Optional[torch.Tensor] = None
    test_indices: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.features) if self.features is not None else 0

    def get_num_enterprise_features(self) -> int:
        """Number of features from enterprise data."""
        if self.enterprise_data.schema is None:
            return 0
        return len(self.enterprise_data.schema.get_feature_fields())

    def get_num_weather_features(self) -> int:
        """Number of weather features included."""
        return len(self.weather_features)

    def normalize(self) -> "CombinedDataset":
        """Normalize features and targets to zero mean, unit variance."""
        if self.features is not None:
            self.feature_means = self.features.mean(dim=0)
            self.feature_stds = self.features.std(dim=0) + 1e-8
            self.features = (self.features - self.feature_means) / self.feature_stds

        if self.targets is not None:
            self.target_means = self.targets.mean(dim=0)
            self.target_stds = self.targets.std(dim=0) + 1e-8
            self.targets = (self.targets - self.target_means) / self.target_stds

        return self

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.target_means is not None and self.target_stds is not None:
            return predictions * self.target_stds + self.target_means
        return predictions

    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        temporal_split: bool = True
    ) -> None:
        """
        Create train/val/test splits.

        If temporal_split=True, uses time-based splitting (recommended for
        forecasting tasks). Otherwise uses random splitting.
        """
        n = len(self)

        if temporal_split and self.timestamps is not None:
            # Sort by time and split
            sorted_indices = torch.argsort(self.timestamps)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            self.train_indices = sorted_indices[:train_end]
            self.val_indices = sorted_indices[train_end:val_end]
            self.test_indices = sorted_indices[val_end:]
        else:
            # Random split
            indices = torch.randperm(n)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            self.train_indices = indices[:train_end]
            self.val_indices = indices[train_end:val_end]
            self.test_indices = indices[val_end:]

    def get_pytorch_datasets(self) -> tuple:
        """Return PyTorch TensorDatasets for train/val/test."""
        from torch.utils.data import TensorDataset

        if self.train_indices is None:
            self.create_splits()

        train_ds = TensorDataset(
            self.features[self.train_indices],
            self.targets[self.train_indices]
        )
        val_ds = TensorDataset(
            self.features[self.val_indices],
            self.targets[self.val_indices]
        )
        test_ds = TensorDataset(
            self.features[self.test_indices],
            self.targets[self.test_indices]
        )

        return train_ds, val_ds, test_ds

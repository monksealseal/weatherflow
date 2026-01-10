"""
Retail Enterprise Models

Specialized models for retail enterprises that need to combine their
business data with weather intelligence.

Key Use Cases:
1. Demand Forecasting - Predict how many winter jackets sell on a cold December day
2. Inventory Optimization - Keep the right stock levels for weather-sensitive products
3. Margin Protection - Understand weather impact on promotions and pricing
4. Purchasing Behavior - Learn how weather affects customer shopping patterns

The key insight: Weather is one of the biggest external factors affecting retail,
yet most retailers have no systematic way to incorporate it into their planning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np

from .model_builder import (
    EnterpriseModelConfig,
    EnterpriseWeatherModel,
    IndustryVertical,
    ModelPurpose,
    WeatherIntegrationLayer,
    TemporalEncoding,
)
from .data_models import WeatherDataType, RetailDataSchema


class RetailModelTemplates:
    """
    Pre-built model templates for common retail use cases.

    These templates are optimized starting points - retailers can customize
    them for their specific needs.
    """

    @staticmethod
    def list_templates() -> List[str]:
        """List available retail templates."""
        return [
            "demand_forecast_apparel",
            "demand_forecast_beverages",
            "demand_forecast_outdoor",
            "inventory_optimization",
            "margin_prediction",
            "purchasing_behavior",
            "seasonal_planning",
            "promotional_effectiveness"
        ]

    @staticmethod
    def get_template(name: str) -> EnterpriseModelConfig:
        """Get a specific template configuration."""
        templates = {
            "demand_forecast_apparel": RetailModelTemplates._demand_forecast_apparel(),
            "demand_forecast_beverages": RetailModelTemplates._demand_forecast_beverages(),
            "demand_forecast_outdoor": RetailModelTemplates._demand_forecast_outdoor(),
            "inventory_optimization": RetailModelTemplates._inventory_optimization(),
            "margin_prediction": RetailModelTemplates._margin_prediction(),
            "purchasing_behavior": RetailModelTemplates._purchasing_behavior(),
            "seasonal_planning": RetailModelTemplates._seasonal_planning(),
            "promotional_effectiveness": RetailModelTemplates._promotional_effectiveness()
        }

        if name not in templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")

        return templates[name]

    @staticmethod
    def _demand_forecast_apparel() -> EnterpriseModelConfig:
        """
        Demand forecasting for apparel (winter jackets, rain gear, etc.)

        This is the classic use case: predict how many winter jackets
        you'll sell based on weather forecasts.
        """
        return EnterpriseModelConfig(
            name="demand_forecast_apparel",
            description="Predict apparel demand based on weather conditions. "
                       "Optimized for weather-sensitive items like winter jackets, "
                       "rain gear, and summer clothing.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.DEMAND_FORECAST,

            # Enterprise features from retail data
            enterprise_features=[
                "inventory_level",
                "unit_price",
                "unit_cost",
                "is_promoted",
                "discount_percent",
                "category",  # Embedded categorical
            ],

            # Weather features most relevant to apparel
            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.HUMIDITY,
                WeatherDataType.HEATING_DEGREE_DAYS,
                WeatherDataType.COOLING_DEGREE_DAYS,
                WeatherDataType.COMFORT_INDEX,
            ],

            target_variables=["units_sold"],

            # Architecture tuned for demand forecasting
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            weather_integration="attention",  # Weather attention for apparel
            weather_context_window=14,  # 2 weeks of history
            weather_forecast_horizon=7,  # 1 week forecast

            use_temporal_encoding=True,
            use_seasonality=True,  # Important for apparel

            decision_context="Inventory planning and purchasing decisions for "
                           "weather-sensitive apparel items"
        )

    @staticmethod
    def _demand_forecast_beverages() -> EnterpriseModelConfig:
        """
        Demand forecasting for beverages.

        Hot drinks surge in cold weather, cold drinks surge in heat.
        Very temperature-sensitive category.
        """
        return EnterpriseModelConfig(
            name="demand_forecast_beverages",
            description="Predict beverage demand based on temperature and conditions. "
                       "Captures the strong relationship between weather and drink preferences.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.DEMAND_FORECAST,

            enterprise_features=[
                "inventory_level",
                "unit_price",
                "is_promoted",
                "category",  # hot_beverages, cold_beverages, etc.
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.HUMIDITY,
                WeatherDataType.COMFORT_INDEX,
                WeatherDataType.UV_INDEX,
            ],

            target_variables=["units_sold"],

            hidden_dims=[128, 64, 32],  # Simpler model - relationship is more direct
            weather_integration="gating",  # Gating works well for direct relationships
            weather_context_window=3,  # Short context - beverages are immediate
            weather_forecast_horizon=3,

            decision_context="Daily/weekly beverage ordering and inventory management"
        )

    @staticmethod
    def _demand_forecast_outdoor() -> EnterpriseModelConfig:
        """
        Demand forecasting for outdoor/recreation products.

        Camping gear, sports equipment, gardening supplies - all heavily
        dependent on weather conditions.
        """
        return EnterpriseModelConfig(
            name="demand_forecast_outdoor",
            description="Predict outdoor/recreation product demand. Considers both "
                       "current conditions and forecasts since customers plan activities ahead.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.DEMAND_FORECAST,

            enterprise_features=[
                "inventory_level",
                "unit_price",
                "is_promoted",
                "category",
                "gross_margin",
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.UV_INDEX,
                WeatherDataType.CLOUD_COVER,
                WeatherDataType.COMFORT_INDEX,
            ],

            target_variables=["units_sold"],

            hidden_dims=[256, 128, 64],
            weather_integration="attention",
            weather_context_window=7,
            weather_forecast_horizon=14,  # Longer forecast horizon - people plan outdoor activities

            decision_context="Seasonal inventory planning for outdoor recreation products"
        )

    @staticmethod
    def _inventory_optimization() -> EnterpriseModelConfig:
        """
        Inventory optimization model.

        Not just predicting demand, but optimizing inventory levels
        considering weather forecasts, lead times, and holding costs.
        """
        return EnterpriseModelConfig(
            name="inventory_optimization",
            description="Optimize inventory levels for weather-sensitive products. "
                       "Balances stockout risk against holding costs using weather forecasts.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.INVENTORY_OPTIMIZATION,

            enterprise_features=[
                "inventory_level",
                "reorder_point",
                "unit_price",
                "unit_cost",
                "category",
                "gross_margin",
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.SEASONAL_PATTERNS,
                WeatherDataType.WEATHER_VOLATILITY,
            ],

            # Multiple targets for optimization
            target_variables=[
                "optimal_order_quantity",
                "reorder_probability",
                "expected_stockout_days",
            ],

            hidden_dims=[256, 256, 128],
            weather_integration="cross_attention",
            weather_context_window=30,  # Longer history for patterns
            weather_forecast_horizon=14,

            # Use quantile regression for uncertainty
            loss_type="quantile",
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],

            decision_context="Automated reorder recommendations considering "
                           "weather uncertainty"
        )

    @staticmethod
    def _margin_prediction() -> EnterpriseModelConfig:
        """
        Gross margin prediction model.

        Weather affects not just demand but also margins - through
        promotional decisions, markdown timing, and pricing flexibility.
        """
        return EnterpriseModelConfig(
            name="margin_prediction",
            description="Predict gross margins for weather-sensitive categories. "
                       "Understand how weather conditions affect pricing power and markdown needs.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.MARGIN_PREDICTION,

            enterprise_features=[
                "unit_price",
                "unit_cost",
                "inventory_level",
                "is_promoted",
                "discount_percent",
                "category",
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.SEASONAL_PATTERNS,
                WeatherDataType.CLIMATE_NORMALS,  # Compare to normal
            ],

            target_variables=["gross_margin"],

            hidden_dims=[128, 64, 32],
            weather_integration="gating",
            weather_context_window=7,
            weather_forecast_horizon=7,

            decision_context="Pricing and markdown decisions for seasonal items"
        )

    @staticmethod
    def _purchasing_behavior() -> EnterpriseModelConfig:
        """
        Customer purchasing behavior model.

        Understand how weather affects shopping patterns - not just what
        people buy, but when and how they shop.
        """
        return EnterpriseModelConfig(
            name="purchasing_behavior",
            description="Model customer purchasing behavior changes due to weather. "
                       "Predict basket size, visit frequency, and category switching.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.PURCHASING_BEHAVIOR,

            enterprise_features=[
                "store_id",
                "category",
                "is_promoted",
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.COMFORT_INDEX,
                WeatherDataType.STORM_EVENTS,  # Severe weather changes behavior
            ],

            target_variables=[
                "traffic_index",
                "basket_size",
                "conversion_rate",
            ],

            hidden_dims=[256, 128, 64],
            weather_integration="attention",
            weather_context_window=7,
            weather_forecast_horizon=3,

            decision_context="Staffing, marketing, and store operations planning"
        )

    @staticmethod
    def _seasonal_planning() -> EnterpriseModelConfig:
        """
        Long-term seasonal planning model.

        Help retailers plan their seasonal assortments based on
        climate patterns and trends.
        """
        return EnterpriseModelConfig(
            name="seasonal_planning",
            description="Long-range planning for seasonal merchandise using climate patterns. "
                       "Plan assortments months ahead based on expected conditions.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.DEMAND_FORECAST,

            enterprise_features=[
                "category",
                "unit_cost",
                "gross_margin",
            ],

            weather_features=[
                WeatherDataType.CLIMATE_NORMALS,
                WeatherDataType.SEASONAL_PATTERNS,
                WeatherDataType.HISTORICAL_EXTREMES,
                WeatherDataType.TREND_ANALYSIS,
            ],

            target_variables=[
                "seasonal_demand_index",
                "optimal_buy_quantity",
            ],

            hidden_dims=[256, 128],
            weather_integration="concatenate",
            weather_context_window=365,  # Full year of history
            weather_forecast_horizon=90,  # Seasonal planning horizon

            use_seasonality=True,

            decision_context="Seasonal buy planning and assortment decisions"
        )

    @staticmethod
    def _promotional_effectiveness() -> EnterpriseModelConfig:
        """
        Promotional effectiveness model.

        How does weather affect promotional lift? A rain gear promotion
        works great during rainy weather but not during a heatwave.
        """
        return EnterpriseModelConfig(
            name="promotional_effectiveness",
            description="Model how weather conditions affect promotional effectiveness. "
                       "Optimize promotion timing based on weather forecasts.",
            industry=IndustryVertical.RETAIL,
            purpose=ModelPurpose.MARGIN_PREDICTION,

            enterprise_features=[
                "category",
                "discount_percent",
                "unit_price",
                "inventory_level",
            ],

            weather_features=[
                WeatherDataType.TEMPERATURE,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.CLIMATE_NORMALS,
            ],

            target_variables=[
                "promotional_lift",
                "incremental_margin",
            ],

            hidden_dims=[128, 64],
            weather_integration="gating",
            weather_context_window=7,
            weather_forecast_horizon=14,

            decision_context="Promotion scheduling and discount optimization"
        )


class DemandForecastModel(EnterpriseWeatherModel):
    """
    Specialized model for retail demand forecasting.

    Extends the base enterprise model with retail-specific features
    like product embeddings and promotional effect modeling.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Add product category embedding
        self.category_embedding = nn.Embedding(
            num_embeddings=20,  # Number of product categories
            embedding_dim=config.categorical_embedding_dim
        )

        # Promotional effect layer
        self.promo_effect = nn.Sequential(
            nn.Linear(2, 32),  # is_promoted, discount_percent
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Weather sensitivity layer - learns how sensitive this category is to weather
        self.weather_sensitivity = nn.Sequential(
            nn.Linear(config.hidden_dims[0], 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        category_ids: Optional[torch.Tensor] = None,
        promo_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with retail-specific features."""

        # Get base prediction
        base_output = super().forward(
            enterprise_features, weather_features, timestamps, locations
        )

        # Add category embedding effect
        if category_ids is not None:
            cat_emb = self.category_embedding(category_ids)
            # This could modulate the output based on category

        # Add promotional effect
        if promo_features is not None:
            promo_effect = self.promo_effect(promo_features)
            # This could add promotional lift to the prediction

        return base_output


class InventoryOptimizationModel(EnterpriseWeatherModel):
    """
    Specialized model for inventory optimization.

    Outputs not just demand predictions but optimal order quantities
    considering uncertainty from weather forecasts.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Safety stock calculator based on weather uncertainty
        self.safety_stock_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Safety stock must be positive
        )

        # Reorder urgency scorer
        self.urgency_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 urgency score
        )

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        current_inventory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning inventory optimization outputs.

        Returns dict with:
        - demand_forecast: Expected demand
        - safety_stock: Recommended safety stock level
        - reorder_urgency: How urgent is reorder (0-1)
        - optimal_order_qty: Recommended order quantity
        """
        # Get base features through the network
        # (Simplified - would use intermediate features in production)
        base_output = super().forward(
            enterprise_features, weather_features, timestamps, locations
        )

        # Calculate safety stock recommendation
        # (Would use proper intermediate features)
        safety_stock = torch.zeros_like(base_output)

        # Calculate reorder urgency
        urgency = torch.zeros(base_output.size(0), 1)

        return {
            "demand_forecast": base_output,
            "safety_stock": safety_stock,
            "reorder_urgency": urgency,
            "optimal_order_qty": base_output + safety_stock
        }


class MarginPredictionModel(EnterpriseWeatherModel):
    """
    Specialized model for gross margin prediction.

    Helps retailers understand how weather affects their margins -
    through pricing power, markdown needs, and promotional effectiveness.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Markdown risk predictor
        self.markdown_risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Pricing power indicator
        self.pricing_power_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # -1 to 1, negative = need to discount
        )


class PurchasingBehaviorModel(EnterpriseWeatherModel):
    """
    Model customer purchasing behavior changes due to weather.

    Predicts how weather affects traffic, basket size, and what
    categories customers shift between.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Traffic prediction head
        self.traffic_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Basket size prediction head
        self.basket_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Basket size must be positive
        )

        # Category switching probabilities
        self.category_switch_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 12),  # 12 main categories
            nn.Softmax(dim=-1)
        )


# Convenience functions for retailers

def create_winter_jacket_model(company: str) -> Tuple[EnterpriseModelConfig, DemandForecastModel]:
    """
    Create a model specifically for winter jacket demand forecasting.

    This is the canonical example - predict how many winter jackets
    will sell based on weather forecasts.
    """
    config = RetailModelTemplates.get_template("demand_forecast_apparel")
    config.name = f"{company}_winter_jacket_demand"
    config.company = company
    config.description = (
        f"Winter jacket demand forecasting model for {company}. "
        "Predicts daily/weekly sales based on temperature forecasts, "
        "precipitation, and wind chill."
    )
    config.decision_context = (
        "How many winter jackets should we order for December? "
        "When should we markdown unsold inventory?"
    )

    model = DemandForecastModel(config)
    return config, model


def create_beverage_demand_model(company: str) -> Tuple[EnterpriseModelConfig, DemandForecastModel]:
    """Create a model for beverage demand forecasting."""
    config = RetailModelTemplates.get_template("demand_forecast_beverages")
    config.name = f"{company}_beverage_demand"
    config.company = company

    model = DemandForecastModel(config)
    return config, model


def create_inventory_model(company: str) -> Tuple[EnterpriseModelConfig, InventoryOptimizationModel]:
    """Create an inventory optimization model."""
    config = RetailModelTemplates.get_template("inventory_optimization")
    config.name = f"{company}_inventory_optimizer"
    config.company = company

    model = InventoryOptimizationModel(config)
    return config, model

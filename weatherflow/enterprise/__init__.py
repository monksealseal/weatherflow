"""
WeatherFlow Enterprise Module

Enable enterprises to create custom AI models that combine their proprietary
business data with weather and climate intelligence.

Supported Verticals:
- Retail: Inventory optimization, demand forecasting, margin prediction
- Insurance: Risk assessment, pricing optimization, claims prediction

The key insight: Why fine-tune separate weather models OR business forecasting
models when you can build ONE unified model that leverages both?
"""

from .data_models import (
    EnterpriseDataType,
    WeatherDataType,
    RetailDataSchema,
    InsuranceDataSchema,
    EnterpriseDataset,
    CombinedDataset,
)

from .model_builder import (
    EnterpriseModelConfig,
    EnterpriseModelBuilder,
    ModelPurpose,
    IndustryVertical,
)

from .retail import (
    RetailModelTemplates,
    DemandForecastModel,
    InventoryOptimizationModel,
    MarginPredictionModel,
    PurchasingBehaviorModel,
)

from .insurance import (
    InsuranceModelTemplates,
    RiskAssessmentModel,
    PricingOptimizationModel,
    ClaimsPredictionModel,
    PortfolioRiskModel,
)

__all__ = [
    # Data Models
    "EnterpriseDataType",
    "WeatherDataType",
    "RetailDataSchema",
    "InsuranceDataSchema",
    "EnterpriseDataset",
    "CombinedDataset",
    # Model Builder
    "EnterpriseModelConfig",
    "EnterpriseModelBuilder",
    "ModelPurpose",
    "IndustryVertical",
    # Retail
    "RetailModelTemplates",
    "DemandForecastModel",
    "InventoryOptimizationModel",
    "MarginPredictionModel",
    "PurchasingBehaviorModel",
    # Insurance
    "InsuranceModelTemplates",
    "RiskAssessmentModel",
    "PricingOptimizationModel",
    "ClaimsPredictionModel",
    "PortfolioRiskModel",
]

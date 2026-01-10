"""
Insurance Enterprise Models

Specialized models for insurance brokers that need to combine their
policy and claims data with weather and climate intelligence.

Key Use Cases:
1. Risk Assessment - Score locations based on weather/climate exposure
2. Pricing Optimization - Get the best price that properly accounts for risk
3. Claims Prediction - Predict claims likelihood and severity
4. Portfolio Risk - Understand aggregate risk across all locations

The key insight: Insurance brokers need to ensure their clients are getting
prices that PROPERLY account for weather and climate risk at specific locations.
Too high = client overpays. Too low = unexpected losses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_builder import (
    EnterpriseModelConfig,
    EnterpriseWeatherModel,
    IndustryVertical,
    ModelPurpose,
    WeatherIntegrationLayer,
    SpatialEncoding,
)
from .data_models import WeatherDataType, InsuranceDataSchema


class InsuranceModelTemplates:
    """
    Pre-built model templates for common insurance use cases.

    These templates are optimized for insurance brokers who need to
    properly price weather and climate risk for their clients.
    """

    @staticmethod
    def list_templates() -> List[str]:
        """List available insurance templates."""
        return [
            "risk_assessment_property",
            "risk_assessment_flood",
            "risk_assessment_wind",
            "risk_assessment_wildfire",
            "pricing_optimization",
            "claims_prediction",
            "portfolio_risk_analysis",
            "location_scoring",
            "climate_trend_risk"
        ]

    @staticmethod
    def get_template(name: str) -> EnterpriseModelConfig:
        """Get a specific template configuration."""
        templates = {
            "risk_assessment_property": InsuranceModelTemplates._risk_assessment_property(),
            "risk_assessment_flood": InsuranceModelTemplates._risk_assessment_flood(),
            "risk_assessment_wind": InsuranceModelTemplates._risk_assessment_wind(),
            "risk_assessment_wildfire": InsuranceModelTemplates._risk_assessment_wildfire(),
            "pricing_optimization": InsuranceModelTemplates._pricing_optimization(),
            "claims_prediction": InsuranceModelTemplates._claims_prediction(),
            "portfolio_risk_analysis": InsuranceModelTemplates._portfolio_risk_analysis(),
            "location_scoring": InsuranceModelTemplates._location_scoring(),
            "climate_trend_risk": InsuranceModelTemplates._climate_trend_risk()
        }

        if name not in templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")

        return templates[name]

    @staticmethod
    def _risk_assessment_property() -> EnterpriseModelConfig:
        """
        Comprehensive property risk assessment.

        Combines multiple weather perils (wind, hail, flood, wildfire)
        into an overall risk score for a property.
        """
        return EnterpriseModelConfig(
            name="risk_assessment_property",
            description="Comprehensive property risk assessment combining all weather perils. "
                       "Provides location-specific risk scores based on climate history and projections.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "construction_type",
                "year_built",
                "square_footage",
                "elevation",
                "distance_to_coast",
                "flood_zone",
                "coverage_amount",
            ],

            weather_features=[
                WeatherDataType.STORM_EVENTS,
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.WILDFIRE_RISK,
                WeatherDataType.HURRICANE_TRACKS,
                WeatherDataType.TORNADO_RISK,
                WeatherDataType.HAIL_RISK,
                WeatherDataType.WINTER_STORM_RISK,
                WeatherDataType.HISTORICAL_EXTREMES,
                WeatherDataType.CLIMATE_NORMALS,
            ],

            target_variables=["risk_score"],

            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.3,
            weather_integration="cross_attention",
            weather_context_window=365 * 10,  # 10 years of history
            weather_forecast_horizon=365,  # 1 year projection

            use_spatial_encoding=True,
            spatial_encoding_dim=64,  # Important for location-based risk

            # Quantile loss for uncertainty in risk estimates
            loss_type="quantile",
            quantiles=[0.25, 0.5, 0.75, 0.95],

            decision_context="Property insurance underwriting and pricing decisions"
        )

    @staticmethod
    def _risk_assessment_flood() -> EnterpriseModelConfig:
        """
        Flood-specific risk assessment.

        Deep analysis of flood risk considering elevation, proximity to water,
        flood zone, historical floods, and precipitation patterns.
        """
        return EnterpriseModelConfig(
            name="risk_assessment_flood",
            description="Specialized flood risk assessment model. Combines FEMA flood zones, "
                       "elevation data, precipitation history, and climate projections.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "construction_type",
                "elevation",
                "flood_zone",
                "distance_to_coast",
                "coverage_amount",
            ],

            weather_features=[
                WeatherDataType.PRECIPITATION,
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.HURRICANE_TRACKS,
                WeatherDataType.HISTORICAL_EXTREMES,
                WeatherDataType.TREND_ANALYSIS,
            ],

            target_variables=["flood_risk_factor"],

            hidden_dims=[256, 128, 64],
            weather_integration="attention",
            weather_context_window=365 * 20,  # 20 years - floods are rare
            weather_forecast_horizon=365,

            use_spatial_encoding=True,

            decision_context="Flood insurance pricing and coverage decisions"
        )

    @staticmethod
    def _risk_assessment_wind() -> EnterpriseModelConfig:
        """
        Wind damage risk assessment.

        Evaluates risk from hurricanes, tornadoes, severe thunderstorms,
        and other wind events.
        """
        return EnterpriseModelConfig(
            name="risk_assessment_wind",
            description="Wind damage risk assessment including hurricanes, tornadoes, "
                       "and severe thunderstorm winds. Critical for coastal and tornado alley properties.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "construction_type",
                "year_built",
                "square_footage",
                "distance_to_coast",
                "coverage_amount",
            ],

            weather_features=[
                WeatherDataType.WIND_SPEED,
                WeatherDataType.HURRICANE_TRACKS,
                WeatherDataType.TORNADO_RISK,
                WeatherDataType.STORM_EVENTS,
                WeatherDataType.HISTORICAL_EXTREMES,
            ],

            target_variables=["wind_risk_factor"],

            hidden_dims=[256, 128, 64],
            weather_integration="attention",
            weather_context_window=365 * 30,  # 30 years for hurricanes
            weather_forecast_horizon=180,  # 6 month seasonal outlook

            use_spatial_encoding=True,

            decision_context="Wind coverage pricing and deductible recommendations"
        )

    @staticmethod
    def _risk_assessment_wildfire() -> EnterpriseModelConfig:
        """
        Wildfire risk assessment.

        Critical for properties in fire-prone areas. Considers vegetation,
        climate trends, and historical fire patterns.
        """
        return EnterpriseModelConfig(
            name="risk_assessment_wildfire",
            description="Wildfire risk assessment for properties in fire-prone areas. "
                       "Combines vegetation data, drought conditions, and historical fire patterns.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "construction_type",
                "square_footage",
                "elevation",
                "coverage_amount",
            ],

            weather_features=[
                WeatherDataType.WILDFIRE_RISK,
                WeatherDataType.TEMPERATURE,
                WeatherDataType.HUMIDITY,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.PRECIPITATION,
                WeatherDataType.TREND_ANALYSIS,  # Drought trends
            ],

            target_variables=["wildfire_risk_factor"],

            hidden_dims=[256, 128, 64],
            weather_integration="gating",
            weather_context_window=365 * 5,  # 5 years
            weather_forecast_horizon=90,  # Seasonal fire risk

            use_spatial_encoding=True,

            decision_context="Wildfire coverage availability and pricing in high-risk areas"
        )

    @staticmethod
    def _pricing_optimization() -> EnterpriseModelConfig:
        """
        Premium pricing optimization.

        Help brokers get the best price for their clients while
        properly accounting for weather/climate risk.
        """
        return EnterpriseModelConfig(
            name="pricing_optimization",
            description="Optimize insurance premium pricing based on weather/climate risk. "
                       "Ensure clients pay fair prices that properly reflect location-specific exposure.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.PRICING_OPTIMIZATION,

            enterprise_features=[
                "property_type",
                "construction_type",
                "year_built",
                "square_footage",
                "coverage_amount",
                "deductible",
                "flood_zone",
                "distance_to_coast",
                "claims_count",
                "claims_total",
            ],

            weather_features=[
                WeatherDataType.STORM_EVENTS,
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.WILDFIRE_RISK,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.HAIL_RISK,
                WeatherDataType.HISTORICAL_EXTREMES,
                WeatherDataType.TREND_ANALYSIS,
            ],

            target_variables=["premium", "loss_ratio"],

            hidden_dims=[512, 256, 128],
            weather_integration="cross_attention",
            weather_context_window=365 * 10,
            weather_forecast_horizon=365,

            use_spatial_encoding=True,

            # Multiple quantiles for price range
            loss_type="quantile",
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],

            decision_context="Negotiate fair premiums that properly price weather risk"
        )

    @staticmethod
    def _claims_prediction() -> EnterpriseModelConfig:
        """
        Claims prediction model.

        Predict likelihood and severity of claims based on
        weather conditions and property characteristics.
        """
        return EnterpriseModelConfig(
            name="claims_prediction",
            description="Predict insurance claims likelihood and severity based on weather events. "
                       "Help anticipate claims and manage reserves.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.CLAIMS_PREDICTION,

            enterprise_features=[
                "property_type",
                "construction_type",
                "year_built",
                "coverage_amount",
                "deductible",
                "flood_zone",
                "claims_count",  # Historical claims
                "claims_total",
            ],

            weather_features=[
                WeatherDataType.STORM_EVENTS,
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.HAIL_RISK,
                WeatherDataType.TEMPERATURE,  # Freeze damage
                WeatherDataType.PRECIPITATION,
            ],

            target_variables=[
                "claim_probability",
                "expected_claim_amount",
            ],

            hidden_dims=[256, 128, 64],
            weather_integration="attention",
            weather_context_window=30,  # Recent weather
            weather_forecast_horizon=14,

            use_temporal_encoding=True,
            use_spatial_encoding=True,

            decision_context="Claims anticipation and reserve management"
        )

    @staticmethod
    def _portfolio_risk_analysis() -> EnterpriseModelConfig:
        """
        Portfolio-level risk analysis.

        Understand aggregate risk across a portfolio of locations -
        critical for brokers managing multiple properties or clients.
        """
        return EnterpriseModelConfig(
            name="portfolio_risk_analysis",
            description="Analyze aggregate risk across a portfolio of insured properties. "
                       "Identify concentration risks and correlated weather exposures.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.PORTFOLIO_RISK,

            enterprise_features=[
                "property_type",
                "coverage_amount",
                "premium",
                "loss_ratio",
                "flood_zone",
            ],

            weather_features=[
                WeatherDataType.HURRICANE_TRACKS,
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.WILDFIRE_RISK,
                WeatherDataType.TORNADO_RISK,
                WeatherDataType.WEATHER_VOLATILITY,
            ],

            target_variables=[
                "portfolio_var",  # Value at Risk
                "concentration_risk",
                "correlation_factor",
            ],

            hidden_dims=[512, 256, 128],
            weather_integration="cross_attention",
            weather_context_window=365 * 20,
            weather_forecast_horizon=365,

            use_spatial_encoding=True,

            decision_context="Portfolio diversification and reinsurance decisions"
        )

    @staticmethod
    def _location_scoring() -> EnterpriseModelConfig:
        """
        Quick location scoring model.

        Fast scoring of locations for quick quotes and screening.
        Lighter-weight than full risk assessment.
        """
        return EnterpriseModelConfig(
            name="location_scoring",
            description="Quick location-based risk scoring for initial screening and quotes. "
                       "Provides fast, actionable risk scores without full underwriting analysis.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "coverage_amount",
            ],

            weather_features=[
                WeatherDataType.FLOOD_RISK,
                WeatherDataType.WILDFIRE_RISK,
                WeatherDataType.WIND_SPEED,
                WeatherDataType.HAIL_RISK,
            ],

            target_variables=["location_score"],

            hidden_dims=[128, 64],  # Simpler model for speed
            weather_integration="concatenate",
            weather_context_window=365 * 5,
            weather_forecast_horizon=0,  # No forecast needed

            use_spatial_encoding=True,

            decision_context="Quick screening for new business and renewals"
        )

    @staticmethod
    def _climate_trend_risk() -> EnterpriseModelConfig:
        """
        Climate trend risk model.

        Long-term view of how climate change affects risk at locations.
        Critical for long-tail policies and portfolio planning.
        """
        return EnterpriseModelConfig(
            name="climate_trend_risk",
            description="Assess long-term climate trend impacts on insurance risk. "
                       "Project how changing climate patterns will affect future exposure.",
            industry=IndustryVertical.INSURANCE,
            purpose=ModelPurpose.RISK_ASSESSMENT,

            enterprise_features=[
                "property_type",
                "construction_type",
                "year_built",
                "elevation",
                "distance_to_coast",
            ],

            weather_features=[
                WeatherDataType.TREND_ANALYSIS,
                WeatherDataType.CLIMATE_NORMALS,
                WeatherDataType.HISTORICAL_EXTREMES,
                WeatherDataType.SEASONAL_PATTERNS,
            ],

            target_variables=[
                "current_risk_score",
                "projected_risk_5yr",
                "projected_risk_10yr",
                "climate_sensitivity",
            ],

            hidden_dims=[512, 256, 128],
            weather_integration="cross_attention",
            weather_context_window=365 * 30,  # 30 years of climate history
            weather_forecast_horizon=365 * 10,  # 10 year projections

            use_spatial_encoding=True,
            spatial_encoding_dim=128,  # Rich spatial encoding

            decision_context="Long-term portfolio strategy and climate adaptation planning"
        )


class RiskAssessmentModel(EnterpriseWeatherModel):
    """
    Specialized model for insurance risk assessment.

    Extends the base model with insurance-specific features like
    peril decomposition and confidence intervals.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Peril-specific risk heads
        self.flood_risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 risk score
        )

        self.wind_risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.hail_risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.wildfire_risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 confidence score
        )

        # Risk aggregation layer
        self.risk_aggregator = nn.Sequential(
            nn.Linear(4, 32),  # 4 peril scores
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning detailed risk assessment.

        Returns dict with:
        - overall_risk: Combined risk score (0-100)
        - flood_risk: Flood-specific risk (0-100)
        - wind_risk: Wind-specific risk (0-100)
        - hail_risk: Hail-specific risk (0-100)
        - wildfire_risk: Wildfire-specific risk (0-100)
        - confidence: Confidence in the assessment (0-1)
        """
        # Get base output (this runs through the full network)
        # In production, we'd capture intermediate features
        base_output = super().forward(
            enterprise_features, weather_features, timestamps, locations
        )

        # For now, use base output as proxy for final hidden state
        # In production, would properly access intermediate features
        hidden = base_output

        # Individual peril risks (scaled to 0-100)
        flood_risk = self.flood_risk_head(hidden) * 100
        wind_risk = self.wind_risk_head(hidden) * 100
        hail_risk = self.hail_risk_head(hidden) * 100
        wildfire_risk = self.wildfire_risk_head(hidden) * 100

        # Aggregate risk
        peril_scores = torch.cat([flood_risk, wind_risk, hail_risk, wildfire_risk], dim=-1) / 100
        overall_risk = self.risk_aggregator(peril_scores) * 100

        # Confidence score
        confidence = self.confidence_head(hidden)

        return {
            "overall_risk": overall_risk,
            "flood_risk": flood_risk,
            "wind_risk": wind_risk,
            "hail_risk": hail_risk,
            "wildfire_risk": wildfire_risk,
            "confidence": confidence
        }


class PricingOptimizationModel(EnterpriseWeatherModel):
    """
    Specialized model for premium pricing optimization.

    Helps brokers find the right price point that properly
    reflects weather/climate risk.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Expected loss prediction
        self.expected_loss_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Must be positive
        )

        # Risk margin recommendation
        self.risk_margin_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 margin factor
        )

        # Fair premium calculator
        self.premium_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1] + 2, 64),  # + expected_loss + margin
            nn.ReLU(),
            nn.Linear(64, 3)  # Low, mid, high estimates
        )

    def forward(
        self,
        enterprise_features: torch.Tensor,
        weather_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        current_premium: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning pricing recommendations.

        Returns dict with:
        - expected_loss: Expected annual loss
        - risk_margin: Recommended risk margin
        - fair_premium_low: Lower bound of fair premium
        - fair_premium_mid: Best estimate of fair premium
        - fair_premium_high: Upper bound of fair premium
        - premium_adequacy: Is current premium adequate? (if provided)
        """
        base_output = super().forward(
            enterprise_features, weather_features, timestamps, locations
        )

        hidden = base_output

        # Expected loss
        expected_loss = self.expected_loss_head(hidden)

        # Risk margin
        risk_margin = self.risk_margin_head(hidden) * 0.5  # Max 50% margin

        # Fair premium estimates
        premium_input = torch.cat([hidden, expected_loss, risk_margin], dim=-1)
        premium_estimates = self.premium_head(premium_input)

        result = {
            "expected_loss": expected_loss,
            "risk_margin": risk_margin,
            "fair_premium_low": premium_estimates[:, 0:1],
            "fair_premium_mid": premium_estimates[:, 1:2],
            "fair_premium_high": premium_estimates[:, 2:3],
        }

        # Check premium adequacy if current premium provided
        if current_premium is not None:
            adequacy = (current_premium - premium_estimates[:, 1:2]) / premium_estimates[:, 1:2]
            result["premium_adequacy"] = adequacy  # Positive = adequate, negative = underpriced

        return result


class ClaimsPredictionModel(EnterpriseWeatherModel):
    """
    Model for predicting insurance claims based on weather events.

    Predicts both likelihood and severity of claims.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Claim likelihood (probability)
        self.claim_prob_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Claim severity (given a claim occurs)
        self.claim_severity_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Low, expected, high
            nn.Softplus()
        )

        # Peril classification (which peril is most likely)
        self.peril_classifier = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # flood, wind, hail, fire, other
            nn.Softmax(dim=-1)
        )


class PortfolioRiskModel(EnterpriseWeatherModel):
    """
    Model for portfolio-level risk analysis.

    Analyzes risk across multiple locations to understand
    concentration and correlation risks.
    """

    def __init__(self, config: EnterpriseModelConfig):
        super().__init__(config)

        # Portfolio VaR estimation
        self.var_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # VaR at different confidence levels
            nn.Softplus()
        )

        # Concentration risk scorer
        self.concentration_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 concentration score
        )

        # Geographic correlation estimator
        self.correlation_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 to 1 correlation
        )


# Convenience functions for insurance brokers

def create_property_risk_model(
    company: str,
    focus: str = "comprehensive"
) -> Tuple[EnterpriseModelConfig, RiskAssessmentModel]:
    """
    Create a property risk assessment model.

    Args:
        company: Company name
        focus: Risk focus - "comprehensive", "flood", "wind", "wildfire"
    """
    template_map = {
        "comprehensive": "risk_assessment_property",
        "flood": "risk_assessment_flood",
        "wind": "risk_assessment_wind",
        "wildfire": "risk_assessment_wildfire"
    }

    config = InsuranceModelTemplates.get_template(template_map.get(focus, "risk_assessment_property"))
    config.name = f"{company}_property_risk_{focus}"
    config.company = company
    config.description = (
        f"Property risk assessment model for {company}. "
        f"Focus: {focus}. Provides location-specific risk scores "
        "based on weather/climate exposure."
    )

    model = RiskAssessmentModel(config)
    return config, model


def create_pricing_model(company: str) -> Tuple[EnterpriseModelConfig, PricingOptimizationModel]:
    """Create a premium pricing optimization model."""
    config = InsuranceModelTemplates.get_template("pricing_optimization")
    config.name = f"{company}_pricing_optimizer"
    config.company = company
    config.description = (
        f"Premium pricing optimization model for {company}. "
        "Ensures clients get fair prices that properly reflect "
        "weather and climate risk at their specific locations."
    )
    config.decision_context = (
        "Is this premium fair? What premium properly prices the weather risk? "
        "Where are we over/under-pricing relative to actual exposure?"
    )

    model = PricingOptimizationModel(config)
    return config, model


def create_portfolio_model(company: str) -> Tuple[EnterpriseModelConfig, PortfolioRiskModel]:
    """Create a portfolio risk analysis model."""
    config = InsuranceModelTemplates.get_template("portfolio_risk_analysis")
    config.name = f"{company}_portfolio_risk"
    config.company = company

    model = PortfolioRiskModel(config)
    return config, model


def score_location(
    model: RiskAssessmentModel,
    lat: float,
    lon: float,
    property_type: str = "residential_single",
    coverage_amount: float = 500000
) -> Dict[str, float]:
    """
    Quick utility to score a single location.

    Returns risk scores for quick assessment.
    """
    # This would need proper implementation with real weather data
    # For now, returns placeholder structure
    return {
        "latitude": lat,
        "longitude": lon,
        "overall_risk": 0.0,
        "flood_risk": 0.0,
        "wind_risk": 0.0,
        "hail_risk": 0.0,
        "wildfire_risk": 0.0,
        "confidence": 0.0,
        "note": "Requires weather data integration"
    }

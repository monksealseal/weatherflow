"""
Enterprise Model Examples

Concrete, runnable examples showing how retailers and insurance brokers
can build custom AI models combining their data with weather intelligence.

These examples demonstrate the power of the platform and serve as
starting points for enterprise customers.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from .model_builder import (
    EnterpriseModelBuilder,
    EnterpriseModelConfig,
    EnterpriseWeatherModel,
    IndustryVertical,
    ModelPurpose,
    build_enterprise_model,
)
from .data_models import (
    WeatherDataType,
    EnterpriseDataset,
    CombinedDataset,
    RetailDataSchema,
    InsuranceDataSchema,
)
from .retail import (
    RetailModelTemplates,
    DemandForecastModel,
    create_winter_jacket_model,
)
from .insurance import (
    InsuranceModelTemplates,
    RiskAssessmentModel,
    create_property_risk_model,
)


# =============================================================================
# RETAIL EXAMPLES
# =============================================================================

def example_winter_jacket_demand():
    """
    Example: Predicting Winter Jacket Demand

    This is THE canonical example of weather-informed retail forecasting.
    A retailer wants to know: "How many winter jackets should I order for December?"

    The model learns:
    1. How temperature affects demand (colder = more jackets)
    2. How wind chill makes jackets more appealing
    3. Seasonal patterns in jacket shopping
    4. How promotions interact with weather conditions
    """
    print("=" * 60)
    print("EXAMPLE: Winter Jacket Demand Forecasting")
    print("=" * 60)

    # Create the model
    config, model = create_winter_jacket_model(company="ACME Retail")

    print(f"\nModel: {config.name}")
    print(f"Company: {config.company}")
    print(f"Purpose: {config.purpose.value}")
    print(f"\nEnterprise Features: {config.enterprise_features}")
    print(f"Weather Features: {[w.value for w in config.weather_features]}")
    print(f"Target: {config.target_variables}")

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    n_samples = 1000

    # Enterprise features
    inventory = np.random.randint(50, 500, n_samples).astype(np.float32)
    price = np.random.uniform(50, 200, n_samples).astype(np.float32)
    cost = price * np.random.uniform(0.4, 0.6, n_samples).astype(np.float32)
    is_promoted = (np.random.random(n_samples) > 0.8).astype(np.float32)
    discount = is_promoted * np.random.uniform(0.1, 0.3, n_samples).astype(np.float32)

    enterprise_features = torch.tensor(np.stack([
        inventory, price, cost, is_promoted, discount
    ], axis=1))

    # Weather features (simplified - just temperature for demo)
    # In production, would have full weather context window
    temperature = np.random.uniform(20, 70, n_samples).astype(np.float32)
    precipitation = np.random.uniform(0, 2, n_samples).astype(np.float32)
    wind_speed = np.random.uniform(0, 30, n_samples).astype(np.float32)

    # Replicate for context window (simplified)
    n_weather_features = len(config.weather_features) * (
        config.weather_context_window + config.weather_forecast_horizon
    )
    weather_features = torch.randn(n_samples, n_weather_features)

    # Generate target (units sold) based on weather relationship
    # Colder weather = more jackets sold
    temp_effect = -2 * (temperature - 50)  # Below 50F increases demand
    promo_effect = is_promoted * 50 * discount  # Promotions boost sales
    base_demand = 100
    units_sold = np.maximum(0, base_demand + temp_effect + promo_effect + np.random.normal(0, 10, n_samples))
    targets = torch.tensor(units_sold.reshape(-1, 1).astype(np.float32))

    print(f"Training data shape: {enterprise_features.shape}")
    print(f"Weather data shape: {weather_features.shape}")
    print(f"Target shape: {targets.shape}")

    # Forward pass demonstration
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(enterprise_features[:10], weather_features[:10])
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5].squeeze().tolist()}")
        print(f"Sample actuals: {targets[:5].squeeze().tolist()}")

    print("\nModel ready for training!")
    return config, model


def example_beverage_demand():
    """
    Example: Predicting Hot vs Cold Beverage Demand

    A coffee shop chain wants to optimize inventory:
    - Hot days = more iced coffee, lemonade
    - Cold days = more hot coffee, hot chocolate

    Temperature has a very direct effect on beverage preferences.
    """
    print("=" * 60)
    print("EXAMPLE: Beverage Demand by Temperature")
    print("=" * 60)

    config = RetailModelTemplates.get_template("demand_forecast_beverages")
    config.name = "starbeans_beverage_demand"
    config.company = "StarBeans Coffee"
    config.decision_context = (
        "How much hot coffee vs iced coffee should each store stock today? "
        "Minimize waste while ensuring we don't run out during peak hours."
    )

    model = DemandForecastModel(config)

    print(f"\nModel: {config.name}")
    print(f"Weather Features: {[w.value for w in config.weather_features]}")

    print("\nKey Insight: Every 5°F above 75°F shifts ~15% of coffee sales from hot to iced")
    print("The model learns this relationship automatically from your sales + weather data")

    return config, model


def example_inventory_optimization():
    """
    Example: Weather-Aware Inventory Optimization

    A hardware store needs to optimize seasonal inventory:
    - Snow shovels before snowstorms
    - Air conditioners before heat waves
    - Generators before hurricane season

    The model considers weather uncertainty to recommend safety stock.
    """
    print("=" * 60)
    print("EXAMPLE: Weather-Aware Inventory Optimization")
    print("=" * 60)

    config = RetailModelTemplates.get_template("inventory_optimization")
    config.name = "hardware_store_inventory"
    config.company = "BuildRight Hardware"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    print(f"\nModel: {config.name}")
    print(f"Purpose: Optimize stock levels considering weather forecast uncertainty")

    print("\nExample Decision:")
    print("  - 3-day forecast shows possible snowstorm (60% chance)")
    print("  - Current snow shovel inventory: 50 units")
    print("  - Model recommends: Order 200 more units (accounts for uncertainty)")

    print("\nKey Feature: Quantile regression provides confidence intervals")
    print("  - 90% confident demand will be between 80 and 250 units")
    print("  - Order enough to cover the upper bound minus safety margin")

    return config, model


# =============================================================================
# INSURANCE EXAMPLES
# =============================================================================

def example_property_risk_assessment():
    """
    Example: Property Risk Assessment

    An insurance broker needs to assess weather/climate risk for a property:
    - What's the flood risk at this address?
    - What's the wind/hurricane exposure?
    - How does wildfire risk compare to similar properties?

    The model provides location-specific risk scores.
    """
    print("=" * 60)
    print("EXAMPLE: Property Risk Assessment")
    print("=" * 60)

    config, model = create_property_risk_model(
        company="SecureChoice Insurance",
        focus="comprehensive"
    )

    print(f"\nModel: {config.name}")
    print(f"Company: {config.company}")
    print(f"Weather Features: {[w.value for w in config.weather_features]}")

    print("\nExample Output for a Miami Beach Property:")
    example_result = {
        "overall_risk": 78,
        "flood_risk": 85,
        "wind_risk": 82,
        "hail_risk": 15,
        "wildfire_risk": 5,
        "confidence": 0.92,
        "notes": "High exposure to hurricane and flood risk due to coastal location"
    }

    for key, value in example_result.items():
        if isinstance(value, (int, float)):
            if key == "confidence":
                print(f"  {key}: {value:.0%}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    return config, model


def example_pricing_optimization():
    """
    Example: Premium Pricing Optimization

    An insurance broker wants to ensure their client gets a fair price:
    - Is the quoted premium appropriate for the actual weather risk?
    - Where is the carrier over/under-pricing relative to exposure?

    The model analyzes whether premiums properly reflect risk.
    """
    print("=" * 60)
    print("EXAMPLE: Premium Pricing Optimization")
    print("=" * 60)

    config = InsuranceModelTemplates.get_template("pricing_optimization")
    config.name = "broker_pricing_advisor"
    config.company = "TrueRisk Brokers"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    print(f"\nModel: {config.name}")
    print(f"Purpose: Ensure clients pay fair premiums for actual risk")

    print("\nExample Analysis:")
    print("  Property: Commercial warehouse in Houston")
    print("  Carrier Quote: $45,000/year")
    print("  Model Assessment:")
    print("    - Expected Loss: $28,000/year")
    print("    - Risk Margin: 25%")
    print("    - Fair Premium Range: $35,000 - $42,000")
    print("    - Verdict: SLIGHTLY OVERPRICED (+7% above fair value)")
    print("    - Recommendation: Negotiate or seek alternative quotes")

    return config, model


def example_portfolio_risk():
    """
    Example: Portfolio Risk Analysis

    A broker managing multiple properties needs to understand aggregate risk:
    - Where is the portfolio concentrated?
    - What's the correlated weather exposure?
    - What happens if a major hurricane hits?

    The model provides portfolio-level risk metrics.
    """
    print("=" * 60)
    print("EXAMPLE: Portfolio Risk Analysis")
    print("=" * 60)

    config = InsuranceModelTemplates.get_template("portfolio_risk_analysis")
    config.name = "portfolio_weather_exposure"
    config.company = "Diversified Insurance Group"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    print(f"\nModel: {config.name}")
    print(f"Purpose: Understand aggregate weather exposure across portfolio")

    print("\nExample Portfolio Analysis (500 properties):")
    print("  Geographic Distribution:")
    print("    - Florida: 45% (HIGH hurricane concentration)")
    print("    - Texas: 25% (moderate hurricane + hail)")
    print("    - California: 20% (HIGH wildfire exposure)")
    print("    - Other: 10%")

    print("\n  Risk Metrics:")
    print("    - 1-in-100 year loss (VaR 99%): $45M")
    print("    - Concentration Risk Score: 78/100 (HIGH)")
    print("    - Correlation Factor: 0.65 (significant)")

    print("\n  Recommendation:")
    print("    - Portfolio is over-concentrated in hurricane-prone areas")
    print("    - Consider reinsurance for Florida book")
    print("    - Diversify into Midwest (lower correlated exposure)")

    return config, model


def example_climate_trend_analysis():
    """
    Example: Climate Trend Risk Analysis

    Long-term view: How is climate change affecting risk?
    - Are flood zones expanding?
    - Is wildfire season getting longer?
    - How should we adjust pricing for 10-year policies?
    """
    print("=" * 60)
    print("EXAMPLE: Climate Trend Risk Analysis")
    print("=" * 60)

    config = InsuranceModelTemplates.get_template("climate_trend_risk")
    config.name = "climate_risk_projector"
    config.company = "FutureReady Insurance"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    print(f"\nModel: {config.name}")
    print(f"Purpose: Project how climate trends affect future risk")

    print("\nExample 10-Year Projection for Coastal Florida Property:")
    print("  Current Risk Score: 75/100")
    print("  5-Year Projected: 82/100 (+9%)")
    print("  10-Year Projected: 88/100 (+17%)")

    print("\n  Key Drivers:")
    print("    - Sea level rise: +3 inches expected")
    print("    - Hurricane intensity: +8% increase in Cat 4+ probability")
    print("    - Flood zone expansion: Property now in expanded 100-year zone")

    print("\n  Pricing Implication:")
    print("    - Current premium may be adequate for next 2-3 years")
    print("    - Beyond 5 years, expect 15-25% premium increases")
    print("    - Consider climate escalation clauses in long-term policies")

    return config, model


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def quick_start_retail(company: str, use_case: str = "demand") -> Tuple[EnterpriseModelConfig, EnterpriseWeatherModel]:
    """
    Quick start for retailers.

    Args:
        company: Your company name
        use_case: "demand", "inventory", or "margin"

    Returns:
        config, model tuple ready for training
    """
    template_map = {
        "demand": "demand_forecast_apparel",
        "inventory": "inventory_optimization",
        "margin": "margin_prediction"
    }

    template = template_map.get(use_case, "demand_forecast_apparel")
    config = RetailModelTemplates.get_template(template)
    config.company = company
    config.name = f"{company.lower().replace(' ', '_')}_{use_case}"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    return config, model


def quick_start_insurance(company: str, use_case: str = "risk") -> Tuple[EnterpriseModelConfig, EnterpriseWeatherModel]:
    """
    Quick start for insurance brokers.

    Args:
        company: Your company name
        use_case: "risk", "pricing", "claims", or "portfolio"

    Returns:
        config, model tuple ready for training
    """
    template_map = {
        "risk": "risk_assessment_property",
        "pricing": "pricing_optimization",
        "claims": "claims_prediction",
        "portfolio": "portfolio_risk_analysis"
    }

    template = template_map.get(use_case, "risk_assessment_property")
    config = InsuranceModelTemplates.get_template(template)
    config.company = company
    config.name = f"{company.lower().replace(' ', '_')}_{use_case}"

    builder = EnterpriseModelBuilder()
    model = builder.create_model(config)

    return config, model


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================

def run_all_examples():
    """Run all examples to demonstrate the platform capabilities."""
    print("\n" + "=" * 70)
    print("WEATHERFLOW ENTERPRISE CUSTOM MODELS - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows how enterprises can build AI models that")
    print("combine their business data with weather and climate intelligence.")
    print("=" * 70)

    # Retail examples
    print("\n\n### RETAIL VERTICAL ###\n")
    example_winter_jacket_demand()
    print("\n")
    example_beverage_demand()
    print("\n")
    example_inventory_optimization()

    # Insurance examples
    print("\n\n### INSURANCE VERTICAL ###\n")
    example_property_risk_assessment()
    print("\n")
    example_pricing_optimization()
    print("\n")
    example_portfolio_risk()
    print("\n")
    example_climate_trend_analysis()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Enterprises don't need to choose between weather OR business models")
    print("2. Custom models are tailored to specific decisions, not generic forecasting")
    print("3. Weather features are automatically integrated with enterprise data")
    print("4. Models provide uncertainty quantification for better decision-making")
    print("5. Both retailers and insurance brokers benefit from weather intelligence")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()

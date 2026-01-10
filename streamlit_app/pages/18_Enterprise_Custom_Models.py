"""
Enterprise Custom Models - Streamlit Page

This is where enterprises build AI models that combine their business data
with weather and climate intelligence. The only place they can do it.

Two verticals:
1. Retail - Inventory, demand forecasting, margin prediction
2. Insurance - Risk assessment, pricing optimization, claims prediction

The key insight: Enterprises struggle with AI adoption AND with using weather data.
We solve both problems at once with custom, decision-specific models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from weatherflow.enterprise import (
        EnterpriseModelConfig,
        EnterpriseModelBuilder,
        ModelPurpose,
        IndustryVertical,
        RetailDataSchema,
        InsuranceDataSchema,
        WeatherDataType,
        RetailModelTemplates,
        InsuranceModelTemplates,
    )
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Enterprise Custom Models - WeatherFlow",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a5f 0%, #2e7d32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .value-prop {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .vertical-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    .vertical-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    .template-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .feature-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        font-size: 0.8rem;
    }
    .weather-tag {
        display: inline-block;
        background: #fff3e0;
        color: #e65100;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">Enterprise Custom Models</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Build AI models that combine YOUR data with weather intelligence. '
        'The only platform where enterprises can create decision-specific models.</p>',
        unsafe_allow_html=True
    )

    # Value proposition
    st.markdown("""
    <div class="value-prop">
        <h3 style="margin-top:0;">Why Choose Between Weather OR Business Models?</h3>
        <p style="margin-bottom:0;">
            Enterprises struggle to adopt AI. They don't know how to make it work with their data.
            And they don't know how to use weather and climate data to make informed decisions.
            <strong>We solve both problems at once.</strong>
            Create ONE unified model that understands your business AND weather patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Choose Your Vertical",
        "Build Custom Model",
        "Templates & Examples",
        "Your Models"
    ])

    with tab1:
        render_vertical_selection()

    with tab2:
        render_model_builder()

    with tab3:
        render_templates()

    with tab4:
        render_my_models()


def render_vertical_selection():
    """Render the vertical selection page."""
    st.markdown("### Select Your Industry")
    st.markdown("We've optimized our platform for two industries with very different needs:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="vertical-card">
            <h3>Retail</h3>
            <p><strong>Core Focus:</strong> Inventory, Gross Margin, Purchasing Behavior</p>
            <hr>
            <p><strong>Key Questions We Solve:</strong></p>
            <ul>
                <li>How many winter jackets should I order for December?</li>
                <li>How will this heat wave affect beverage sales?</li>
                <li>When should I markdown seasonal inventory?</li>
                <li>How does weather affect my store traffic?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select Retail", key="select_retail", use_container_width=True):
            st.session_state.selected_vertical = "retail"
            st.success("Retail vertical selected! Go to 'Build Custom Model' tab.")

        # Retail use cases
        with st.expander("Retail Use Cases"):
            st.markdown("""
            **Demand Forecasting**
            - Predict sales of weather-sensitive products
            - Plan inventory for seasonal items
            - Optimize purchasing decisions

            **Inventory Optimization**
            - Determine optimal stock levels based on weather forecasts
            - Reduce stockouts during unexpected weather
            - Minimize overstock and markdowns

            **Margin Protection**
            - Understand weather impact on promotional effectiveness
            - Optimize pricing for weather conditions
            - Time markdowns based on weather forecasts

            **Store Operations**
            - Predict store traffic based on weather
            - Optimize staffing for weather-driven demand
            - Plan marketing around weather events
            """)

    with col2:
        st.markdown("""
        <div class="vertical-card">
            <h3>Insurance</h3>
            <p><strong>Core Focus:</strong> Risk Pricing, Location Analysis, Portfolio Management</p>
            <hr>
            <p><strong>Key Questions We Solve:</strong></p>
            <ul>
                <li>Is this premium properly pricing the weather risk?</li>
                <li>What's the flood/wind/fire risk at this location?</li>
                <li>Where is my portfolio concentrated?</li>
                <li>How is climate change affecting my book?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select Insurance", key="select_insurance", use_container_width=True):
            st.session_state.selected_vertical = "insurance"
            st.success("Insurance vertical selected! Go to 'Build Custom Model' tab.")

        # Insurance use cases
        with st.expander("Insurance Use Cases"):
            st.markdown("""
            **Risk Assessment**
            - Score locations for weather/climate exposure
            - Quantify flood, wind, hail, wildfire risk
            - Assess climate trend impacts

            **Pricing Optimization**
            - Ensure premiums properly reflect weather risk
            - Identify over/under-priced policies
            - Get fair prices for clients

            **Claims Prediction**
            - Anticipate claims from weather events
            - Manage reserves proactively
            - Identify high-risk policies

            **Portfolio Management**
            - Understand aggregate weather exposure
            - Identify concentration risks
            - Plan reinsurance needs
            """)


def render_model_builder():
    """Render the model builder interface."""
    st.markdown("### Build Your Custom Model")

    # Check if vertical selected
    selected_vertical = st.session_state.get("selected_vertical", None)

    if not selected_vertical:
        st.info("Please select a vertical in the 'Choose Your Vertical' tab first.")
        return

    st.markdown(f"**Selected Vertical:** {selected_vertical.title()}")

    # Model configuration form
    with st.form("model_config"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Model Identity")
            model_name = st.text_input(
                "Model Name",
                placeholder="e.g., winter_jacket_demand_2024"
            )
            company = st.text_input(
                "Company Name",
                placeholder="Your company name"
            )
            description = st.text_area(
                "Description",
                placeholder="What decision will this model support?",
                height=100
            )

        with col2:
            st.markdown("#### Model Purpose")
            if selected_vertical == "retail":
                purposes = [
                    ("Demand Forecast", "demand_forecast"),
                    ("Inventory Optimization", "inventory_optimization"),
                    ("Margin Prediction", "margin_prediction"),
                    ("Purchasing Behavior", "purchasing_behavior")
                ]
            else:
                purposes = [
                    ("Risk Assessment", "risk_assessment"),
                    ("Pricing Optimization", "pricing_optimization"),
                    ("Claims Prediction", "claims_prediction"),
                    ("Portfolio Risk", "portfolio_risk")
                ]

            purpose = st.selectbox(
                "What are you trying to predict?",
                options=[p[1] for p in purposes],
                format_func=lambda x: next(p[0] for p in purposes if p[1] == x)
            )

        st.markdown("---")

        # Enterprise features selection
        st.markdown("#### Your Enterprise Data")
        st.markdown("Select the data elements from YOUR business that you'll provide:")

        if selected_vertical == "retail":
            enterprise_features = render_retail_features()
        else:
            enterprise_features = render_insurance_features()

        st.markdown("---")

        # Weather features selection
        st.markdown("#### Weather & Climate Features")
        st.markdown("Select weather data to combine with your business data:")

        weather_features = render_weather_features(selected_vertical, purpose)

        st.markdown("---")

        # Target variables
        st.markdown("#### What to Predict (Target Variables)")
        if selected_vertical == "retail":
            targets = render_retail_targets(purpose)
        else:
            targets = render_insurance_targets(purpose)

        st.markdown("---")

        # Advanced configuration
        with st.expander("Advanced Configuration"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)

            with adv_col1:
                st.markdown("**Architecture**")
                hidden_dims = st.text_input(
                    "Hidden Dimensions",
                    value="256, 128, 64",
                    help="Comma-separated list of hidden layer sizes"
                )
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)

            with adv_col2:
                st.markdown("**Weather Integration**")
                integration = st.selectbox(
                    "Integration Method",
                    ["concatenate", "attention", "gating", "cross_attention"],
                    index=1,
                    help="How to combine weather and business features"
                )
                context_window = st.number_input(
                    "Weather History (days)",
                    min_value=1,
                    max_value=365,
                    value=14
                )
                forecast_horizon = st.number_input(
                    "Weather Forecast (days)",
                    min_value=0,
                    max_value=90,
                    value=7
                )

            with adv_col3:
                st.markdown("**Training**")
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 1e-4, 1e-3, 1e-2],
                    value=1e-3
                )
                batch_size = st.selectbox(
                    "Batch Size",
                    [16, 32, 64, 128, 256],
                    index=2
                )
                loss_type = st.selectbox(
                    "Loss Function",
                    ["mse", "mae", "huber", "quantile"]
                )

        # Submit button
        submitted = st.form_submit_button("Create Model", use_container_width=True)

        if submitted:
            if not model_name or not company:
                st.error("Please provide a model name and company name.")
            elif not enterprise_features:
                st.error("Please select at least one enterprise data feature.")
            elif not weather_features:
                st.error("Please select at least one weather feature.")
            else:
                # Create the model configuration
                create_model_config(
                    name=model_name,
                    company=company,
                    description=description,
                    vertical=selected_vertical,
                    purpose=purpose,
                    enterprise_features=enterprise_features,
                    weather_features=weather_features,
                    targets=targets,
                    hidden_dims=[int(x.strip()) for x in hidden_dims.split(",")],
                    dropout=dropout,
                    integration=integration,
                    context_window=context_window,
                    forecast_horizon=forecast_horizon,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    loss_type=loss_type
                )


def render_retail_features() -> List[str]:
    """Render retail enterprise feature selection."""
    col1, col2, col3 = st.columns(3)

    features = []

    with col1:
        st.markdown("**Identifiers & Location**")
        if st.checkbox("Store ID", value=True):
            features.append("store_id")
        if st.checkbox("Product/SKU ID", value=True):
            features.append("product_id")
        if st.checkbox("Category"):
            features.append("category")
        if st.checkbox("Store Location (lat/lon)"):
            features.extend(["latitude", "longitude"])

    with col2:
        st.markdown("**Sales & Inventory**")
        if st.checkbox("Inventory Level", value=True):
            features.append("inventory_level")
        if st.checkbox("Reorder Point"):
            features.append("reorder_point")
        if st.checkbox("Historical Sales"):
            features.append("historical_sales")

    with col3:
        st.markdown("**Pricing & Promotions**")
        if st.checkbox("Unit Price", value=True):
            features.append("unit_price")
        if st.checkbox("Unit Cost"):
            features.append("unit_cost")
        if st.checkbox("Gross Margin"):
            features.append("gross_margin")
        if st.checkbox("Is Promoted"):
            features.append("is_promoted")
        if st.checkbox("Discount Percent"):
            features.append("discount_percent")

    return features


def render_insurance_features() -> List[str]:
    """Render insurance enterprise feature selection."""
    col1, col2, col3 = st.columns(3)

    features = []

    with col1:
        st.markdown("**Property Details**")
        if st.checkbox("Property Type", value=True):
            features.append("property_type")
        if st.checkbox("Construction Type"):
            features.append("construction_type")
        if st.checkbox("Year Built"):
            features.append("year_built")
        if st.checkbox("Square Footage"):
            features.append("square_footage")

    with col2:
        st.markdown("**Location & Risk**")
        if st.checkbox("Location (lat/lon)", value=True):
            features.extend(["latitude", "longitude"])
        if st.checkbox("Elevation"):
            features.append("elevation")
        if st.checkbox("Distance to Coast"):
            features.append("distance_to_coast")
        if st.checkbox("Flood Zone"):
            features.append("flood_zone")

    with col3:
        st.markdown("**Coverage & History**")
        if st.checkbox("Coverage Amount", value=True):
            features.append("coverage_amount")
        if st.checkbox("Deductible"):
            features.append("deductible")
        if st.checkbox("Claims Count"):
            features.append("claims_count")
        if st.checkbox("Claims Total"):
            features.append("claims_total")
        if st.checkbox("Current Premium"):
            features.append("premium")

    return features


def render_weather_features(vertical: str, purpose: str) -> List[str]:
    """Render weather feature selection."""
    col1, col2, col3 = st.columns(3)

    features = []

    with col1:
        st.markdown("**Current/Forecast Weather**")
        if st.checkbox("Temperature", value=True):
            features.append("temperature")
        if st.checkbox("Precipitation", value=True):
            features.append("precipitation")
        if st.checkbox("Humidity"):
            features.append("humidity")
        if st.checkbox("Wind Speed"):
            features.append("wind_speed")
        if st.checkbox("Cloud Cover"):
            features.append("cloud_cover")

    with col2:
        st.markdown("**Extreme Events**")
        if st.checkbox("Storm Events"):
            features.append("storm_events")
        if st.checkbox("Flood Risk", value=(vertical == "insurance")):
            features.append("flood_risk")
        if st.checkbox("Wildfire Risk", value=(vertical == "insurance")):
            features.append("wildfire_risk")
        if st.checkbox("Hurricane Tracks"):
            features.append("hurricane_tracks")
        if st.checkbox("Hail Risk"):
            features.append("hail_risk")

    with col3:
        st.markdown("**Climate Patterns**")
        if st.checkbox("Climate Normals"):
            features.append("climate_normals")
        if st.checkbox("Seasonal Patterns"):
            features.append("seasonal_patterns")
        if st.checkbox("Historical Extremes"):
            features.append("historical_extremes")
        if st.checkbox("Heating Degree Days", value=(vertical == "retail")):
            features.append("heating_degree_days")
        if st.checkbox("Cooling Degree Days", value=(vertical == "retail")):
            features.append("cooling_degree_days")

    return features


def render_retail_targets(purpose: str) -> List[str]:
    """Render retail target variable selection."""
    targets_by_purpose = {
        "demand_forecast": [
            ("Units Sold", "units_sold"),
            ("Revenue", "revenue"),
            ("Traffic", "traffic")
        ],
        "inventory_optimization": [
            ("Optimal Order Quantity", "optimal_order_qty"),
            ("Reorder Probability", "reorder_probability"),
            ("Stockout Risk", "stockout_risk")
        ],
        "margin_prediction": [
            ("Gross Margin", "gross_margin"),
            ("Markdown Risk", "markdown_risk"),
            ("Pricing Power", "pricing_power")
        ],
        "purchasing_behavior": [
            ("Conversion Rate", "conversion_rate"),
            ("Basket Size", "basket_size"),
            ("Visit Frequency", "visit_frequency")
        ]
    }

    options = targets_by_purpose.get(purpose, targets_by_purpose["demand_forecast"])

    selected = st.multiselect(
        "Select target variables to predict",
        options=[t[1] for t in options],
        format_func=lambda x: next(t[0] for t in options if t[1] == x),
        default=[options[0][1]]
    )

    return selected


def render_insurance_targets(purpose: str) -> List[str]:
    """Render insurance target variable selection."""
    targets_by_purpose = {
        "risk_assessment": [
            ("Overall Risk Score", "risk_score"),
            ("Flood Risk Factor", "flood_risk_factor"),
            ("Wind Risk Factor", "wind_risk_factor"),
            ("Wildfire Risk Factor", "wildfire_risk_factor")
        ],
        "pricing_optimization": [
            ("Fair Premium", "fair_premium"),
            ("Expected Loss", "expected_loss"),
            ("Loss Ratio", "loss_ratio")
        ],
        "claims_prediction": [
            ("Claim Probability", "claim_probability"),
            ("Expected Claim Amount", "expected_claim_amount"),
            ("Claim Severity", "claim_severity")
        ],
        "portfolio_risk": [
            ("Portfolio VaR", "portfolio_var"),
            ("Concentration Risk", "concentration_risk"),
            ("Correlation Factor", "correlation_factor")
        ]
    }

    options = targets_by_purpose.get(purpose, targets_by_purpose["risk_assessment"])

    selected = st.multiselect(
        "Select target variables to predict",
        options=[t[1] for t in options],
        format_func=lambda x: next(t[0] for t in options if t[1] == x),
        default=[options[0][1]]
    )

    return selected


def create_model_config(
    name: str,
    company: str,
    description: str,
    vertical: str,
    purpose: str,
    enterprise_features: List[str],
    weather_features: List[str],
    targets: List[str],
    hidden_dims: List[int],
    dropout: float,
    integration: str,
    context_window: int,
    forecast_horizon: int,
    learning_rate: float,
    batch_size: int,
    loss_type: str
):
    """Create and save model configuration."""

    config = {
        "name": name,
        "company": company,
        "description": description,
        "vertical": vertical,
        "purpose": purpose,
        "enterprise_features": enterprise_features,
        "weather_features": weather_features,
        "target_variables": targets,
        "architecture": {
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout,
            "weather_integration": integration,
            "weather_context_window": context_window,
            "weather_forecast_horizon": forecast_horizon
        },
        "training": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "loss_type": loss_type
        },
        "created_at": datetime.now().isoformat(),
        "status": "configured"
    }

    # Save to session state
    if "enterprise_models" not in st.session_state:
        st.session_state.enterprise_models = []

    st.session_state.enterprise_models.append(config)

    st.success(f"Model '{name}' created successfully!")

    # Show model summary
    st.markdown("### Model Summary")

    sum_col1, sum_col2 = st.columns(2)

    with sum_col1:
        st.markdown("**Enterprise Features:**")
        for feat in enterprise_features:
            st.markdown(f'<span class="feature-tag">{feat}</span>', unsafe_allow_html=True)

    with sum_col2:
        st.markdown("**Weather Features:**")
        for feat in weather_features:
            st.markdown(f'<span class="weather-tag">{feat}</span>', unsafe_allow_html=True)

    st.markdown("**Targets:**")
    st.write(", ".join(targets))

    # Next steps
    st.markdown("---")
    st.markdown("### Next Steps")
    st.markdown("""
    1. **Upload Your Data** - Provide your enterprise data (CSV or connect to your database)
    2. **Train the Model** - We'll combine your data with weather data and train
    3. **Deploy & Predict** - Use your model to make decisions
    """)

    # Show config JSON
    with st.expander("View Configuration JSON"):
        st.json(config)


def render_templates():
    """Render the templates and examples page."""
    st.markdown("### Pre-Built Templates")
    st.markdown("Start with an optimized template and customize for your needs.")

    tab1, tab2 = st.tabs(["Retail Templates", "Insurance Templates"])

    with tab1:
        render_retail_templates()

    with tab2:
        render_insurance_templates()


def render_retail_templates():
    """Render retail templates."""
    templates = [
        {
            "name": "Winter Jacket Demand",
            "template_id": "demand_forecast_apparel",
            "description": "Predict how many winter jackets will sell based on temperature forecasts. "
                          "The classic weather-retail use case.",
            "decision": "How many winter jackets should I order for December?",
            "weather_features": ["Temperature", "Wind Speed", "Heating Degree Days"],
            "example_insight": "When temperatures drop 10°F below normal, jacket sales increase 40%"
        },
        {
            "name": "Beverage Demand",
            "template_id": "demand_forecast_beverages",
            "description": "Predict hot vs cold beverage demand based on temperature and humidity.",
            "decision": "How much hot coffee vs iced coffee should I stock?",
            "weather_features": ["Temperature", "Humidity", "Comfort Index"],
            "example_insight": "Every 5°F above 75°F shifts 15% of coffee sales from hot to iced"
        },
        {
            "name": "Inventory Optimizer",
            "template_id": "inventory_optimization",
            "description": "Optimize stock levels for weather-sensitive products with uncertainty quantification.",
            "decision": "What's my optimal order quantity given the weather forecast uncertainty?",
            "weather_features": ["Temperature", "Precipitation", "Weather Volatility"],
            "example_insight": "High forecast uncertainty? Order more safety stock."
        },
        {
            "name": "Promotional Effectiveness",
            "template_id": "promotional_effectiveness",
            "description": "Understand how weather affects your promotional lift.",
            "decision": "When should I run my rain gear promotion?",
            "weather_features": ["Precipitation", "Climate Normals"],
            "example_insight": "Rain gear promos during rainy weeks see 3x normal lift"
        }
    ]

    for template in templates:
        with st.container():
            st.markdown(f"""
            <div class="template-card">
                <h4>{template['name']}</h4>
                <p>{template['description']}</p>
                <p><strong>Key Decision:</strong> <em>{template['decision']}</em></p>
                <p><strong>Weather Features:</strong> {', '.join(template['weather_features'])}</p>
                <p><strong>Example Insight:</strong> {template['example_insight']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Use This Template", key=f"use_{template['template_id']}"):
                st.session_state.selected_vertical = "retail"
                st.session_state.selected_template = template['template_id']
                st.success(f"Template '{template['name']}' selected! Go to 'Build Custom Model' tab to customize.")


def render_insurance_templates():
    """Render insurance templates."""
    templates = [
        {
            "name": "Property Risk Assessment",
            "template_id": "risk_assessment_property",
            "description": "Comprehensive risk scoring combining flood, wind, hail, and wildfire exposure.",
            "decision": "What's the true weather risk at this property?",
            "weather_features": ["Flood Risk", "Wind Speed", "Hail Risk", "Wildfire Risk", "Historical Extremes"],
            "example_insight": "This coastal property has 3x average hurricane exposure"
        },
        {
            "name": "Pricing Optimization",
            "template_id": "pricing_optimization",
            "description": "Ensure premiums properly reflect weather/climate risk at each location.",
            "decision": "Is this premium fair for the actual weather risk?",
            "weather_features": ["Storm Events", "Flood Risk", "Climate Trends"],
            "example_insight": "This policy is underpriced by 15% given flood zone changes"
        },
        {
            "name": "Claims Prediction",
            "template_id": "claims_prediction",
            "description": "Predict claim likelihood and severity based on weather events.",
            "decision": "Should I increase reserves before this storm season?",
            "weather_features": ["Storm Events", "Hurricane Tracks", "Temperature"],
            "example_insight": "Hurricane season forecast suggests 20% higher claims"
        },
        {
            "name": "Climate Trend Risk",
            "template_id": "climate_trend_risk",
            "description": "Long-term view of how climate change affects your book.",
            "decision": "How will my portfolio risk change over the next 10 years?",
            "weather_features": ["Climate Normals", "Trend Analysis", "Historical Extremes"],
            "example_insight": "Flood risk in this region projected to increase 40% by 2035"
        }
    ]

    for template in templates:
        with st.container():
            st.markdown(f"""
            <div class="template-card">
                <h4>{template['name']}</h4>
                <p>{template['description']}</p>
                <p><strong>Key Decision:</strong> <em>{template['decision']}</em></p>
                <p><strong>Weather Features:</strong> {', '.join(template['weather_features'])}</p>
                <p><strong>Example Insight:</strong> {template['example_insight']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Use This Template", key=f"use_{template['template_id']}"):
                st.session_state.selected_vertical = "insurance"
                st.session_state.selected_template = template['template_id']
                st.success(f"Template '{template['name']}' selected! Go to 'Build Custom Model' tab to customize.")


def render_my_models():
    """Render the user's created models."""
    st.markdown("### Your Custom Models")

    models = st.session_state.get("enterprise_models", [])

    if not models:
        st.info("You haven't created any models yet. Go to 'Build Custom Model' to create your first model!")

        # Show example models
        st.markdown("---")
        st.markdown("### Example: What a Model Looks Like")

        example_model = {
            "name": "acme_winter_jacket_demand",
            "company": "ACME Retail",
            "vertical": "retail",
            "purpose": "demand_forecast",
            "enterprise_features": ["inventory_level", "unit_price", "is_promoted", "category"],
            "weather_features": ["temperature", "precipitation", "wind_speed", "heating_degree_days"],
            "target_variables": ["units_sold"],
            "status": "trained",
            "metrics": {
                "mape": 8.5,
                "r2": 0.92
            }
        }

        render_model_card(example_model, is_example=True)

    else:
        for i, model in enumerate(models):
            render_model_card(model, index=i)


def render_model_card(model: Dict, index: int = 0, is_example: bool = False):
    """Render a single model card."""
    status_colors = {
        "configured": "#ffc107",
        "training": "#17a2b8",
        "trained": "#28a745",
        "deployed": "#007bff"
    }

    status = model.get("status", "configured")
    status_color = status_colors.get(status, "#6c757d")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"""
        ### {model['name']}
        **Company:** {model['company']} | **Vertical:** {model['vertical'].title()} | **Purpose:** {model['purpose'].replace('_', ' ').title()}

        **Enterprise Features:** {', '.join(model['enterprise_features'][:5])}{'...' if len(model['enterprise_features']) > 5 else ''}

        **Weather Features:** {', '.join(model['weather_features'][:5])}{'...' if len(model['weather_features']) > 5 else ''}

        **Targets:** {', '.join(model['target_variables'])}
        """)

    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <span style="background: {status_color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">
                {status.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if status == "trained" and "metrics" in model:
            st.metric("MAPE", f"{model['metrics']['mape']}%")
            st.metric("R²", f"{model['metrics']['r2']:.2f}")

    # Action buttons
    if not is_example:
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("Upload Data", key=f"upload_{index}"):
                st.info("Data upload feature coming soon!")

        with btn_col2:
            if st.button("Train Model", key=f"train_{index}"):
                st.info("Training feature coming soon!")

        with btn_col3:
            if st.button("View Details", key=f"details_{index}"):
                with st.expander("Model Details", expanded=True):
                    st.json(model)

    st.markdown("---")


def render_demo_visualization():
    """Render a demo visualization showing the power of combined models."""
    st.markdown("### See the Power of Combined Models")

    # Create synthetic demo data
    dates = pd.date_range(start="2023-01-01", periods=90, freq="D")

    # Simulate temperature (winter to spring transition)
    temps = 30 + 20 * np.sin(np.linspace(0, np.pi, 90)) + np.random.normal(0, 5, 90)

    # Simulate jacket sales (inversely related to temp with some noise)
    base_demand = 100
    temp_effect = -2 * (temps - 50)  # Higher when cold
    promo_effect = np.where(np.random.random(90) > 0.9, 30, 0)  # Random promos
    sales = np.maximum(0, base_demand + temp_effect + promo_effect + np.random.normal(0, 10, 90))

    # Model predictions (better when using weather)
    weather_model_pred = sales + np.random.normal(0, 5, 90)  # Good prediction
    no_weather_pred = np.full(90, sales.mean()) + np.random.normal(0, 15, 90)  # Poor prediction

    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Winter Jacket Sales vs Temperature", "Model Comparison"),
        vertical_spacing=0.15
    )

    # Top plot: Sales and Temperature
    fig.add_trace(
        go.Scatter(x=dates, y=sales, name="Actual Sales", line=dict(color="#1f77b4", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=temps, name="Temperature (°F)", line=dict(color="#ff7f0e", width=2), yaxis="y2"),
        row=1, col=1
    )

    # Bottom plot: Model comparison
    fig.add_trace(
        go.Scatter(x=dates, y=sales, name="Actual", line=dict(color="#1f77b4", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=weather_model_pred, name="With Weather Data", line=dict(color="#2ca02c", width=2, dash="dash")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=no_weather_pred, name="Without Weather Data", line=dict(color="#d62728", width=2, dash="dot")),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics comparison
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.markdown("#### With Weather + Enterprise Data")
        st.metric("MAPE", "8.5%", "-41.5%", delta_color="inverse")
        st.metric("R²", "0.92", "+0.47")

    with metric_col2:
        st.markdown("#### Enterprise Data Only")
        st.metric("MAPE", "50%")
        st.metric("R²", "0.45")


# Run the app
if __name__ == "__main__":
    main()

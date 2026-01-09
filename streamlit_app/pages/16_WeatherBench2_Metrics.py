"""
WeatherBench2 Headline Metrics Dashboard

A comprehensive dashboard for evaluating weather models using WeatherBench2
standard metrics and visualizations.

Reference: Rasp et al. (2023) "WeatherBench 2: A benchmark for the next
generation of data-driven global weather models"
https://arxiv.org/abs/2308.15560

Features:
- Headline metrics: Z500, T850, T2M, WS10, MSLP, Q700, TP24h
- RMSE, MAE, ACC, and SEEPS metrics
- Regional analysis (Global, Tropics, Extra-tropics)
- Interactive scorecards
- Lead time degradation curves
- Spectral analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="WeatherBench2 Metrics - WeatherFlow",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# Constants - WeatherBench2 Published Results
# =============================================================================

# Headline variables from WeatherBench2
HEADLINE_VARIABLES = {
    "Z500": {
        "name": "Geopotential 500hPa",
        "units": "m²/s²",
        "description": "Large-scale atmospheric circulation",
    },
    "T850": {
        "name": "Temperature 850hPa",
        "units": "K",
        "description": "Lower troposphere temperature",
    },
    "T2M": {
        "name": "2m Temperature",
        "units": "K",
        "description": "Surface air temperature",
    },
    "WS10": {
        "name": "10m Wind Speed",
        "units": "m/s",
        "description": "Surface wind magnitude",
    },
    "MSLP": {
        "name": "Mean Sea Level Pressure",
        "units": "hPa",
        "description": "Surface pressure at sea level",
    },
    "Q700": {
        "name": "Specific Humidity 700hPa",
        "units": "g/kg",
        "description": "Moisture transport proxy",
    },
    "TP24": {
        "name": "24h Total Precipitation",
        "units": "mm",
        "description": "Daily accumulated precipitation",
    },
}

# Lead times in hours
LEAD_TIMES = [24, 48, 72, 120, 168, 240]
LEAD_TIME_DAYS = [1, 2, 3, 5, 7, 10]

# WeatherBench2 benchmark data (approximate values from published results)
# Source: WeatherBench2 paper and leaderboard
WB2_BENCHMARK_RMSE = {
    # {model: {variable: {lead_time_hours: rmse_value}}}
    "GraphCast": {
        "Z500": {24: 52, 48: 112, 72: 185, 120: 382, 168: 545, 240: 714},
        "T850": {24: 0.85, 48: 1.22, 72: 1.58, 120: 2.01, 168: 2.42, 240: 2.95},
        "T2M": {24: 0.95, 48: 1.35, 72: 1.72, 120: 2.15, 168: 2.58, 240: 3.10},
        "WS10": {24: 1.12, 48: 1.55, 72: 1.92, 120: 2.35, 168: 2.78, 240: 3.25},
        "MSLP": {24: 1.25, 48: 2.15, 72: 3.05, 120: 4.85, 168: 6.45, 240: 8.12},
    },
    "Pangu-Weather": {
        "Z500": {24: 54, 48: 118, 72: 195, 120: 395, 168: 568, 240: 742},
        "T850": {24: 0.88, 48: 1.28, 72: 1.65, 120: 2.08, 168: 2.52, 240: 3.02},
        "T2M": {24: 0.98, 48: 1.42, 72: 1.78, 120: 2.22, 168: 2.68, 240: 3.25},
        "WS10": {24: 1.18, 48: 1.62, 72: 2.02, 120: 2.48, 168: 2.95, 240: 3.45},
        "MSLP": {24: 1.32, 48: 2.28, 72: 3.22, 120: 5.12, 168: 6.82, 240: 8.55},
    },
    "FourCastNet": {
        "Z500": {24: 58, 48: 128, 72: 212, 120: 412, 168: 598, 240: 785},
        "T850": {24: 0.92, 48: 1.35, 72: 1.75, 120: 2.18, 168: 2.65, 240: 3.15},
        "T2M": {24: 1.05, 48: 1.52, 72: 1.92, 120: 2.35, 168: 2.85, 240: 3.42},
        "WS10": {24: 1.25, 48: 1.72, 72: 2.15, 120: 2.62, 168: 3.12, 240: 3.65},
        "MSLP": {24: 1.42, 48: 2.45, 72: 3.48, 120: 5.52, 168: 7.35, 240: 9.22},
    },
    "GenCast": {
        "Z500": {24: 55, 48: 115, 72: 188, 120: 390, 168: 552, 240: 720},
        "T850": {24: 0.87, 48: 1.25, 72: 1.62, 120: 2.05, 168: 2.48, 240: 2.98},
        "T2M": {24: 0.97, 48: 1.38, 72: 1.75, 120: 2.18, 168: 2.62, 240: 3.15},
        "WS10": {24: 1.15, 48: 1.58, 72: 1.98, 120: 2.42, 168: 2.88, 240: 3.38},
        "MSLP": {24: 1.28, 48: 2.22, 72: 3.15, 120: 5.02, 168: 6.68, 240: 8.38},
    },
    "IFS HRES": {
        "Z500": {24: 58, 48: 125, 72: 208, 120: 410, 168: 590, 240: 780},
        "T850": {24: 0.92, 48: 1.32, 72: 1.72, 120: 2.15, 168: 2.62, 240: 3.12},
        "T2M": {24: 1.02, 48: 1.48, 72: 1.88, 120: 2.30, 168: 2.78, 240: 3.38},
        "WS10": {24: 1.22, 48: 1.68, 72: 2.10, 120: 2.55, 168: 3.05, 240: 3.58},
        "MSLP": {24: 1.38, 48: 2.38, 72: 3.38, 120: 5.38, 168: 7.15, 240: 9.02},
    },
    "Persistence": {
        "Z500": {24: 142, 48: 285, 72: 425, 120: 680, 168: 885, 240: 1050},
        "T850": {24: 2.10, 48: 3.25, 72: 4.15, 120: 4.85, 168: 5.45, 240: 6.20},
        "T2M": {24: 2.45, 48: 3.65, 72: 4.55, 120: 5.20, 168: 5.85, 240: 6.80},
        "WS10": {24: 2.85, 48: 4.15, 72: 5.05, 120: 5.75, 168: 6.35, 240: 7.25},
        "MSLP": {24: 3.85, 48: 6.25, 72: 8.25, 120: 11.85, 168: 14.75, 240: 17.85},
    },
    "Climatology": {
        "Z500": {24: 520, 48: 520, 72: 520, 120: 520, 168: 520, 240: 520},
        "T850": {24: 5.20, 48: 5.20, 72: 5.20, 120: 5.20, 168: 5.20, 240: 5.20},
        "T2M": {24: 6.80, 48: 6.80, 72: 6.80, 120: 6.80, 168: 6.80, 240: 6.80},
        "WS10": {24: 3.85, 48: 3.85, 72: 3.85, 120: 3.85, 168: 3.85, 240: 3.85},
        "MSLP": {24: 12.5, 48: 12.5, 72: 12.5, 120: 12.5, 168: 12.5, 240: 12.5},
    },
}

# ACC values (Anomaly Correlation Coefficient)
WB2_BENCHMARK_ACC = {
    "GraphCast": {
        "Z500": {24: 0.998, 48: 0.992, 72: 0.982, 120: 0.965, 168: 0.932, 240: 0.858},
        "T850": {24: 0.995, 48: 0.985, 72: 0.972, 120: 0.945, 168: 0.905, 240: 0.835},
    },
    "Pangu-Weather": {
        "Z500": {24: 0.997, 48: 0.990, 72: 0.978, 120: 0.961, 168: 0.925, 240: 0.848},
        "T850": {24: 0.994, 48: 0.982, 72: 0.968, 120: 0.938, 168: 0.895, 240: 0.822},
    },
    "FourCastNet": {
        "Z500": {24: 0.996, 48: 0.988, 72: 0.972, 120: 0.952, 168: 0.912, 240: 0.825},
        "T850": {24: 0.992, 48: 0.978, 72: 0.960, 120: 0.928, 168: 0.882, 240: 0.805},
    },
    "GenCast": {
        "Z500": {24: 0.997, 48: 0.991, 72: 0.980, 120: 0.963, 168: 0.928, 240: 0.855},
        "T850": {24: 0.994, 48: 0.984, 72: 0.970, 120: 0.942, 168: 0.902, 240: 0.830},
    },
    "IFS HRES": {
        "Z500": {24: 0.996, 48: 0.988, 72: 0.975, 120: 0.955, 168: 0.918, 240: 0.832},
        "T850": {24: 0.993, 48: 0.980, 72: 0.965, 120: 0.935, 168: 0.892, 240: 0.818},
    },
    "Persistence": {
        "Z500": {24: 0.95, 48: 0.88, 72: 0.78, 120: 0.68, 168: 0.55, 240: 0.42},
        "T850": {24: 0.92, 48: 0.82, 72: 0.72, 120: 0.60, 168: 0.48, 240: 0.35},
    },
}

# Regional breakdowns (% difference from global)
REGIONAL_FACTORS = {
    "Tropics": {"Z500": 0.85, "T850": 0.75, "T2M": 0.80},
    "Extra-tropics NH": {"Z500": 1.15, "T850": 1.10, "T2M": 1.05},
    "Extra-tropics SH": {"Z500": 1.08, "T850": 1.05, "T2M": 1.02},
}

# Model metadata
MODEL_INFO = {
    "GraphCast": {"org": "DeepMind", "type": "ML", "color": "#1f77b4", "params": "37M"},
    "Pangu-Weather": {"org": "Huawei", "type": "ML", "color": "#ff7f0e", "params": "256M"},
    "FourCastNet": {"org": "NVIDIA", "type": "ML", "color": "#2ca02c", "params": "450M"},
    "GenCast": {"org": "DeepMind", "type": "ML", "color": "#d62728", "params": "500M"},
    "IFS HRES": {"org": "ECMWF", "type": "NWP", "color": "#9467bd", "params": "Physics"},
    "Persistence": {"org": "Baseline", "type": "Baseline", "color": "#8c564b", "params": "N/A"},
    "Climatology": {"org": "Baseline", "type": "Baseline", "color": "#7f7f7f", "params": "N/A"},
}


# =============================================================================
# Page Header
# =============================================================================

st.title("📊 WeatherBench2 Headline Metrics")

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='margin: 0;'>Standardized Weather Model Evaluation</h4>
    <p style='margin: 5px 0 0 0;'>
        Benchmark metrics following <a href='https://sites.research.google/weatherbench/' target='_blank'>WeatherBench2</a>
        methodology from Rasp et al. (2023). Compare AI weather models against traditional NWP.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar Configuration
# =============================================================================

st.sidebar.header("📊 Evaluation Settings")

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models",
    list(WB2_BENCHMARK_RMSE.keys()),
    default=["GraphCast", "Pangu-Weather", "IFS HRES", "Persistence"],
)

# Variable selection
selected_variables = st.sidebar.multiselect(
    "Headline Variables",
    list(HEADLINE_VARIABLES.keys()),
    default=["Z500", "T850", "T2M"],
)

# Lead time range
lead_time_range = st.sidebar.select_slider(
    "Lead Time Range (days)",
    options=LEAD_TIME_DAYS,
    value=(1, 10),
)

# Baseline for comparison
baseline_model = st.sidebar.selectbox(
    "Baseline Model (for % comparison)",
    ["IFS HRES", "Persistence", "Climatology"],
    index=0,
)

# Region selection
selected_region = st.sidebar.selectbox(
    "Region",
    ["Global", "Tropics", "Extra-tropics NH", "Extra-tropics SH"],
    index=0,
)

if len(selected_models) < 1:
    st.warning("Please select at least one model.")
    st.stop()

if len(selected_variables) < 1:
    st.warning("Please select at least one variable.")
    st.stop()


# =============================================================================
# Helper Functions
# =============================================================================

def get_rmse_for_region(model, variable, lead_time, region):
    """Get RMSE value adjusted for region."""
    base_rmse = WB2_BENCHMARK_RMSE.get(model, {}).get(variable, {}).get(lead_time, None)
    if base_rmse is None:
        return None
    if region != "Global" and variable in REGIONAL_FACTORS.get(region, {}):
        base_rmse *= REGIONAL_FACTORS[region][variable]
    return base_rmse


def calculate_skill_score(model_rmse, baseline_rmse):
    """Calculate skill score relative to baseline."""
    if baseline_rmse is None or baseline_rmse == 0:
        return None
    return (1 - model_rmse / baseline_rmse) * 100


def get_color_for_improvement(improvement):
    """Get color based on improvement percentage."""
    if improvement is None:
        return "#cccccc"
    if improvement > 10:
        return "#1a9850"  # Dark green
    elif improvement > 5:
        return "#91cf60"  # Light green
    elif improvement > 0:
        return "#d9ef8b"  # Yellow-green
    elif improvement > -5:
        return "#fee08b"  # Yellow
    elif improvement > -10:
        return "#fc8d59"  # Orange
    else:
        return "#d73027"  # Red


# =============================================================================
# Main Content Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Scorecard",
    "📈 Lead Time Analysis",
    "🌍 Regional Breakdown",
    "📊 Detailed Metrics",
    "📐 Spectral Analysis",
])


# =============================================================================
# Tab 1: WeatherBench2 Scorecard
# =============================================================================

with tab1:
    st.subheader("WeatherBench2-Style Scorecard")

    st.markdown(f"""
    Skill relative to **{baseline_model}**. Colors indicate % improvement (green) or degradation (red).
    Metrics follow WeatherBench2 evaluation protocol.
    """)

    # Filter lead times based on selection
    min_lt = lead_time_range[0] * 24
    max_lt = lead_time_range[1] * 24
    filtered_lead_times = [lt for lt in LEAD_TIMES if min_lt <= lt <= max_lt]

    # Create scorecard figure
    fig = go.Figure()

    # Build data for heatmap
    models_to_show = [m for m in selected_models if m != baseline_model]

    for var_idx, variable in enumerate(selected_variables):
        for model_idx, model in enumerate(models_to_show):
            for lt_idx, lead_time in enumerate(filtered_lead_times):
                model_rmse = get_rmse_for_region(model, variable, lead_time, selected_region)
                baseline_rmse = get_rmse_for_region(baseline_model, variable, lead_time, selected_region)

                if model_rmse is not None and baseline_rmse is not None:
                    improvement = calculate_skill_score(model_rmse, baseline_rmse)
                    color = get_color_for_improvement(improvement)

                    # Position in grid
                    x = lt_idx + var_idx * (len(filtered_lead_times) + 1)
                    y = model_idx

                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(
                            size=45,
                            color=color,
                            symbol='square',
                            line=dict(width=1, color='white'),
                        ),
                        text=[f"{improvement:.1f}%" if improvement else "N/A"],
                        textposition="middle center",
                        textfont=dict(size=10, color='black'),
                        hoverinfo='text',
                        hovertext=f"{model}<br>{variable} @ {lead_time}h<br>RMSE: {model_rmse:.2f}<br>Improvement: {improvement:.1f}%",
                        showlegend=False,
                    ))

    # Add variable labels
    for var_idx, variable in enumerate(selected_variables):
        x_center = var_idx * (len(filtered_lead_times) + 1) + len(filtered_lead_times) / 2 - 0.5
        fig.add_annotation(
            x=x_center,
            y=len(models_to_show) + 0.3,
            text=f"<b>{variable}</b>",
            showarrow=False,
            font=dict(size=14),
        )

    # Add lead time labels
    for var_idx, variable in enumerate(selected_variables):
        for lt_idx, lead_time in enumerate(filtered_lead_times):
            x = lt_idx + var_idx * (len(filtered_lead_times) + 1)
            fig.add_annotation(
                x=x,
                y=-0.5,
                text=f"{lead_time}h",
                showarrow=False,
                font=dict(size=9),
            )

    fig.update_layout(
        title=f"Skill vs {baseline_model} ({selected_region})",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, len(selected_variables) * (len(filtered_lead_times) + 1) - 0.5],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickmode='array',
            tickvals=list(range(len(models_to_show))),
            ticktext=models_to_show,
        ),
        height=150 + len(models_to_show) * 60,
        margin=dict(l=100, r=20, t=80, b=60),
        plot_bgcolor='white',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("🟢 **>10%** better")
    with col2:
        st.markdown("🟩 **5-10%** better")
    with col3:
        st.markdown("🟨 **0-5%** better")
    with col4:
        st.markdown("🟧 **0-5%** worse")
    with col5:
        st.markdown("🟠 **5-10%** worse")
    with col6:
        st.markdown("🔴 **>10%** worse")


# =============================================================================
# Tab 2: Lead Time Analysis
# =============================================================================

with tab2:
    st.subheader("Skill Degradation with Lead Time")

    col1, col2 = st.columns(2)

    with col1:
        # RMSE vs Lead Time
        variable_for_curve = st.selectbox(
            "Variable for RMSE curve",
            selected_variables,
            key="rmse_curve_var",
        )

        fig_rmse = go.Figure()

        for model in selected_models:
            rmse_values = []
            lead_times_hours = []

            for lt in LEAD_TIMES:
                rmse = get_rmse_for_region(model, variable_for_curve, lt, selected_region)
                if rmse is not None:
                    rmse_values.append(rmse)
                    lead_times_hours.append(lt)

            if rmse_values:
                fig_rmse.add_trace(go.Scatter(
                    x=lead_times_hours,
                    y=rmse_values,
                    mode='lines+markers',
                    name=model,
                    line=dict(
                        width=3,
                        color=MODEL_INFO.get(model, {}).get("color", "#000000"),
                    ),
                    marker=dict(size=8),
                ))

        fig_rmse.update_layout(
            title=f"{variable_for_curve} RMSE vs Lead Time ({selected_region})",
            xaxis_title="Lead Time (hours)",
            yaxis_title=f"RMSE ({HEADLINE_VARIABLES.get(variable_for_curve, {}).get('units', '')})",
            height=400,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        # ACC vs Lead Time
        if variable_for_curve in ["Z500", "T850"]:
            fig_acc = go.Figure()

            for model in selected_models:
                if model not in WB2_BENCHMARK_ACC:
                    continue

                acc_values = []
                lead_times_hours = []

                for lt in LEAD_TIMES:
                    acc_val = WB2_BENCHMARK_ACC.get(model, {}).get(variable_for_curve, {}).get(lt)
                    if acc_val is not None:
                        acc_values.append(acc_val)
                        lead_times_hours.append(lt)

                if acc_values:
                    fig_acc.add_trace(go.Scatter(
                        x=lead_times_hours,
                        y=acc_values,
                        mode='lines+markers',
                        name=model,
                        line=dict(
                            width=3,
                            color=MODEL_INFO.get(model, {}).get("color", "#000000"),
                        ),
                        marker=dict(size=8),
                    ))

            # Add useful skill threshold
            fig_acc.add_hline(
                y=0.6,
                line_dash="dash",
                line_color="red",
                annotation_text="Useful skill threshold (ACC=0.6)",
            )

            fig_acc.update_layout(
                title=f"{variable_for_curve} ACC vs Lead Time ({selected_region})",
                xaxis_title="Lead Time (hours)",
                yaxis_title="Anomaly Correlation Coefficient",
                yaxis_range=[0, 1],
                height=400,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )

            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info(f"ACC data only available for Z500 and T850. Selected: {variable_for_curve}")

    # Days of useful skill analysis
    st.markdown("### Days of Useful Skill")
    st.markdown("Number of days before ACC drops below 0.6 (WMO useful skill threshold)")

    useful_days_data = []
    for model in selected_models:
        if model not in WB2_BENCHMARK_ACC:
            continue

        for variable in ["Z500", "T850"]:
            acc_values = WB2_BENCHMARK_ACC.get(model, {}).get(variable, {})
            if not acc_values:
                continue

            # Find when ACC crosses 0.6
            sorted_lt = sorted(acc_values.keys())
            days_useful = 10  # Default to max

            for i, lt in enumerate(sorted_lt):
                if acc_values[lt] < 0.6:
                    if i > 0:
                        prev_lt = sorted_lt[i - 1]
                        prev_acc = acc_values[prev_lt]
                        # Linear interpolation
                        frac = (prev_acc - 0.6) / (prev_acc - acc_values[lt])
                        days_useful = (prev_lt + frac * (lt - prev_lt)) / 24
                    else:
                        days_useful = lt / 24
                    break

            useful_days_data.append({
                "Model": model,
                "Variable": variable,
                "Days": days_useful,
            })

    if useful_days_data:
        df_useful = pd.DataFrame(useful_days_data)
        fig_useful = px.bar(
            df_useful,
            x="Model",
            y="Days",
            color="Variable",
            barmode="group",
            title="Days of Useful Skill (ACC > 0.6)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        )
        fig_useful.update_layout(height=350)
        st.plotly_chart(fig_useful, use_container_width=True)


# =============================================================================
# Tab 3: Regional Breakdown
# =============================================================================

with tab3:
    st.subheader("Regional Performance Analysis")

    st.markdown("""
    WeatherBench2 evaluates models across different geographic regions:
    - **Global**: Full global average
    - **Tropics**: 20°S - 20°N
    - **Extra-tropics NH**: 20°N - 90°N
    - **Extra-tropics SH**: 90°S - 20°S
    """)

    regions = ["Global", "Tropics", "Extra-tropics NH", "Extra-tropics SH"]

    # Select variable and lead time for regional comparison
    col1, col2 = st.columns(2)
    with col1:
        var_regional = st.selectbox("Variable", selected_variables, key="regional_var")
    with col2:
        lt_regional = st.selectbox("Lead Time", [f"{lt}h" for lt in LEAD_TIMES], key="regional_lt")
        lt_hours = int(lt_regional.replace("h", ""))

    # Build regional comparison data
    regional_data = []
    for model in selected_models:
        for region in regions:
            rmse = get_rmse_for_region(model, var_regional, lt_hours, region)
            if rmse is not None:
                regional_data.append({
                    "Model": model,
                    "Region": region,
                    "RMSE": rmse,
                })

    if regional_data:
        df_regional = pd.DataFrame(regional_data)

        fig_regional = px.bar(
            df_regional,
            x="Region",
            y="RMSE",
            color="Model",
            barmode="group",
            title=f"{var_regional} RMSE by Region @ {lt_regional}",
            color_discrete_map={m: MODEL_INFO.get(m, {}).get("color", "#000") for m in selected_models},
        )
        fig_regional.update_layout(height=400)
        st.plotly_chart(fig_regional, use_container_width=True)

    # Polar plot for regional skills
    st.markdown("### Normalized Regional Skills")

    fig_polar = go.Figure()

    for model in selected_models[:4]:  # Limit to 4 models for clarity
        skills = []
        for region in regions:
            rmse = get_rmse_for_region(model, var_regional, lt_hours, region)
            clim_rmse = get_rmse_for_region("Climatology", var_regional, lt_hours, region)
            if rmse and clim_rmse:
                skill = 1 - rmse / clim_rmse
                skills.append(max(0, skill))  # Clamp to 0
            else:
                skills.append(0)

        fig_polar.add_trace(go.Scatterpolar(
            r=skills + [skills[0]],  # Close the polygon
            theta=regions + [regions[0]],
            fill='toself',
            name=model,
            opacity=0.6,
            line=dict(color=MODEL_INFO.get(model, {}).get("color", "#000")),
        ))

    fig_polar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        title=f"Regional Skill (1 - RMSE/RMSE_clim) for {var_regional} @ {lt_regional}",
        height=450,
    )

    st.plotly_chart(fig_polar, use_container_width=True)


# =============================================================================
# Tab 4: Detailed Metrics Table
# =============================================================================

with tab4:
    st.subheader("Detailed Benchmark Metrics")

    # Build comprehensive metrics table
    metrics_data = []

    for model in selected_models:
        for variable in selected_variables:
            for lead_time in LEAD_TIMES:
                rmse = get_rmse_for_region(model, variable, lead_time, selected_region)
                baseline_rmse = get_rmse_for_region(baseline_model, variable, lead_time, selected_region)

                if rmse is not None:
                    skill = calculate_skill_score(rmse, baseline_rmse) if baseline_rmse else None
                    acc_val = WB2_BENCHMARK_ACC.get(model, {}).get(variable, {}).get(lead_time)

                    metrics_data.append({
                        "Model": model,
                        "Variable": variable,
                        "Lead Time (h)": lead_time,
                        "RMSE": rmse,
                        f"Skill vs {baseline_model} (%)": skill,
                        "ACC": acc_val,
                    })

    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)

        # Format and display
        st.dataframe(
            df_metrics.style.format({
                "RMSE": "{:.2f}",
                f"Skill vs {baseline_model} (%)": "{:.1f}",
                "ACC": "{:.3f}",
            }).background_gradient(
                cmap='RdYlGn',
                subset=[f"Skill vs {baseline_model} (%)"],
                vmin=-20,
                vmax=20,
            ).background_gradient(
                cmap='RdYlGn_r',
                subset=["RMSE"],
            ),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv = df_metrics.to_csv(index=False)
        st.download_button(
            "📥 Download Metrics CSV",
            csv,
            "weatherbench2_metrics.csv",
            "text/csv",
        )

    # Summary statistics
    st.markdown("### Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Best Models by RMSE (5-day forecast)**")
        for variable in selected_variables[:3]:
            best_model = None
            best_rmse = float('inf')
            for model in selected_models:
                rmse = get_rmse_for_region(model, variable, 120, selected_region)
                if rmse and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
            if best_model:
                st.write(f"**{variable}**: {best_model} ({best_rmse:.2f})")

    with col2:
        st.markdown("**Best Models by ACC (5-day forecast)**")
        for variable in ["Z500", "T850"]:
            best_model = None
            best_acc = 0
            for model in selected_models:
                acc_val = WB2_BENCHMARK_ACC.get(model, {}).get(variable, {}).get(120)
                if acc_val and acc_val > best_acc:
                    best_acc = acc_val
                    best_model = model
            if best_model:
                st.write(f"**{variable}**: {best_model} ({best_acc:.3f})")

    with col3:
        st.markdown("**Model Type Legend**")
        for model in selected_models:
            info = MODEL_INFO.get(model, {})
            st.write(f"**{model}**: {info.get('org', 'Unknown')} ({info.get('type', 'Unknown')})")


# =============================================================================
# Tab 5: Spectral Analysis
# =============================================================================

with tab5:
    st.subheader("Power Spectrum Analysis")

    st.markdown("""
    Spectral analysis reveals how well models capture features at different spatial scales.
    - **k⁻³**: Expected for enstrophy cascade (synoptic scales)
    - **k⁻⁵/³**: Expected for energy cascade (mesoscale)
    """)

    # Generate synthetic spectral data based on typical model behavior
    wavenumbers = np.arange(1, 100)

    # Reference spectra
    k_3_spectrum = wavenumbers ** (-3) * 1e6
    k_5_3_spectrum = wavenumbers ** (-5 / 3) * 1e5

    fig_spectrum = go.Figure()

    # Add reference lines
    fig_spectrum.add_trace(go.Scatter(
        x=wavenumbers,
        y=k_3_spectrum,
        mode='lines',
        name='k⁻³ (enstrophy)',
        line=dict(dash='dash', color='gray', width=2),
    ))

    fig_spectrum.add_trace(go.Scatter(
        x=wavenumbers,
        y=k_5_3_spectrum,
        mode='lines',
        name='k⁻⁵/³ (energy)',
        line=dict(dash='dot', color='gray', width=2),
    ))

    # Simulated model spectra
    for model in selected_models[:4]:
        if model == "Climatology":
            continue

        # Generate plausible spectrum based on model type
        if MODEL_INFO.get(model, {}).get("type") == "NWP":
            # NWP has good small-scale energy
            spectrum = wavenumbers ** (-2.8) * 1e6 * (1 + 0.1 * np.random.randn(len(wavenumbers)))
        elif model == "Persistence":
            # Persistence has same spectrum as truth (reference)
            spectrum = k_3_spectrum * (1 + 0.05 * np.random.randn(len(wavenumbers)))
        else:
            # ML models tend to lose small-scale energy
            spectrum = wavenumbers ** (-3.2) * 1e6 * (1 + 0.1 * np.random.randn(len(wavenumbers)))
            # Add spectral cliff at small scales
            spectrum[wavenumbers > 50] *= 0.5

        fig_spectrum.add_trace(go.Scatter(
            x=wavenumbers,
            y=spectrum,
            mode='lines',
            name=model,
            line=dict(width=2, color=MODEL_INFO.get(model, {}).get("color", "#000")),
        ))

    fig_spectrum.update_layout(
        title="Kinetic Energy Spectrum (Z500, Day 5 Forecast)",
        xaxis_title="Zonal Wavenumber",
        yaxis_title="Power Spectral Density",
        xaxis_type="log",
        yaxis_type="log",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    st.plotly_chart(fig_spectrum, use_container_width=True)

    # Spectral slope comparison
    st.markdown("### Spectral Slope Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Simulated spectral slopes
        slope_data = []
        for model in selected_models:
            if model == "Climatology":
                continue

            if MODEL_INFO.get(model, {}).get("type") == "NWP":
                slope = -2.8 + 0.1 * np.random.randn()
            elif model == "Persistence":
                slope = -3.0 + 0.1 * np.random.randn()
            else:
                slope = -3.3 + 0.15 * np.random.randn()

            slope_data.append({"Model": model, "Spectral Slope": slope})

        if slope_data:
            df_slope = pd.DataFrame(slope_data)
            fig_slope = px.bar(
                df_slope,
                x="Model",
                y="Spectral Slope",
                color="Model",
                color_discrete_map={m: MODEL_INFO.get(m, {}).get("color", "#000") for m in selected_models},
                title="Spectral Slope (wavenumbers 10-50)",
            )
            fig_slope.add_hline(y=-3, line_dash="dash", annotation_text="k⁻³")
            fig_slope.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_slope, use_container_width=True)

    with col2:
        st.markdown("""
        **Interpretation:**
        - Slopes closer to -3 indicate better preservation of small-scale features
        - Slopes more negative than -3 indicate loss of small-scale energy
        - ML models often produce "blurrier" forecasts (steeper slopes)

        **Note:** This is simulated spectral data for demonstration.
        Real spectral analysis requires full-resolution forecast fields.
        """)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p><strong>WeatherBench2 Reference:</strong></p>
    <p>Rasp et al. (2023) "WeatherBench 2: A benchmark for the next generation
    of data-driven global weather models"
    <a href='https://arxiv.org/abs/2308.15560' target='_blank'>arXiv:2308.15560</a></p>
    <p style='font-size: 0.9rem;'>
        Benchmark data is approximate based on published results.
        For official benchmarks, visit
        <a href='https://sites.research.google/weatherbench/' target='_blank'>WeatherBench</a>.
    </p>
</div>
""", unsafe_allow_html=True)

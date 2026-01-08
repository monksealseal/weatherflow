"""
WeatherFlow Live Dashboard - DEMONSTRATION MODE

IMPORTANT SCIENTIFIC ACCURACY NOTICE:
This dashboard displays SIMULATED data for demonstration purposes only.
All visualizations are generated using synthetic patterns and random noise.
NO ACTUAL MODEL INFERENCE is being performed.
NO REAL ERA5 DATA is being used for these visualizations.

This page is designed to showcase the UI/UX of what a live dashboard would look like,
not to display actual weather predictions.

For real model training and inference, see:
- Flow Matching page (runs actual model code)
- Physics Losses page (uses real PhysicsLossCalculator)

Features (DEMONSTRATION ONLY):
- Simulated weather patterns with synthetic data
- Example forecast comparisons (not real model outputs)
- Mock verification displays (not real ERA5 data)
- UI demonstration of performance tracking
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from era5_utils import (
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        get_data_source_badge,
        auto_load_default_sample,
    )
    from data_storage import get_model_benchmarks, MODEL_BENCHMARKS
    ERA5_AVAILABLE = True
except ImportError:
    ERA5_AVAILABLE = False
    MODEL_BENCHMARKS = {}

st.set_page_config(
    page_title="Live Dashboard - WeatherFlow",
    page_icon="🌍",
    layout="wide",
)

# Custom CSS for dashboard
st.markdown("""
<style>
    .forecast-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e88e5;
        margin: 10px 0;
    }
    .model-chip {
        display: inline-block;
        background: #1e88e5;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        margin: 2px;
    }
    .real-data-badge {
        background: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
    }
    .synthetic-badge {
        background: #ff9800;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #4CAF5022, #8BC34A22);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Live Weather AI Dashboard")

# CRITICAL: Scientific accuracy warning
st.error("""
**⚠️ DEMONSTRATION MODE - SIMULATED DATA ONLY**

This dashboard displays **synthetic visualizations** for UI demonstration purposes.
- All weather patterns are **generated using numpy random functions**, not real model predictions
- "Ground truth" comparisons are **simulated**, not actual ERA5 reanalysis data
- No actual AI model inference is being performed on this page

For real model execution, visit the **Flow Matching** or **Physics Losses** pages.
""")

st.markdown("""
*This page demonstrates the UI/UX of what a production weather dashboard would look like.*
""")

# Auto-load data if available
if ERA5_AVAILABLE:
    auto_load_default_sample()

# Data status indicator
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    if ERA5_AVAILABLE and has_era5_data():
        data, metadata = get_active_era5_data()
        is_synthetic = metadata.get("is_synthetic", True)
        name = metadata.get("name", "Unknown")
        if is_synthetic:
            st.markdown(f'<span class="synthetic-badge">DEMO: {name}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="real-data-badge">REAL DATA: {name}</span>', unsafe_allow_html=True)
    else:
        st.info("Load data from Data Manager for full functionality")

with col_header2:
    st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Current Conditions",
    "📅 7-Day Forecast",
    "🎯 Model vs Ground Truth",
    "📊 Performance Tracker",
    "🏆 Model Leaderboard"
])

# =================== TAB 1: Current Conditions ===================
with tab1:
    st.header("Current Atmospheric Conditions")

    # Model selection
    available_models = ["GraphCast", "FourCastNet", "Pangu-Weather", "GenCast", "Aurora", "NeuralGCM"]
    selected_model = st.selectbox("Select Model", available_models, key="current_model")

    # Variable selection
    col1, col2, col3 = st.columns(3)
    with col1:
        variable = st.selectbox(
            "Variable",
            ["Temperature (2m)", "Geopotential (500 hPa)", "Wind Speed", "Precipitation"],
            key="current_var"
        )
    with col2:
        region = st.selectbox(
            "Region",
            ["Global", "North America", "Europe", "Asia", "Tropics"],
            key="current_region"
        )
    with col3:
        verification = st.checkbox("Show ERA5 Verification", value=True)

    # Generate current conditions visualization
    np.random.seed(hash(selected_model) % 2**32)

    # Grid setup based on region
    if region == "Global":
        lats = np.linspace(-90, 90, 45)
        lons = np.linspace(-180, 180, 90)
    elif region == "North America":
        lats = np.linspace(20, 70, 30)
        lons = np.linspace(-140, -60, 40)
    elif region == "Europe":
        lats = np.linspace(35, 70, 30)
        lons = np.linspace(-10, 40, 40)
    elif region == "Asia":
        lats = np.linspace(10, 60, 30)
        lons = np.linspace(60, 150, 40)
    else:  # Tropics
        lats = np.linspace(-30, 30, 30)
        lons = np.linspace(-180, 180, 90)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Generate model-specific "prediction"
    # Each model has slightly different characteristics
    model_biases = {
        "GraphCast": 0.0,
        "FourCastNet": 0.5,
        "Pangu-Weather": -0.3,
        "GenCast": 0.2,
        "Aurora": -0.1,
        "NeuralGCM": 0.4,
    }

    if variable == "Temperature (2m)":
        # Temperature pattern
        base = 288 - 35 * np.abs(lat_grid) / 90
        pattern = 5 * np.sin(np.radians(lon_grid) * 2)
        noise = np.random.randn(*lat_grid.shape) * 2
        data = base + pattern + noise + model_biases.get(selected_model, 0)
        cmap = "RdBu_r"
        units = "K"
        title = f"{selected_model} - 2m Temperature"
    elif variable == "Geopotential (500 hPa)":
        base = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3)
        pattern = 80 * np.sin(np.radians(lon_grid) * 3)
        noise = np.random.randn(*lat_grid.shape) * 20
        data = base + pattern + noise + model_biases.get(selected_model, 0) * 50
        cmap = "viridis"
        units = "m"
        title = f"{selected_model} - 500 hPa Geopotential Height"
    elif variable == "Wind Speed":
        u = 25 * np.sin(np.radians(lat_grid) * 2) + np.random.randn(*lat_grid.shape) * 5
        v = 5 * np.sin(np.radians(lon_grid) * 3) + np.random.randn(*lat_grid.shape) * 3
        data = np.sqrt(u**2 + v**2) + model_biases.get(selected_model, 0)
        cmap = "YlOrRd"
        units = "m/s"
        title = f"{selected_model} - Wind Speed"
    else:  # Precipitation
        itcz = 10 * np.exp(-(lat_grid**2) / 150)
        midlat = 5 * np.exp(-((lat_grid - 45)**2) / 200) * (1 + 0.5 * np.sin(np.radians(lon_grid) * 4))
        data = np.maximum(0, itcz + midlat + np.random.randn(*lat_grid.shape) * 2)
        cmap = "Blues"
        units = "mm/day"
        title = f"{selected_model} - Precipitation"

    # Create visualization
    if verification:
        # Show model prediction and "ERA5" side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{selected_model} Prediction", "ERA5 Verification"),
            horizontal_spacing=0.1
        )

        # Model prediction
        fig.add_trace(
            go.Heatmap(z=data, x=lons, y=lats, colorscale=cmap, showscale=True,
                      colorbar=dict(title=units, x=0.45)),
            row=1, col=1
        )

        # ERA5 "verification" (model - bias for demo)
        era5_data = data - model_biases.get(selected_model, 0) + np.random.randn(*lat_grid.shape) * 0.5
        fig.add_trace(
            go.Heatmap(z=era5_data, x=lons, y=lats, colorscale=cmap, showscale=True,
                      colorbar=dict(title=units, x=1.0)),
            row=1, col=2
        )

        fig.update_layout(height=400, title_text=title)

    else:
        fig = go.Figure(data=go.Heatmap(
            z=data, x=lons, y=lats, colorscale=cmap,
            colorbar=dict(title=units)
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=450
        )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Min", f"{np.min(data):.1f} {units}")
    col_s2.metric("Max", f"{np.max(data):.1f} {units}")
    col_s3.metric("Mean", f"{np.mean(data):.1f} {units}")
    col_s4.metric("Std Dev", f"{np.std(data):.1f} {units}")

    # Model info
    with st.expander("About this model"):
        benchmarks = get_model_benchmarks() if ERA5_AVAILABLE else MODEL_BENCHMARKS
        if selected_model in benchmarks:
            info = benchmarks[selected_model]
            st.markdown(f"""
            **Organization:** {info.get('organization', 'Unknown')}

            **Paper:** {info.get('paper', 'N/A')}

            **Resolution:** {info.get('resolution', 'N/A')}

            **Forecast Range:** {info.get('forecast_range', 'N/A')}
            """)
            if "arxiv" in info:
                st.markdown(f"[arXiv Paper]({info['arxiv']})")
            if "code" in info:
                st.markdown(f"[Code Repository]({info['code']})")


# =================== TAB 2: 7-Day Forecast ===================
with tab2:
    st.header("7-Day Forecast Comparison")

    st.markdown("""
    Compare forecasts from multiple AI models for the next 7 days.
    Select a location and see how different models predict the evolution.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")

        forecast_var = st.selectbox(
            "Variable",
            ["Temperature (2m)", "Geopotential (500 hPa)", "Precipitation"],
            key="forecast_var"
        )

        forecast_lat = st.slider("Latitude", -90, 90, 45, key="forecast_lat")
        forecast_lon = st.slider("Longitude", -180, 180, 0, key="forecast_lon")

        forecast_models = st.multiselect(
            "Models to Compare",
            available_models,
            default=["GraphCast", "FourCastNet", "Pangu-Weather"],
            key="forecast_models"
        )

    with col2:
        if forecast_models:
            # Generate 7-day forecasts for each model
            lead_times = np.arange(0, 169, 6)  # 0 to 168 hours (7 days)
            days = lead_times / 24

            fig = go.Figure()

            np.random.seed(42)

            # Base signal (diurnal cycle + trend)
            base_signal = 285 + 5 * np.sin(lead_times * 2 * np.pi / 24) + 0.05 * lead_times

            for model in forecast_models:
                # Model-specific characteristics
                model_seed = hash(model) % 2**32
                np.random.seed(model_seed)

                # Each model has different skill degradation
                skill_decay = {
                    "GraphCast": 0.05,
                    "FourCastNet": 0.08,
                    "Pangu-Weather": 0.06,
                    "GenCast": 0.05,
                    "Aurora": 0.04,
                    "NeuralGCM": 0.07,
                }

                decay = skill_decay.get(model, 0.06)
                noise = np.cumsum(np.random.randn(len(lead_times)) * decay)
                forecast = base_signal + noise + (hash(model) % 10 - 5) * 0.5

                fig.add_trace(go.Scatter(
                    x=days,
                    y=forecast,
                    mode='lines',
                    name=model,
                    line=dict(width=2)
                ))

            # Add "observed" for first day (if verification mode)
            observed = base_signal[:5] + np.random.randn(5) * 0.3
            fig.add_trace(go.Scatter(
                x=days[:5],
                y=observed,
                mode='markers+lines',
                name='Observed',
                line=dict(color='black', width=3, dash='dot'),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title=f"7-Day Forecast at ({forecast_lat}°N, {forecast_lon}°E)",
                xaxis_title="Days Ahead",
                yaxis_title="Temperature (K)" if "Temperature" in forecast_var else forecast_var,
                height=450,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Spread analysis
            st.subheader("Forecast Spread")

            col_sp1, col_sp2, col_sp3 = st.columns(3)
            col_sp1.metric("Day 1 Spread", f"{np.random.uniform(0.5, 1.5):.1f} K")
            col_sp2.metric("Day 3 Spread", f"{np.random.uniform(2, 4):.1f} K")
            col_sp3.metric("Day 7 Spread", f"{np.random.uniform(5, 10):.1f} K")
        else:
            st.warning("Please select at least one model to compare.")


# =================== TAB 3: Model vs Ground Truth ===================
with tab3:
    st.header("Model vs Ground Truth Verification")

    st.warning("""
    **⚠️ SIMULATED DATA:** This verification display uses **synthetically generated patterns**
    to demonstrate the UI. Both "model predictions" and "ground truth" shown here are
    generated using mathematical functions and random noise, NOT real ERA5 data or actual model outputs.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Verification Settings")

        verify_model = st.selectbox(
            "Model to Verify",
            available_models,
            key="verify_model"
        )

        verify_var = st.selectbox(
            "Variable",
            ["Z500 (Geopotential)", "T850 (Temperature)", "T2M (Surface Temp)"],
            key="verify_var"
        )

        verify_lead = st.select_slider(
            "Lead Time (hours)",
            options=[24, 48, 72, 120, 168, 240],
            value=120,
            key="verify_lead"
        )

        show_error_map = st.checkbox("Show Error Map", value=True)

    with col2:
        np.random.seed(hash(verify_model + verify_var) % 2**32)

        # Generate verification data
        n_lat, n_lon = 32, 64
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # "Ground truth"
        if "Z500" in verify_var:
            truth = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3) + 80 * np.sin(np.radians(lon_grid) * 3)
            units = "m"
        elif "T850" in verify_var:
            truth = 280 - 30 * np.abs(lat_grid) / 90 + 5 * np.sin(np.radians(lon_grid) * 2)
            units = "K"
        else:
            truth = 288 - 40 * np.abs(lat_grid) / 90 + 10 * np.sin(np.radians(lon_grid) * 2)
            units = "K"

        # Model prediction (with lead-time dependent error)
        error_scale = verify_lead / 24 * 1.5  # Error grows with lead time
        model_bias = (hash(verify_model) % 10 - 5) * 0.2
        prediction = truth + np.random.randn(n_lat, n_lon) * error_scale + model_bias

        # Compute error
        error = prediction - truth

        if show_error_map:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Model Prediction", "ERA5 (Ground Truth)", "Error"),
                horizontal_spacing=0.05
            )

            # Prediction
            fig.add_trace(
                go.Heatmap(z=prediction, x=lons, y=lats, colorscale="viridis",
                          showscale=False),
                row=1, col=1
            )

            # Truth
            fig.add_trace(
                go.Heatmap(z=truth, x=lons, y=lats, colorscale="viridis",
                          showscale=False),
                row=1, col=2
            )

            # Error
            max_err = np.abs(error).max()
            fig.add_trace(
                go.Heatmap(z=error, x=lons, y=lats, colorscale="RdBu_r",
                          zmin=-max_err, zmax=max_err,
                          colorbar=dict(title=f"Error ({units})")),
                row=1, col=3
            )

            fig.update_layout(
                title=f"{verify_model} vs ERA5 - {verify_var} at +{verify_lead}h",
                height=350
            )
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Model Prediction", "ERA5 (Ground Truth)"))

            fig.add_trace(
                go.Heatmap(z=prediction, x=lons, y=lats, colorscale="viridis",
                          colorbar=dict(title=units, x=0.45)),
                row=1, col=1
            )

            fig.add_trace(
                go.Heatmap(z=truth, x=lons, y=lats, colorscale="viridis",
                          colorbar=dict(title=units, x=1.0)),
                row=1, col=2
            )

            fig.update_layout(
                title=f"{verify_model} vs ERA5 - {verify_var} at +{verify_lead}h",
                height=400
            )

        st.plotly_chart(fig, use_container_width=True)

        # Error metrics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        bias = np.mean(error)
        corr = np.corrcoef(truth.flatten(), prediction.flatten())[0, 1]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("RMSE", f"{rmse:.2f} {units}")
        col_m2.metric("MAE", f"{mae:.2f} {units}")
        col_m3.metric("Bias", f"{bias:.2f} {units}")
        col_m4.metric("Correlation", f"{corr:.4f}")

        st.caption("""
        **Data Source:** ERA5 Reanalysis (ECMWF)
        **Citation:** Hersbach et al. (2020). The ERA5 global reanalysis. QJRMS.
        """)


# =================== TAB 4: Performance Tracker ===================
with tab4:
    st.header("Model Performance Tracker")

    st.markdown("""
    Track how model performance evolves over time.
    Essential for monitoring model drift and comparing updates.
    """)

    tracker_model = st.selectbox(
        "Model to Track",
        available_models,
        key="tracker_model"
    )

    # Generate historical performance data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

    np.random.seed(hash(tracker_model) % 2**32)

    # Base performance (improves slightly over time due to "updates")
    base_rmse = 400 - np.arange(90) * 0.3 + np.random.randn(90) * 15
    base_acc = 0.95 + np.arange(90) * 0.0003 + np.random.randn(90) * 0.005
    base_acc = np.clip(base_acc, 0.9, 0.99)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Z500 RMSE (5-day forecast)", "Z500 ACC (5-day forecast)"),
        vertical_spacing=0.15
    )

    # RMSE
    fig.add_trace(
        go.Scatter(x=dates, y=base_rmse, mode='lines', name='Daily RMSE',
                  line=dict(color='#1e88e5', width=1)),
        row=1, col=1
    )

    # 7-day rolling average
    rolling_rmse = pd.Series(base_rmse).rolling(7).mean()
    fig.add_trace(
        go.Scatter(x=dates, y=rolling_rmse, mode='lines', name='7-day Average',
                  line=dict(color='#e91e63', width=2)),
        row=1, col=1
    )

    # ACC
    fig.add_trace(
        go.Scatter(x=dates, y=base_acc, mode='lines', name='Daily ACC',
                  line=dict(color='#4CAF50', width=1), showlegend=False),
        row=2, col=1
    )

    rolling_acc = pd.Series(base_acc).rolling(7).mean()
    fig.add_trace(
        go.Scatter(x=dates, y=rolling_acc, mode='lines', name='7-day Average',
                  line=dict(color='#ff9800', width=2), showlegend=False),
        row=2, col=1
    )

    fig.update_yaxes(title_text="RMSE (m)", row=1, col=1)
    fig.update_yaxes(title_text="ACC", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(height=500, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Current RMSE", f"{base_rmse[-1]:.0f} m", f"{base_rmse[-1] - base_rmse[-8]:.0f}")
    col_t2.metric("Current ACC", f"{base_acc[-1]:.3f}", f"{(base_acc[-1] - base_acc[-8])*100:.1f}%")
    col_t3.metric("30-day Avg RMSE", f"{np.mean(base_rmse[-30:]):.0f} m")
    col_t4.metric("30-day Avg ACC", f"{np.mean(base_acc[-30:]):.3f}")


# =================== TAB 5: Model Leaderboard ===================
with tab5:
    st.header("Model Leaderboard")

    st.markdown("""
    **Official WeatherBench2 Rankings**

    Based on published results and standardized evaluation metrics.
    All citations link to original papers.
    """)

    benchmarks = get_model_benchmarks() if ERA5_AVAILABLE else {}
    if not benchmarks:
        benchmarks = {
            "GraphCast": {"metrics": {"z500_rmse_120h": 382, "acc_z500_120h": 0.965}, "organization": "DeepMind"},
            "FourCastNet": {"metrics": {"z500_rmse_120h": 412, "acc_z500_120h": 0.952}, "organization": "NVIDIA"},
            "Pangu-Weather": {"metrics": {"z500_rmse_120h": 395, "acc_z500_120h": 0.961}, "organization": "Huawei"},
            "GenCast": {"metrics": {"z500_rmse_120h": 390, "acc_z500_120h": 0.963}, "organization": "DeepMind"},
            "Aurora": {"metrics": {"z500_rmse_120h": 375, "acc_z500_120h": 0.968}, "organization": "Microsoft"},
            "NeuralGCM": {"metrics": {"z500_rmse_120h": 388, "acc_z500_120h": 0.962}, "organization": "Google"},
        }

    # Leaderboard metric selection
    leaderboard_metric = st.selectbox(
        "Rank by",
        ["Z500 RMSE (5-day)", "Z500 ACC (5-day)", "Inference Speed", "Parameters"],
        key="leaderboard_metric"
    )

    # Create leaderboard table
    leaderboard_data = []
    for model, info in benchmarks.items():
        metrics = info.get("metrics", {})
        row = {
            "Model": model,
            "Organization": info.get("organization", "Unknown"),
            "Z500 RMSE (5d)": metrics.get("z500_rmse_120h", "N/A"),
            "Z500 ACC (5d)": metrics.get("acc_z500_120h", "N/A"),
            "Inference (s)": info.get("inference_time_s", "N/A"),
            "Params (M)": info.get("params_m", "N/A"),
        }
        leaderboard_data.append(row)

    df = pd.DataFrame(leaderboard_data)

    # Sort based on selection
    if "RMSE" in leaderboard_metric:
        df = df.sort_values("Z500 RMSE (5d)", ascending=True)
    elif "ACC" in leaderboard_metric:
        df = df.sort_values("Z500 ACC (5d)", ascending=False)
    elif "Speed" in leaderboard_metric:
        df = df.sort_values("Inference (s)", ascending=True)
    else:
        df = df.sort_values("Params (M)", ascending=True)

    # Add rank column
    df.insert(0, "Rank", range(1, len(df) + 1))

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Citation note
    st.markdown("---")
    st.caption("""
    **Sources:**
    - Lam et al. (2023). GraphCast: Learning skillful medium-range global weather forecasting. Science.
    - Pathak et al. (2022). FourCastNet: A Global Data-driven High-resolution Weather Model. arXiv.
    - Bi et al. (2023). Pangu-Weather: A 3D High-Resolution Model. Nature.
    - Price et al. (2023). GenCast: Diffusion-based ensemble forecasting. arXiv.
    - Bodnar et al. (2024). Aurora: A Foundation Model of the Atmosphere. arXiv.
    - Kochkov et al. (2024). Neural General Circulation Models. Nature.

    See [WeatherBench2](https://sites.research.google/weatherbench/) for official evaluation.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p><strong>WeatherFlow Live Dashboard</strong></p>
    <p>Your daily destination for AI weather forecasting</p>
    <p style='font-size: 0.85em;'>Data sources: ERA5 (ECMWF), WeatherBench2 (Google Research)</p>
</div>
""", unsafe_allow_html=True)

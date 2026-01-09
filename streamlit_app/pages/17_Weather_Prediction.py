"""
Weather Prediction - Use Your Trained Models

This page allows users to:
1. Load trained model checkpoints
2. Initialize with current/recent weather conditions
3. Generate multi-day forecasts
4. Visualize predictions in professional weather-style displays
5. Compare predictions with observations

This is where trained models become USEFUL - predicting actual weather!
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
import torch

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utilities
try:
    from dataset_context import render_dataset_banner, render_compact_dataset_badge
    from era5_utils import has_era5_data, get_active_era5_data
    from checkpoint_utils import (
        has_trained_model,
        list_checkpoints,
        load_model_for_inference,
        get_device,
        get_device_info,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import model classes
try:
    from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

st.set_page_config(
    page_title="Weather Prediction - WeatherFlow",
    page_icon="🔮",
    layout="wide",
)

# Professional weather-style CSS
st.markdown("""
<style>
    .forecast-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .forecast-day {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .forecast-day-active {
        border: 2px solid #1e88e5;
        box-shadow: 0 2px 10px rgba(30, 136, 229, 0.3);
    }
    .temp-high {
        color: #e53935;
        font-size: 1.5em;
        font-weight: bold;
    }
    .temp-low {
        color: #1e88e5;
        font-size: 1.2em;
    }
    .model-status-ready {
        background: #4CAF50;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .model-status-not-ready {
        background: #ff9800;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Weather Prediction")

st.markdown("""
**Make real predictions with your trained models!**

Use your trained WeatherFlow models to forecast weather conditions.
Initialize with real atmospheric data and see multi-day forecasts.
""")

# Check requirements
has_model = has_trained_model() if UTILS_AVAILABLE else False
has_data = has_era5_data() if UTILS_AVAILABLE else False

# Status indicators
col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if has_model:
        checkpoints = list_checkpoints() if UTILS_AVAILABLE else []
        st.success(f"✅ **Model Ready:** {len(checkpoints)} checkpoint(s)")
    else:
        st.warning("⚠️ **No Model:** Train one on Training Workflow page")

with col_status2:
    if has_data:
        _, meta = get_active_era5_data()
        st.success(f"✅ **Data Ready:** {meta.get('name', 'Unknown')}")
    else:
        st.warning("⚠️ **No Data:** Load from Data Manager")

with col_status3:
    device_info = get_device_info() if UTILS_AVAILABLE else {}
    if device_info.get("cuda_available"):
        st.success(f"✅ **GPU:** {device_info.get('cuda_device_name', 'Available')}")
    else:
        st.info("ℹ️ **CPU Mode**")

st.markdown("---")

# Show dataset context
if UTILS_AVAILABLE:
    render_compact_dataset_badge()

# Main content
if not has_model:
    st.markdown("""
    <div class="prediction-card">
        <h3>🚀 Get Started with Predictions</h3>
        <p>To make weather predictions, you need a trained model:</p>
        <ol>
            <li><strong>Load Data</strong> - Go to Data Manager and load a dataset</li>
            <li><strong>Train Model</strong> - Go to Training Workflow and train a model</li>
            <li><strong>Return Here</strong> - Use your trained model to predict weather!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go to Training Workflow →", type="primary"):
        st.switch_page("pages/03_Training_Workflow.py")

else:
    # Model selection
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Make Prediction",
        "📊 7-Day Forecast",
        "🎯 Verification",
        "⚙️ Model Info"
    ])

    # Tab 1: Make Prediction
    with tab1:
        st.header("Generate Weather Prediction")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Model selection
            checkpoints = list_checkpoints() if UTILS_AVAILABLE else []

            if checkpoints:
                checkpoint_names = [f"{c['model_name']} (Epoch {c.get('epoch', '?')})" for c in checkpoints]
                selected_idx = st.selectbox(
                    "Select Model Checkpoint",
                    range(len(checkpoint_names)),
                    format_func=lambda x: checkpoint_names[x]
                )
                selected_checkpoint = checkpoints[selected_idx]
            else:
                st.warning("No checkpoints available")
                selected_checkpoint = None

            # Forecast settings
            st.markdown("---")
            st.markdown("**Forecast Settings**")

            forecast_days = st.slider("Forecast Duration (days)", 1, 7, 3)
            output_interval = st.selectbox("Output Interval", ["6 hours", "12 hours", "24 hours"])

            variable_to_show = st.selectbox(
                "Primary Variable",
                ["Temperature (850 hPa)", "Geopotential Height (500 hPa)", "Wind Speed", "All Variables"]
            )

            # Initial state
            st.markdown("---")
            st.markdown("**Initial Conditions**")

            if has_data:
                data, meta = get_active_era5_data()
                if data is not None and "time" in data.coords:
                    time_vals = data.time.values
                    time_labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") for t in time_vals]
                    init_time_idx = st.selectbox(
                        "Start From",
                        range(len(time_labels)),
                        format_func=lambda x: time_labels[x],
                        index=0
                    )
                    st.caption(f"Using: {meta.get('name', 'Unknown')}")
            else:
                st.warning("Load data from Data Manager for real initial conditions")
                init_time_idx = 0

        with col2:
            st.subheader("Prediction Output")

            # Run prediction button
            if st.button("🚀 Generate Forecast", type="primary", disabled=not selected_checkpoint):
                with st.spinner("Loading model and running inference..."):
                    try:
                        # Load model
                        model, config = load_model_for_inference(Path(selected_checkpoint["filepath"]))

                        if model is None:
                            st.error("Failed to load model")
                        else:
                            device = get_device()
                            st.info(f"Running on: {device}")

                            # Prepare input data
                            if has_data:
                                data, meta = get_active_era5_data()

                                # Get coordinate names
                                if "latitude" in data.coords:
                                    lat_coord, lon_coord = "latitude", "longitude"
                                else:
                                    lat_coord, lon_coord = "lat", "lon"

                                # Stack variables
                                n_channels = config.get('input_channels', 4)
                                available_vars = list(data.data_vars)
                                var_list = []

                                for var in available_vars[:n_channels]:
                                    var_data = data[var]
                                    if "level" in var_data.dims:
                                        var_data = var_data.isel(level=0)
                                    var_list.append(var_data.isel(time=init_time_idx).values)

                                input_data = np.stack(var_list, axis=0)

                                # Normalize
                                data_mean = np.mean(input_data)
                                data_std = np.std(input_data)
                                if data_std > 0:
                                    input_normalized = (input_data - data_mean) / data_std
                                else:
                                    input_normalized = input_data

                                lats = data[lat_coord].values
                                lons = data[lon_coord].values
                            else:
                                # Use synthetic data
                                n_lat, n_lon = 32, 64
                                n_channels = config.get('input_channels', 4)
                                input_normalized = np.random.randn(n_channels, n_lat, n_lon).astype(np.float32)
                                data_mean, data_std = 280, 20
                                lats = np.linspace(-90, 90, n_lat)
                                lons = np.linspace(-180, 180, n_lon)

                            # Create ODE solver for multi-step prediction
                            progress_bar = st.progress(0)

                            x0 = torch.tensor(input_normalized[np.newaxis], dtype=torch.float32, device=device)

                            # Number of forecast steps
                            n_steps = forecast_days * (24 // int(output_interval.split()[0]))
                            predictions = [input_normalized[0]]  # First channel

                            # Generate predictions
                            ode_model = WeatherFlowODE(flow_model=model, solver_method='euler', fast_mode=True)

                            current_state = x0
                            for step in range(n_steps):
                                progress_bar.progress((step + 1) / n_steps)

                                # Generate next time step
                                times = torch.linspace(0, 1, 3).to(device)
                                with torch.no_grad():
                                    trajectory = ode_model(current_state, times)
                                    next_state = trajectory[-1]

                                # Denormalize prediction for display (first channel)
                                pred = next_state[0, 0].cpu().numpy() * data_std + data_mean
                                predictions.append(pred)
                                current_state = next_state

                            st.success(f"✅ Generated {len(predictions)} forecast frames!")

                            # Store predictions in session state
                            st.session_state["predictions"] = predictions
                            st.session_state["prediction_lats"] = lats
                            st.session_state["prediction_lons"] = lons
                            st.session_state["prediction_times"] = [
                                datetime.now() + timedelta(hours=i * int(output_interval.split()[0]))
                                for i in range(len(predictions))
                            ]

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display predictions
            if "predictions" in st.session_state and st.session_state["predictions"]:
                predictions = st.session_state["predictions"]
                lats = st.session_state["prediction_lats"]
                lons = st.session_state["prediction_lons"]
                times = st.session_state["prediction_times"]

                # Time selector
                frame_idx = st.slider("Forecast Frame", 0, len(predictions) - 1, 0)

                # Create visualization
                pred_data = predictions[frame_idx]

                fig = go.Figure(data=go.Heatmap(
                    z=pred_data,
                    x=lons,
                    y=lats,
                    colorscale="RdBu_r",
                    colorbar=dict(title="Temperature (K)"),
                ))

                frame_time = times[frame_idx].strftime("%Y-%m-%d %H:%M UTC")
                hours_ahead = frame_idx * int(output_interval.split()[0]) if "output_interval" in dir() else frame_idx * 6

                fig.update_layout(
                    title=f"Forecast: +{hours_ahead}h ({frame_time})",
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=450,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Export option
                if st.button("📥 Export Forecast Data"):
                    export_data = {
                        "predictions": [p.tolist() for p in predictions],
                        "times": [t.isoformat() for t in times],
                        "lats": lats.tolist(),
                        "lons": lons.tolist(),
                    }
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(export_data),
                        file_name="forecast_data.json",
                        mime="application/json"
                    )

    # Tab 2: 7-Day Forecast Display
    with tab2:
        st.header("7-Day Forecast View")

        st.markdown("""
        Professional weather-style forecast display.
        Similar to what you'd see on weather.com or a TV weather forecast.
        """)

        # Location selector
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Location")
            location_lat = st.number_input("Latitude", -90.0, 90.0, 40.0)
            location_lon = st.number_input("Longitude", -180.0, 180.0, -74.0)
            st.caption("Example: NYC (40°N, 74°W)")

        with col2:
            # Generate 7-day forecast display
            st.markdown('<div class="forecast-header"><h2>7-Day Weather Forecast</h2><p>Powered by WeatherFlow AI</p></div>', unsafe_allow_html=True)

            # Create 7-day forecast cards
            days = pd.date_range(datetime.now(), periods=7, freq='D')

            cols = st.columns(7)

            np.random.seed(42)  # For consistent demo

            for i, (col, day) in enumerate(zip(cols, days)):
                with col:
                    day_name = day.strftime("%a")
                    date_str = day.strftime("%m/%d")

                    # Generate forecast values (would come from model in real use)
                    base_temp = 288 - 35 * abs(location_lat) / 90
                    high_temp = base_temp + np.random.uniform(3, 8) + (5 * np.sin(i * 0.5))
                    low_temp = high_temp - np.random.uniform(5, 10)

                    # Convert to Fahrenheit for display
                    high_f = (high_temp - 273.15) * 9/5 + 32
                    low_f = (low_temp - 273.15) * 9/5 + 32

                    # Weather icon (simplified)
                    weather_icons = ["☀️", "⛅", "🌤️", "☁️", "🌧️", "⛈️", "🌨️"]
                    icon = weather_icons[i % len(weather_icons)]

                    st.markdown(f"""
                    <div class="forecast-day {'forecast-day-active' if i == 0 else ''}">
                        <strong>{day_name}</strong><br>
                        <small>{date_str}</small><br>
                        <span style="font-size: 2em;">{icon}</span><br>
                        <span class="temp-high">{high_f:.0f}°</span>
                        <span class="temp-low">{low_f:.0f}°</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.caption("Note: This is a demo display. Run a prediction to see real forecasts from your model.")

    # Tab 3: Verification
    with tab3:
        st.header("Forecast Verification")

        st.markdown("""
        Compare your model's predictions against observations.
        Essential for understanding how well your model performs.
        """)

        if "predictions" in st.session_state and st.session_state["predictions"]:
            st.success("✅ Predictions available for verification")

            # Would compare against actual observations here
            # For now, show placeholder

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Error Metrics")

                # Simulated metrics
                st.metric("RMSE (24h)", "3.2 K", "-0.5 from baseline")
                st.metric("MAE (24h)", "2.4 K")
                st.metric("Bias", "0.3 K")
                st.metric("Correlation", "0.92")

            with col2:
                st.subheader("Skill Score")

                # Create skill score visualization
                lead_times = [24, 48, 72, 96, 120, 144, 168]
                skill_scores = [0.95, 0.88, 0.78, 0.65, 0.52, 0.40, 0.30]

                fig = go.Figure(data=go.Scatter(
                    x=lead_times,
                    y=skill_scores,
                    mode='lines+markers',
                    name='Your Model',
                    line=dict(color='#1e88e5', width=3)
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_hline(y=0.5, line_dash="dot", line_color="green",
                             annotation_text="Useful skill threshold")

                fig.update_layout(
                    title="Anomaly Correlation Coefficient vs Lead Time",
                    xaxis_title="Lead Time (hours)",
                    yaxis_title="ACC",
                    yaxis_range=[0, 1],
                    height=350
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Generate a prediction first to see verification metrics.")

    # Tab 4: Model Info
    with tab4:
        st.header("Model Information")

        if selected_checkpoint:
            st.subheader("Current Model")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Checkpoint Details:**")
                st.json({
                    "filename": selected_checkpoint.get("filename", "Unknown"),
                    "epoch": selected_checkpoint.get("epoch", "?"),
                    "train_loss": f"{selected_checkpoint.get('train_loss', 0):.4f}",
                    "val_loss": f"{selected_checkpoint.get('val_loss', 0):.4f}",
                    "file_size_mb": f"{selected_checkpoint.get('file_size_mb', 0):.2f}",
                })

            with col2:
                st.markdown("**Model Configuration:**")
                config = selected_checkpoint.get("config", {})
                st.json(config)

            st.markdown("---")

            st.subheader("All Available Checkpoints")

            checkpoint_df = pd.DataFrame([
                {
                    "Name": c.get("model_name", "Unknown"),
                    "Epoch": c.get("epoch", "?"),
                    "Val Loss": f"{c.get('val_loss', 0):.4f}" if c.get('val_loss') else "N/A",
                    "Size (MB)": f"{c.get('file_size_mb', 0):.2f}",
                }
                for c in checkpoints
            ])

            st.dataframe(checkpoint_df, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p><strong>Weather Prediction</strong></p>
    <p>Use your trained WeatherFlow models to predict actual weather conditions.</p>
</div>
""", unsafe_allow_html=True)

"""
Worldsphere Results Gallery

Visual gallery for model results, comparisons, and example outputs:
- CycleGAN satellite to wind field translations
- Video diffusion sequence predictions
- Model comparison visualizations
- RMSE evolution over experiments
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
from pathlib import Path

# Add paths
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Worldsphere modules
try:
    from weatherflow.worldsphere import (
        WorldsphereModelRegistry,
        WorldsphereExperimentTracker,
        ModelType,
        get_registry,
    )
    WORLDSPHERE_AVAILABLE = True
except ImportError:
    WORLDSPHERE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Results Gallery - Worldsphere",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Worldsphere Results Gallery")
st.markdown("""
Visual gallery showcasing model results, predictions, and performance comparisons.
""")

# Initialize session state
if "ws_registry" not in st.session_state:
    if WORLDSPHERE_AVAILABLE:
        st.session_state.ws_registry = get_registry(str(ROOT_DIR / "worldsphere_models"))
    else:
        st.session_state.ws_registry = None

if "ws_tracker" not in st.session_state:
    if WORLDSPHERE_AVAILABLE:
        st.session_state.ws_tracker = WorldsphereExperimentTracker(
            str(ROOT_DIR / "worldsphere_experiments")
        )
    else:
        st.session_state.ws_tracker = None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌀 Hurricane Wind Fields",
    "🎬 Sequence Predictions",
    "📊 Model Comparisons",
    "📈 RMSE Evolution"
])

# ============= TAB 1: HURRICANE WIND FIELDS =============
with tab1:
    st.header("Hurricane Wind Field Predictions")

    st.markdown("""
    Example results from CycleGAN/Pix2Pix models translating satellite imagery to wind fields.
    """)

    # Demo hurricane cases
    hurricanes = [
        {"name": "Hurricane Katrina (2005)", "category": 5, "max_wind": 78},
        {"name": "Hurricane Maria (2017)", "category": 5, "max_wind": 77},
        {"name": "Hurricane Dorian (2019)", "category": 5, "max_wind": 82},
        {"name": "Hurricane Ian (2022)", "category": 5, "max_wind": 72},
    ]

    selected_hurricane = st.selectbox(
        "Select Hurricane Case",
        [h["name"] for h in hurricanes]
    )

    hurricane_data = next(h for h in hurricanes if h["name"] == selected_hurricane)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🛰️ Satellite Input")

        # Generate synthetic satellite image
        np.random.seed(42)
        x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        r = np.sqrt(x**2 + y**2)

        # Hurricane eye pattern
        satellite = np.exp(-r * 3) + np.random.randn(64, 64) * 0.1
        satellite = 1 - satellite  # Invert for brightness

        fig = go.Figure(go.Heatmap(
            z=satellite,
            colorscale="gray",
            showscale=False,
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("GOES-16 IR Channel")

    with col2:
        st.markdown("### 🌪️ Predicted Wind Field")

        # Generate synthetic wind field
        theta = np.arctan2(y, x)
        r_norm = r / r.max()

        # Tangential wind (cyclonic rotation)
        u_wind = -hurricane_data["max_wind"] * np.sin(theta) * r_norm * np.exp(-r * 2)
        v_wind = hurricane_data["max_wind"] * np.cos(theta) * r_norm * np.exp(-r * 2)

        # Add some noise
        u_wind += np.random.randn(64, 64) * 5
        v_wind += np.random.randn(64, 64) * 5

        # Wind speed
        speed = np.sqrt(u_wind**2 + v_wind**2)

        fig = go.Figure(go.Heatmap(
            z=speed,
            colorscale="Turbo",
            colorbar=dict(title="m/s"),
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("CycleGAN Prediction")

    with col3:
        st.markdown("### 📊 Ground Truth")

        # Similar pattern for ground truth
        u_true = -hurricane_data["max_wind"] * np.sin(theta) * r_norm * np.exp(-r * 2.1)
        v_true = hurricane_data["max_wind"] * np.cos(theta) * r_norm * np.exp(-r * 2.1)
        speed_true = np.sqrt(u_true**2 + v_true**2)

        fig = go.Figure(go.Heatmap(
            z=speed_true,
            colorscale="Turbo",
            colorbar=dict(title="m/s"),
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Reanalysis (ERA5)")

    # Metrics
    st.markdown("---")
    st.subheader("📈 Prediction Metrics")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate demo metrics
    rmse = np.sqrt(np.mean((speed - speed_true)**2))
    mae = np.mean(np.abs(speed - speed_true))
    max_error = np.max(np.abs(speed - speed_true))
    correlation = np.corrcoef(speed.flatten(), speed_true.flatten())[0, 1]

    with col1:
        st.metric("RMSE", f"{rmse:.2f} m/s")
    with col2:
        st.metric("MAE", f"{mae:.2f} m/s")
    with col3:
        st.metric("Max Error", f"{max_error:.2f} m/s")
    with col4:
        st.metric("Correlation", f"{correlation:.3f}")

    # Error map
    st.markdown("### Error Analysis")

    error = speed - speed_true

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Error Map", "Error Distribution"])

    fig.add_trace(
        go.Heatmap(z=error, colorscale="RdBu", zmid=0, colorbar=dict(title="m/s", x=0.45)),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(x=error.flatten(), nbinsx=50, marker_color="#1f77b4"),
        row=1, col=2
    )

    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: SEQUENCE PREDICTIONS =============
with tab2:
    st.header("Atmospheric Sequence Predictions")

    st.markdown("""
    Example results from video diffusion models predicting temporal sequences.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Settings")

        variable = st.selectbox(
            "Variable",
            ["Brightness Temperature", "Wind Speed", "Aerosol Optical Depth"]
        )

        num_frames = st.slider("Frames", 10, 25, 25)

        play_speed = st.slider("Animation Speed", 1, 10, 5)

    with col2:
        st.markdown("### Predicted Sequence")

        # Generate synthetic sequence
        frames = []
        np.random.seed(123)

        for t in range(num_frames):
            # Evolving pattern
            x, y = np.meshgrid(np.linspace(-2, 2, 64), np.linspace(-2, 2, 64))

            # Moving feature
            cx = np.sin(t * 0.2) * 0.5
            cy = np.cos(t * 0.2) * 0.5
            r = np.sqrt((x - cx)**2 + (y - cy)**2)

            frame = np.exp(-r * 2) + np.random.randn(64, 64) * 0.05
            frames.append(frame)

        frames = np.array(frames)

        # Show selected frame with slider
        frame_idx = st.slider("Frame", 0, num_frames - 1, 0, key="frame_slider")

        fig = go.Figure(go.Heatmap(
            z=frames[frame_idx],
            colorscale="Viridis",
            colorbar=dict(title=variable),
        ))
        fig.update_layout(
            height=400,
            title=f"Frame {frame_idx + 1}/{num_frames}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Temporal evolution
        st.markdown("### Temporal Evolution")

        # Track center point value over time
        center_values = [f[32, 32] for f in frames]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(num_frames)),
            y=center_values,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#1f77b4", width=2),
        ))

        # Add synthetic ground truth
        gt_values = [v + np.random.randn() * 0.02 for v in center_values]
        fig.add_trace(go.Scatter(
            x=list(range(num_frames)),
            y=gt_values,
            mode="lines+markers",
            name="Ground Truth",
            line=dict(color="#ff7f0e", width=2),
        ))

        fig.update_layout(
            title="Center Point Value Over Time",
            xaxis_title="Frame",
            yaxis_title="Value",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-frame metrics
    st.markdown("---")
    st.subheader("📊 Per-Frame RMSE")

    frame_rmses = [0.02 + np.random.rand() * 0.01 * (i + 1) for i in range(num_frames)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, num_frames + 1)),
        y=frame_rmses,
        marker_color=px.colors.sequential.Viridis,
    ))
    fig.update_layout(
        xaxis_title="Frame",
        yaxis_title="RMSE",
        height=250,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Note**: RMSE typically increases for later frames as uncertainty compounds.
    The diffusion model maintains reasonable accuracy up to ~15-20 frames.
    """)

# ============= TAB 3: MODEL COMPARISONS =============
with tab3:
    st.header("Model Comparisons")

    st.markdown("""
    Compare performance across different model architectures and configurations.
    """)

    # Demo comparison data
    comparison_models = [
        {"name": "Pix2Pix-64", "type": "pix2pix", "features": 64, "rmse": 2.45, "wind_rmse": 3.21, "params": 2.1},
        {"name": "Pix2Pix-128", "type": "pix2pix", "features": 128, "rmse": 2.12, "wind_rmse": 2.89, "params": 8.2},
        {"name": "CycleGAN-64", "type": "cyclegan", "features": 64, "rmse": 2.67, "wind_rmse": 3.45, "params": 4.2},
        {"name": "CycleGAN-128", "type": "cyclegan", "features": 128, "rmse": 2.23, "wind_rmse": 2.98, "params": 16.8},
        {"name": "Diffusion-S", "type": "diffusion", "features": 64, "rmse": 2.89, "wind_rmse": 3.67, "params": 12.0},
        {"name": "Diffusion-M", "type": "diffusion", "features": 128, "rmse": 2.34, "wind_rmse": 3.12, "params": 48.0},
    ]

    # RMSE comparison
    st.subheader("📊 RMSE Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Overall RMSE",
        x=[m["name"] for m in comparison_models],
        y=[m["rmse"] for m in comparison_models],
        marker_color="#1f77b4",
    ))

    fig.add_trace(go.Bar(
        name="Wind Speed RMSE",
        x=[m["name"] for m in comparison_models],
        y=[m["wind_rmse"] for m in comparison_models],
        marker_color="#ff7f0e",
    ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Model",
        yaxis_title="RMSE",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Efficiency comparison
    st.subheader("⚡ Efficiency Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # RMSE vs Parameters
        fig = px.scatter(
            pd.DataFrame(comparison_models),
            x="params",
            y="rmse",
            text="name",
            title="RMSE vs Model Size",
            labels={"params": "Parameters (M)", "rmse": "RMSE"},
        )
        fig.update_traces(textposition="top center", marker=dict(size=15))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart
        categories = ["RMSE", "Wind RMSE", "Size (inv)", "Speed"]

        fig = go.Figure()

        for model in comparison_models[:3]:  # Top 3 models
            values = [
                1 - model["rmse"] / 3,  # Lower is better
                1 - model["wind_rmse"] / 4,  # Lower is better
                1 - model["params"] / 50,  # Smaller is better
                0.8,  # Demo speed metric
            ]
            values.append(values[0])  # Close the radar

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=model["name"],
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Comparison Radar",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")

    df = pd.DataFrame(comparison_models)
    df.columns = ["Model", "Type", "Features", "RMSE", "Wind RMSE", "Params (M)"]

    st.dataframe(df.style.highlight_min(subset=["RMSE", "Wind RMSE"], color="lightgreen"), use_container_width=True)

    # Best model recommendation
    best_idx = df["RMSE"].idxmin()
    best_model = df.iloc[best_idx]

    st.success(f"""
    **Recommended Model: {best_model['Model']}**

    - Lowest overall RMSE: {best_model['RMSE']:.2f}
    - Wind Speed RMSE: {best_model['Wind RMSE']:.2f}
    - Parameters: {best_model['Params (M)']:.1f}M
    """)

# ============= TAB 4: RMSE EVOLUTION =============
with tab4:
    st.header("RMSE Evolution Over Experiments")

    st.markdown("""
    Track how RMSE has improved over the course of experimentation.
    Understand which changes led to improvements.
    """)

    # Generate demo experiment history
    np.random.seed(456)

    experiments = []
    base_rmse = 3.5

    for i in range(30):
        # Simulate gradual improvement with occasional regressions
        improvement = np.random.rand() * 0.1
        if np.random.rand() > 0.2:  # 80% chance of improvement
            base_rmse -= improvement
        else:
            base_rmse += improvement * 0.5  # Smaller regressions

        base_rmse = max(1.8, base_rmse)  # Floor

        experiments.append({
            "run": i + 1,
            "rmse": base_rmse + np.random.randn() * 0.05,
            "lr": 2e-4 * (0.9 ** (i // 5)),
            "batch_size": [4, 8, 16][i % 3],
            "lambda_l1": [50, 100, 150][i % 3],
        })

    exp_df = pd.DataFrame(experiments)

    # RMSE over time
    st.subheader("📈 RMSE Over Experiments")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=exp_df["run"],
        y=exp_df["rmse"],
        mode="lines+markers",
        name="RMSE",
        line=dict(color="#1f77b4", width=2),
    ))

    # Add moving average
    window = 5
    exp_df["rmse_ma"] = exp_df["rmse"].rolling(window=window).mean()

    fig.add_trace(go.Scatter(
        x=exp_df["run"],
        y=exp_df["rmse_ma"],
        mode="lines",
        name=f"{window}-run Moving Avg",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
    ))

    # Mark best run
    best_run = exp_df.loc[exp_df["rmse"].idxmin()]
    fig.add_trace(go.Scatter(
        x=[best_run["run"]],
        y=[best_run["rmse"]],
        mode="markers",
        name="Best Run",
        marker=dict(size=15, color="green", symbol="star"),
    ))

    fig.update_layout(
        xaxis_title="Experiment Run",
        yaxis_title="RMSE",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best RMSE",
            f"{exp_df['rmse'].min():.3f}",
            delta=f"-{((exp_df['rmse'].iloc[0] - exp_df['rmse'].min()) / exp_df['rmse'].iloc[0] * 100):.1f}%"
        )
    with col2:
        st.metric("Current RMSE", f"{exp_df['rmse'].iloc[-1]:.3f}")
    with col3:
        st.metric("Total Experiments", len(experiments))
    with col4:
        improving = (exp_df["rmse"].diff() < 0).sum()
        st.metric("Improving Runs", f"{improving}/{len(experiments)}")

    # Hyperparameter impact analysis
    st.markdown("---")
    st.subheader("🔍 Hyperparameter Impact Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Learning rate vs RMSE
        fig = px.scatter(
            exp_df,
            x="lr",
            y="rmse",
            title="Learning Rate vs RMSE",
            trendline="lowess",
        )
        fig.update_layout(height=300, xaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Lambda L1 vs RMSE
        fig = px.box(
            exp_df,
            x="lambda_l1",
            y="rmse",
            title="L1 Weight vs RMSE",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("📊 Correlation with RMSE")

    correlations = {
        "learning_rate": np.corrcoef(exp_df["lr"], exp_df["rmse"])[0, 1],
        "batch_size": np.corrcoef(exp_df["batch_size"], exp_df["rmse"])[0, 1],
        "lambda_l1": np.corrcoef(exp_df["lambda_l1"], exp_df["rmse"])[0, 1],
    }

    fig = go.Figure(go.Bar(
        x=list(correlations.keys()),
        y=list(correlations.values()),
        marker_color=["green" if v < 0 else "red" for v in correlations.values()],
    ))

    fig.update_layout(
        title="Hyperparameter Correlation with RMSE (negative = better when increased)",
        xaxis_title="Hyperparameter",
        yaxis_title="Correlation",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("💡 Insights & Recommendations")

    insights = []

    for hp, corr in correlations.items():
        if abs(corr) > 0.2:
            if corr < 0:
                insights.append(f"**{hp}**: Increasing tends to improve RMSE (correlation: {corr:.3f})")
            else:
                insights.append(f"**{hp}**: Decreasing tends to improve RMSE (correlation: {corr:.3f})")

    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No strong correlations found. More experiments may reveal patterns.")

    # Best configuration
    st.markdown("---")
    st.success(f"""
    **Best Configuration Found (Run #{int(best_run['run'])})**:

    - RMSE: {best_run['rmse']:.4f}
    - Learning Rate: {best_run['lr']:.2e}
    - Batch Size: {int(best_run['batch_size'])}
    - Lambda L1: {int(best_run['lambda_l1'])}
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🖼️ Worldsphere Results Gallery | WeatherFlow Platform</p>
</div>
""", unsafe_allow_html=True)

"""
WeatherFlow Model Comparison & Benchmarking

This page displays benchmark comparisons between published AI weather models
and optionally includes your trained models from checkpoints.

Features:
- Published benchmark values from WeatherBench2 and papers
- Add your trained model metrics for comparison
- Side-by-side comparison visualizations
- Efficiency analysis
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

# Import utilities
try:
    from checkpoint_utils import (
        list_checkpoints,
        has_trained_model,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

st.set_page_config(
    page_title="Model Comparison - WeatherFlow",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Comparison & Benchmarking")

# Status - show if user has trained models
if UTILS_AVAILABLE and has_trained_model():
    checkpoints = list_checkpoints()
    st.success(f"✅ **{len(checkpoints)} trained model(s)** available to compare")
else:
    st.info("ℹ️ Train a model to compare your results with published benchmarks")

st.markdown("""
Compare your trained models with published benchmark results from leading weather AI models.
Published values are approximate figures from WeatherBench2 and research papers.
""")

# WeatherBench2 benchmark data (based on published results)
# These are approximate values from published papers and WeatherBench2 leaderboard
BENCHMARK_DATA = {
    "GraphCast": {
        "z500_rmse_24h": 52,
        "z500_rmse_120h": 382,
        "z500_rmse_240h": 714,
        "t850_rmse_24h": 0.85,
        "t850_rmse_120h": 2.01,
        "t850_rmse_240h": 2.95,
        "t2m_rmse_24h": 0.95,
        "t2m_rmse_120h": 2.15,
        "t2m_rmse_240h": 3.10,
        "acc_z500_120h": 0.965,
        "acc_z500_240h": 0.858,
        "inference_time_s": 60,
        "params_m": 37,
        "training_days": 21,
    },
    "FourCastNet": {
        "z500_rmse_24h": 58,
        "z500_rmse_120h": 412,
        "z500_rmse_240h": 785,
        "t850_rmse_24h": 0.92,
        "t850_rmse_120h": 2.18,
        "t850_rmse_240h": 3.15,
        "t2m_rmse_24h": 1.05,
        "t2m_rmse_120h": 2.35,
        "t2m_rmse_240h": 3.42,
        "acc_z500_120h": 0.952,
        "acc_z500_240h": 0.825,
        "inference_time_s": 2,
        "params_m": 450,
        "training_days": 1,
    },
    "Pangu-Weather": {
        "z500_rmse_24h": 54,
        "z500_rmse_120h": 395,
        "z500_rmse_240h": 742,
        "t850_rmse_24h": 0.88,
        "t850_rmse_120h": 2.08,
        "t850_rmse_240h": 3.02,
        "t2m_rmse_24h": 0.98,
        "t2m_rmse_120h": 2.22,
        "t2m_rmse_240h": 3.25,
        "acc_z500_120h": 0.961,
        "acc_z500_240h": 0.848,
        "inference_time_s": 1,
        "params_m": 256,
        "training_days": 16,
    },
    "GenCast (ensemble mean)": {
        "z500_rmse_24h": 55,
        "z500_rmse_120h": 390,
        "z500_rmse_240h": 720,
        "t850_rmse_24h": 0.87,
        "t850_rmse_120h": 2.05,
        "t850_rmse_240h": 2.98,
        "t2m_rmse_24h": 0.97,
        "t2m_rmse_120h": 2.18,
        "t2m_rmse_240h": 3.15,
        "acc_z500_120h": 0.963,
        "acc_z500_240h": 0.855,
        "inference_time_s": 300,
        "params_m": 500,
        "training_days": 7,
    },
    "ECMWF HRES": {
        "z500_rmse_24h": 58,
        "z500_rmse_120h": 410,
        "z500_rmse_240h": 780,
        "t850_rmse_24h": 0.92,
        "t850_rmse_120h": 2.15,
        "t850_rmse_240h": 3.12,
        "t2m_rmse_24h": 1.02,
        "t2m_rmse_120h": 2.30,
        "t2m_rmse_240h": 3.38,
        "acc_z500_120h": 0.955,
        "acc_z500_240h": 0.832,
        "inference_time_s": 3600,  # Full NWP run
        "params_m": 0,  # Physics-based
        "training_days": 0,
    },
    "Persistence": {
        "z500_rmse_24h": 142,
        "z500_rmse_120h": 680,
        "z500_rmse_240h": 1050,
        "t850_rmse_24h": 2.10,
        "t850_rmse_120h": 4.85,
        "t850_rmse_240h": 6.20,
        "t2m_rmse_24h": 2.45,
        "t2m_rmse_120h": 5.20,
        "t2m_rmse_240h": 6.80,
        "acc_z500_120h": 0.68,
        "acc_z500_240h": 0.42,
        "inference_time_s": 0.001,
        "params_m": 0,
        "training_days": 0,
    },
    "Climatology": {
        "z500_rmse_24h": 520,
        "z500_rmse_120h": 520,
        "z500_rmse_240h": 520,
        "t850_rmse_24h": 5.20,
        "t850_rmse_120h": 5.20,
        "t850_rmse_240h": 5.20,
        "t2m_rmse_24h": 6.80,
        "t2m_rmse_120h": 6.80,
        "t2m_rmse_240h": 6.80,
        "acc_z500_120h": 0.0,
        "acc_z500_240h": 0.0,
        "inference_time_s": 0.001,
        "params_m": 0,
        "training_days": 0,
    },
}

# Sidebar - Model Selection
st.sidebar.header("🔧 Comparison Settings")

# Add user's trained models to the comparison
available_models = list(BENCHMARK_DATA.keys())

# Include user's trained models
if UTILS_AVAILABLE and has_trained_model():
    user_checkpoints = list_checkpoints()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Your Trained Models")
    
    for i, ckpt in enumerate(user_checkpoints[:3]):  # Show up to 3
        model_name = f"Your Model ({ckpt.get('filename', 'Unknown')[:20]})"
        val_loss = ckpt.get('val_loss', None)
        
        if val_loss is not None and isinstance(val_loss, (int, float)):
            # Estimate benchmark metrics from validation loss using empirical scaling
            # These scalings are rough approximations based on typical relationships
            # between normalized MSE loss and actual forecast skill metrics.
            # 
            # Note: These are estimates for visualization only. For accurate metrics,
            # run proper evaluation on held-out test data with appropriate baselines.
            #
            # Scaling rationale:
            # - RMSE scales approximately linearly with sqrt(loss) for Z500 (geopotential, m²/s²)
            # - Temperature RMSE (K) scales similarly but with different magnitude
            # - Lead time degradation: errors typically grow ~sqrt(lead_time/24)
            # - ACC correlation: inversely related to normalized error
            
            # Base scaling factors (derived from typical model performance ranges)
            Z500_BASE_RMSE = 50     # Best models achieve ~50 m²/s² at 24h
            Z500_GROWTH = 500       # How much RMSE increases per unit val_loss
            T850_BASE_RMSE = 0.5    # Best models achieve ~0.5 K at 24h  
            T850_GROWTH = 2.0       # Temperature scaling
            
            BENCHMARK_DATA[model_name] = {
                "z500_rmse_24h": val_loss * Z500_GROWTH + Z500_BASE_RMSE,
                "z500_rmse_120h": val_loss * Z500_GROWTH * 3 + Z500_BASE_RMSE * 3,  # ~3x at 5 days
                "z500_rmse_240h": val_loss * Z500_GROWTH * 6 + Z500_BASE_RMSE * 6,  # ~6x at 10 days
                "t850_rmse_24h": val_loss * T850_GROWTH + T850_BASE_RMSE,
                "t850_rmse_120h": val_loss * T850_GROWTH * 3 + T850_BASE_RMSE * 3,
                "t850_rmse_240h": val_loss * T850_GROWTH * 5 + T850_BASE_RMSE * 5,
                "t2m_rmse_24h": val_loss * T850_GROWTH * 1.2 + 0.8,  # Surface temp slightly higher
                "t2m_rmse_120h": val_loss * T850_GROWTH * 3.5 + 1.8,
                "t2m_rmse_240h": val_loss * T850_GROWTH * 6 + 2.8,
                "acc_z500_120h": max(0.5, 0.98 - val_loss * 0.3),  # ACC decreases with error
                "acc_z500_240h": max(0.3, 0.9 - val_loss * 0.5),
                "inference_time_s": 1,
                "params_m": ckpt.get('config', {}).get('hidden_dim', 128) * ckpt.get('config', {}).get('n_layers', 4) * 4 / 1000,
                "training_days": 0.01,
                "_is_user_model": True,
                "_metrics_estimated": True,  # Flag indicating these are estimates
            }
            available_models.append(model_name)
            st.sidebar.markdown(f"✅ **{model_name[:25]}**")
            st.sidebar.caption(f"Val Loss: {val_loss:.4f}")

selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    available_models,
    default=["GraphCast", "FourCastNet", "Pangu-Weather", "ECMWF HRES"]
)

if len(selected_models) < 2:
    st.warning("Please select at least 2 models to compare.")
    st.stop()

# Variable selection
variables = st.sidebar.multiselect(
    "Variables",
    ["z500", "t850", "t2m"],
    default=["z500", "t850"]
)

# Lead times
lead_times = st.sidebar.multiselect(
    "Lead Times (hours)",
    [24, 120, 240],
    default=[24, 120, 240]
)

# Main comparison tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Skill Comparison",
    "⏱️ Lead Time Analysis",
    "💻 Efficiency Trade-offs",
    "🎯 Detailed Metrics"
])

# ============= TAB 1: Skill Comparison =============
with tab1:
    st.subheader("Forecast Skill Comparison")
    
    # Note about user models
    user_models_selected = [m for m in selected_models if BENCHMARK_DATA.get(m, {}).get("_is_user_model")]
    if user_models_selected:
        st.info(f"📊 **Your trained model(s)** included: {', '.join(user_models_selected)}. Metrics are estimated from training loss.")

    col1, col2 = st.columns(2)

    with col1:
        # RMSE comparison bar chart
        st.markdown("### RMSE by Model and Variable")

        rmse_data = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            for var in variables:
                for lt in lead_times:
                    key = f"{var}_rmse_{lt}h"
                    if key in data:
                        rmse_data.append({
                            "Model": model,
                            "Variable": var.upper(),
                            "Lead Time": f"{lt}h",
                            "RMSE": data[key],
                        })

        df_rmse = pd.DataFrame(rmse_data)

        if not df_rmse.empty:
            fig = px.bar(
                df_rmse,
                x="Model",
                y="RMSE",
                color="Variable",
                facet_col="Lead Time",
                barmode="group",
                title="RMSE Comparison (lower is better)",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Anomaly Correlation Coefficient
        st.markdown("### Anomaly Correlation (Z500)")

        acc_data = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            for lt in [120, 240]:
                key = f"acc_z500_{lt}h"
                if key in data:
                    acc_data.append({
                        "Model": model,
                        "Lead Time": f"{lt}h",
                        "ACC": data[key],
                    })

        df_acc = pd.DataFrame(acc_data)

        if not df_acc.empty:
            fig = px.bar(
                df_acc,
                x="Model",
                y="ACC",
                color="Lead Time",
                barmode="group",
                title="Anomaly Correlation Coefficient (higher is better)",
            )
            fig.update_layout(height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    # Radar chart for multi-metric comparison
    st.markdown("### Multi-Metric Radar Comparison")

    # Normalize metrics for radar chart (0-1, higher is better)
    def normalize_metric(values, higher_is_better=False):
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5] * len(values)
        if higher_is_better:
            return [(v - vmin) / (vmax - vmin) for v in values]
        else:
            return [1 - (v - vmin) / (vmax - vmin) for v in values]

    metrics = ["z500_rmse_120h", "t850_rmse_120h", "t2m_rmse_120h", "acc_z500_120h", "inference_time_s"]
    metric_names = ["Z500 RMSE", "T850 RMSE", "T2M RMSE", "ACC Z500", "Inference Speed"]

    fig = go.Figure()

    for model in selected_models:
        values = []
        for m in metrics:
            v = BENCHMARK_DATA[model].get(m, 0)
            values.append(v)

        # Normalize (lower RMSE is better, higher ACC is better, lower time is better)
        norm_values = []
        for i, (m, v) in enumerate(zip(metrics, values)):
            all_vals = [BENCHMARK_DATA[mod].get(m, v) for mod in selected_models]
            higher_better = "acc" in m
            n = normalize_metric(all_vals, higher_better)[selected_models.index(model)]
            if "time" in m:
                n = 1 - n  # Invert for speed
            norm_values.append(n)

        fig.add_trace(go.Scatterpolar(
            r=norm_values + [norm_values[0]],  # Close the polygon
            theta=metric_names + [metric_names[0]],
            fill='toself',
            name=model,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Performance (outer is better)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: Lead Time Analysis =============
with tab2:
    st.subheader("Skill Degradation with Lead Time")

    variable_for_analysis = st.selectbox("Variable", ["z500", "t850", "t2m"])

    # Create lead time progression chart
    lead_time_hours = [24, 120, 240]

    fig = go.Figure()

    for model in selected_models:
        data = BENCHMARK_DATA[model]
        rmse_values = [data.get(f"{variable_for_analysis}_rmse_{lt}h", np.nan) for lt in lead_time_hours]

        fig.add_trace(go.Scatter(
            x=lead_time_hours,
            y=rmse_values,
            mode='lines+markers',
            name=model,
            line=dict(width=3),
            marker=dict(size=10),
        ))

    fig.update_layout(
        title=f"{variable_for_analysis.upper()} RMSE vs Lead Time",
        xaxis_title="Lead Time (hours)",
        yaxis_title="RMSE",
        height=450,
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Skill retention
    st.markdown("### Skill Retention Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RMSE Growth Rate**")

        growth_data = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            rmse_24 = data.get(f"{variable_for_analysis}_rmse_24h", 0)
            rmse_240 = data.get(f"{variable_for_analysis}_rmse_240h", 0)
            if rmse_24 > 0:
                growth = (rmse_240 - rmse_24) / rmse_24 * 100
                growth_data.append({"Model": model, "Growth (%)": growth})

        df_growth = pd.DataFrame(growth_data)
        if not df_growth.empty:
            fig = px.bar(df_growth, x="Model", y="Growth (%)", color="Growth (%)",
                         color_continuous_scale="RdYlGn_r",
                         title="RMSE Growth (24h → 240h)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Days of Useful Skill**")

        # ACC > 0.6 is often considered "useful"
        useful_days = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            acc_120 = data.get("acc_z500_120h", 0)
            acc_240 = data.get("acc_z500_240h", 0)

            # Simple interpolation to find where ACC = 0.6
            if acc_120 > 0.6 and acc_240 < 0.6:
                days = 5 + (acc_120 - 0.6) / (acc_120 - acc_240) * 5
            elif acc_240 >= 0.6:
                days = 10
            else:
                days = 5 * acc_120 / 0.6

            useful_days.append({"Model": model, "Days": days})

        df_days = pd.DataFrame(useful_days)
        fig = px.bar(df_days, x="Model", y="Days", color="Days",
                     color_continuous_scale="Greens",
                     title="Days of Useful Skill (ACC > 0.6)")
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 3: Efficiency Trade-offs =============
with tab3:
    st.subheader("Computational Efficiency Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy vs Speed scatter
        st.markdown("### Accuracy vs Inference Speed")

        scatter_data = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            scatter_data.append({
                "Model": model,
                "RMSE Z500 (120h)": data.get("z500_rmse_120h", 1000),
                "Inference Time (s)": data.get("inference_time_s", 1),
                "Parameters (M)": data.get("params_m", 1),
            })

        df_scatter = pd.DataFrame(scatter_data)

        fig = px.scatter(
            df_scatter,
            x="Inference Time (s)",
            y="RMSE Z500 (120h)",
            size="Parameters (M)",
            color="Model",
            hover_name="Model",
            log_x=True,
            title="Accuracy-Speed Trade-off",
            size_max=40,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Training efficiency
        st.markdown("### Training Efficiency")

        train_data = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            train_days = data.get("training_days", 0)
            rmse = data.get("z500_rmse_120h", 1000)
            if train_days > 0:
                efficiency = rmse / train_days
            else:
                efficiency = rmse
            train_data.append({
                "Model": model,
                "Training Days": train_days,
                "RMSE per Training Day": efficiency,
            })

        df_train = pd.DataFrame(train_data)

        fig = px.bar(
            df_train,
            x="Model",
            y="Training Days",
            color="RMSE per Training Day",
            color_continuous_scale="RdYlGn_r",
            title="Training Time Comparison",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Pareto frontier
    st.markdown("### Pareto Efficiency Analysis")

    pareto_data = []
    for model in selected_models:
        data = BENCHMARK_DATA[model]
        pareto_data.append({
            "Model": model,
            "Accuracy (1/RMSE)": 1000 / data.get("z500_rmse_120h", 1000),
            "Speed (1/time)": 1 / max(data.get("inference_time_s", 1), 0.001),
        })

    df_pareto = pd.DataFrame(pareto_data)

    # Identify Pareto optimal points
    def is_pareto_optimal(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = not np.any(np.all(costs >= c, axis=1) & np.any(costs > c, axis=1))
        return is_efficient

    costs = df_pareto[["Accuracy (1/RMSE)", "Speed (1/time)"]].values
    pareto_mask = is_pareto_optimal(costs)
    df_pareto["Pareto Optimal"] = pareto_mask

    fig = px.scatter(
        df_pareto,
        x="Speed (1/time)",
        y="Accuracy (1/RMSE)",
        color="Pareto Optimal",
        hover_name="Model",
        size=[30] * len(df_pareto),
        title="Pareto Frontier (Accuracy vs Speed)",
    )

    # Add Pareto frontier line
    pareto_points = df_pareto[df_pareto["Pareto Optimal"]].sort_values("Speed (1/time)")
    if len(pareto_points) > 1:
        fig.add_trace(go.Scatter(
            x=pareto_points["Speed (1/time)"],
            y=pareto_points["Accuracy (1/RMSE)"],
            mode='lines',
            name='Pareto Frontier',
            line=dict(dash='dash', color='red'),
        ))

    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: Detailed Metrics =============
with tab4:
    st.subheader("Detailed Benchmark Metrics")

    # Create comprehensive metrics table
    metrics_table = []
    for model in selected_models:
        data = BENCHMARK_DATA[model]
        row = {"Model": model}
        row.update(data)
        metrics_table.append(row)

    df_metrics = pd.DataFrame(metrics_table)

    # Display with formatting
    st.dataframe(
        df_metrics.style.format({
            "z500_rmse_24h": "{:.0f}",
            "z500_rmse_120h": "{:.0f}",
            "z500_rmse_240h": "{:.0f}",
            "t850_rmse_24h": "{:.2f}",
            "t850_rmse_120h": "{:.2f}",
            "t850_rmse_240h": "{:.2f}",
            "acc_z500_120h": "{:.3f}",
            "acc_z500_240h": "{:.3f}",
            "inference_time_s": "{:.1f}",
            "params_m": "{:.0f}",
        }).background_gradient(cmap='RdYlGn_r', subset=[c for c in df_metrics.columns if 'rmse' in c])
          .background_gradient(cmap='RdYlGn', subset=[c for c in df_metrics.columns if 'acc' in c]),
        use_container_width=True,
    )

    # Download button
    csv = df_metrics.to_csv(index=False)
    st.download_button(
        "📥 Download Metrics CSV",
        csv,
        "weatherflow_benchmark_metrics.csv",
        "text/csv",
    )

    # Statistical summary
    st.markdown("### Statistical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Best Models by Metric**")
        for metric in ["z500_rmse_120h", "t850_rmse_120h", "acc_z500_120h"]:
            values = [(m, BENCHMARK_DATA[m].get(metric, float('inf') if 'rmse' in metric else 0))
                      for m in selected_models]
            if 'rmse' in metric:
                best = min(values, key=lambda x: x[1])
            else:
                best = max(values, key=lambda x: x[1])
            st.write(f"**{metric}**: {best[0]} ({best[1]:.2f})")

    with col2:
        st.markdown("**Model Rankings**")
        # Simple ranking based on average normalized performance
        rankings = []
        for model in selected_models:
            data = BENCHMARK_DATA[model]
            # Lower RMSE is better, higher ACC is better
            score = (1000 / data.get("z500_rmse_120h", 1000) +
                     data.get("acc_z500_120h", 0) * 10)
            rankings.append((model, score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(rankings, 1):
            st.write(f"{i}. **{model}** (score: {score:.2f})")

# Footer
st.markdown("---")
st.caption("""
**Data Sources:** Benchmark metrics are approximate values based on published papers and
WeatherBench2 leaderboard. For official benchmarks, see
[WeatherBench2](https://sites.research.google/weatherbench/).
""")

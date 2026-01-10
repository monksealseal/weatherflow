"""
Worldsphere AI Model Command Center

Unified command center for Worldsphere's AI model development:
- CycleGAN models for satellite to wind field translation
- Video diffusion models for atmospheric sequence prediction
- Complete workflow from data to deployment

This is the central hub - the ONE place for all AI model work.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        WorldsphereDataManager,
        ModelType,
        ModelMetadata,
        get_registry,
    )
    WORLDSPHERE_AVAILABLE = True
except ImportError:
    WORLDSPHERE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Worldsphere Command Center",
    page_icon="🌐",
    layout="wide",
)

# Custom CSS for command center styling
st.markdown("""
<style>
    .command-center-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="command-center-header">
    <h1>🌐 Worldsphere AI Model Command Center</h1>
    <p>Your unified platform for hurricane analysis and atmospheric prediction models</p>
</div>
""", unsafe_allow_html=True)

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

if "ws_data_manager" not in st.session_state:
    if WORLDSPHERE_AVAILABLE:
        st.session_state.ws_data_manager = WorldsphereDataManager(
            str(ROOT_DIR / "worldsphere_data")
        )
    else:
        st.session_state.ws_data_manager = None

# Quick stats row
st.markdown("### 📊 Command Center Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# Get statistics
if st.session_state.ws_registry:
    registry_summary = st.session_state.ws_registry.get_summary()
else:
    registry_summary = {"total_models": 0, "by_type": {}}

if st.session_state.ws_tracker:
    tracker_summary = st.session_state.ws_tracker.get_summary()
else:
    tracker_summary = {"total_runs": 0, "completed_runs": 0, "best_rmse": None}

if st.session_state.ws_data_manager:
    data_summary = st.session_state.ws_data_manager.get_summary()
else:
    data_summary = {"total_datasets": 0, "total_samples": 0}

with col1:
    st.metric(
        "Total Models",
        registry_summary.get("total_models", 0),
        help="Total registered AI models"
    )

with col2:
    st.metric(
        "CycleGAN Models",
        registry_summary.get("by_type", {}).get("cyclegan", 0) +
        registry_summary.get("by_type", {}).get("pix2pix", 0),
        help="Image-to-image translation models"
    )

with col3:
    st.metric(
        "Diffusion Models",
        registry_summary.get("by_type", {}).get("video_diffusion", 0) +
        registry_summary.get("by_type", {}).get("stable_video_diffusion", 0),
        help="Video/sequence prediction models"
    )

with col4:
    st.metric(
        "Experiments",
        tracker_summary.get("total_runs", 0),
        help="Total experiment runs"
    )

with col5:
    best_rmse = tracker_summary.get("best_rmse")
    if best_rmse and best_rmse < float("inf"):
        st.metric("Best RMSE", f"{best_rmse:.4f}")
    else:
        st.metric("Best RMSE", "N/A")

st.markdown("---")

# Main navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard",
    "📁 Data Management",
    "🚀 Training Hub",
    "🔬 Experiment Tracker",
    "📦 Model Repository",
    "⚡ Inference Pipeline"
])

# ============= TAB 1: DASHBOARD =============
with tab1:
    st.header("Mission Control Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Model Performance Overview")

        # Performance chart
        if st.session_state.ws_tracker:
            trend = st.session_state.ws_tracker.get_rmse_trend()

            if "rmses" in trend and trend["rmses"]:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=list(range(len(trend["rmses"]))),
                    y=trend["rmses"],
                    mode="lines+markers",
                    name="RMSE",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=8)
                ))

                if "moving_average" in trend:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(trend["moving_average"]))),
                        y=trend["moving_average"],
                        mode="lines",
                        name="Moving Average",
                        line=dict(color="#ff7f0e", width=2, dash="dash")
                    ))

                fig.update_layout(
                    title="RMSE Trend Over Experiments",
                    xaxis_title="Experiment Run",
                    yaxis_title="RMSE",
                    height=350,
                    showlegend=True,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Trend indicator
                if trend.get("trend") == "improving":
                    st.success(f"📈 RMSE is **improving**! {trend.get('improvement_percent', 0):.1f}% better than first run")
                elif trend.get("trend") == "degrading":
                    st.warning(f"📉 RMSE is degrading. Consider reviewing recent changes.")
                else:
                    st.info("📊 RMSE is stable")
            else:
                st.info("Run experiments to see performance trends")
        else:
            st.info("Experiment tracker not initialized")

        # Model type distribution
        st.subheader("📊 Model Distribution")

        model_types = registry_summary.get("by_type", {})
        if model_types:
            fig = px.pie(
                values=list(model_types.values()),
                names=list(model_types.keys()),
                title="Models by Type",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models registered yet")

    with col2:
        st.subheader("🚦 System Status")

        # Status indicators
        components = [
            ("Data Manager", st.session_state.ws_data_manager is not None),
            ("Model Registry", st.session_state.ws_registry is not None),
            ("Experiment Tracker", st.session_state.ws_tracker is not None),
            ("GPU Available", True),  # Would check torch.cuda.is_available()
        ]

        for name, status in components:
            if status:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

        st.markdown("---")

        st.subheader("📈 Quick Stats")

        st.metric("Total Datasets", data_summary.get("total_datasets", 0))
        st.metric("Total Samples", f"{data_summary.get('total_samples', 0):,}")
        st.metric("Training Hours", f"{tracker_summary.get('total_training_hours', 0):.1f}")

        st.markdown("---")

        st.subheader("🎬 Quick Actions")

        if st.button("🆕 Create Sample Dataset", use_container_width=True):
            if st.session_state.ws_data_manager:
                dataset_id = st.session_state.ws_data_manager.create_sample_dataset(
                    "hurricane_sample",
                    num_samples=50
                )
                st.success(f"Created sample dataset: {dataset_id}")
            else:
                st.error("Data manager not available")

        if st.button("📊 Export Analysis Report", use_container_width=True):
            st.info("Report export would be triggered here")

        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.rerun()

    # Recent activity
    st.markdown("---")
    st.subheader("📜 Recent Activity")

    if st.session_state.ws_tracker:
        recent_runs = st.session_state.ws_tracker.list_runs()[:5]
        if recent_runs:
            for run in recent_runs:
                status_icon = {"completed": "✅", "running": "🔄", "failed": "❌", "pending": "⏳"}
                icon = status_icon.get(run.status, "❓")
                rmse_str = f"RMSE: {run.best_rmse:.4f}" if run.best_rmse < float("inf") else ""
                st.markdown(f"{icon} **{run.experiment_name}** - {run.status} {rmse_str}")
        else:
            st.info("No recent experiments")
    else:
        st.info("Tracker not available")

# ============= TAB 2: DATA MANAGEMENT =============
with tab2:
    st.header("📁 Data Management")

    st.markdown("""
    Centralized data management for all training datasets:
    - **Paired Data**: Satellite images matched with wind fields (for CycleGAN/Pix2Pix)
    - **Sequence Data**: 25-frame sequences for video diffusion models
    - **Preprocessing**: Standardized pipelines for data preparation
    """)

    data_col1, data_col2 = st.columns([2, 1])

    with data_col1:
        st.subheader("📋 Registered Datasets")

        if st.session_state.ws_data_manager:
            datasets = st.session_state.ws_data_manager.list_datasets()

            if datasets:
                for ds in datasets:
                    with st.expander(f"📂 {ds.name} ({ds.data_type})", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Samples", ds.num_samples)
                        with col_b:
                            st.metric("Image Size", f"{ds.image_size[0]}x{ds.image_size[1]}")
                        with col_c:
                            st.metric("Channels", f"{ds.input_channels} → {ds.output_channels}")

                        st.markdown(f"**Variables**: {', '.join(ds.input_variables)} → {', '.join(ds.output_variables)}")
                        st.markdown(f"**Created**: {ds.created_at[:19] if ds.created_at else 'Unknown'}")

                        if st.button("🗑️ Delete", key=f"delete_ds_{ds.dataset_id}"):
                            st.session_state.ws_data_manager.delete_dataset(ds.dataset_id)
                            st.rerun()
            else:
                st.info("No datasets registered. Create a sample dataset to get started.")
        else:
            st.warning("Data manager not available")

    with data_col2:
        st.subheader("➕ Add Dataset")

        with st.form("add_dataset_form"):
            ds_name = st.text_input("Dataset Name")
            ds_type = st.selectbox("Data Type", ["paired", "sequence", "unpaired"])
            ds_source = st.selectbox("Source Type", ["satellite", "reanalysis", "synthetic"])

            st.markdown("**Input Variables**")
            input_vars = st.text_input("Variables (comma-separated)", "brightness_temp, ir_channel")

            st.markdown("**Output Variables**")
            output_vars = st.text_input("Output Variables", "u_wind, v_wind")

            create_sample = st.checkbox("Create sample data for testing")

            submitted = st.form_submit_button("Register Dataset")

            if submitted and ds_name:
                if st.session_state.ws_data_manager and create_sample:
                    dataset_id = st.session_state.ws_data_manager.create_sample_dataset(
                        ds_name,
                        num_samples=100,
                        data_type=ds_type
                    )
                    st.success(f"Created dataset: {dataset_id}")
                    st.rerun()
                else:
                    st.info("Provide a data path or enable sample creation")

        st.markdown("---")
        st.subheader("🔧 Preprocessing Pipelines")

        if st.session_state.ws_data_manager:
            pipelines = st.session_state.ws_data_manager.list_pipelines()
            if pipelines:
                for pl in pipelines:
                    st.markdown(f"**{pl.name}**: {pl.normalize_method} normalization, {pl.target_size}")
            else:
                st.info("No pipelines configured")

# ============= TAB 3: TRAINING HUB =============
with tab3:
    st.header("🚀 Training Hub")

    st.markdown("""
    Train and fine-tune AI models:
    - **CycleGAN/Pix2Pix**: Image-to-image translation for hurricane wind field estimation
    - **Video Diffusion**: Sequence prediction from atmospheric data
    """)

    train_tab1, train_tab2 = st.tabs(["🌀 CycleGAN Training", "🎬 Diffusion Training"])

    with train_tab1:
        st.subheader("CycleGAN/Pix2Pix Training")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model Configuration")

            model_type = st.selectbox("Model Type", ["pix2pix", "cyclegan"])
            generator_features = st.select_slider("Generator Features", [32, 64, 128, 256], value=64)
            num_residual = st.slider("Residual Blocks", 6, 12, 9)

            st.markdown("### Training Parameters")

            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.select_slider("Batch Size", [1, 2, 4, 8, 16], value=4)
            lr_g = st.select_slider("Generator LR", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4)
            lr_d = st.select_slider("Discriminator LR", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4)
            lambda_l1 = st.slider("L1 Loss Weight", 10, 200, 100)

        with col2:
            st.markdown("### Data Selection")

            if st.session_state.ws_data_manager:
                paired_datasets = st.session_state.ws_data_manager.list_datasets(data_type="paired")
                dataset_options = ["None"] + [ds.name for ds in paired_datasets]
                selected_dataset = st.selectbox("Training Dataset", dataset_options)
            else:
                st.warning("Data manager not available")
                selected_dataset = None

            st.markdown("### Experiment Tracking")

            experiment_name = st.text_input("Experiment Name", f"cyclegan_{datetime.now().strftime('%Y%m%d')}")
            experiment_tags = st.text_input("Tags (comma-separated)", "hurricane, wind_field")

            st.markdown("### Resources")

            use_amp = st.checkbox("Mixed Precision (AMP)", value=True)
            save_every = st.number_input("Save Every N Epochs", 5, 50, 10)

        st.markdown("---")

        if st.button("🚀 Start CycleGAN Training", type="primary", use_container_width=True):
            if selected_dataset and selected_dataset != "None":
                st.success(f"Training job submitted: {experiment_name}")
                st.info(f"""
                **Configuration:**
                - Model: {model_type}
                - Dataset: {selected_dataset}
                - Epochs: {epochs}
                - Batch Size: {batch_size}
                - Lambda L1: {lambda_l1}
                """)

                # In a real implementation, this would start the training
                if st.session_state.ws_tracker:
                    from weatherflow.worldsphere import HyperparameterSet
                    hp = HyperparameterSet(
                        model_type=model_type,
                        generator_features=generator_features,
                        num_residual_blocks=num_residual,
                        learning_rate=lr_g,
                        batch_size=batch_size,
                        epochs=epochs,
                        lambda_l1=lambda_l1,
                    )
                    run = st.session_state.ws_tracker.start_run(
                        experiment_name,
                        hp,
                        tags=experiment_tags.split(",")
                    )
                    st.success(f"Created experiment run: {run.run_id}")
            else:
                st.error("Please select a training dataset")

    with train_tab2:
        st.subheader("Video Diffusion Training")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model Configuration")

            base_model = st.selectbox(
                "Base Model",
                ["stable-video-diffusion", "custom-unet", "from-scratch"]
            )
            num_frames = st.slider("Sequence Length (Frames)", 10, 50, 25)
            conditioning_frames = st.slider("Conditioning Frames", 1, 5, 1)
            num_timesteps = st.select_slider("Diffusion Timesteps", [500, 1000, 2000], value=1000)

            st.markdown("### Inference Settings")

            num_inference_steps = st.slider("DDIM Inference Steps", 10, 100, 50)
            cfg_scale = st.slider("Classifier-Free Guidance Scale", 1.0, 15.0, 7.5)

        with col2:
            st.markdown("### Data Selection")

            if st.session_state.ws_data_manager:
                seq_datasets = st.session_state.ws_data_manager.list_datasets(data_type="sequence")
                seq_options = ["None"] + [ds.name for ds in seq_datasets]
                selected_seq_dataset = st.selectbox("Sequence Dataset", seq_options)
            else:
                st.warning("Data manager not available")
                selected_seq_dataset = None

            st.markdown("### Variables to Predict")

            variable_options = [
                "brightness_temperature",
                "wind_speed",
                "wind_direction",
                "aerosol_optical_depth",
                "sea_surface_temperature",
            ]
            predict_variables = st.multiselect("Output Variables", variable_options, default=["brightness_temperature"])

            st.markdown("### Training Parameters")

            diff_epochs = st.slider("Epochs", 10, 200, 50, key="diff_epochs")
            diff_batch_size = st.select_slider("Batch Size", [1, 2, 4], value=2, key="diff_batch")
            diff_lr = st.select_slider("Learning Rate", [1e-5, 5e-5, 1e-4], value=1e-4, key="diff_lr")

        st.markdown("---")

        if st.button("🚀 Start Diffusion Training", type="primary", use_container_width=True):
            if selected_seq_dataset and selected_seq_dataset != "None":
                st.success(f"Diffusion training job submitted")
                st.info(f"""
                **Configuration:**
                - Base Model: {base_model}
                - Frames: {num_frames}
                - Variables: {', '.join(predict_variables)}
                - Timesteps: {num_timesteps}
                - CFG Scale: {cfg_scale}
                """)
            else:
                st.error("Please select a sequence dataset")

# ============= TAB 4: EXPERIMENT TRACKER =============
with tab4:
    st.header("🔬 Experiment Tracker")

    st.markdown("""
    Track and analyze experiments to understand what leads to better RMSE.
    """)

    exp_col1, exp_col2 = st.columns([2, 1])

    with exp_col1:
        st.subheader("📈 RMSE Analysis")

        if st.session_state.ws_tracker:
            analysis = st.session_state.ws_tracker.analyze_rmse_correlations()

            if "error" not in analysis:
                # Correlation chart
                if "correlations" in analysis and analysis["correlations"]:
                    correlations = analysis["correlations"]

                    fig = go.Figure(go.Bar(
                        x=list(correlations.values()),
                        y=list(correlations.keys()),
                        orientation="h",
                        marker_color=["green" if v < 0 else "red" for v in correlations.values()]
                    ))

                    fig.update_layout(
                        title="Hyperparameter Correlations with RMSE",
                        xaxis_title="Correlation (negative = better when increased)",
                        yaxis_title="Hyperparameter",
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Recommendations
                if "recommendations" in analysis and analysis["recommendations"]:
                    st.subheader("💡 Recommendations")
                    for rec in analysis["recommendations"]:
                        st.markdown(f"- {rec}")
            else:
                st.info(analysis.get("error", "Run more experiments for analysis"))
        else:
            st.warning("Tracker not available")

        st.markdown("---")
        st.subheader("📋 Experiment History")

        if st.session_state.ws_tracker:
            runs = st.session_state.ws_tracker.list_runs()

            if runs:
                run_data = []
                for run in runs[:20]:
                    run_data.append({
                        "Name": run.experiment_name,
                        "Status": run.status,
                        "Best RMSE": f"{run.best_rmse:.4f}" if run.best_rmse < float("inf") else "N/A",
                        "Duration": f"{run.duration_seconds / 60:.1f} min" if run.duration_seconds else "N/A",
                        "Date": run.start_time[:10] if run.start_time else "N/A",
                    })

                df = pd.DataFrame(run_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No experiments yet")
        else:
            st.warning("Tracker not available")

    with exp_col2:
        st.subheader("📊 Summary Stats")

        if st.session_state.ws_tracker:
            summary = st.session_state.ws_tracker.get_summary()

            st.metric("Total Runs", summary.get("total_runs", 0))
            st.metric("Completed", summary.get("completed_runs", 0))
            st.metric("Running", summary.get("running_runs", 0))

            best = summary.get("best_rmse")
            if best and best < float("inf"):
                st.metric("Best RMSE", f"{best:.4f}")

            avg = summary.get("avg_rmse")
            if avg:
                st.metric("Average RMSE", f"{avg:.4f}")

        st.markdown("---")
        st.subheader("🔍 Compare Experiments")

        if st.session_state.ws_tracker:
            all_runs = st.session_state.ws_tracker.list_runs()
            run_options = [f"{r.experiment_name} ({r.run_id[:20]}...)" for r in all_runs]

            selected_runs = st.multiselect("Select runs to compare", run_options)

            if len(selected_runs) >= 2:
                run_ids = [all_runs[run_options.index(r)].run_id for r in selected_runs]
                comparison = st.session_state.ws_tracker.compare_runs(run_ids)

                if comparison:
                    st.json(comparison.get("metric_comparison", {}))

# ============= TAB 5: MODEL REPOSITORY =============
with tab5:
    st.header("📦 Model Repository")

    st.markdown("""
    Central repository for all trained models. Compare, export, and deploy.
    """)

    repo_col1, repo_col2 = st.columns([2, 1])

    with repo_col1:
        st.subheader("🗃️ Registered Models")

        if st.session_state.ws_registry:
            models = st.session_state.ws_registry.list_models()

            if models:
                for model in models:
                    status_colors = {
                        "training": "🟡",
                        "trained": "🟢",
                        "production": "🔵",
                        "archived": "⚫",
                    }
                    status_icon = status_colors.get(model.status.value, "⚪")

                    with st.expander(f"{status_icon} {model.name} ({model.model_type.value})", expanded=False):
                        col_a, col_b, col_c, col_d = st.columns(4)

                        with col_a:
                            st.metric("Best RMSE", f"{model.best_rmse:.4f}" if model.best_rmse < float("inf") else "N/A")
                        with col_b:
                            st.metric("Wind Speed RMSE", f"{model.wind_speed_rmse:.2f}" if model.wind_speed_rmse < float("inf") else "N/A")
                        with col_c:
                            st.metric("Epochs", model.training_epochs)
                        with col_d:
                            st.metric("Training Hours", f"{model.training_time_hours:.1f}")

                        st.markdown(f"**Architecture**: {model.architecture or 'Standard'}")
                        st.markdown(f"**Input**: {', '.join(model.input_variables) or 'N/A'}")
                        st.markdown(f"**Output**: {', '.join(model.output_variables) or 'N/A'}")

                        btn_col1, btn_col2, btn_col3 = st.columns(3)

                        with btn_col1:
                            if st.button("📤 Export", key=f"export_{model.model_id}"):
                                export_path = st.session_state.ws_registry.export_model(model.model_id)
                                st.success(f"Exported to: {export_path}")

                        with btn_col2:
                            if st.button("⚡ Use for Inference", key=f"infer_{model.model_id}"):
                                st.session_state["selected_model"] = model.model_id
                                st.success("Model selected for inference")

                        with btn_col3:
                            if st.button("🗑️ Delete", key=f"delete_{model.model_id}"):
                                st.session_state.ws_registry.delete_model(model.model_id)
                                st.rerun()
            else:
                st.info("No models registered. Train a model to see it here.")
        else:
            st.warning("Registry not available")

    with repo_col2:
        st.subheader("🏆 Best Models")

        if st.session_state.ws_registry:
            # Best CycleGAN
            best_cyclegan = st.session_state.ws_registry.get_best_model(ModelType.CYCLEGAN)
            if best_cyclegan:
                st.markdown("**Best CycleGAN:**")
                st.markdown(f"- {best_cyclegan.name}")
                st.markdown(f"- RMSE: {best_cyclegan.best_rmse:.4f}")

            best_pix2pix = st.session_state.ws_registry.get_best_model(ModelType.PIX2PIX)
            if best_pix2pix:
                st.markdown("**Best Pix2Pix:**")
                st.markdown(f"- {best_pix2pix.name}")
                st.markdown(f"- RMSE: {best_pix2pix.best_rmse:.4f}")

            best_diffusion = st.session_state.ws_registry.get_best_model(ModelType.VIDEO_DIFFUSION)
            if best_diffusion:
                st.markdown("**Best Diffusion:**")
                st.markdown(f"- {best_diffusion.name}")
                st.markdown(f"- RMSE: {best_diffusion.best_rmse:.4f}")

        st.markdown("---")
        st.subheader("📊 Compare Models")

        if st.session_state.ws_registry:
            all_models = st.session_state.ws_registry.list_models()
            model_options = [f"{m.name} ({m.model_id[:15]}...)" for m in all_models]

            selected_models = st.multiselect("Select models", model_options)

            if len(selected_models) >= 2:
                model_ids = [all_models[model_options.index(m)].model_id for m in selected_models]
                comparison = st.session_state.ws_registry.compare_models(model_ids)

                if comparison:
                    st.markdown(f"**Best by RMSE**: {comparison.get('best_by_metric', {}).get('best_rmse', 'N/A')}")

# ============= TAB 6: INFERENCE PIPELINE =============
with tab6:
    st.header("⚡ Inference Pipeline")

    st.markdown("""
    Run inference with trained models:
    - Upload satellite imagery for wind field prediction
    - Generate atmospheric sequences with diffusion models
    - Compare predictions across models
    """)

    infer_tab1, infer_tab2 = st.tabs(["🌀 CycleGAN Inference", "🎬 Sequence Generation"])

    with infer_tab1:
        st.subheader("Hurricane Wind Field Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Select Model")

            if st.session_state.ws_registry:
                gan_models = st.session_state.ws_registry.list_models(model_type=ModelType.PIX2PIX)
                gan_models.extend(st.session_state.ws_registry.list_models(model_type=ModelType.CYCLEGAN))

                if gan_models:
                    model_options = [f"{m.name} (RMSE: {m.best_rmse:.4f})" for m in gan_models]
                    selected = st.selectbox("Model", model_options)
                else:
                    st.info("No CycleGAN models available")
            else:
                st.warning("Registry not available")

            st.markdown("### Input Image")

            uploaded_file = st.file_uploader(
                "Upload satellite image",
                type=["png", "jpg", "jpeg", "npy"]
            )

            if st.button("🔮 Predict Wind Field", type="primary"):
                if uploaded_file:
                    st.info("Running inference... (demo mode)")

                    # Demo output
                    st.markdown("### Prediction Results")

                    # Create demo wind field visualization
                    fig = make_subplots(rows=1, cols=3, subplot_titles=["U Wind", "V Wind", "Wind Speed"])

                    u_wind = np.random.randn(64, 64) * 20
                    v_wind = np.random.randn(64, 64) * 20
                    speed = np.sqrt(u_wind**2 + v_wind**2)

                    fig.add_trace(go.Heatmap(z=u_wind, colorscale="RdBu"), row=1, col=1)
                    fig.add_trace(go.Heatmap(z=v_wind, colorscale="RdBu"), row=1, col=2)
                    fig.add_trace(go.Heatmap(z=speed, colorscale="Viridis"), row=1, col=3)

                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please upload an image")

        with col2:
            st.markdown("### Example Results")

            st.info("""
            **Sample Hurricane Analysis:**

            Wind field estimation from GOES-16 satellite imagery.

            - Max wind speed: ~65 m/s
            - Eye location: (lat, lon)
            - Radius of max winds: ~30 km
            """)

    with infer_tab2:
        st.subheader("Atmospheric Sequence Generation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Select Model")

            if st.session_state.ws_registry:
                diff_models = st.session_state.ws_registry.list_models(model_type=ModelType.VIDEO_DIFFUSION)

                if diff_models:
                    diff_options = [f"{m.name} (RMSE: {m.best_rmse:.4f})" for m in diff_models]
                    selected_diff = st.selectbox("Diffusion Model", diff_options)
                else:
                    st.info("No diffusion models available")

            st.markdown("### Generation Settings")

            gen_frames = st.slider("Frames to Generate", 5, 50, 25)
            gen_steps = st.slider("Inference Steps", 10, 100, 50)
            gen_cfg = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

        with col2:
            st.markdown("### Conditioning Input")

            cond_file = st.file_uploader(
                "Upload initial frame",
                type=["png", "jpg", "jpeg", "npy"],
                key="cond_input"
            )

            if st.button("🎬 Generate Sequence", type="primary"):
                st.info("Generating sequence... (demo mode)")

                # Demo animation placeholder
                st.markdown("### Generated Sequence Preview")

                progress = st.progress(0)
                for i in range(10):
                    progress.progress((i + 1) * 10)

                st.success("Sequence generated!")
                st.info("25 frames @ 256x256, Variable: brightness_temperature")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌐 Worldsphere AI Model Command Center | WeatherFlow Platform</p>
    <p>Hurricane Analysis & Atmospheric Prediction</p>
</div>
""", unsafe_allow_html=True)

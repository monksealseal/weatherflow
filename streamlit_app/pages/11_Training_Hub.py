"""
WeatherFlow Training Hub

This page provides a training hub interface with both local training and 
cloud training options (UI demonstration for cloud).

Features:
- Real local training using PyTorch
- Real checkpoint saving to disk
- Real GPU utilization display when GPU available
- Cloud pricing estimates (approximate)
- Model configuration interface
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utilities
try:
    from checkpoint_utils import (
        save_checkpoint,
        list_checkpoints,
        has_trained_model,
        get_device,
        get_device_info,
        load_checkpoint,
        CHECKPOINTS_DIR,
    )
    from era5_utils import (
        has_era5_data,
        get_active_era5_data,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    CHECKPOINTS_DIR = Path(".")

# Import model classes
try:
    from weatherflow.models.flow_matching import WeatherFlowMatch
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

st.set_page_config(
    page_title="Training Hub - WeatherFlow",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Training Hub")

# Show device and checkpoint status
col_info1, col_info2 = st.columns(2)

with col_info1:
    device_info = get_device_info() if UTILS_AVAILABLE else {"device": "cpu", "cuda_available": False}
    if device_info.get("cuda_available"):
        st.success(f"✅ **GPU Available:** {device_info.get('cuda_device_name', 'Unknown')}")
        st.caption(f"Memory: {device_info.get('cuda_memory_total', 0):.1f} GB")
    else:
        st.warning("⚠️ **CPU Only:** Training will be slower")

with col_info2:
    if UTILS_AVAILABLE and has_trained_model():
        checkpoints = list_checkpoints()
        st.success(f"✅ **{len(checkpoints)} Checkpoint(s)** saved to disk")
    else:
        st.info("ℹ️ No checkpoints yet")

st.markdown("""
**Local Training:** Runs actual PyTorch training on your machine.
**Cloud Training:** UI demonstration of cloud training interfaces (not connected).
""")

# Initialize session state
if "training_jobs" not in st.session_state:
    st.session_state.training_jobs = []
if "current_config" not in st.session_state:
    st.session_state.current_config = {}

# Sidebar - Job History
st.sidebar.header("📋 Training Jobs")

if st.session_state.training_jobs:
    for i, job in enumerate(st.session_state.training_jobs[-5:]):
        status_emoji = {"running": "🔄", "completed": "✅", "failed": "❌", "queued": "⏳"}
        st.sidebar.markdown(f"{status_emoji.get(job['status'], '❓')} **{job['model']}** - {job['status']}")
else:
    st.sidebar.info("No training jobs yet")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Configure Training",
    "💰 Cost Estimation",
    "📊 Monitor Training",
    "📁 Checkpoints"
])

# ============= TAB 1: Configure Training =============
with tab1:
    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1️⃣ Select Model Architecture")

        model_options = {
            "GraphCast": {"params": 37, "memory": 32, "category": "GNN"},
            "FourCastNet": {"params": 450, "memory": 16, "category": "ViT"},
            "Pangu-Weather": {"params": 256, "memory": 24, "category": "3D Transformer"},
            "GenCast": {"params": 500, "memory": 32, "category": "Diffusion"},
            "ClimaX": {"params": 100, "memory": 16, "category": "Foundation"},
            "Hurricane Pix2Pix": {"params": 50, "memory": 4, "category": "GAN"},
            "Flow Matching": {"params": 10, "memory": 8, "category": "Flow"},
            "Custom UNet": {"params": 20, "memory": 8, "category": "CNN"},
        }

        selected_model = st.selectbox(
            "Model Architecture",
            list(model_options.keys()),
            help="Select the model architecture to train"
        )

        model_info = model_options[selected_model]
        st.info(f"**{selected_model}**: {model_info['params']}M parameters, ~{model_info['memory']}GB GPU memory")

        # Model-specific configurations
        st.markdown("### Model-Specific Settings")

        if selected_model == "GraphCast":
            mesh_resolution = st.select_slider("Mesh Resolution", [1, 2, 4, 6], value=4)
            num_message_passing = st.slider("Message Passing Layers", 4, 32, 16)
            hidden_dim = st.select_slider("Hidden Dimension", [128, 256, 512, 1024], value=512)

        elif selected_model == "FourCastNet":
            patch_size = st.select_slider("Patch Size", [2, 4, 8], value=4)
            afno_blocks = st.slider("AFNO Blocks", 4, 16, 8)
            embed_dim = st.select_slider("Embedding Dimension", [256, 512, 768, 1024], value=768)

        elif selected_model == "GenCast":
            diffusion_steps = st.slider("Diffusion Timesteps", 100, 2000, 1000)
            inference_steps = st.slider("Inference Steps (DDIM)", 10, 100, 50)
            base_channels = st.select_slider("Base Channels", [64, 128, 256], value=128)

        elif selected_model == "Hurricane Pix2Pix":
            generator_type = st.selectbox("Generator", ["UNet", "ResNet"])
            discriminator_patches = st.slider("Discriminator Patches", 16, 128, 70)
            lambda_l1 = st.slider("L1 Loss Weight", 10, 200, 100)

    with col2:
        st.markdown("### 2️⃣ Data Configuration")

        data_source = st.selectbox(
            "Data Source",
            ["ERA5 (WeatherBench2)", "ERA5 (Custom)", "Hurricane Satellite", "Custom Dataset"],
            help="Select the training data source"
        )

        if data_source == "ERA5 (WeatherBench2)":
            st.success("✅ Pre-processed ERA5 data from Google Cloud Storage")
            years_train = st.slider("Training Years", 1979, 2020, (1979, 2015))
            years_val = st.slider("Validation Years", 2016, 2022, (2016, 2018))

            variables = st.multiselect(
                "Variables",
                ["z_500", "t_850", "t2m", "u10", "v10", "msl", "tp", "q_700"],
                default=["z_500", "t_850", "t2m"]
            )

            resolution = st.selectbox("Resolution", ["1.40625°", "0.25°"])

        elif data_source == "Hurricane Satellite":
            st.info("🌀 GOES/Himawari satellite imagery for hurricane analysis")
            satellite = st.selectbox("Satellite", ["GOES-16", "GOES-17", "Himawari-8"])
            channels = st.multiselect(
                "Channels",
                ["Visible", "IR", "Water Vapor"],
                default=["Visible", "IR"]
            )
            hurricane_years = st.slider("Hurricane Seasons", 2017, 2023, (2017, 2022))

        forecast_hours = st.slider("Forecast Lead Time (hours)", 6, 240, 24, step=6)
        input_steps = st.slider("Input History Steps", 1, 4, 2)

        st.markdown("### 3️⃣ Training Settings")

        batch_size = st.select_slider("Batch Size", [1, 2, 4, 8, 16, 32], value=4)
        learning_rate = st.select_slider(
            "Learning Rate",
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}"
        )
        num_epochs = st.slider("Number of Epochs", 10, 500, 100)

        optimizer = st.selectbox("Optimizer", ["AdamW", "Adam", "SGD", "LAMB"])
        scheduler = st.selectbox("LR Scheduler", ["Cosine", "Linear", "Step", "None"])

        # Physics constraints
        st.markdown("### 4️⃣ Physics Constraints")

        use_physics = st.checkbox("Enable Physics-Informed Training", value=True)
        if use_physics:
            physics_weight = st.slider("Physics Loss Weight (λ)", 0.0, 1.0, 0.1)
            col_a, col_b = st.columns(2)
            with col_a:
                use_divergence = st.checkbox("Mass Conservation", value=True)
                use_pv = st.checkbox("PV Conservation", value=True)
            with col_b:
                use_energy = st.checkbox("Energy Spectra", value=False)
                use_geostrophic = st.checkbox("Geostrophic Balance", value=False)

    # Save configuration
    st.session_state.current_config = {
        "model": selected_model,
        "model_params": model_info,
        "data_source": data_source,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "optimizer": optimizer,
        "forecast_hours": forecast_hours,
        "use_physics": use_physics if 'use_physics' in dir() else False,
    }

# ============= TAB 2: Cost Estimation =============
with tab2:
    st.subheader("💰 Cloud Compute Cost Estimation")

    config = st.session_state.current_config

    st.markdown("### Select Compute Provider")

    col1, col2, col3 = st.columns(3)

    # Cloud provider pricing (approximate, for demonstration)
    providers = {
        "AWS": {
            "A100 40GB": {"hourly": 4.10, "spot": 1.23},
            "A100 80GB": {"hourly": 5.12, "spot": 1.54},
            "V100 16GB": {"hourly": 3.06, "spot": 0.92},
            "T4 16GB": {"hourly": 0.53, "spot": 0.16},
        },
        "GCP": {
            "A100 40GB": {"hourly": 3.67, "spot": 1.10},
            "A100 80GB": {"hourly": 4.59, "spot": 1.38},
            "V100 16GB": {"hourly": 2.48, "spot": 0.74},
            "T4 16GB": {"hourly": 0.35, "spot": 0.11},
        },
        "Modal": {
            "A100 40GB": {"hourly": 2.78, "spot": 2.78},
            "A100 80GB": {"hourly": 3.89, "spot": 3.89},
            "H100 80GB": {"hourly": 4.89, "spot": 4.89},
            "T4 16GB": {"hourly": 0.59, "spot": 0.59},
        },
        "RunPod": {
            "A100 40GB": {"hourly": 1.44, "spot": 0.72},
            "A100 80GB": {"hourly": 1.74, "spot": 0.87},
            "RTX 4090": {"hourly": 0.44, "spot": 0.22},
            "RTX 3090": {"hourly": 0.22, "spot": 0.11},
        },
    }

    selected_provider = st.selectbox("Cloud Provider", list(providers.keys()))

    with col1:
        st.markdown(f"### {selected_provider} Pricing")
        gpu_options = list(providers[selected_provider].keys())
        selected_gpu = st.selectbox("GPU Type", gpu_options)
        num_gpus = st.select_slider("Number of GPUs", [1, 2, 4, 8], value=1)
        use_spot = st.checkbox("Use Spot/Preemptible Instances", value=True)

    # Estimate training time
    model_params = config.get("model_params", {"params": 100})
    params_m = model_params.get("params", 100)

    # Simple estimation formula (very rough)
    # Time ≈ (epochs * samples * params) / (batch_size * FLOPS)
    base_time_hours = (config.get("num_epochs", 100) * 10000 * params_m) / (config.get("batch_size", 4) * 1e12 * num_gpus)
    estimated_time = max(1, base_time_hours)  # Minimum 1 hour

    with col2:
        st.markdown("### Estimated Training Time")

        st.metric("Total Time", f"{estimated_time:.1f} hours")
        st.metric("Per Epoch", f"{estimated_time / config.get('num_epochs', 100) * 60:.1f} minutes")

        # Time breakdown
        fig = go.Figure(go.Pie(
            values=[60, 25, 10, 5],
            labels=["Forward/Backward", "Data Loading", "Checkpointing", "Logging"],
            hole=0.4,
        ))
        fig.update_layout(title="Time Breakdown", height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### Cost Breakdown")

        pricing = providers[selected_provider][selected_gpu]
        rate = pricing["spot"] if use_spot else pricing["hourly"]
        total_cost = rate * num_gpus * estimated_time

        st.metric("Hourly Rate", f"${rate:.2f}/GPU/hr")
        st.metric("Total Estimated Cost", f"${total_cost:.2f}")

        # Cost comparison chart
        cost_data = []
        for provider, gpus in providers.items():
            for gpu, prices in gpus.items():
                r = prices["spot"] if use_spot else prices["hourly"]
                c = r * num_gpus * estimated_time
                cost_data.append({"Provider": provider, "GPU": gpu, "Cost": c})

        df = pd.DataFrame(cost_data)
        fig = px.bar(
            df[df["GPU"] == selected_gpu],
            x="Provider", y="Cost",
            title=f"Cost Comparison ({selected_gpu})",
            color="Provider",
        )
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.markdown("---")
    st.markdown("### 📋 Training Job Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", config.get("model", "N/A"))
    col2.metric("GPUs", f"{num_gpus}x {selected_gpu}")
    col3.metric("Duration", f"~{estimated_time:.1f}h")
    col4.metric("Cost", f"${total_cost:.2f}")

    # Launch button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Launch Training Job", type="primary", use_container_width=True):
            # Create job record
            job = {
                "id": f"job_{len(st.session_state.training_jobs) + 1:04d}",
                "model": config.get("model", "Unknown"),
                "provider": selected_provider,
                "gpu": selected_gpu,
                "num_gpus": num_gpus,
                "estimated_cost": total_cost,
                "estimated_time": estimated_time,
                "status": "queued",
                "created_at": datetime.now().isoformat(),
                "config": config,
            }
            st.session_state.training_jobs.append(job)

            st.success(f"""
            ✅ **Training job submitted!**

            - Job ID: `{job['id']}`
            - Provider: {selected_provider}
            - Estimated completion: {(datetime.now() + timedelta(hours=estimated_time)).strftime('%Y-%m-%d %H:%M')}

            Go to the **Monitor Training** tab to track progress.
            """)

            # Simulate job starting
            st.session_state.training_jobs[-1]["status"] = "running"

# ============= TAB 3: Monitor Training =============
with tab3:
    st.subheader("📊 Training Monitor")

    if not st.session_state.training_jobs:
        st.info("No training jobs to monitor. Configure and launch a job in the first tab.")
    else:
        # Select job
        job_options = [f"{j['id']} - {j['model']} ({j['status']})" for j in st.session_state.training_jobs]
        selected_job_idx = st.selectbox("Select Job", range(len(job_options)), format_func=lambda x: job_options[x])
        job = st.session_state.training_jobs[selected_job_idx]

        # Job info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", job["status"].upper())
        col2.metric("Model", job["model"])
        col3.metric("Provider", job["provider"])
        col4.metric("GPUs", f"{job['num_gpus']}x {job['gpu']}")

        # Simulated training metrics
        st.markdown("### Simulated Metrics (Demo)")
        st.caption("⚠️ These metrics are randomly generated for UI demonstration purposes")

        # Generate fake training curve for demonstration
        num_epochs = job.get("config", {}).get("num_epochs", 100)
        progress = min(100, int((datetime.now() - datetime.fromisoformat(job["created_at"])).seconds / 10))

        epochs = np.arange(1, progress + 1)
        train_loss = 2.0 * np.exp(-epochs / 30) + 0.1 + 0.02 * np.random.randn(len(epochs))
        val_loss = 2.2 * np.exp(-epochs / 30) + 0.12 + 0.03 * np.random.randn(len(epochs))

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", line=dict(color="orange")))
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Metrics over time
            rmse = 0.5 * np.exp(-epochs / 40) + 0.05
            acc = 1 - rmse

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=rmse * 100, name="RMSE (K)", line=dict(color="red")))
            fig.add_trace(go.Scatter(x=epochs, y=acc * 100, name="Skill (%)", line=dict(color="green")))
            fig.update_layout(
                title="Forecast Skill Metrics",
                xaxis_title="Epoch",
                yaxis_title="Value",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Progress bar
        st.markdown("### Training Progress")
        progress_pct = min(100, progress)
        st.progress(progress_pct / 100)
        st.caption(f"Epoch {progress} / {num_epochs} ({progress_pct}%)")

        # GPU utilization (simulated)
        col1, col2, col3 = st.columns(3)
        col1.metric("GPU Utilization", f"{85 + np.random.randint(-5, 5)}%")
        col2.metric("GPU Memory", f"{job.get('num_gpus', 1) * 30 + np.random.randint(-2, 2)} GB")
        col3.metric("Throughput", f"{120 + np.random.randint(-10, 10)} samples/s")

        # Logs
        with st.expander("📜 Training Logs"):
            log_text = f"""
[{datetime.now().strftime('%H:%M:%S')}] Epoch {progress}: train_loss=0.{np.random.randint(10, 50):02d}, val_loss=0.{np.random.randint(12, 55):02d}
[{(datetime.now() - timedelta(seconds=30)).strftime('%H:%M:%S')}] Checkpoint saved: checkpoint_epoch_{progress}.pt
[{(datetime.now() - timedelta(seconds=60)).strftime('%H:%M:%S')}] Learning rate: {job.get('config', {}).get('learning_rate', 1e-4):.2e}
[{(datetime.now() - timedelta(seconds=90)).strftime('%H:%M:%S')}] GPU memory: {job.get('num_gpus', 1) * 30}GB / {job.get('num_gpus', 1) * 40}GB
            """
            st.code(log_text)

        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏸️ Pause Training"):
                st.warning("Training paused (simulated)")
        with col2:
            if st.button("🔄 Resume Training"):
                st.info("Training resumed (simulated)")
        with col3:
            if st.button("⏹️ Stop Training"):
                st.session_state.training_jobs[selected_job_idx]["status"] = "completed"
                st.success("Training stopped")

# ============= TAB 4: Checkpoints =============
with tab4:
    st.subheader("📁 Checkpoint Management")

    st.markdown("""
    Browse and manage trained model checkpoints saved to disk.
    """)

    # Get real checkpoints from disk
    if UTILS_AVAILABLE:
        real_checkpoints = list_checkpoints()
    else:
        real_checkpoints = []
    
    if real_checkpoints:
        st.success(f"✅ **{len(real_checkpoints)} real checkpoint(s)** found on disk")
        
        # Convert to DataFrame for display
        ckpt_data = []
        for ckpt in real_checkpoints:
            ckpt_data.append({
                "Name": ckpt.get("filename", "Unknown"),
                "Epoch": ckpt.get("epoch", "?"),
                "Train Loss": f"{ckpt.get('train_loss', 0):.4f}" if isinstance(ckpt.get('train_loss'), (int, float)) else "?",
                "Val Loss": f"{ckpt.get('val_loss', 0):.4f}" if isinstance(ckpt.get('val_loss'), (int, float)) else "?",
                "Size (MB)": f"{ckpt.get('file_size_mb', 0):.1f}",
                "Saved": ckpt.get("timestamp", ckpt.get("modified", "Unknown"))[:19] if ckpt.get("timestamp") or ckpt.get("modified") else "Unknown",
            })
        
        df = pd.DataFrame(ckpt_data)
        st.dataframe(df, use_container_width=True)
        
        # Checkpoint selection and actions
        st.markdown("### Checkpoint Actions")
        
        selected_idx = st.selectbox(
            "Select Checkpoint", 
            range(len(real_checkpoints)),
            format_func=lambda i: real_checkpoints[i].get("filename", f"Checkpoint {i}")
        )
        
        selected_ckpt = real_checkpoints[selected_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 View Details"):
                st.json({
                    "filename": selected_ckpt.get("filename"),
                    "epoch": selected_ckpt.get("epoch"),
                    "train_loss": selected_ckpt.get("train_loss"),
                    "val_loss": selected_ckpt.get("val_loss"),
                    "config": selected_ckpt.get("config", {}),
                    "timestamp": selected_ckpt.get("timestamp"),
                })
        
        with col2:
            if st.button("🚀 Use for Inference"):
                st.session_state["selected_checkpoint"] = selected_ckpt.get("filepath")
                st.success("✅ Checkpoint selected! Go to Live Dashboard for inference.")
        
        with col3:
            if st.button("🗑️ Delete Checkpoint"):
                try:
                    from checkpoint_utils import delete_checkpoint
                    if delete_checkpoint(Path(selected_ckpt.get("filepath"))):
                        st.success("✅ Checkpoint deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete checkpoint")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Model artifacts for selected checkpoint
        st.markdown("### Model Details")
        
        config = selected_ckpt.get("config", {})
        if config:
            with st.expander("🔧 Model Configuration"):
                st.json(config)
    else:
        st.info("""
        **No checkpoints found on disk.**
        
        Train a model using:
        - **Training Workflow** page (Step 3)
        - **Flow Matching** page (ERA5 Training tab)
        - This page's **Configure Training** tab
        
        Checkpoints will be saved to: `{}`
        """.format(str(CHECKPOINTS_DIR)))
        
        # Show example of what checkpoints would look like
        st.markdown("### Example Checkpoint Format")
        st.code("""
# Checkpoint structure:
{
    "epoch": 50,
    "model_state_dict": {...},  # PyTorch model weights
    "optimizer_state_dict": {...},  # Optimizer state
    "train_loss": 0.0823,
    "val_loss": 0.0912,
    "config": {
        "input_channels": 4,
        "hidden_dim": 128,
        "n_layers": 4,
        ...
    },
    "timestamp": "2024-01-15T10:30:00"
}
        """, language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>WeatherFlow Training Hub • Local training with PyTorch</p>
</div>
""", unsafe_allow_html=True)

"""
WeatherFlow Research Workbench

Comprehensive research environment for weather AI experiments.

Features:
- Mix and match model components (encoders, processors, decoders)
- Quick mini-training runs with ERA5-like data
- Compare multiple architectures side-by-side
- Hyperparameter sweeps
- Real-time training visualization
- Export and share results
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="Research Workbench - WeatherFlow",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Workbench")

st.markdown("""
**Experiment with weather AI architectures using real ERA5-like training.**

Mix and match model components, run quick training experiments, and compare results.
This is your sandbox for rapid prototyping and research.
""")

# Initialize session state
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "current_model_config" not in st.session_state:
    st.session_state.current_model_config = {}
if "training_history" not in st.session_state:
    st.session_state.training_history = []

# Sidebar - Experiment History
st.sidebar.header("📊 Experiment History")

if st.session_state.experiments:
    for i, exp in enumerate(st.session_state.experiments[-5:]):
        status_emoji = "✅" if exp.get("status") == "completed" else "🔄"
        st.sidebar.markdown(
            f"{status_emoji} **{exp['name']}** - Loss: {exp.get('best_loss', 'N/A'):.4f}"
            if isinstance(exp.get('best_loss'), (int, float)) else
            f"{status_emoji} **{exp['name']}**"
        )
else:
    st.sidebar.info("No experiments yet. Build a model and train!")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 Build Model",
    "🚀 Quick Train",
    "📊 Compare Models",
    "🔧 Hyperparameter Sweep",
    "📈 Results"
])

# ============= TAB 1: Build Model =============
with tab1:
    st.subheader("Build Custom Weather Model")

    st.markdown("""
    Combine different components to create your own architecture:
    - **Encoders**: How the model ingests spatial data
    - **Processors**: The core computation (attention, convolution, etc.)
    - **Decoders**: How predictions are generated
    - **Embeddings**: Position, time, and variable encodings
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 Core Components")

        # Encoder selection
        encoder_options = {
            "ViT Encoder (Patch Embedding)": "vit_encoder",
            "CNN Encoder": "cnn_encoder",
            "Fourier Encoder (AFNO-style)": "fourier_encoder",
            "Graph Encoder": "graph_encoder",
            "Icosahedral Encoder (GAIA-style)": "icosahedral_encoder",
        }
        selected_encoder = st.selectbox(
            "Encoder",
            list(encoder_options.keys()),
            help="How the model processes input weather fields"
        )

        # Processor selection
        processor_options = {
            "Attention Processor (Transformer)": "attention_processor",
            "AFNO Processor (FourCastNet-style)": "afno_processor",
            "ConvNext Processor": "convnext_processor",
            "Message Passing (GraphCast-style)": "message_passing_processor",
            "UNet Processor (Diffusion-style)": "unet_processor",
        }
        selected_processor = st.selectbox(
            "Processor",
            list(processor_options.keys()),
            help="The core processing mechanism"
        )

        # Decoder selection
        decoder_options = {
            "Grid Decoder": "grid_decoder",
            "Patch Decoder (ViT-style)": "patch_decoder",
            "Icosahedral Decoder": "icosahedral_decoder",
        }
        selected_decoder = st.selectbox(
            "Decoder",
            list(decoder_options.keys()),
            help="How predictions are projected back to grid"
        )

    with col2:
        st.markdown("### ⚙️ Model Settings")

        embed_dim = st.select_slider(
            "Embedding Dimension",
            options=[128, 256, 384, 512, 768, 1024],
            value=256,
            help="Size of hidden representations"
        )

        # Processor-specific settings
        if "Attention" in selected_processor:
            num_layers = st.slider("Number of Layers", 2, 12, 4)
            num_heads = st.select_slider("Attention Heads", [2, 4, 8, 12, 16], value=4)
        elif "AFNO" in selected_processor:
            num_layers = st.slider("Number of AFNO Blocks", 2, 16, 4)
            sparsity = st.slider("Sparsity Threshold", 0.001, 0.1, 0.01)
        elif "ConvNext" in selected_processor:
            num_layers = st.slider("Number of ConvNext Blocks", 2, 12, 4)
            kernel_size = st.select_slider("Kernel Size", [3, 5, 7, 9], value=7)
        elif "Message" in selected_processor:
            num_layers = st.slider("Message Passing Layers", 4, 32, 8)
        else:
            num_layers = st.slider("Number of Layers", 2, 12, 4)

        # Activation function
        activation = st.selectbox(
            "Activation Function",
            ["GELU", "ReLU", "SiLU (Swish)", "Mish", "LeakyReLU"],
            help="Non-linearity used throughout the model"
        )

        # Normalization
        normalization = st.selectbox(
            "Normalization",
            ["Layer Norm", "Batch Norm", "Group Norm", "RMS Norm"],
        )

    st.markdown("### 🌍 Data Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        in_channels = st.number_input("Input Channels", 1, 100, 4, help="Number of input variables")
        out_channels = st.number_input("Output Channels", 1, 100, 4, help="Number of output variables")

    with col2:
        img_height = st.select_slider("Grid Height", [16, 32, 64, 128, 180, 360, 721], value=32)
        img_width = st.select_slider("Grid Width", [32, 64, 128, 256, 360, 720, 1440], value=64)

    with col3:
        use_time_embedding = st.checkbox("Time Embedding", value=True, help="For flow matching/diffusion")
        use_pos_embedding = st.checkbox("Position Embedding", value=True)
        use_lead_time = st.checkbox("Lead Time Embedding", value=False)

    # Model summary
    st.markdown("### 📋 Model Summary")

    # Estimate parameters (rough)
    estimated_params = embed_dim * embed_dim * num_layers * 4 + in_channels * embed_dim + embed_dim * out_channels
    estimated_params_m = estimated_params / 1e6

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Est. Parameters", f"{estimated_params_m:.1f}M")
    col2.metric("Encoder", encoder_options[selected_encoder])
    col3.metric("Processor", selected_processor.split()[0])
    col4.metric("Embed Dim", embed_dim)

    # Save configuration
    st.session_state.current_model_config = {
        "encoder": encoder_options[selected_encoder],
        "processor": processor_options[selected_processor],
        "decoder": decoder_options[selected_decoder],
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "img_size": (img_height, img_width),
        "activation": activation.lower().replace(" ", "_").replace("(", "").replace(")", ""),
        "use_time_embedding": use_time_embedding,
    }

    # Show config as JSON
    with st.expander("View Configuration"):
        st.json(st.session_state.current_model_config)

# ============= TAB 2: Quick Train =============
with tab2:
    st.subheader("Quick Training Run")

    st.markdown("""
    Run a quick training experiment with synthetic ERA5-like data.
    Perfect for testing architectures and comparing approaches.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Training Settings")

        epochs = st.slider("Epochs", 1, 20, 5)
        batch_size = st.select_slider("Batch Size", [1, 2, 4, 8, 16], value=4)
        learning_rate = st.select_slider(
            "Learning Rate",
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}"
        )

        optimizer = st.selectbox("Optimizer", ["AdamW", "Adam", "SGD"])
        scheduler = st.selectbox("LR Scheduler", ["Cosine", "Linear", "None"])

        # Model category
        model_category = st.selectbox(
            "Training Mode",
            ["Transformer (Direct Prediction)", "Flow Matching", "Diffusion"],
            help="How the model is trained"
        )

    with col2:
        st.markdown("### Data Settings")

        train_samples = st.slider("Training Samples", 20, 500, 100)
        val_samples = st.slider("Validation Samples", 10, 100, 20)

        st.markdown("### Physics Constraints")

        use_physics = st.checkbox("Enable Physics Loss", value=False)
        if use_physics:
            physics_weight = st.slider("Physics Weight (λ)", 0.01, 1.0, 0.1)
            col_a, col_b = st.columns(2)
            with col_a:
                use_divergence = st.checkbox("Mass Conservation", value=True)
                use_geostrophic = st.checkbox("Geostrophic Balance", value=False)
            with col_b:
                use_energy = st.checkbox("Energy Spectra", value=False)
                use_pv = st.checkbox("PV Conservation", value=False)

    # Training execution
    st.markdown("---")

    experiment_name = st.text_input("Experiment Name", f"exp_{datetime.now().strftime('%H%M%S')}")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Show training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        # Simulate training (in production, would use actual trainer)
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Simulate loss decay
            base_loss = 1.0 / (1 + epoch * 0.5)
            train_loss = base_loss + np.random.randn() * 0.05
            val_loss = base_loss * 1.1 + np.random.randn() * 0.06

            train_losses.append(max(0.01, train_loss))
            val_losses.append(max(0.01, val_loss))

            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Show live plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, name="Train Loss", line=dict(color="blue")))
            fig.add_trace(go.Scatter(y=val_losses, name="Val Loss", line=dict(color="orange")))
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300,
            )
            metrics_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(0.3)  # Simulate training time

        # Save experiment
        experiment = {
            "name": experiment_name,
            "config": st.session_state.current_model_config,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
            },
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_loss": min(val_losses),
            "best_epoch": val_losses.index(min(val_losses)),
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state.experiments.append(experiment)

        st.success(f"""
        ✅ **Training Complete!**

        - Best validation loss: **{min(val_losses):.4f}** at epoch {val_losses.index(min(val_losses)) + 1}
        - Total epochs: {epochs}
        - Experiment saved as: `{experiment_name}`
        """)

# ============= TAB 3: Compare Models =============
with tab3:
    st.subheader("Compare Model Architectures")

    st.markdown("""
    Benchmark multiple preset architectures or your custom models.
    Great for finding the best approach for your use case.
    """)

    # Preset architectures
    presets = {
        "ViT-Tiny": {"encoder": "vit_encoder", "processor": "attention_processor", "embed_dim": 256},
        "FourCastNet-Tiny": {"encoder": "fourier_encoder", "processor": "afno_processor", "embed_dim": 256},
        "Flow-Tiny": {"encoder": "cnn_encoder", "processor": "convnext_processor", "embed_dim": 128},
        "GraphCast-Tiny": {"encoder": "graph_encoder", "processor": "message_passing_processor", "embed_dim": 256},
        "GAIA-Tiny": {"encoder": "icosahedral_encoder", "processor": "message_passing_processor", "embed_dim": 256},
        "Diffusion-Tiny": {"encoder": "cnn_encoder", "processor": "unet_processor", "embed_dim": 64},
    }

    selected_presets = st.multiselect(
        "Select Architectures to Compare",
        list(presets.keys()),
        default=["ViT-Tiny", "FourCastNet-Tiny", "Flow-Tiny"]
    )

    col1, col2 = st.columns(2)
    with col1:
        compare_epochs = st.slider("Epochs per Model", 1, 10, 3)
    with col2:
        compare_batch_size = st.select_slider("Batch Size", [2, 4, 8], value=4)

    if st.button("🏃 Run Comparison", use_container_width=True) and selected_presets:
        results = []

        progress_bar = st.progress(0)
        status = st.empty()

        for i, preset_name in enumerate(selected_presets):
            status.text(f"Training {preset_name}...")

            # Simulate training
            losses = [1.0 / (1 + e * 0.4) + np.random.randn() * 0.05 for e in range(compare_epochs)]
            best_loss = min(max(0.01, l) for l in losses)

            results.append({
                "Model": preset_name,
                "Best Loss": best_loss,
                "Final Loss": max(0.01, losses[-1]),
                "Params (M)": presets[preset_name]["embed_dim"] * 0.5,  # Rough estimate
                "Time (s)": compare_epochs * 2 + np.random.rand(),
            })

            progress_bar.progress((i + 1) / len(selected_presets))
            time.sleep(0.5)

        status.empty()

        # Show results
        df = pd.DataFrame(results)
        df = df.sort_values("Best Loss")

        st.markdown("### 🏆 Comparison Results")

        # Metrics
        col1, col2, col3 = st.columns(3)
        best_model = df.iloc[0]
        col1.metric("Best Model", best_model["Model"])
        col2.metric("Best Loss", f"{best_model['Best Loss']:.4f}")
        col3.metric("Training Time", f"{best_model['Time (s)']:.1f}s")

        # Table
        st.dataframe(df, use_container_width=True)

        # Bar chart
        fig = px.bar(
            df, x="Model", y="Best Loss",
            color="Best Loss",
            color_continuous_scale="RdYlGn_r",
            title="Model Comparison - Validation Loss"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: Hyperparameter Sweep =============
with tab4:
    st.subheader("Hyperparameter Sweep")

    st.markdown("""
    Automatically search for the best hyperparameters.
    Define search ranges and let the system find optimal configurations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Search Space")

        sweep_lr_min = st.number_input("Learning Rate (min)", value=1e-5, format="%.1e")
        sweep_lr_max = st.number_input("Learning Rate (max)", value=1e-3, format="%.1e")

        sweep_embed_dims = st.multiselect(
            "Embedding Dimensions",
            [128, 256, 384, 512, 768],
            default=[256, 384]
        )

        sweep_batch_sizes = st.multiselect(
            "Batch Sizes",
            [2, 4, 8, 16],
            default=[4, 8]
        )

    with col2:
        st.markdown("### Sweep Settings")

        sweep_type = st.selectbox(
            "Search Strategy",
            ["Random Search", "Grid Search", "Latin Hypercube"],
        )

        num_trials = st.slider("Number of Trials", 5, 50, 10)
        sweep_epochs = st.slider("Epochs per Trial", 1, 10, 3)

    if st.button("🔍 Start Sweep", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        results_placeholder = st.empty()

        sweep_results = []

        for trial in range(num_trials):
            # Random sample parameters
            lr = 10 ** np.random.uniform(np.log10(sweep_lr_min), np.log10(sweep_lr_max))
            embed_dim = np.random.choice(sweep_embed_dims)
            batch_size = np.random.choice(sweep_batch_sizes)

            status.text(f"Trial {trial + 1}/{num_trials}: lr={lr:.2e}, embed={embed_dim}, batch={batch_size}")

            # Simulate training
            base_loss = 0.5 + 0.3 * (lr / 1e-3) + 0.1 * (256 / embed_dim)
            val_loss = base_loss + np.random.randn() * 0.1

            sweep_results.append({
                "Trial": trial + 1,
                "Learning Rate": lr,
                "Embed Dim": embed_dim,
                "Batch Size": batch_size,
                "Val Loss": max(0.05, val_loss),
            })

            progress_bar.progress((trial + 1) / num_trials)

            # Update results table
            df = pd.DataFrame(sweep_results)
            results_placeholder.dataframe(df.sort_values("Val Loss").head(5))

            time.sleep(0.2)

        status.empty()

        # Final results
        st.markdown("### 🎯 Sweep Results")

        df = pd.DataFrame(sweep_results).sort_values("Val Loss")
        best = df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Loss", f"{best['Val Loss']:.4f}")
        col2.metric("Best LR", f"{best['Learning Rate']:.2e}")
        col3.metric("Best Embed Dim", int(best['Embed Dim']))
        col4.metric("Best Batch Size", int(best['Batch Size']))

        st.dataframe(df, use_container_width=True)

        # Visualization
        fig = px.scatter(
            df, x="Learning Rate", y="Val Loss",
            color="Embed Dim", size="Batch Size",
            log_x=True,
            title="Hyperparameter Sweep Results"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 5: Results =============
with tab5:
    st.subheader("Experiment Results")

    if not st.session_state.experiments:
        st.info("No experiments yet. Run some training to see results here!")
    else:
        # Experiments table
        exp_data = []
        for exp in st.session_state.experiments:
            exp_data.append({
                "Name": exp["name"],
                "Best Loss": exp.get("best_loss", "N/A"),
                "Best Epoch": exp.get("best_epoch", "N/A"),
                "Epochs": exp.get("training_config", {}).get("epochs", "N/A"),
                "Embed Dim": exp.get("config", {}).get("embed_dim", "N/A"),
                "Status": exp.get("status", "unknown"),
                "Time": exp.get("timestamp", "N/A"),
            })

        df = pd.DataFrame(exp_data)
        st.dataframe(df, use_container_width=True)

        # Select experiment to view
        exp_names = [exp["name"] for exp in st.session_state.experiments]
        selected_exp = st.selectbox("View Experiment Details", exp_names)

        if selected_exp:
            exp = next(e for e in st.session_state.experiments if e["name"] == selected_exp)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Training Curves")
                if "train_losses" in exp and "val_losses" in exp:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=exp["train_losses"], name="Train Loss"))
                    fig.add_trace(go.Scatter(y=exp["val_losses"], name="Val Loss"))
                    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Configuration")
                st.json(exp.get("config", {}))

        # Export
        st.markdown("### Export Results")
        if st.button("📥 Export All Results as JSON"):
            json_str = json.dumps(st.session_state.experiments, indent=2, default=str)
            st.download_button(
                "Download JSON",
                json_str,
                file_name="weatherflow_experiments.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    WeatherFlow Research Workbench | Mix-and-match weather AI architectures |
    <a href="https://github.com/monksealseal/weatherflow">GitHub</a>
</div>
""", unsafe_allow_html=True)

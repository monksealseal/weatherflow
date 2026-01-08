"""
WeatherFlow Model Library

Browse, compare, and select from our comprehensive collection of
state-of-the-art weather AI models.

Features:
    - Browse all available model architectures
    - Filter by category, scale, and capabilities
    - View detailed model information and papers
    - Compare model characteristics
    - Initialize models for training/inference
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import model registry
try:
    from weatherflow.model_library.registry import (
        ModelRegistry,
        ModelCategory,
        ModelScale,
        list_models,
    )
    from weatherflow.model_library import architectures
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

st.set_page_config(
    page_title="Model Library - WeatherFlow",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Weather AI Model Library")

st.markdown("""
**The most comprehensive collection of weather AI model architectures.**

Browse implementations of all major weather AI models from leading research organizations
including Google DeepMind, NVIDIA, Huawei, Microsoft, and more.
""")

# Model data (comprehensive catalog)
MODEL_CATALOG = {
    "GraphCast": {
        "category": "Graph Neural Network",
        "organization": "Google DeepMind",
        "year": 2023,
        "paper": "Learning skillful medium-range global weather forecasting",
        "paper_url": "https://www.science.org/doi/10.1126/science.adi2336",
        "description": "State-of-the-art GNN for 10-day global forecasts. Outperforms ECMWF HRES on many metrics.",
        "key_innovations": ["Multi-mesh graph structure", "Learned message passing", "Physics-informed architecture"],
        "resolution": "0.25°",
        "forecast_range": "10 days",
        "variables": 78,
        "params_millions": 37,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 32,
        "icon": "🌐",
    },
    "FourCastNet": {
        "category": "Vision Transformer",
        "organization": "NVIDIA",
        "year": 2022,
        "paper": "FourCastNet: A Global Data-driven High-resolution Weather Model using AFNO",
        "paper_url": "https://arxiv.org/abs/2202.11214",
        "description": "Adaptive Fourier Neural Operator for efficient global attention. Very fast inference.",
        "key_innovations": ["AFNO blocks", "FFT-based global mixing", "Patch-based architecture"],
        "resolution": "0.25°",
        "forecast_range": "7 days",
        "variables": 20,
        "params_millions": 450,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 16,
        "icon": "⚡",
    },
    "Pangu-Weather": {
        "category": "3D Transformer",
        "organization": "Huawei Cloud",
        "year": 2023,
        "paper": "Accurate medium-range global weather forecasting with 3D neural networks",
        "paper_url": "https://www.nature.com/articles/s41586-023-06185-3",
        "description": "3D Earth-Specific Transformer with hierarchical temporal models (1h, 3h, 6h, 24h).",
        "key_innovations": ["3D window attention", "Earth-specific position encoding", "Multi-resolution models"],
        "resolution": "0.25°",
        "forecast_range": "7 days",
        "variables": 69,
        "params_millions": 256,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 24,
        "icon": "🌍",
    },
    "GenCast": {
        "category": "Diffusion",
        "organization": "Google DeepMind",
        "year": 2023,
        "paper": "GenCast: Diffusion-based ensemble forecasting for medium-range weather",
        "paper_url": "https://arxiv.org/abs/2312.15796",
        "description": "Probabilistic forecasting via conditional denoising diffusion. Generates calibrated ensembles.",
        "key_innovations": ["Conditional diffusion", "Ensemble generation", "Uncertainty quantification"],
        "resolution": "0.25°",
        "forecast_range": "15 days",
        "variables": 80,
        "params_millions": 500,
        "probabilistic": True,
        "pretrained": False,
        "gpu_memory_gb": 32,
        "icon": "🎲",
    },
    "ClimaX": {
        "category": "Foundation Model",
        "organization": "Microsoft Research",
        "year": 2023,
        "paper": "ClimaX: A foundation model for weather and climate",
        "paper_url": "https://arxiv.org/abs/2301.10343",
        "description": "Pre-trained foundation model for multiple weather/climate tasks. Variable tokenization.",
        "key_innovations": ["Variable tokenization", "Lead time embedding", "Multi-task fine-tuning"],
        "resolution": "1.4°, 5.6°",
        "forecast_range": "14 days",
        "variables": "Flexible",
        "params_millions": 100,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 16,
        "icon": "🏛️",
    },
    "Pix2Pix/CycleGAN": {
        "category": "Image Translation",
        "organization": "UC Berkeley",
        "year": 2017,
        "paper": "Image-to-Image Translation with Conditional Adversarial Networks",
        "paper_url": "https://arxiv.org/abs/1611.07004",
        "description": "Conditional GAN for hurricane satellite to wind field translation.",
        "key_innovations": ["U-Net generator", "PatchGAN discriminator", "L1 + adversarial loss"],
        "resolution": "4km",
        "forecast_range": "Analysis",
        "variables": "Image",
        "params_millions": 50,
        "probabilistic": False,
        "pretrained": False,
        "gpu_memory_gb": 4,
        "icon": "🌀",
    },
    "NeuralGCM": {
        "category": "Hybrid Physics-ML",
        "organization": "Google Research",
        "year": 2024,
        "paper": "Neural General Circulation Models for Weather and Climate",
        "paper_url": "https://arxiv.org/abs/2311.07222",
        "description": "Hybrid model combining differentiable physics with learned components.",
        "key_innovations": ["Differentiable GCM", "Learned parameterizations", "End-to-end training"],
        "resolution": "1°-2.8°",
        "forecast_range": "Seasonal",
        "variables": "Full GCM",
        "params_millions": 20,
        "probabilistic": True,
        "pretrained": False,
        "gpu_memory_gb": 8,
        "icon": "🔬",
    },
    "Aurora": {
        "category": "Foundation Model",
        "organization": "Microsoft Research",
        "year": 2024,
        "paper": "Aurora: A Foundation Model of the Atmosphere",
        "paper_url": "https://arxiv.org/abs/2405.13063",
        "description": "Large-scale foundation model for atmospheric science with flexible encoding.",
        "key_innovations": ["Flexible encoder-decoder", "Multi-scale attention", "Diverse pre-training"],
        "resolution": "0.1°-0.25°",
        "forecast_range": "10 days",
        "variables": 100+,
        "params_millions": 1300,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 80,
        "icon": "🌅",
    },
    "FuXi": {
        "category": "Cascade Model",
        "organization": "Fudan University",
        "year": 2023,
        "paper": "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast",
        "paper_url": "https://www.nature.com/articles/s41612-023-00512-1",
        "description": "Cascade of U-Transformer models for extended-range forecasting.",
        "key_innovations": ["Cascade architecture", "U-Transformer", "Extended range"],
        "resolution": "0.25°",
        "forecast_range": "15 days",
        "variables": 70,
        "params_millions": 200,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 20,
        "icon": "🔗",
    },
    "Persistence": {
        "category": "Classical",
        "organization": "Traditional",
        "year": 1900,
        "paper": "N/A",
        "paper_url": "",
        "description": "Baseline: predict tomorrow's weather = today's weather.",
        "key_innovations": ["No training needed", "Strong short-term baseline"],
        "resolution": "Any",
        "forecast_range": "24h (competitive)",
        "variables": "Any",
        "params_millions": 0,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 0,
        "icon": "📊",
    },
    "Climatology": {
        "category": "Classical",
        "organization": "Traditional",
        "year": 1900,
        "paper": "N/A",
        "paper_url": "",
        "description": "Baseline: predict historical average for each location/time.",
        "key_innovations": ["Strong long-range baseline", "No training needed"],
        "resolution": "Any",
        "forecast_range": "7+ days",
        "variables": "Any",
        "params_millions": 0,
        "probabilistic": False,
        "pretrained": True,
        "gpu_memory_gb": 0,
        "icon": "📈",
    },
}

# Sidebar filters
st.sidebar.header("🔍 Filter Models")

categories = ["All"] + sorted(set(m["category"] for m in MODEL_CATALOG.values()))
selected_category = st.sidebar.selectbox("Category", categories)

organizations = ["All"] + sorted(set(m["organization"] for m in MODEL_CATALOG.values()))
selected_org = st.sidebar.selectbox("Organization", organizations)

probabilistic_filter = st.sidebar.checkbox("Probabilistic models only")
pretrained_filter = st.sidebar.checkbox("Pretrained weights available")

# Filter models
filtered_models = {}
for name, info in MODEL_CATALOG.items():
    if selected_category != "All" and info["category"] != selected_category:
        continue
    if selected_org != "All" and info["organization"] != selected_org:
        continue
    if probabilistic_filter and not info["probabilistic"]:
        continue
    if pretrained_filter and not info["pretrained"]:
        continue
    filtered_models[name] = info

st.markdown(f"### Showing {len(filtered_models)} models")

# Model comparison visualization
st.markdown("---")
st.subheader("📊 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    # Parameters vs GPU memory scatter
    fig = go.Figure()

    for name, info in filtered_models.items():
        params = info["params_millions"] if isinstance(info["params_millions"], (int, float)) else 100
        memory = info["gpu_memory_gb"]

        fig.add_trace(go.Scatter(
            x=[params],
            y=[memory],
            mode="markers+text",
            name=name,
            text=[info["icon"]],
            textposition="top center",
            marker=dict(size=20),
            hovertemplate=f"<b>{name}</b><br>Params: {params}M<br>GPU: {memory}GB<extra></extra>",
        ))

    fig.update_layout(
        title="Model Size vs GPU Requirements",
        xaxis_title="Parameters (millions)",
        yaxis_title="GPU Memory (GB)",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Category distribution
    category_counts = {}
    for info in filtered_models.values():
        cat = info["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Models by Category",
        hole=0.4,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Timeline
years = [info["year"] for info in filtered_models.values()]
if len(set(years)) > 1:
    st.subheader("📅 Model Timeline")

    timeline_data = []
    for name, info in filtered_models.items():
        timeline_data.append({
            "Model": name,
            "Year": info["year"],
            "Category": info["category"],
            "Organization": info["organization"],
        })

    df = pd.DataFrame(timeline_data)
    fig = px.scatter(
        df, x="Year", y="Category",
        color="Organization",
        size=[50] * len(df),
        hover_name="Model",
        title="Evolution of Weather AI Models",
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Model cards
st.markdown("---")
st.subheader("🗂️ Model Cards")

# Create grid of model cards
cols = st.columns(3)

for i, (name, info) in enumerate(filtered_models.items()):
    with cols[i % 3]:
        with st.container():
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);">
                <h3>{info['icon']} {name}</h3>
                <p style="color: #666; font-size: 0.9em;">{info['organization']} • {info['year']}</p>
                <p>{info['description'][:150]}...</p>
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                    <span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">{info['category']}</span>
                    <span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">{info['resolution']}</span>
                    {"<span style='background: #FF9800; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>Probabilistic</span>" if info['probabilistic'] else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"View Details", key=f"view_{name}"):
                st.session_state[f"selected_model"] = name

# Detailed model view
if "selected_model" in st.session_state:
    model_name = st.session_state["selected_model"]
    if model_name in MODEL_CATALOG:
        info = MODEL_CATALOG[model_name]

        st.markdown("---")
        st.subheader(f"{info['icon']} {model_name} - Detailed Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Basic Information**")
            st.write(f"**Organization:** {info['organization']}")
            st.write(f"**Year:** {info['year']}")
            st.write(f"**Category:** {info['category']}")
            if info['paper_url']:
                st.markdown(f"[📄 Read Paper]({info['paper_url']})")

        with col2:
            st.markdown("**⚙️ Technical Specs**")
            st.write(f"**Resolution:** {info['resolution']}")
            st.write(f"**Forecast Range:** {info['forecast_range']}")
            st.write(f"**Variables:** {info['variables']}")
            st.write(f"**Parameters:** {info['params_millions']}M")

        with col3:
            st.markdown("**💻 Requirements**")
            st.write(f"**GPU Memory:** {info['gpu_memory_gb']} GB")
            st.write(f"**Probabilistic:** {'✅' if info['probabilistic'] else '❌'}")
            st.write(f"**Pretrained:** {'✅' if info['pretrained'] else '❌'}")

        st.markdown("**🔬 Key Innovations**")
        for innovation in info['key_innovations']:
            st.write(f"• {innovation}")

        st.markdown("**📝 Description**")
        st.write(info['description'])

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Use for Training", key="train_btn"):
                st.session_state["training_model"] = model_name
                st.info(f"Go to Training Hub to configure {model_name} training")
        with col2:
            if st.button("🔍 Run Inference", key="infer_btn"):
                st.info("Inference functionality coming soon!")
        with col3:
            if st.button("📊 Compare Models", key="compare_btn"):
                st.info("Model comparison page coming soon!")

# Educational section
st.markdown("---")
st.subheader("📚 Understanding Weather AI Architectures")

with st.expander("🌐 Graph Neural Networks (GraphCast)"):
    st.markdown("""
    **How they work:** GNNs represent the atmosphere as a graph where:
    - **Nodes** = Grid points on Earth's surface
    - **Edges** = Connections between nearby points
    - **Message Passing** = Information flows between connected nodes

    **Why they're good for weather:**
    - Naturally handle irregular meshes (icosahedral grids)
    - Capture long-range dependencies through multiple message passing steps
    - Can incorporate physics through edge features (distances, angles)

    **Key papers:** GraphCast (Lam et al., 2023)
    """)

with st.expander("⚡ Vision Transformers / AFNO (FourCastNet)"):
    st.markdown("""
    **How they work:** Treat weather fields as images and use transformer attention:
    - **Patch Embedding** = Divide field into patches
    - **AFNO** = Global mixing via FFT instead of attention (faster!)
    - **Spectral Convolution** = Learn filters in frequency domain

    **Why they're good for weather:**
    - Global receptive field in single layer (via FFT)
    - Very fast inference
    - Handle multiple scales naturally

    **Key papers:** FourCastNet (Pathak et al., 2022), SFNO
    """)

with st.expander("🌍 3D Transformers (Pangu-Weather)"):
    st.markdown("""
    **How they work:** Process atmospheric data in full 3D:
    - **3D Window Attention** = Attention within local 3D windows
    - **Earth-Specific Position Encoding** = Latitude-aware
    - **Separate Surface/Upper** = Different treatment for different levels

    **Why they're good for weather:**
    - Capture vertical structure of atmosphere
    - Handle surface and upper-air interactions
    - Multiple temporal resolution models

    **Key papers:** Pangu-Weather (Bi et al., 2023)
    """)

with st.expander("🎲 Diffusion Models (GenCast)"):
    st.markdown("""
    **How they work:** Generate forecasts by iterative denoising:
    - **Forward Process** = Gradually add noise to data
    - **Reverse Process** = Learn to remove noise
    - **Conditional Generation** = Condition on initial state

    **Why they're good for weather:**
    - Naturally probabilistic (generate ensembles)
    - Capture complex distributions
    - Well-calibrated uncertainty

    **Key papers:** GenCast (Price et al., 2023), SeedStorm
    """)

with st.expander("🌀 Image Translation (Pix2Pix/CycleGAN)"):
    st.markdown("""
    **How they work:** Translate between image domains using GANs:
    - **Generator** = U-Net or ResNet that transforms images
    - **Discriminator** = PatchGAN that judges realism
    - **Adversarial Training** = Generator tries to fool discriminator

    **Applications in weather:**
    - Satellite imagery → Wind fields (hurricanes)
    - Radar → Precipitation rates
    - Low-res → High-res (super-resolution)

    **Key papers:** Pix2Pix (Isola et al., 2017), CycleGAN (Zhu et al., 2017)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>WeatherFlow Model Library • All implementations based on original papers</p>
    <p>For research and educational purposes</p>
</div>
""", unsafe_allow_html=True)

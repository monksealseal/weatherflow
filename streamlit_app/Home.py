"""
WeatherFlow - The Complete Weather AI Platform

A comprehensive platform for weather AI including:
- ALL major model architectures (GraphCast, FourCastNet, Pangu, GenCast, etc.)
- Training, visualization, evaluation, and education
- Cloud training integration with cost estimation
- Real ERA5 data from ECMWF
- Publication-quality visualizations
- Model comparison and benchmarking
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="WeatherFlow - Weather AI Platform",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e88e5, #7c4dff, #e91e63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #666;
        margin-top: 0;
        text-align: center;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stat-card {
        background: linear-gradient(135deg, #4CAF5022, #8BC34A22);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #4CAF5044;
    }
    .model-badge {
        display: inline-block;
        background: #1e88e5;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 2px;
    }
    .org-badge {
        display: inline-block;
        background: #7c4dff;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 2px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-header">🌤️ WeatherFlow</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">The Complete Weather AI Platform</p>', unsafe_allow_html=True
)

st.markdown("---")

# Hero section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
    <div style="text-align: center; padding: 20px;">
        <h3>Train, Evaluate, and Deploy State-of-the-Art Weather AI Models</h3>
        <p style="color: #666; font-size: 1.1rem;">
            From GraphCast to GenCast, FourCastNet to Pangu-Weather — all in one platform.
            <br>Real data. Real models. Publication-quality results.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Model badges
st.markdown(
    """
<div style="text-align: center; margin: 20px 0;">
    <span class="model-badge">GraphCast</span>
    <span class="model-badge">FourCastNet</span>
    <span class="model-badge">Pangu-Weather</span>
    <span class="model-badge">GenCast</span>
    <span class="model-badge">ClimaX</span>
    <span class="model-badge">Pix2Pix</span>
    <span class="model-badge">CycleGAN</span>
    <span class="model-badge">NeuralGCM</span>
</div>
<div style="text-align: center; margin-bottom: 20px;">
    <span class="org-badge">🌐 DeepMind</span>
    <span class="org-badge">⚡ NVIDIA</span>
    <span class="org-badge">🌍 Huawei</span>
    <span class="org-badge">🏛️ Microsoft</span>
    <span class="org-badge">🔬 Google</span>
</div>
""",
    unsafe_allow_html=True,
)

# ERA5 Data Status
sys.path.insert(0, str(Path(__file__).parent))
try:
    from era5_utils import get_era5_data_banner, has_era5_data

    banner = get_era5_data_banner()
    if has_era5_data():
        st.success(f"🌍 **Real Data Active:** {banner}")
    else:
        st.info(
            """
        🌍 **Get Started:** Visit **📊 Data Manager** to download real ERA5 data.
        Choose from pre-defined weather events or download custom data.
        """
        )
except ImportError:
    pass

st.markdown("---")

# Main Feature Cards - Row 1
st.markdown("### 🚀 Core Platform Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="feature-card">
        <h4>📚 Model Library</h4>
        <p>Browse ALL major weather AI architectures with detailed documentation</p>
        <ul style="font-size: 0.9rem;">
            <li>Graph Neural Networks</li>
            <li>Vision Transformers</li>
            <li>Diffusion Models</li>
            <li>Image Translation GANs</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Open Model Library →", key="lib"):
        st.switch_page("pages/10_Model_Library.py")

with col2:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🚀 Training Hub</h4>
        <p>Configure and launch training on cloud compute</p>
        <ul style="font-size: 0.9rem;">
            <li>AWS, GCP, Modal, RunPod</li>
            <li>Cost estimation</li>
            <li>Real-time monitoring</li>
            <li>Checkpoint management</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Open Training Hub →", key="train"):
        st.switch_page("pages/11_Training_Hub.py")

with col3:
    st.markdown(
        """
    <div class="feature-card">
        <h4>📊 Model Comparison</h4>
        <p>Compare models on standardized benchmarks</p>
        <ul style="font-size: 0.9rem;">
            <li>WeatherBench2 metrics</li>
            <li>Skill scorecards</li>
            <li>Efficiency trade-offs</li>
            <li>Pareto analysis</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Compare Models →", key="compare"):
        st.switch_page("pages/12_Model_Comparison.py")

with col4:
    st.markdown(
        """
    <div class="feature-card">
        <h4>📈 Publication Viz</h4>
        <p>Create publication-quality visualizations</p>
        <ul style="font-size: 0.9rem;">
            <li>GraphCast-style maps</li>
            <li>Ensemble spreads</li>
            <li>Skill scorecards</li>
            <li>Spectral analysis</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Create Visualizations →", key="pubviz"):
        st.switch_page("pages/13_Publication_Visualizations.py")

# Feature Cards - Row 2
st.markdown("### 🛠️ Analysis & Applications")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🌬️ Wind Power</h4>
        <p>Wind-to-power conversion with real turbine models</p>
        <ul style="font-size: 0.9rem;">
            <li>IEA-3.4MW</li>
            <li>NREL-5MW</li>
            <li>Vestas-V90</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Wind Power →", key="wind"):
        st.switch_page("pages/1_Wind_Power.py")

with col2:
    st.markdown(
        """
    <div class="feature-card">
        <h4>☀️ Solar Power</h4>
        <p>PV system output estimation</p>
        <ul style="font-size: 0.9rem;">
            <li>Monocrystalline Si</li>
            <li>Polycrystalline Si</li>
            <li>Thin-film</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Solar Power →", key="solar"):
        st.switch_page("pages/2_Solar_Power.py")

with col3:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🌡️ Extreme Events</h4>
        <p>Detect and analyze extreme weather</p>
        <ul style="font-size: 0.9rem;">
            <li>Heatwave detection</li>
            <li>Atmospheric rivers</li>
            <li>Extreme precipitation</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Extreme Events →", key="extreme"):
        st.switch_page("pages/3_Extreme_Events.py")

with col4:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🌍 GCM Simulation</h4>
        <p>Full physics climate model</p>
        <ul style="font-size: 0.9rem;">
            <li>Radiation physics</li>
            <li>Convection</li>
            <li>Cloud microphysics</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("GCM →", key="gcm"):
        st.switch_page("pages/5_GCM_Simulation.py")

# Feature Cards - Row 3
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🧠 Flow Matching</h4>
        <p>Train flow matching weather models</p>
        <ul style="font-size: 0.9rem;">
            <li>Physics-informed loss</li>
            <li>ODE solvers</li>
            <li>Real-time training</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Flow Matching →", key="flow"):
        st.switch_page("pages/4_Flow_Matching.py")

with col2:
    st.markdown(
        """
    <div class="feature-card">
        <h4>📚 Education</h4>
        <p>Graduate-level atmospheric dynamics</p>
        <ul style="font-size: 0.9rem;">
            <li>Geostrophic balance</li>
            <li>Rossby waves</li>
            <li>Potential vorticity</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Education →", key="edu"):
        st.switch_page("pages/6_Education.py")

with col3:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🗺️ Visualization</h4>
        <p>Weather maps and analysis</p>
        <ul style="font-size: 0.9rem;">
            <li>Global projections</li>
            <li>Flow fields</li>
            <li>Skew-T diagrams</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Visualization →", key="viz"):
        st.switch_page("pages/7_Visualization.py")

with col4:
    st.markdown(
        """
    <div class="feature-card">
        <h4>⚗️ Physics Losses</h4>
        <p>Physics-informed constraints</p>
        <ul style="font-size: 0.9rem;">
            <li>Mass conservation</li>
            <li>Energy spectra</li>
            <li>PV conservation</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Physics Losses →", key="physics"):
        st.switch_page("pages/8_Physics_Losses.py")

st.markdown("### 🧬 GAIA Platform")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown(
        """
    <div class="feature-card">
        <h4>🧬 GAIA Function Studio</h4>
        <p>Every GAIA function, visualized and demo-ready</p>
        <ul style="font-size: 0.9rem;">
            <li>Model components</li>
            <li>Data pipeline utilities</li>
            <li>Training & evaluation tools</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Explore GAIA →", key="gaia"):
        st.switch_page("pages/14_GAIA_Functions.py")

st.markdown("---")

# Statistics
st.markdown("### 📈 Platform Statistics")

stat_cols = st.columns(6)
stats = [
    ("15+", "Model Architectures"),
    ("5", "Cloud Providers"),
    ("78+", "ERA5 Variables"),
    ("10", "Preprocessing Pipelines"),
    ("50+", "Visualizations"),
    ("100+", "Python Modules"),
]

for col, (value, label) in zip(stat_cols, stats):
    with col:
        st.markdown(
            f"""
        <div class="stat-card">
            <h2 style="margin: 0; color: #4CAF50;">{value}</h2>
            <p style="margin: 0; color: #666;">{label}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Model Architecture Overview
st.markdown("### 🏗️ Available Model Architectures")

with st.expander("**Graph Neural Networks** - GraphCast style", expanded=False):
    st.markdown(
        """
    | Model | Organization | Key Features |
    |-------|--------------|--------------|
    | GraphCast | DeepMind | Multi-mesh GNN, 0.25° resolution, 10-day forecasts |
    | IcosahedralNet | WeatherFlow | Icosahedral graph structure, spherical geometry |

    **Implementation:** `weatherflow.model_library.architectures.graphcast`
    """
    )

with st.expander("**Vision Transformers** - FourCastNet style", expanded=False):
    st.markdown(
        """
    | Model | Organization | Key Features |
    |-------|--------------|--------------|
    | FourCastNet | NVIDIA | AFNO blocks, FFT-based attention, very fast |
    | SFNO | - | Spherical Fourier Neural Operator |

    **Implementation:** `weatherflow.model_library.architectures.fourcastnet`
    """
    )

with st.expander("**3D Transformers** - Pangu-Weather style", expanded=False):
    st.markdown(
        """
    | Model | Organization | Key Features |
    |-------|--------------|--------------|
    | Pangu-Weather | Huawei | 3D Earth-Specific Transformer, 1h/3h/6h/24h models |
    | SwinTransformer3D | WeatherFlow | 3D window attention for atmospheric data |

    **Implementation:** `weatherflow.model_library.architectures.pangu`
    """
    )

with st.expander("**Diffusion Models** - GenCast style", expanded=False):
    st.markdown(
        """
    | Model | Organization | Key Features |
    |-------|--------------|--------------|
    | GenCast | DeepMind | Conditional diffusion, ensemble generation |
    | WeatherDiffusion | WeatherFlow | DDPM/DDIM schedulers, probabilistic forecasts |

    **Implementation:** `weatherflow.model_library.architectures.gencast`
    """
    )

with st.expander("**Image Translation** - Hurricane wind fields", expanded=False):
    st.markdown(
        """
    | Model | Original Paper | Application |
    |-------|----------------|-------------|
    | Pix2Pix | Isola et al. 2017 | Satellite → wind field (paired) |
    | CycleGAN | Zhu et al. 2017 | Satellite → wind field (unpaired) |
    | HurricaneWindField | WeatherFlow | End-to-end hurricane analysis |

    **Implementation:** `weatherflow.model_library.architectures.image_translation`
    """
    )

with st.expander("**Foundation Models** - ClimaX style", expanded=False):
    st.markdown(
        """
    | Model | Organization | Key Features |
    |-------|--------------|--------------|
    | ClimaX | Microsoft | Variable tokenization, multi-task, fine-tunable |
    | Aurora | Microsoft | Large-scale foundation model for atmosphere |

    **Implementation:** `weatherflow.model_library.architectures.climax`
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>WeatherFlow - The Complete Weather AI Platform</h4>
    <p>
        Built for the weather AI community | All models based on original papers
        <br>
        GraphCast • FourCastNet • Pangu-Weather • GenCast • ClimaX • Pix2Pix • CycleGAN
    </p>
    <p style="font-size: 0.9rem;">
        🌐 Real ERA5 data from ECMWF | ☁️ Cloud training on AWS, GCP, Modal, RunPod
    </p>
</div>
""",
    unsafe_allow_html=True,
)

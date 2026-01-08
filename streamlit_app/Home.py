"""
WeatherFlow - Weather AI Learning & Demonstration Platform

An educational platform for exploring weather AI concepts.

IMPORTANT: This platform includes both:
- Real implementations (Flow Matching models, Physics Loss functions)
- UI demonstrations (Live Dashboard, Training Hub, Experiments)

Each page has a banner indicating whether it runs real code or displays simulated data.
See individual page docstrings for details.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

# Try to import utilities
try:
    from era5_utils import (
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        auto_load_default_sample,
        get_data_source_badge,
    )
    from data_storage import get_data_status, get_model_benchmarks
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

st.set_page_config(
    page_title="WeatherFlow - Weather AI Research Platform",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the unified platform look
st.markdown("""
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
    .tagline {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    .value-prop {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .workflow-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #1e88e5;
        margin: 15px 0;
        transition: transform 0.2s;
    }
    .workflow-card:hover {
        transform: translateX(5px);
    }
    .feature-section {
        background: #fafafa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .model-badge {
        display: inline-block;
        background: #1e88e5;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px;
    }
    .org-badge {
        display: inline-block;
        background: #7c4dff;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px;
    }
    .stat-box {
        background: linear-gradient(135deg, #4CAF5022, #8BC34A22);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .real-data-emphasis {
        background: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 20px 0;
    }
    .cta-button {
        background: linear-gradient(90deg, #1e88e5, #7c4dff);
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Auto-load data on startup
if UTILS_AVAILABLE:
    auto_load_default_sample()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<p class="main-header">🌤️ WeatherFlow</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Weather AI Learning & Demonstration Platform</p>', unsafe_allow_html=True)

# Transparency notice
with st.expander("ℹ️ **Important: About This Platform**", expanded=False):
    st.markdown("""
    ### What's Real vs. Simulated

    **Pages with REAL model execution:**
    - ✅ **Flow Matching** - Runs actual PyTorch forward/backward passes
    - ✅ **Physics Losses** - Uses real PhysicsLossCalculator computations
    - ✅ **Data Manager** - Downloads real ERA5 reanalysis data

    **Pages with UI DEMONSTRATIONS (simulated data):**
    - ⚠️ **Live Dashboard** - Synthetic weather patterns for UI demo
    - ⚠️ **Training Hub** - Simulated training, no cloud jobs launched
    - ⚠️ **Training Workflow** - UI demonstration with fake loss curves
    - ⚠️ **Experiments** - Randomly generated ablation results
    - ⚠️ **Model Comparison** - Hardcoded benchmark values from papers

    Each page displays a prominent banner indicating its status.
    """)

# Data status banner
if UTILS_AVAILABLE and has_era5_data():
    data, metadata = get_active_era5_data()
    is_synthetic = metadata.get("is_synthetic", True) if metadata else True
    name = metadata.get("name", "Unknown") if metadata else "Unknown"

    if is_synthetic:
        st.warning(f"**Demo Mode:** Using synthetic data ({name}). Download real ERA5 data from Data Manager for research.")
    else:
        st.success(f"**Real Data Active:** {name} (ERA5 Reanalysis)")
else:
    st.info("**Welcome!** Visit the Data Manager to load ERA5 data and unlock all features.")

# =============================================================================
# VALUE PROPOSITION
# =============================================================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="value-prop">
        <h2>🎯 ERA5 Data Access</h2>
        <p>Access ERA5 reanalysis data samples from ECMWF.
        Real atmospheric observations available through the Data Manager.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="value-prop">
        <h2>🧠 Model Implementations</h2>
        <p>Explore flow matching models and physics loss functions.
        Based on published research with proper citations.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="value-prop">
        <h2>📊 Learning Platform</h2>
        <p>Educational tools for understanding weather AI.
        UI demonstrations and real physics computations.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# QUICK START WORKFLOW
# =============================================================================
st.markdown("---")
st.markdown("## 🚀 Quick Start: Your Daily Workflow")

st.markdown("""
Whether you're checking today's forecasts, training a new model, or catching up on research,
WeatherFlow has you covered.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="workflow-card">
        <h3>📍 Step 1: Check the Dashboard</h3>
        <p>See current weather predictions from multiple AI models.
        Compare forecasts for the next 7 days. Verify against ground truth.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Live Dashboard →", key="btn_dashboard", type="primary"):
        st.switch_page("pages/01_Live_Dashboard.py")

    st.markdown("""
    <div class="workflow-card">
        <h3>📰 Step 2: Stay Updated</h3>
        <p>Latest papers from arXiv, Nature, Science.
        Model releases and community highlights.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Research Feed →", key="btn_research"):
        st.switch_page("pages/02_Research_Feed.py")

with col2:
    st.markdown("""
    <div class="workflow-card">
        <h3>🏋️ Step 3: Train Your Model</h3>
        <p>End-to-end workflow: select data, configure model, train, evaluate.
        All on real ERA5 data with proper benchmarking.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Training →", key="btn_training", type="primary"):
        st.switch_page("pages/03_Training_Workflow.py")

    st.markdown("""
    <div class="workflow-card">
        <h3>📊 Step 4: Compare & Analyze</h3>
        <p>See how your model stacks up against GraphCast, FourCastNet, and others.
        WeatherBench2 metrics, regional analysis, error maps.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Compare Models →", key="btn_compare"):
        st.switch_page("pages/12_Model_Comparison.py")

# =============================================================================
# DATA SOURCES AND TRANSPARENCY
# =============================================================================
st.markdown("""
<div class="real-data-emphasis">
    <h3>📋 Data Sources & Transparency Notice</h3>
    <p><strong>WeatherFlow uses the following data sources:</strong></p>
    <ul>
        <li><strong>ERA5 Reanalysis</strong> - ECMWF's gold-standard atmospheric dataset (Hersbach et al., 2020)</li>
        <li><strong>WeatherBench2</strong> - Google Research's standardized benchmark (Rasp et al., 2023)</li>
        <li><strong>Published Metrics</strong> - Model comparison data extracted from original papers</li>
    </ul>
    <p><strong>⚠️ Important:</strong> Many pages show <em>UI demonstrations</em> with simulated data.
    Look for warning banners on each page indicating whether data is real or simulated.
    Pages running actual model code include: Flow Matching, Physics Losses.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL SHOWCASE
# =============================================================================
st.markdown("---")
st.markdown("## 🏗️ Supported Model Architectures")

st.markdown("""
<div style="text-align: center; margin: 20px 0;">
    <span class="model-badge">GraphCast</span>
    <span class="model-badge">FourCastNet</span>
    <span class="model-badge">Pangu-Weather</span>
    <span class="model-badge">GenCast</span>
    <span class="model-badge">ClimaX</span>
    <span class="model-badge">Aurora</span>
    <span class="model-badge">NeuralGCM</span>
    <span class="model-badge">Flow Matching</span>
</div>
<div style="text-align: center; margin-bottom: 20px;">
    <span class="org-badge">🌐 DeepMind</span>
    <span class="org-badge">⚡ NVIDIA</span>
    <span class="org-badge">🌍 Huawei</span>
    <span class="org-badge">🏛️ Microsoft</span>
    <span class="org-badge">🔬 Google</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FEATURE SECTIONS
# =============================================================================
st.markdown("---")
st.markdown("## 📚 Platform Features")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Research Tools",
    "⚡ Applications",
    "📖 Education",
    "🛠️ Advanced"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Core Research Features

        **📊 Data Manager**
        Load and manage ERA5 reanalysis data. Pre-bundled sample datasets
        for key weather events (Hurricane Katrina, European Heat Wave, etc.).

        **🏋️ Training Hub**
        Configure and train models on real data. Multiple architectures,
        customizable hyperparameters, real-time loss visualization.

        **📈 Model Comparison**
        Compare against WeatherBench2 benchmarks. Skill scores by variable,
        lead time degradation, regional analysis.
        """)

    with col2:
        st.markdown("""
        ### Analysis & Visualization

        **🗺️ Publication Visualizations**
        Create publication-quality figures matching top journal standards.
        Maps, scorecards, spectral analysis, ensemble spreads.

        **🎬 Animation Export**
        Generate GIF animations of weather evolution.
        Easy download for presentations and papers.

        **🔬 Research Workbench**
        Rapid prototyping environment for custom architectures.
        Mix and match components, quick training runs.
        """)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Renewable Energy

        **🌬️ Wind Power Calculator**
        Convert wind forecasts to power output using real turbine models
        (IEA-3.4MW, NREL-5MW, Vestas-V90).

        **☀️ Solar Power Calculator**
        PV system output estimation with pvlib integration.
        Multiple panel types and tracking options.
        """)

    with col2:
        st.markdown("""
        ### Extreme Events

        **🌡️ Extreme Event Detection**
        Physics-based algorithms for heatwaves, atmospheric rivers,
        extreme precipitation. Case studies on real ERA5 data.

        **🌍 GCM Simulation**
        Full-physics General Circulation Model. Standalone climate
        simulation for understanding atmospheric dynamics.
        """)

with tab3:
    st.markdown("""
    ### Graduate-Level Atmospheric Dynamics

    Interactive learning tools based on classic texts (Holton, Vallis):

    - **Geostrophic Balance** - The fundamental balance in large-scale flow
    - **Rossby Waves** - Planetary waves and their dispersion
    - **Potential Vorticity** - The master variable of atmospheric dynamics
    - **Practice Problems** - Worked examples with solutions

    **Perfect for:** Graduate courses in atmospheric science, self-study for ML researchers
    entering weather prediction.
    """)

with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### GAIA Architecture Studio

        Explore every component of the GAIA weather model:
        - Grid encoders (icosahedral, lat-lon)
        - Processor blocks (GNN, attention, spectral)
        - Decoder heads (deterministic, ensemble, diffusion)
        - Training utilities (curriculum learning, physics losses)
        """)

    with col2:
        st.markdown("""
        ### Physics-Informed Losses

        Constrain your ML models with atmospheric physics:
        - Mass conservation (divergence)
        - PV conservation
        - Energy spectra (power laws)
        - Geostrophic balance

        All with ERA5 data validation.
        """)

# =============================================================================
# STATISTICS
# =============================================================================
st.markdown("---")
st.markdown("## 📊 Platform Statistics")

stat_cols = st.columns(6)
stats = [
    ("8+", "Model Architectures"),
    ("6", "Sample Datasets"),
    ("78+", "ERA5 Variables"),
    ("7", "Pressure Levels"),
    ("50+", "Visualizations"),
    ("1", "Platform"),
]

for col, (value, label) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <h2 style="margin: 0; color: #4CAF50;">{value}</h2>
            <p style="margin: 0; color: #666; font-size: 0.9em;">{label}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns([1, 2, 1])

with col_footer2:
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>WeatherFlow</h3>
        <p><em>The platform weather AI researchers want to use every day.</em></p>
        <p style="font-size: 0.9rem;">
            Real data from ECMWF ERA5 | Models from DeepMind, NVIDIA, Huawei, Microsoft
            <br>
            Benchmarks from WeatherBench2 | All citations included
        </p>
        <p style="font-size: 0.8rem; color: #999;">
            Built for the weather AI community
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.page_link("pages/01_Live_Dashboard.py", label="📍 Live Dashboard")
st.sidebar.page_link("pages/02_Research_Feed.py", label="📰 Research Feed")
st.sidebar.page_link("pages/03_Training_Workflow.py", label="🏋️ Training Workflow")
st.sidebar.page_link("pages/0_Data_Manager.py", label="📊 Data Manager")

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Status")
if UTILS_AVAILABLE:
    status = get_data_status() if 'get_data_status' in dir() else {}
    st.sidebar.metric("Samples Available", f"{status.get('available_samples', 0)}/{status.get('total_samples', 6)}")
    st.sidebar.metric("Storage Used", f"{status.get('storage_used_mb', 0):.1f} MB")

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

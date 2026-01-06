"""
WeatherFlow - Comprehensive Weather Prediction & Analysis Platform

This Streamlit app provides interactive access to ALL Python functionality
in the WeatherFlow repository, including:
- Flow Matching Weather Models
- Renewable Energy Forecasting (Wind & Solar)
- Extreme Event Detection
- General Circulation Model (GCM)
- Graduate-Level Atmospheric Dynamics Education
- Physics-Informed Machine Learning
- Model Training & Evaluation
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="WeatherFlow",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e88e5, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">WeatherFlow</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-Informed Flow Matching for Weather Prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# ERA5 Data Status Banner
sys.path.insert(0, str(Path(__file__).parent))
try:
    from era5_utils import get_era5_data_banner, has_era5_data
    banner = get_era5_data_banner()
    if has_era5_data():
        st.success(f"🌍 {banner}")
    else:
        st.info(f"""
        🌍 **Start Here:** Go to **📊 Data Manager** to download real ERA5 data.
        
        Choose from pre-defined weather events (hurricanes, heat waves, etc.) or download custom data.
        All app features can use actual ECMWF atmospheric observations.
        """)
except ImportError:
    pass

# Overview
st.markdown("""
### Welcome to WeatherFlow Interactive Platform

This web application provides **live, interactive access** to all the Python functionality
in the WeatherFlow repository. Every feature runs actual Python code - no simulations or mockups.

**All functionality uses REAL ERA5 data** from ECMWF when available, or clearly indicates demo mode.
""")

# Feature Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌬️ Renewable Energy")
    st.markdown("""
    - **Wind Power Calculator**: Convert wind forecasts to power output
    - **Solar Power Calculator**: PV system power estimation
    - Multiple turbine/panel configurations
    - Real physics-based models
    """)
    if st.button("Go to Wind Power →", key="wind"):
        st.switch_page("pages/1_Wind_Power.py")

with col2:
    st.markdown("### 🌡️ Extreme Events")
    st.markdown("""
    - **Heatwave Detection**: Temperature threshold analysis
    - **Atmospheric Rivers**: IVT-based detection
    - **Extreme Precipitation**: Percentile analysis
    - Event characterization & statistics
    """)
    if st.button("Go to Extreme Events →", key="extreme"):
        st.switch_page("pages/3_Extreme_Events.py")

with col3:
    st.markdown("### 🧠 ML Models")
    st.markdown("""
    - **Flow Matching**: Train weather prediction models
    - **Physics-Informed**: Conservation constraints
    - **ODE Solvers**: Continuous trajectory generation
    - Interactive training dashboard
    """)
    if st.button("Go to Flow Matching →", key="flow"):
        st.switch_page("pages/4_Flow_Matching.py")

st.markdown("---")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### 🌍 GCM Simulation")
    st.markdown("""
    - **Full GCM**: General Circulation Model
    - Radiation, convection, cloud physics
    - Configurable resolution & CO2
    - Real-time diagnostics
    """)
    if st.button("Go to GCM →", key="gcm"):
        st.switch_page("pages/5_GCM_Simulation.py")

with col5:
    st.markdown("### 📚 Education")
    st.markdown("""
    - **Graduate Dynamics**: Interactive learning
    - Rossby waves, geostrophic balance
    - Potential vorticity visualization
    - Step-by-step problem solving
    """)
    if st.button("Go to Education →", key="edu"):
        st.switch_page("pages/6_Education.py")

with col6:
    st.markdown("### 📊 Visualization")
    st.markdown("""
    - **Weather Maps**: Global projections
    - **Flow Fields**: Vector visualization
    - **Skew-T Diagrams**: Atmospheric profiles
    - Animation & comparison tools
    """)
    if st.button("Go to Visualization →", key="viz"):
        st.switch_page("pages/7_Visualization.py")

st.markdown("---")

# Quick Stats
st.markdown("### 📈 Platform Capabilities")

stat_cols = st.columns(6)
stats = [
    ("152", "Python Files"),
    ("8+", "ML Models"),
    ("3", "Turbine Types"),
    ("3", "Panel Types"),
    ("3", "Event Detectors"),
    ("6", "GCM Physics")
]

for col, (value, label) in zip(stat_cols, stats):
    with col:
        st.metric(label=label, value=value)

# Module Overview
st.markdown("---")
st.markdown("### 🗂️ Complete Module Index")

with st.expander("Core WeatherFlow Library", expanded=False):
    st.markdown("""
    | Module | Description | Status |
    |--------|-------------|--------|
    | `weatherflow.models` | Flow matching, physics-guided attention | ✅ Active |
    | `weatherflow.training` | FlowTrainer, metrics, utilities | ✅ Active |
    | `weatherflow.physics` | Physics losses, atmospheric constraints | ✅ Active |
    | `weatherflow.data` | ERA5 dataset loading, streaming | ✅ Active |
    | `weatherflow.solvers` | ODE, Langevin, Riemannian solvers | ✅ Active |
    | `weatherflow.manifolds` | Spherical geometry | ✅ Active |
    | `weatherflow.education` | Graduate atmospheric dynamics | ✅ Active |
    | `weatherflow.utils` | Visualization, evaluation | ✅ Active |
    """)

with st.expander("Applications", expanded=False):
    st.markdown("""
    | Application | Description | Status |
    |-------------|-------------|--------|
    | `applications.renewable_energy.wind_power` | Wind turbine power curves | ✅ Active |
    | `applications.renewable_energy.solar_power` | PV system modeling | ✅ Active |
    | `applications.extreme_event_analysis` | Heatwave, AR, precipitation | ✅ Active |
    """)

with st.expander("General Circulation Model (GCM)", expanded=False):
    st.markdown("""
    | Component | Description | Status |
    |-----------|-------------|--------|
    | `gcm.core.model` | Main GCM class | ✅ Active |
    | `gcm.core.dynamics` | Atmospheric dynamics | ✅ Active |
    | `gcm.physics.radiation` | Radiation scheme | ✅ Active |
    | `gcm.physics.convection` | Convection parameterization | ✅ Active |
    | `gcm.physics.cloud_microphysics` | Cloud processes | ✅ Active |
    | `gcm.physics.boundary_layer` | PBL scheme | ✅ Active |
    | `gcm.physics.ocean` | Ocean mixed layer | ✅ Active |
    | `gcm.grid` | Spherical & vertical grids | ✅ Active |
    """)

with st.expander("Foundation Model & Experiments", expanded=False):
    st.markdown("""
    | Module | Description | Status |
    |--------|-------------|--------|
    | `foundation_model.models` | FlowAtmosphere, FlowFormer | ✅ Active |
    | `foundation_model.training` | Distributed trainer | ✅ Active |
    | `foundation_model.adaptation` | PEFT fine-tuning | ✅ Active |
    | `experiments.ablation_study` | Architecture ablations | ✅ Active |
    | `experiments.weatherbench2_evaluation` | Benchmark evaluation | ✅ Active |
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>WeatherFlow - Built with Streamlit | All Python code runs live</p>
    <p>Navigate using the sidebar to explore all features</p>
</div>
""", unsafe_allow_html=True)

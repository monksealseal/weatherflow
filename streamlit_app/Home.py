"""
WeatherFlow - Weather AI Platform

A unified platform for weather AI that makes it clear what users should do:
1. Load Data (quick demo or real ERA5)
2. Train Models (with cost estimates)
3. Make Predictions (use trained models)
4. Analyze Results (visualizations and evaluations)

This home page provides a clear journey through these steps.
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
        has_era5_data,
        get_active_era5_data,
        auto_load_default_sample,
    )
    from data_storage import get_data_status
    from checkpoint_utils import has_trained_model, list_checkpoints
    from dataset_context import render_dataset_banner, get_dataset_summary, render_workflow_progress
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

st.set_page_config(
    page_title="WeatherFlow - Weather AI Platform",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, clean CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e88e5, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .journey-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        border-left: 5px solid #1e88e5;
        transition: all 0.3s ease;
    }
    .journey-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .journey-card.completed {
        border-left-color: #4CAF50;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    .journey-card.active {
        border-left-color: #ff9800;
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    }
    .step-number {
        display: inline-block;
        width: 36px;
        height: 36px;
        background: #1e88e5;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 36px;
        font-weight: bold;
        margin-right: 12px;
    }
    .step-number.completed {
        background: #4CAF50;
    }
    .step-number.active {
        background: #ff9800;
    }
    .quick-action {
        background: linear-gradient(135deg, #1e88e5, #1565c0);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 10px 0;
    }
    .quick-action:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.4);
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .status-ready {
        background: #4CAF50;
        color: white;
    }
    .status-pending {
        background: #ff9800;
        color: white;
    }
    .status-none {
        background: #9e9e9e;
        color: white;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .feature-box {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .highlight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Auto-load data on startup if available
if UTILS_AVAILABLE:
    auto_load_default_sample()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-title">🌤️ WeatherFlow</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Weather AI Made Simple: Load Data → Train Models → Predict Weather</p>', unsafe_allow_html=True)

# =============================================================================
# STATUS CHECK
# =============================================================================
# Determine current status
has_data = False
has_model = False
data_name = "None"

if UTILS_AVAILABLE:
    has_data = has_era5_data()
    has_model = has_trained_model()
    if has_data:
        _, meta = get_active_era5_data()
        data_name = meta.get("name", "Unknown") if meta else "Unknown"

# Show current status prominently
st.markdown("### Your Current Status")

status_cols = st.columns(3)

with status_cols[0]:
    if has_data:
        st.markdown(f'<span class="status-pill status-ready">✅ Data Loaded</span> {data_name}', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-none">📊 No Data</span> Load data to start', unsafe_allow_html=True)

with status_cols[1]:
    if has_model:
        checkpoints = list_checkpoints() if UTILS_AVAILABLE else []
        st.markdown(f'<span class="status-pill status-ready">✅ Model Trained</span> {len(checkpoints)} checkpoint(s)', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-none">🧠 No Model</span> Train a model', unsafe_allow_html=True)

with status_cols[2]:
    if has_model:
        st.markdown('<span class="status-pill status-ready">✅ Ready to Predict</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-pending">🔮 Needs Model</span>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# MAIN USER JOURNEY
# =============================================================================
st.markdown("## 🚀 Your Journey")

# Determine current step
if not has_data:
    current_step = 1
elif not has_model:
    current_step = 2
else:
    current_step = 3

# Step 1: Load Data
step1_class = "completed" if has_data else "active" if current_step == 1 else ""
step1_num_class = "completed" if has_data else "active" if current_step == 1 else ""

st.markdown(f"""
<div class="journey-card {step1_class}">
    <span class="step-number {step1_num_class}">1</span>
    <strong style="font-size: 1.2em;">Load Your Data</strong>
    {"<span class='status-pill status-ready' style='margin-left: 10px;'>✅ Complete</span>" if has_data else ""}
    <p style="margin-top: 10px; color: #666;">
        Start with instant demo data or download real ERA5 weather observations.
        Upload your own images for GAN-style training.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Quick Demo (Instant)", type="primary" if current_step == 1 else "secondary", use_container_width=True):
        st.switch_page("pages/0_Data_Manager.py")
with col2:
    if st.button("🌍 Real ERA5 Data", use_container_width=True):
        st.switch_page("pages/0_Data_Manager.py")
with col3:
    if st.button("🖼️ Upload Images", use_container_width=True):
        st.switch_page("pages/0_Data_Manager.py")

st.markdown("")

# Step 2: Train Model
step2_class = "completed" if has_model else "active" if current_step == 2 else ""
step2_num_class = "completed" if has_model else "active" if current_step == 2 else ""

st.markdown(f"""
<div class="journey-card {step2_class}">
    <span class="step-number {step2_num_class}">2</span>
    <strong style="font-size: 1.2em;">Train Your Model</strong>
    {"<span class='status-pill status-ready' style='margin-left: 10px;'>✅ Complete</span>" if has_model else ""}
    <p style="margin-top: 10px; color: #666;">
        Train weather AI models on your data. Choose from multiple architectures.
        See estimated training time and cost before you start.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏃 Quick Training (Demo)", type="primary" if current_step == 2 else "secondary", use_container_width=True, disabled=not has_data):
        st.switch_page("pages/03_Training_Workflow.py")
with col2:
    if st.button("⚙️ Full Training Config", use_container_width=True, disabled=not has_data):
        st.switch_page("pages/03_Training_Workflow.py")
with col3:
    if st.button("💰 Estimate Cloud Cost", use_container_width=True):
        st.switch_page("pages/03_Training_Workflow.py")

st.markdown("")

# Step 3: Make Predictions
step3_class = "active" if current_step == 3 else ""
step3_num_class = "active" if current_step == 3 else ""

st.markdown(f"""
<div class="journey-card {step3_class}">
    <span class="step-number {step3_num_class}">3</span>
    <strong style="font-size: 1.2em;">Make Predictions</strong>
    <p style="margin-top: 10px; color: #666;">
        Use your trained model to forecast weather! See 7-day forecasts,
        compare with observations, and export your predictions.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔮 Generate Forecast", type="primary" if current_step == 3 else "secondary", use_container_width=True, disabled=not has_model):
        st.switch_page("pages/17_Weather_Prediction.py")
with col2:
    if st.button("📊 View Dashboard", use_container_width=True):
        st.switch_page("pages/01_Live_Dashboard.py")
with col3:
    if st.button("📈 Compare Models", use_container_width=True):
        st.switch_page("pages/12_Model_Comparison.py")

# =============================================================================
# QUICK START RECOMMENDATION
# =============================================================================
st.markdown("---")

st.markdown("### 💡 Recommended Next Step")

if current_step == 1:
    st.markdown("""
    <div class="highlight-box">
        <h3>👉 Load Demo Data Now!</h3>
        <p>Get started instantly with synthetic weather data. Perfect for learning the platform.</p>
        <p><strong>Time required:</strong> Instant (< 1 second)</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Load Demo Data & Get Started", type="primary"):
        st.switch_page("pages/0_Data_Manager.py")

elif current_step == 2:
    st.markdown("""
    <div class="highlight-box">
        <h3>👉 Train Your First Model!</h3>
        <p>You have data loaded. Now train a weather AI model on it.</p>
        <p><strong>Quick demo training:</strong> ~2-5 minutes on CPU</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🏃 Start Quick Training", type="primary"):
        st.switch_page("pages/03_Training_Workflow.py")

else:
    st.markdown("""
    <div class="highlight-box">
        <h3>👉 Make Your First Prediction!</h3>
        <p>You have a trained model. Use it to forecast weather!</p>
        <p><strong>Try it:</strong> Generate a 7-day forecast right now</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔮 Generate Weather Forecast", type="primary"):
        st.switch_page("pages/17_Weather_Prediction.py")

# =============================================================================
# FEATURES OVERVIEW
# =============================================================================
st.markdown("---")
st.markdown("## 📚 Platform Features")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Data Sources",
    "🧠 Models",
    "📊 Visualizations",
    "☁️ Cloud Training"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Built-in Data Sources
        - **Quick Demo Data** - Instant synthetic weather data
        - **ERA5 Reanalysis** - Real ECMWF observations via WeatherBench2
        - **Sample Datasets** - Hurricane Katrina, European Heat Wave, etc.

        ### Custom Data
        - **Image Upload** - Train on your own source/target image pairs
        - **GAN Training** - Perfect for satellite-to-radar translation
        """)
    with col2:
        st.markdown("""
        ### Data Sources Info
        - **Dynamical.org** - Modern weather data infrastructure
        - **WeatherBench2** - Google Research benchmark
        - **ERA5 via Icechunk** - Advanced cloud-native access

        [Learn more about data sources →](https://dynamical.org/updates/)
        """)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Supported Architectures
        - **Flow Matching** - State-of-the-art generative model
        - **UNet** - Classic encoder-decoder
        - **FourCastNet-style** - Fourier Neural Operators
        - **GraphCast-style** - Graph Neural Networks
        - **Vision Transformers** - Attention-based

        All architectures work with both weather data AND image pairs!
        """)
    with col2:
        st.markdown("""
        ### Model Features
        - **Physics-Informed Losses** - Enforce atmospheric constraints
        - **Multi-scale Training** - Start coarse, refine
        - **Checkpointing** - Save and resume training
        - **Export to ONNX** - Deploy anywhere
        """)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Weather Visualizations
        - **7-Day Forecast Display** - Professional weather.com style
        - **Global Maps** - Temperature, wind, precipitation
        - **Animations** - Watch weather evolve

        ### Verification
        - **Model vs Observations** - Compare predictions to truth
        - **Error Metrics** - RMSE, MAE, Correlation
        - **Skill Scores** - ACC by lead time
        """)
    with col2:
        st.markdown("""
        ### Publication Quality
        - **Export to PNG/PDF** - High resolution
        - **GIF Animations** - For presentations
        - **Configurable Colormaps** - Match journal standards

        ### Benchmarking
        - **WeatherBench2 Metrics** - Compare to published models
        - **Leaderboard** - See where you stand
        """)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Cost Estimation
        Know exactly what training will cost BEFORE you start:
        - **GPU Selection** - T4, A100, H100
        - **Time Estimates** - Hours per epoch
        - **Total Cost** - No surprises

        ### Example Costs (GCP)
        | Job Size | Time | Cost |
        |----------|------|------|
        | Demo | 30 min | ~$0.20 |
        | Medium | 4 hours | ~$15 |
        | Large | 48 hours | ~$400 |
        """)
    with col2:
        st.markdown("""
        ### Multi-Node Training
        - **Auto-configuration** - Optimal GPU count
        - **Distributed Training** - Scale to 8+ GPUs
        - **Cost Optimization** - Best price/performance

        ### Coming Soon
        - **One-click GCP Deploy**
        - **Job Monitoring Dashboard**
        - **Auto-shutdown on completion**
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("""
    **Data Sources**
    - ERA5 (ECMWF)
    - WeatherBench2 (Google)
    - Dynamical.org
    """)

with col_footer2:
    st.markdown("""
    **Models**
    - GraphCast-style
    - FourCastNet-style
    - Flow Matching
    """)

with col_footer3:
    st.markdown("""
    **Support**
    - [Documentation](https://github.com/weatherflow)
    - [GitHub Issues](https://github.com/weatherflow/issues)
    """)

st.markdown("---")
st.caption(f"WeatherFlow v1.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🌤️ WeatherFlow")
st.sidebar.markdown("---")

# Show dataset status in sidebar
st.sidebar.markdown("### Current Status")

if has_data:
    st.sidebar.success(f"📊 **Data:** {data_name}")
else:
    st.sidebar.warning("📊 **Data:** Not loaded")

if has_model:
    st.sidebar.success("🧠 **Model:** Ready")
else:
    st.sidebar.warning("🧠 **Model:** Not trained")

st.sidebar.markdown("---")

st.sidebar.markdown("### Quick Navigation")
st.sidebar.page_link("pages/0_Data_Manager.py", label="📊 Load Data")
st.sidebar.page_link("pages/03_Training_Workflow.py", label="🏃 Train Model")
st.sidebar.page_link("pages/17_Weather_Prediction.py", label="🔮 Make Predictions")
st.sidebar.page_link("pages/01_Live_Dashboard.py", label="📺 Dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("### All Pages")
st.sidebar.page_link("pages/12_Model_Comparison.py", label="📈 Model Comparison")
st.sidebar.page_link("pages/04_Visualization_Studio.py", label="🎨 Visualizations")
st.sidebar.page_link("pages/6_Education.py", label="📚 Education")

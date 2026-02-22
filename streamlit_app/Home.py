"""
WeatherFlow - Weather AI Platform

Redesigned for instant impact and the weather AI community.
Philosophy: Show value in 10 seconds, not 10 minutes.
"""

import streamlit as st
import sys
from pathlib import Path

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
    from data_storage import (
        get_data_status,
        initialize_sample_data,
        load_sample_data,
        SAMPLE_DATASETS,
        MODEL_BENCHMARKS,
    )
    from checkpoint_utils import has_trained_model, list_checkpoints

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    SAMPLE_DATASETS = {}
    MODEL_BENCHMARKS = {}

# Plotly for hero visualization
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="WeatherFlow - AI Weather Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start clean
)

# Professional, minimal CSS
st.markdown(
    """
<style>
    /* Hide default streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0066cc 0%, #00a3cc 50%, #00cc99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.4rem;
        color: #555;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Credibility badge */
    .credibility-bar {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 20px 0 35px 0;
        flex-wrap: wrap;
    }
    .cred-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #666;
        font-size: 0.95rem;
    }
    .cred-icon {
        font-size: 1.2rem;
    }

    /* Live data badge */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Quick action cards */
    .action-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
    }
    .action-card:hover {
        border-color: #0066cc;
        box-shadow: 0 8px 25px rgba(0, 102, 204, 0.12);
        transform: translateY(-2px);
    }
    .action-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    .action-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
    }
    .action-desc {
        font-size: 0.95rem;
        color: #6b7280;
        line-height: 1.5;
    }

    /* Benchmark comparison */
    .benchmark-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 15px;
    }
    .benchmark-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .benchmark-model {
        font-weight: 500;
        color: #1f2937;
    }
    .benchmark-org {
        font-size: 0.85rem;
        color: #9ca3af;
    }
    .benchmark-metric {
        font-weight: 600;
        color: #0066cc;
    }

    /* Citation box */
    .citation-box {
        background: #f8fafc;
        border-left: 4px solid #0066cc;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0;
        font-size: 0.9rem;
        color: #475569;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 50px;
        margin: 30px 0;
        flex-wrap: wrap;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Section cards */
    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.04);
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }
    .section-desc {
        color: #6b7280;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #0f172a;
    }

    .steps-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 10px;
    }

    .step-card {
        background: #f8fafc;
        border: 1px dashed #cbd5f5;
        border-radius: 14px;
        padding: 16px;
    }
    .step-card h4 {
        margin: 0 0 6px 0;
        color: #1e293b;
        font-size: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# AUTO-LOAD DATA ON FIRST VISIT - THE KEY TO INSTANT VALUE
# (Improvement 7: wrap expensive load in st.cache_data)
# =============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load_sample(name: str):
    """Cache the downloaded NCEP sample so restarts are instant."""
    if not DATA_STORAGE_AVAILABLE:
        return None, None
    initialize_sample_data(name)
    return load_sample_data(name)


DATA_STORAGE_AVAILABLE = UTILS_AVAILABLE  # alias for cache helper

if UTILS_AVAILABLE and "first_visit_data_loaded" not in st.session_state:
    try:
        auto_load_default_sample()
        if not has_era5_data():
            data, meta = _cached_load_sample("ncep_reanalysis_2013")
            if data is not None:
                st.session_state["era5_data"] = data
                st.session_state["era5_metadata"] = meta
                st.session_state["active_sample"] = "ncep_reanalysis_2013"
        st.session_state["first_visit_data_loaded"] = True
    except Exception:
        st.session_state["first_visit_data_loaded"] = True

# =============================================================================
# HERO SECTION
# =============================================================================
st.markdown('<h1 class="hero-title">WeatherFlow</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Train state-of-the-art weather AI models on real NCEP/NCAR reanalysis data</p>',
    unsafe_allow_html=True,
)

# Credibility bar
st.markdown(
    """
<div class="credibility-bar">
    <div class="cred-item">
        <span class="cred-icon">🔬</span>
        <span>Real NCEP/NCAR Data</span>
    </div>
    <div class="cred-item">
        <span class="cred-icon">📊</span>
        <span>WeatherBench2 Compatible</span>
    </div>
    <div class="cred-item">
        <span class="cred-icon">🧠</span>
        <span>Flow Matching & GraphCast Architectures</span>
    </div>
    <div class="cred-item">
        <span class="cred-icon">📄</span>
        <span>Peer-Reviewed Citations</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HERO VISUALIZATION - THE WOW MOMENT
# =============================================================================
has_data = has_era5_data() if UTILS_AVAILABLE else False

if has_data and PLOTLY_AVAILABLE:
    try:
        data, meta = get_active_era5_data()
        if data is not None:
            # Get the first variable and create a beautiful visualization
            var_name = list(data.data_vars)[0]
            var_data = data[var_name]

            # Handle dimensions
            if "time" in var_data.dims:
                var_data = var_data.isel(time=-1)  # Latest timestep
            if "level" in var_data.dims:
                var_data = var_data.isel(level=0)

            # Get coordinates
            if "latitude" in data.coords:
                lats, lons = data.latitude.values, data.longitude.values
            else:
                lats, lons = data.lat.values, data.lon.values

            # Create the hero visualization
            col_viz, col_info = st.columns([2, 1])

            with col_viz:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=var_data.values,
                        x=lons,
                        y=lats,
                        colorscale="RdBu_r",
                        colorbar=dict(
                            title=dict(
                                text=(
                                    f"{var_name} (K)"
                                    if "temp" in var_name.lower() or var_name == "air"
                                    else var_name
                                ),
                                side="right",
                            ),
                            thickness=15,
                            len=0.7,
                        ),
                        hoverongaps=False,
                    )
                )

                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="system-ui, -apple-system, sans-serif"),
                )

                st.plotly_chart(fig, use_container_width=True)

            with col_info:
                # Data info card
                st.markdown(
                    f"""
                <div class="live-badge">
                    <div class="live-dot"></div>
                    LIVE REAL DATA
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(f"### {meta.get('name', 'Weather Data')}")

                # Key stats
                n_times = len(data.time) if "time" in data.coords else 1
                n_vars = len(list(data.data_vars))

                st.markdown(
                    f"""
                **Source:** NCEP/NCAR Reanalysis
                **Time Steps:** {n_times:,}
                **Variables:** {n_vars}
                **Coverage:** {meta.get('region', 'Global')}
                """
                )

                # Citation
                citation = meta.get(
                    "citation",
                    SAMPLE_DATASETS.get("ncep_reanalysis_2013", {}).get("citation", ""),
                )
                if citation:
                    st.markdown(
                        f"""
                    <div class="citation-box">
                        <strong>Citation:</strong><br>
                        {citation}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.warning(f"Visualization could not be rendered: {e}")
        st.info("Navigate to the Data Manager to reload your dataset.")
else:
    # No data yet - show a compelling call to action
    st.markdown(
        """
    <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 16px; margin: 20px 0;">
        <div style="font-size: 4rem; margin-bottom: 20px;">🌍</div>
        <h2 style="color: #0369a1; margin-bottom: 12px;">Ready to Explore Real Weather Data?</h2>
        <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto 25px auto;">
            Download NCEP/NCAR reanalysis data (7 MB) and start training AI weather models in minutes.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Download Real Data & Get Started",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Downloading NCEP/NCAR Reanalysis data from NOAA..."):
                if UTILS_AVAILABLE and initialize_sample_data("ncep_reanalysis_2013"):
                    data, meta = load_sample_data("ncep_reanalysis_2013")
                    if data is not None:
                        st.session_state["era5_data"] = data
                        st.session_state["era5_metadata"] = meta
                        st.session_state["active_sample"] = "ncep_reanalysis_2013"
                        st.success("Real weather data loaded!")
                        st.balloons()
                        st.rerun()
                else:
                    st.error("Download failed. Please try again.")

st.markdown("---")

# =============================================================================
# OPERATIONAL SNAPSHOT
# =============================================================================
st.markdown("## Operational Snapshot")

has_model = has_trained_model() if UTILS_AVAILABLE else False
data_status = get_data_status() if UTILS_AVAILABLE else {}
available_samples = data_status.get("available_samples", 0)
total_samples = data_status.get("total_samples", 0)

snapshot_col1, snapshot_col2, snapshot_col3 = st.columns(3)

with snapshot_col1:
    st.markdown(
        f"""
    <div class="section-card">
        <div class="section-title">Data Ready</div>
        <p class="section-desc">
            {"Live NCEP/NCAR reanalysis loaded and ready to explore." if has_data else "No dataset loaded yet. Start with the 7 MB NCEP sample."}
        </p>
        <div class="status-pill">📦 {available_samples}/{total_samples} datasets cached</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("📊 Open Data Manager", use_container_width=True):
        st.switch_page("pages/0_Data_Manager.py")

with snapshot_col2:
    st.markdown(
        f"""
    <div class="section-card">
        <div class="section-title">Model Workspace</div>
        <p class="section-desc">
            {"A trained checkpoint is available for forecasting." if has_model else "Train a Flow Matching or GraphCast-style model in minutes."}
        </p>
        <div class="status-pill">🧠 {"Checkpoint ready" if has_model else "No checkpoint yet"}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button(
        "🧠 Launch Training",
        type="primary" if has_data else "secondary",
        use_container_width=True,
        disabled=not has_data,
    ):
        st.switch_page("pages/03_Training_Workflow.py")

with snapshot_col3:
    st.markdown(
        f"""
    <div class="section-card">
        <div class="section-title">Forecasting Suite</div>
        <p class="section-desc">
            Compare WeatherFlow outputs against GraphCast, Pangu-Weather, and FourCastNet baselines.
        </p>
        <div class="status-pill">🔮 {"Ready to predict" if has_model else "Train first to unlock"}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button(
        "🔮 Run Predictions",
        type="primary" if has_model else "secondary",
        use_container_width=True,
        disabled=not has_model,
    ):
        st.switch_page("pages/17_Weather_Prediction.py")

st.markdown("---")

# =============================================================================
# GUIDED WORKFLOW
# =============================================================================
st.markdown("## Guided Workflow")
st.markdown("Follow the streamlined path below to go from data to publishable results.")

st.markdown(
    f"""
<div class="steps-grid">
    <div class="step-card">
        <h4>1. Load real data</h4>
        <p class="section-desc">Pull NCEP/NCAR or ERA-Interim samples with one click.</p>
        <div class="status-pill">{"✅ Loaded" if has_data else "⏳ Pending"}</div>
    </div>
    <div class="step-card">
        <h4>2. Train a model</h4>
        <p class="section-desc">Choose Flow Matching, UNet, or GraphCast-style training.</p>
        <div class="status-pill">{"✅ Trained" if has_model else "⏳ Pending"}</div>
    </div>
    <div class="step-card">
        <h4>3. Benchmark & export</h4>
        <p class="section-desc">Compare skill scores and export figures for reports.</p>
        <div class="status-pill">{"✅ Ready" if has_model else "🔒 Unlock after training"}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# BENCHMARKS - CREDIBILITY FOR THE WEATHER AI COMMUNITY
# =============================================================================
st.markdown("## WeatherBench2 Leaderboard")
st.markdown(
    "*Train your models and compare against published state-of-the-art results*"
)

benchmark_cols = st.columns([1, 1])
benchmark_data = MODEL_BENCHMARKS if UTILS_AVAILABLE else {}

with benchmark_cols[0]:
    st.markdown("#### Z500 RMSE (24h forecast)")

    if benchmark_data:
        sorted_models = sorted(
            benchmark_data.items(),
            key=lambda item: item[1]["metrics"].get("z500_rmse_24h", 999),
        )
        for model, details in sorted_models:
            rmse = details["metrics"].get("z500_rmse_24h", "—")
            organization = details.get("organization", "Unknown")
            col_name, col_metric = st.columns([3, 1])
            with col_name:
                st.markdown(
                    f"**{model}** <span style='color: #9ca3af; font-size: 0.85rem;'>({organization})</span>",
                    unsafe_allow_html=True,
                )
            with col_metric:
                st.markdown(
                    f"<span style='color: #0066cc; font-weight: 600;'>{rmse}m</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "Benchmark metadata is unavailable. Load data utilities to see rankings."
        )

with benchmark_cols[1]:
    st.markdown("#### Key Features of Top Models")
    if benchmark_data:
        table_rows = []
        for model, details in benchmark_data.items():
            table_rows.append(
                f"| **{model}** | {details.get('resolution', '—')} | {details.get('params_m', '—')}M |"
            )
        st.markdown(
            "\n".join(
                [
                    "| Model | Resolution | Parameters |",
                    "|-------|------------|------------|",
                    *table_rows,
                    "",
                    "*WeatherFlow supports Flow Matching, UNet, and custom architectures*",
                ]
            )
        )
    else:
        st.markdown("Add benchmark metadata to display model feature comparisons.")

st.markdown("---")

# =============================================================================
# QUICK ACCESS TO ADVANCED FEATURES
# =============================================================================
st.markdown("## Advanced Capabilities")

adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)

with adv_col1:
    st.markdown("#### 🌊 Flow Matching")
    st.markdown("State-of-the-art generative models for probabilistic forecasts")
    if st.button("Explore Flow Matching", use_container_width=True):
        st.switch_page("pages/4_Flow_Matching.py")

with adv_col2:
    st.markdown("#### ⚡ Physics Constraints")
    st.markdown("Enforce conservation laws in your neural networks")
    if st.button("Physics Losses", use_container_width=True):
        st.switch_page("pages/8_Physics_Losses.py")

with adv_col3:
    st.markdown("#### 📈 Model Comparison")
    st.markdown("Benchmark against GraphCast, Pangu-Weather, FourCastNet")
    if st.button("Compare Models", use_container_width=True):
        st.switch_page("pages/12_Model_Comparison.py")

with adv_col4:
    st.markdown("#### 🎨 Visualizations")
    st.markdown("Publication-quality weather maps and animations")
    if st.button("Visualization Studio", use_container_width=True):
        st.switch_page("pages/04_Visualization_Studio.py")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown(
        """
    **Data Sources**
    - NCEP/NCAR Reanalysis (Kalnay et al., 1996)
    - ERA-Interim (Dee et al., 2011)
    - ERA5 via WeatherBench2
    """
    )

with footer_col2:
    st.markdown(
        """
    **Model Architectures**
    - Flow Matching (Lipman et al., 2023)
    - GraphCast-style GNNs
    - FourCastNet-style FNOs
    """
    )

with footer_col3:
    st.markdown(
        """
    **Resources**
    - [WeatherBench2 Benchmark](https://sites.research.google/weatherbench)
    - [GraphCast Paper](https://arxiv.org/abs/2212.12794)
    - [NCEP Reanalysis](https://psl.noaa.gov/data/reanalysis/)
    """
    )

st.markdown("---")

# --- Version & environment info ---
import torch as _torch
_device_label = "GPU" if _torch.cuda.is_available() else "CPU"
st.caption(
    f"WeatherFlow v0.4.3 | {_device_label} | Real Data Only | Built for the Weather AI Research Community"
)

# =============================================================================
# ENHANCED SIDEBAR (Improvement 8 & 9)
# =============================================================================
with st.sidebar:
    st.markdown("## Quick Access")

    # System status indicators (Improvement 8)
    if has_data:
        st.success("Data Loaded")
    else:
        st.warning("Load Data First")

    if has_model:
        st.success("Model Ready")
    else:
        st.info("No Model Yet")

    st.markdown(f"**Runtime:** {_device_label}")

    st.markdown("---")

    st.markdown("### Core Workflow")
    st.page_link("pages/0_Data_Manager.py", label="Data Manager")
    st.page_link("pages/03_Training_Workflow.py", label="Training")
    st.page_link("pages/17_Weather_Prediction.py", label="Predictions")

    st.markdown("### Analysis")
    st.page_link("pages/12_Model_Comparison.py", label="Benchmarks")
    st.page_link("pages/04_Visualization_Studio.py", label="Visualizations")

    st.markdown("### Advanced")
    st.page_link("pages/4_Flow_Matching.py", label="Flow Matching")
    st.page_link("pages/8_Physics_Losses.py", label="Physics Losses")
    st.page_link("pages/24_FLUX_FineTuning.py", label="FLUX Fine-Tuning")

    # Keyboard shortcuts help (Improvement 9)
    st.markdown("---")
    with st.expander("Keyboard Shortcuts"):
        st.markdown("""
| Shortcut | Action |
|----------|--------|
| `R` | Rerun page |
| `C` | Clear cache |
| `Ctrl+Shift+R` | Hard rerun |
""")

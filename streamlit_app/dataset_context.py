"""
Dataset Context Component

A reusable component that displays the current dataset status prominently
on every page. This ensures users always know what data they're working with.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from era5_utils import has_era5_data, get_active_era5_data
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


def render_dataset_banner():
    """
    Render a prominent dataset banner at the top of any page.
    This ensures users always know what data they're working with.
    """
    if not UTILS_AVAILABLE:
        st.warning("Dataset utilities not available")
        return None, None

    if has_era5_data():
        data, metadata = get_active_era5_data()
        is_synthetic = metadata.get("is_synthetic", True) if metadata else True
        name = metadata.get("name", "Unknown") if metadata else "Unknown"

        # Determine data type styling
        if is_synthetic:
            badge_color = "#ff9800"
            badge_text = "DEMO DATA"
            icon = "🔶"
        else:
            badge_color = "#4CAF50"
            badge_text = "REAL ERA5"
            icon = "✅"

        # Get data info
        n_vars = len(list(data.data_vars)) if data is not None else 0
        n_times = len(data.time) if data is not None and hasattr(data, 'time') else 0

        # Create styled banner
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {badge_color}22 0%, {badge_color}11 100%);
            border: 2px solid {badge_color};
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 20px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="
                        background: {badge_color};
                        color: white;
                        padding: 4px 12px;
                        border-radius: 15px;
                        font-size: 0.85em;
                        font-weight: bold;
                    ">{icon} {badge_text}</span>
                    <span style="margin-left: 10px; font-weight: bold; font-size: 1.1em;">{name}</span>
                </div>
                <div style="color: #666; font-size: 0.9em;">
                    {n_vars} variables | {n_times} time steps
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return data, metadata
    else:
        st.markdown("""
        <div style="
            background: #f0f0f0;
            border: 2px dashed #999;
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
            text-align: center;
        ">
            <span style="font-size: 1.2em;">📊 No dataset loaded</span>
            <br>
            <span style="color: #666;">Go to <strong>Data Manager</strong> to load a dataset</span>
        </div>
        """, unsafe_allow_html=True)
        return None, None


def render_compact_dataset_badge():
    """
    Render a compact dataset badge for sidebars.
    """
    if not UTILS_AVAILABLE:
        return None, None

    if has_era5_data():
        data, metadata = get_active_era5_data()
        is_synthetic = metadata.get("is_synthetic", True) if metadata else True
        name = metadata.get("name", "Unknown") if metadata else "Unknown"

        if is_synthetic:
            st.sidebar.warning(f"📊 **Demo:** {name}")
        else:
            st.sidebar.success(f"📊 **Real:** {name}")

        return data, metadata
    else:
        st.sidebar.info("📊 No data loaded")
        return None, None


def get_dataset_summary() -> dict:
    """
    Get a summary of the current dataset for display.
    """
    if not UTILS_AVAILABLE or not has_era5_data():
        return {
            "loaded": False,
            "name": "None",
            "type": "none",
            "variables": [],
            "time_steps": 0,
        }

    data, metadata = get_active_era5_data()

    return {
        "loaded": True,
        "name": metadata.get("name", "Unknown") if metadata else "Unknown",
        "type": "synthetic" if metadata.get("is_synthetic", True) else "real",
        "variables": list(data.data_vars) if data is not None else [],
        "time_steps": len(data.time) if data is not None and hasattr(data, 'time') else 0,
        "region": metadata.get("region", "Global") if metadata else "Global",
        "period": f"{metadata.get('start_date', '?')} to {metadata.get('end_date', '?')}" if metadata else "Unknown",
    }


def render_workflow_progress(current_step: int, total_steps: int = 4):
    """
    Render a workflow progress indicator.

    Steps:
    1. Load Data
    2. Configure Model
    3. Train
    4. Predict/Analyze
    """
    steps = ["📊 Data", "⚙️ Configure", "🏃 Train", "🔮 Predict"]

    cols = st.columns(total_steps)

    for i, (col, step) in enumerate(zip(cols, steps)):
        step_num = i + 1
        with col:
            if step_num < current_step:
                st.success(f"✅ {step}")
            elif step_num == current_step:
                st.info(f"📍 {step}")
            else:
                st.markdown(f"⬜ {step}")


def ensure_dataset_or_redirect():
    """
    Check if a dataset is loaded, and show a redirect message if not.
    Returns True if data is available, False otherwise.
    """
    if not UTILS_AVAILABLE:
        st.error("Dataset utilities not available")
        return False

    if not has_era5_data():
        st.warning("""
        **No dataset loaded!**

        Please go to the **Data Manager** page to load a dataset first.
        """)

        if st.button("Go to Data Manager →", type="primary"):
            st.switch_page("pages/0_Data_Manager.py")

        return False

    return True

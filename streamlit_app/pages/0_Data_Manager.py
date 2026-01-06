"""
Data Manager - Central hub for weather data in WeatherFlow.

This page provides full transparency about:
- What data is being used
- Where it comes from
- How to select different subsets
- How it connects to other pages
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Data Manager - WeatherFlow",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Manager")

st.markdown("""
This is your central hub for weather data. Here you can:
- See exactly what data the app is using
- Understand where it comes from (full transparency)
- Select time periods and variables for analysis
- Preview the data before using it in other pages
""")

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
STATS_FILE = DATA_DIR / "normalization_stats.json"
DATA_FILE = DATA_DIR / "era5_sample.nc"


def load_metadata():
    """Load metadata about the downloaded data."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return None


def load_stats():
    """Load normalization statistics."""
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            return json.load(f)
    return None


@st.cache_data
def load_data():
    """Load the ERA5 dataset."""
    if DATA_FILE.exists():
        import xarray as xr
        return xr.open_dataset(DATA_FILE)
    return None


# Check if data exists
metadata = load_metadata()
stats = load_stats()

if metadata is None:
    st.warning("⚠️ No data has been downloaded yet!")

    st.markdown("""
    ### How to Download Data

    Run the following command from the repository root:

    ```bash
    python streamlit_app/prepare_data.py
    ```

    This will download **2 months of ERA5 data** from WeatherBench2 (Google Cloud Storage).

    **Key points:**
    - ✅ **FREE** - No API key required
    - ✅ **Public** - From Google's WeatherBench2 project
    - ✅ **Real data** - Actual ERA5 reanalysis
    - ✅ **~50-100 MB** - Small enough for demos

    ### About the Data Source

    [WeatherBench2](https://github.com/google-research/weatherbench2) is Google's benchmark
    for weather forecasting ML models. It hosts processed ERA5 data on Google Cloud Storage
    for free public access.

    **ERA5** is the European Centre for Medium-Range Weather Forecasts (ECMWF) reanalysis
    dataset - considered the gold standard for historical weather data.
    """)

    st.info("After downloading, refresh this page to see your data!")
    st.stop()


# Data is available - show it!
st.success("✅ Data loaded successfully!")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Overview",
    "🔍 Data Preview",
    "⚙️ Selection",
    "📈 Statistics"
])

# Tab 1: Overview
with tab1:
    st.header("Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Information")

        source_data = {
            "Source": metadata.get("source", "Unknown"),
            "License": metadata.get("license", "Unknown"),
            "Downloaded": metadata.get("downloaded_at", "Unknown")[:19].replace("T", " "),
            "File Size": f"{metadata.get('file_size_mb', 0):.1f} MB"
        }

        for key, value in source_data.items():
            st.markdown(f"**{key}:** {value}")

        st.markdown("---")

        st.markdown("""
        **Citation:**
        > Hersbach et al. (2020). The ERA5 global reanalysis.
        > Quarterly Journal of the Royal Meteorological Society.
        """)

    with col2:
        st.subheader("Data Specifications")

        time_info = metadata.get("time_range", {})
        grid_info = metadata.get("grid", {})

        specs = {
            "Time Period": f"{time_info.get('start', '')[:10]} to {time_info.get('end', '')[:10]}",
            "Time Steps": time_info.get("n_steps", "Unknown"),
            "Frequency": f"{time_info.get('frequency_hours', 6)} hours",
            "Grid Size": f"{grid_info.get('latitude', '?')} × {grid_info.get('longitude', '?')}",
            "Resolution": grid_info.get("resolution", "64x32")
        }

        for key, value in specs.items():
            st.markdown(f"**{key}:** {value}")

    st.subheader("Variables Available")

    variables = metadata.get("variables", [])
    levels = metadata.get("pressure_levels", [])

    var_descriptions = {
        "temperature": ("🌡️ Temperature", "Air temperature at pressure levels", "Kelvin (K)"),
        "geopotential": ("📊 Geopotential", "Gravitational potential energy per unit mass", "m²/s²"),
        "u_component_of_wind": ("💨 U-Wind", "Eastward wind component", "m/s"),
        "v_component_of_wind": ("💨 V-Wind", "Northward wind component", "m/s"),
    }

    var_cols = st.columns(len(variables))
    for i, var in enumerate(variables):
        with var_cols[i]:
            info = var_descriptions.get(var, ("❓ " + var, "No description", "Unknown"))
            st.markdown(f"### {info[0]}")
            st.markdown(f"*{info[1]}*")
            st.markdown(f"**Units:** {info[2]}")

    st.markdown("---")

    st.subheader("Pressure Levels")
    level_cols = st.columns(len(levels))
    for i, level in enumerate(levels):
        with level_cols[i]:
            altitude_km = {1000: "~0", 850: "~1.5", 700: "~3", 500: "~5.5", 300: "~9", 200: "~12"}.get(level, "?")
            st.metric(f"{level} hPa", f"~{altitude_km} km altitude")


# Tab 2: Data Preview
with tab2:
    st.header("Data Preview")

    ds = load_data()

    if ds is not None:
        # Selection for preview
        col1, col2, col3 = st.columns(3)

        with col1:
            preview_var = st.selectbox(
                "Variable",
                options=list(ds.data_vars),
                format_func=lambda x: x.replace("_", " ").title()
            )

        with col2:
            preview_level = st.selectbox(
                "Pressure Level (hPa)",
                options=sorted([int(l) for l in ds.level.values])
            )

        with col3:
            time_idx = st.slider(
                "Time Step",
                0, len(ds.time) - 1, 0,
                help="Slide to see different time steps"
            )

        # Get the data slice
        data_slice = ds[preview_var].isel(time=time_idx).sel(level=preview_level)
        time_val = pd.Timestamp(ds.time.values[time_idx]).strftime("%Y-%m-%d %H:%M UTC")

        st.markdown(f"**Showing:** {preview_var.replace('_', ' ').title()} at {preview_level} hPa — {time_val}")

        # Create the plot
        fig = go.Figure(data=go.Heatmap(
            z=data_slice.values,
            x=ds.longitude.values,
            y=ds.latitude.values,
            colorscale='RdBu_r' if 'temperature' in preview_var else 'Viridis',
            colorbar=dict(title=preview_var.split('_')[0])
        ))

        fig.update_layout(
            title=f"{preview_var.replace('_', ' ').title()} at {preview_level} hPa",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show raw values
        with st.expander("View Raw Data Values"):
            st.dataframe(
                pd.DataFrame(
                    data_slice.values,
                    index=[f"{lat:.1f}°" for lat in ds.latitude.values],
                    columns=[f"{lon:.1f}°" for lon in ds.longitude.values]
                ),
                height=300
            )


# Tab 3: Selection
with tab3:
    st.header("Data Selection")

    st.markdown("""
    Configure which subset of data to use across the app. These selections will be
    saved and used by other pages (Wind Power, Solar Power, etc.).
    """)

    ds = load_data()

    if ds is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Time Range")

            # Convert to datetime for the slider
            times = pd.to_datetime(ds.time.values)
            min_time = times.min().to_pydatetime()
            max_time = times.max().to_pydatetime()

            selected_range = st.slider(
                "Select Time Range",
                min_value=min_time,
                max_value=max_time,
                value=(min_time, max_time),
                format="YYYY-MM-DD"
            )

            n_selected_steps = len(times[(times >= selected_range[0]) & (times <= selected_range[1])])
            st.info(f"Selected {n_selected_steps} time steps ({n_selected_steps * 6} hours)")

        with col2:
            st.subheader("Variables")

            selected_vars = st.multiselect(
                "Select Variables to Use",
                options=list(ds.data_vars),
                default=list(ds.data_vars),
                format_func=lambda x: x.replace("_", " ").title()
            )

            st.subheader("Pressure Levels")

            selected_levels = st.multiselect(
                "Select Pressure Levels (hPa)",
                options=sorted([int(l) for l in ds.level.values]),
                default=sorted([int(l) for l in ds.level.values])
            )

        # Save selection to session state
        if st.button("💾 Save Selection", type="primary"):
            st.session_state['data_selection'] = {
                'time_range': selected_range,
                'variables': selected_vars,
                'levels': selected_levels
            }
            st.success("Selection saved! Other pages will now use this data subset.")

        # Show current selection
        if 'data_selection' in st.session_state:
            st.markdown("---")
            st.subheader("Current Active Selection")
            sel = st.session_state['data_selection']
            st.json({
                'time_range': [str(sel['time_range'][0]), str(sel['time_range'][1])],
                'variables': sel['variables'],
                'levels': sel['levels']
            })


# Tab 4: Statistics
with tab4:
    st.header("Data Statistics")

    if stats:
        st.markdown("""
        These statistics are computed from the downloaded data and used for
        normalizing inputs to ML models. Proper normalization is critical for
        training stability.
        """)

        # Create stats table
        stats_df = pd.DataFrame(stats).T
        stats_df.index.name = "Variable"

        st.dataframe(
            stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                'max': '{:.2f}'
            }),
            use_container_width=True
        )

        # Visualize distributions
        st.subheader("Variable Distributions")

        ds = load_data()
        if ds is not None:
            var_to_plot = st.selectbox(
                "Select Variable for Distribution",
                options=list(ds.data_vars),
                key="dist_var"
            )

            # Sample data for histogram (full data would be too large)
            sample_data = ds[var_to_plot].values.flatten()
            sample_data = sample_data[~np.isnan(sample_data)]
            if len(sample_data) > 100000:
                sample_data = np.random.choice(sample_data, 100000, replace=False)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sample_data,
                nbinsx=50,
                name=var_to_plot
            ))

            # Add mean and std lines
            mean_val = stats[var_to_plot]['mean']
            std_val = stats[var_to_plot]['std']

            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_val:.1f}")
            fig.add_vline(x=mean_val - std_val, line_dash="dot", line_color="orange",
                         annotation_text="-1 std")
            fig.add_vline(x=mean_val + std_val, line_dash="dot", line_color="orange",
                         annotation_text="+1 std")

            fig.update_layout(
                title=f"Distribution of {var_to_plot.replace('_', ' ').title()}",
                xaxis_title="Value",
                yaxis_title="Count",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)


# Sidebar info
st.sidebar.header("Data Status")

if metadata:
    st.sidebar.success("✅ Data Loaded")
    st.sidebar.markdown(f"""
    **Source:** {metadata.get('source', 'Unknown')}

    **Time Range:**
    {metadata.get('time_range', {}).get('start', '')[:10]} to
    {metadata.get('time_range', {}).get('end', '')[:10]}

    **Variables:** {len(metadata.get('variables', []))}

    **File Size:** {metadata.get('file_size_mb', 0):.1f} MB
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How Data Flows

1. **Data Manager** (this page)
   - Download and configure data

2. **Other Pages** read from here:
   - Wind Power → uses wind components
   - Solar Power → uses temperature
   - Flow Matching → uses all variables
   - GCM Simulation → uses full state
""")

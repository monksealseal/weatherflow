"""
Data Manager - Central hub for REAL ERA5 weather data in WeatherFlow.

Downloads actual ERA5 reanalysis data from WeatherBench2 (Google Cloud Storage).
NO SYNTHETIC DATA - only real atmospheric observations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Data Manager - WeatherFlow", page_icon="📊", layout="wide"
)

st.title("📊 Data Manager")

st.markdown(
    """
**Real ERA5 Reanalysis Data** from the European Centre for Medium-Range Weather Forecasts (ECMWF).

This page downloads and manages actual atmospheric observations - no synthetic data.
"""
)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "era5_cache.nc"
METADATA_FILE = DATA_DIR / "metadata.json"

# WeatherBench2 ERA5 URL (public, free access)
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


@st.cache_resource
def load_era5_from_weatherbench2():
    """
    Load ERA5 data directly from WeatherBench2 Google Cloud Storage.
    This is REAL ERA5 reanalysis data from ECMWF.
    """
    import xarray as xr

    status = st.empty()
    status.info("🌐 Connecting to WeatherBench2 ERA5 dataset...")

    # Try multiple methods to access the data
    ds = None
    method_used = None

    # Method 1: Direct zarr with anonymous GCS access
    try:
        ds = xr.open_zarr(
            WEATHERBENCH2_URL, storage_options={"anon": True}, consolidated=True
        )
        method_used = "GCS Anonymous"
    except Exception as e1:
        status.warning(f"Method 1 failed: {type(e1).__name__}")

        # Method 2: via gcsfs
        try:
            import gcsfs

            fs = gcsfs.GCSFileSystem(token="anon")
            mapper = fs.get_mapper(WEATHERBENCH2_URL)
            ds = xr.open_zarr(mapper, consolidated=True)
            method_used = "gcsfs"
        except Exception as e2:
            status.warning(f"Method 2 failed: {type(e2).__name__}")

            # Method 3: HTTP fallback
            try:
                import fsspec

                http_url = WEATHERBENCH2_URL.replace(
                    "gs://", "https://storage.googleapis.com/"
                )
                fs = fsspec.filesystem("http")
                mapper = fs.get_mapper(http_url)
                ds = xr.open_zarr(mapper, consolidated=True)
                method_used = "HTTP"
            except Exception as e3:
                status.error(f"All connection methods failed. Last error: {e3}")
                return None, None

    if ds is not None:
        status.success(f"✅ Connected to WeatherBench2 via {method_used}")

        metadata = {
            "source": "WeatherBench2 ERA5 (ECMWF Reanalysis)",
            "url": WEATHERBENCH2_URL,
            "is_synthetic": False,
            "connection_method": method_used,
            "available_variables": list(ds.data_vars)[:20],
            "time_range": {
                "start": str(ds.time.values[0]),
                "end": str(ds.time.values[-1]),
            },
            "grid": {
                "latitude": int(ds.latitude.size),
                "longitude": int(ds.longitude.size),
            },
            "citation": "Hersbach et al. (2020). The ERA5 global reanalysis. Q J R Meteorol Soc.",
        }

        return ds, metadata

    return None, None


@st.cache_data
def load_subset(_ds, variables, levels, start_date, end_date):
    """Load a specific subset of the ERA5 data."""
    subset = _ds[variables].sel(time=slice(start_date, end_date), level=levels)
    return subset.load()


# Main content
ds, metadata = load_era5_from_weatherbench2()

if ds is None:
    st.error(
        """
    ❌ Could not connect to WeatherBench2.

    This app requires internet access to download real ERA5 data.

    **If running locally**, ensure you have network access and try:
    ```bash
    pip install xarray zarr gcsfs fsspec requests aiohttp
    ```

    **If on Streamlit Cloud**, this should work automatically.
    Please check your connection and refresh the page.
    """
    )
    st.stop()

# Data loaded successfully
st.success("✅ Connected to REAL ERA5 data from WeatherBench2")

# Show data source prominently
st.markdown(
    """
---
### Data Source Verification

| Property | Value |
|----------|-------|
| **Source** | WeatherBench2 / ECMWF ERA5 |
| **Type** | REAL atmospheric reanalysis |
| **Synthetic** | ❌ NO |
| **License** | Copernicus Climate Data Store |
"""
)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Overview", "🔍 Preview Data", "⚙️ Select Subset", "📈 Statistics"]
)

with tab1:
    st.header("ERA5 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About ERA5")
        st.markdown(
            """
        **ERA5** is the fifth generation ECMWF atmospheric reanalysis of the global climate.

        - Produced by the Copernicus Climate Change Service (C3S)
        - Hourly data from 1940 to present
        - 0.25° horizontal resolution (original)
        - WeatherBench2 provides regridded versions for ML

        **This is NOT synthetic data** - these are actual atmospheric observations
        combined with numerical weather prediction models.
        """
        )

        st.markdown(
            f"""
        **Citation:**
        > {metadata.get('citation', 'Hersbach et al. (2020)')}
        """
        )

    with col2:
        st.subheader("Available in this Dataset")

        st.markdown(
            f"**Grid Size:** {metadata['grid']['latitude']} × {metadata['grid']['longitude']}"
        )

        # Show available variables
        st.markdown("**Variables:**")
        available_vars = list(ds.data_vars)
        var_cols = st.columns(3)
        for i, var in enumerate(available_vars[:15]):
            var_cols[i % 3].write(f"• {var}")

        if len(available_vars) > 15:
            st.write(f"... and {len(available_vars) - 15} more")

        # Show levels if available
        if "level" in ds.coords:
            st.markdown(f"**Pressure Levels:** {list(ds.level.values)} hPa")

with tab2:
    st.header("Data Preview")

    col1, col2, col3 = st.columns(3)

    with col1:
        preview_var = st.selectbox(
            "Variable",
            options=[v for v in ds.data_vars if "level" in ds[v].dims],
            index=0,
        )

    with col2:
        if "level" in ds.coords:
            preview_level = st.selectbox(
                "Pressure Level (hPa)",
                options=sorted([int(l) for l in ds.level.values]),
                index=2,  # Default to 500 hPa region
            )
        else:
            preview_level = None

    with col3:
        # Date selection
        min_time = pd.Timestamp(ds.time.values[0])
        max_time = pd.Timestamp(ds.time.values[-1])

        preferred_date = pd.Timestamp("2023-01-15")
        default_date = max(min(preferred_date, max_time), min_time).date()

        preview_date = st.date_input(
            "Date",
            value=default_date,
            min_value=min_time.date(),
            max_value=max_time.date(),
        )

    if st.button("Load Preview", type="primary"):
        with st.spinner("Loading real ERA5 data..."):
            try:
                # Select the specific time slice
                time_str = preview_date.strftime("%Y-%m-%d")

                if preview_level:
                    data_slice = (
                        ds[preview_var]
                        .sel(time=time_str, level=preview_level, method="nearest")
                        .isel(time=0)
                        .load()
                    )
                else:
                    data_slice = (
                        ds[preview_var]
                        .sel(time=time_str, method="nearest")
                        .isel(time=0)
                        .load()
                    )

                # Create plot
                fig = go.Figure(
                    data=go.Heatmap(
                        z=data_slice.values,
                        x=ds.longitude.values,
                        y=ds.latitude.values,
                        colorscale=(
                            "RdBu_r" if "temperature" in preview_var else "Viridis"
                        ),
                        colorbar=dict(title=preview_var),
                    )
                )

                title = f"REAL ERA5: {preview_var}"
                if preview_level:
                    title += f" at {preview_level} hPa"
                title += f" - {time_str}"

                fig.update_layout(
                    title=title,
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    f"📊 This is REAL ERA5 data - atmospheric observations from {time_str}"
                )

            except Exception as e:
                st.error(f"Error loading data: {e}")

with tab3:
    st.header("Select Data Subset")

    st.markdown(
        """
    Configure which data to use across the app. This selection will be stored
    and used by other pages (Wind Power, Flow Matching, etc.).
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time Range")

        start_date = st.date_input(
            "Start Date", value=pd.Timestamp("2023-01-01"), key="start"
        )

        end_date = st.date_input(
            "End Date", value=pd.Timestamp("2023-02-28"), key="end"
        )

        # Calculate time steps
        days = (end_date - start_date).days
        time_steps = days * 4  # 6-hourly = 4 per day
        st.info(f"≈ {time_steps} time steps ({days} days × 4 per day)")

    with col2:
        st.subheader("Variables")

        # Common ML variables
        default_vars = [
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
        available = [v for v in default_vars if v in ds.data_vars]

        selected_vars = st.multiselect(
            "Select Variables", options=list(ds.data_vars), default=available
        )

        st.subheader("Pressure Levels")

        if "level" in ds.coords:
            all_levels = sorted([int(l) for l in ds.level.values])
            default_levels = [l for l in [1000, 850, 500, 200] if l in all_levels]

            selected_levels = st.multiselect(
                "Select Levels (hPa)", options=all_levels, default=default_levels
            )
        else:
            selected_levels = []

    if st.button("💾 Save Selection for App", type="primary"):
        st.session_state["era5_selection"] = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "variables": selected_vars,
            "levels": selected_levels,
            "source": "WeatherBench2 ERA5 (REAL DATA)",
        }
        st.success("✅ Selection saved! Other pages will use this real ERA5 data.")

        # Show what was saved
        st.json(st.session_state["era5_selection"])

with tab4:
    st.header("Data Statistics")

    st.markdown("Statistics computed from REAL ERA5 observations.")

    stat_var = st.selectbox(
        "Select Variable for Statistics",
        options=[v for v in ds.data_vars if "level" in ds[v].dims][:10],
        key="stat_var",
    )

    stat_level = st.selectbox(
        "Select Level (hPa)",
        options=(
            sorted([int(l) for l in ds.level.values]) if "level" in ds.coords else [0]
        ),
        key="stat_level",
    )

    if st.button("Compute Statistics (samples 2023)"):
        with st.spinner("Computing statistics from real ERA5 data..."):
            try:
                # Sample a year of data
                sample = (
                    ds[stat_var]
                    .sel(time=slice("2023-01-01", "2023-12-31"), level=stat_level)
                    .load()
                )

                values = sample.values.flatten()
                values = values[~np.isnan(values)]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{np.mean(values):.2f}")
                col2.metric("Std Dev", f"{np.std(values):.2f}")
                col3.metric("Min", f"{np.min(values):.2f}")
                col4.metric("Max", f"{np.max(values):.2f}")

                # Histogram
                fig = go.Figure(
                    data=go.Histogram(
                        x=np.random.choice(values, min(50000, len(values))), nbinsx=50
                    )
                )
                fig.update_layout(
                    title=f"Distribution of {stat_var} at {stat_level} hPa (Real ERA5 Data)",
                    xaxis_title="Value",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing statistics: {e}")


# Sidebar
st.sidebar.header("Data Source")
st.sidebar.success("✅ REAL ERA5 Data")
st.sidebar.markdown(
    f"""
**Source:** WeatherBench2

**Provider:** ECMWF

**Type:** Reanalysis

**Synthetic:** ❌ NO

---

This is actual atmospheric data from
satellite observations, weather stations,
and numerical weather prediction models.
"""
)

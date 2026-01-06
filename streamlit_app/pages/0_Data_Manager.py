"""
Data Manager - Central hub for REAL ERA5 weather data in WeatherFlow.

Downloads actual ERA5 reanalysis data from WeatherBench2 (Google Cloud Storage).
NO SYNTHETIC DATA - only real atmospheric observations.

Features:
- Pre-downloaded sample datasets for key weather events
- On-demand data download with progress indicators
- All app pages use this data source
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Data Manager - WeatherFlow", page_icon="📊", layout="wide"
)

st.title("📊 Data Manager")

st.markdown(
    """
**Real ERA5 Reanalysis Data** from the European Centre for Medium-Range Weather Forecasts (ECMWF).

This page manages actual atmospheric observations - no synthetic data. 
Choose from pre-downloaded sample datasets or download custom data on-demand.
"""
)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# WeatherBench2 ERA5 URL (public, free access)
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Pre-defined sample datasets (curated weather events)
SAMPLE_DATASETS = {
    "tropical_cyclones_2005": {
        "name": "Tropical Cyclones (2005)",
        "description": "Hurricane Katrina and active Atlantic hurricane season",
        "start_date": "2005-08-15",
        "end_date": "2005-09-30",
        "region": "Atlantic Basin",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200],
        "events": ["Hurricane Katrina (Aug 23-31)", "Hurricane Rita (Sep 17-26)"],
    },
    "heat_wave_europe_2003": {
        "name": "European Heat Wave (2003)",
        "description": "Record-breaking heat wave affecting Western Europe",
        "start_date": "2003-07-20",
        "end_date": "2003-08-20",
        "region": "Europe",
        "variables": ["temperature", "geopotential"],
        "pressure_levels": [1000, 850, 500],
        "events": ["Peak temperatures Aug 4-13", "Over 70,000 excess deaths"],
    },
    "atmospheric_rivers_2017": {
        "name": "Atmospheric Rivers (2017)",
        "description": "Pineapple Express events impacting US West Coast",
        "start_date": "2017-01-01",
        "end_date": "2017-02-28",
        "region": "US West Coast",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 700, 500],
        "events": ["Oroville Dam emergency (Feb 7-14)", "Multiple AR landfalls"],
    },
    "polar_vortex_2019": {
        "name": "Polar Vortex Event (2019)",
        "description": "Extreme cold outbreak in North America",
        "start_date": "2019-01-20",
        "end_date": "2019-02-10",
        "region": "North America",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200, 50],
        "events": ["Record lows Jan 27-31", "Stratospheric warming precursor"],
    },
    "sudden_stratospheric_warming_2018": {
        "name": "Sudden Stratospheric Warming (2018)",
        "description": "Major SSW event with Beast from the East",
        "start_date": "2018-02-01",
        "end_date": "2018-03-15",
        "region": "Northern Hemisphere",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200, 50, 10],
        "events": ["SSW onset Feb 11-12", "European cold wave late Feb"],
    },
    "general_sample_2023": {
        "name": "General Sample (2023)",
        "description": "Standard sample for testing and development",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "region": "Global",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 700, 500, 300, 200],
        "events": ["General atmospheric patterns", "Multiple weather systems"],
    },
}


def get_sample_data_file(sample_id: str) -> Path:
    """Get the path to sample data file."""
    return SAMPLES_DIR / f"{sample_id}.nc"


def get_sample_metadata_file(sample_id: str) -> Path:
    """Get the path to sample metadata file."""
    return SAMPLES_DIR / f"{sample_id}_metadata.json"


def is_sample_downloaded(sample_id: str) -> bool:
    """Check if a sample dataset has been downloaded."""
    return get_sample_data_file(sample_id).exists()


def get_downloaded_samples() -> list:
    """Get list of downloaded sample datasets."""
    return [sid for sid in SAMPLE_DATASETS.keys() if is_sample_downloaded(sid)]


def load_sample_data(sample_id: str):
    """Load a downloaded sample dataset."""
    import xarray as xr
    data_file = get_sample_data_file(sample_id)
    if data_file.exists():
        return xr.open_dataset(data_file)
    return None


def connect_to_weatherbench2():
    """
    Connect to WeatherBench2 without loading data.
    Returns the dataset handle for lazy loading.
    """
    import xarray as xr
    
    ds = None
    method_used = None
    error_msg = None

    # Method 1: Direct zarr with anonymous GCS access
    try:
        ds = xr.open_zarr(
            WEATHERBENCH2_URL, storage_options={"anon": True}, consolidated=True
        )
        method_used = "GCS Anonymous"
    except Exception as e1:
        error_msg = str(e1)
        
        # Method 2: via gcsfs
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token="anon")
            mapper = fs.get_mapper(WEATHERBENCH2_URL)
            ds = xr.open_zarr(mapper, consolidated=True)
            method_used = "gcsfs"
        except Exception as e2:
            error_msg = str(e2)
            
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
                error_msg = str(e3)

    if ds is not None:
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
        return ds, metadata, None
    
    return None, None, error_msg


def download_sample_dataset(sample_id: str, ds, progress_callback=None):
    """
    Download a sample dataset from WeatherBench2.
    
    Args:
        sample_id: ID of the sample dataset
        ds: Connected xarray dataset
        progress_callback: Function to call with progress updates (0-100)
    """
    import xarray as xr
    
    sample_info = SAMPLE_DATASETS[sample_id]
    
    # Select the subset
    variables = sample_info["variables"]
    available_vars = [v for v in variables if v in ds.data_vars]
    
    if not available_vars:
        raise ValueError(f"None of the requested variables are available: {variables}")
    
    if progress_callback:
        progress_callback(10, "Selecting data subset...")
    
    # Select time range
    subset = ds[available_vars].sel(
        time=slice(sample_info["start_date"], sample_info["end_date"])
    )
    
    # Select pressure levels if available
    if "level" in ds.coords:
        available_levels = [l for l in sample_info["pressure_levels"] if l in ds.level.values]
        if available_levels:
            subset = subset.sel(level=available_levels)
    
    if progress_callback:
        progress_callback(30, "Loading data from WeatherBench2...")
    
    # Load the data
    loaded_data = subset.load()
    
    if progress_callback:
        progress_callback(80, "Saving to disk...")
    
    # Save to file
    data_file = get_sample_data_file(sample_id)
    loaded_data.to_netcdf(data_file)
    
    # Save metadata
    metadata = {
        **sample_info,
        "downloaded_at": datetime.now().isoformat(),
        "source": "WeatherBench2 ERA5 (ECMWF Reanalysis)",
        "is_synthetic": False,
        "file_size_mb": data_file.stat().st_size / (1024 * 1024),
        "actual_variables": available_vars,
        "time_steps": len(loaded_data.time),
    }
    
    with open(get_sample_metadata_file(sample_id), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if progress_callback:
        progress_callback(100, "Complete!")
    
    return loaded_data


# Initialize session state
if "era5_data" not in st.session_state:
    st.session_state["era5_data"] = None
if "era5_metadata" not in st.session_state:
    st.session_state["era5_metadata"] = None
if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = None
if "wb2_connected" not in st.session_state:
    st.session_state["wb2_connected"] = False
if "wb2_dataset" not in st.session_state:
    st.session_state["wb2_dataset"] = None


# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Sample Datasets", "🌐 Custom Download", "🔍 Preview Data", "📈 Statistics", "ℹ️ About ERA5"
])


# Tab 1: Sample Datasets
with tab1:
    st.header("Pre-Defined Sample Datasets")
    
    st.markdown("""
    Choose from curated weather event datasets. These are designed to showcase 
    interesting atmospheric phenomena and can be used across all app features.
    
    **Note:** Data must be downloaded before use. Downloaded data is cached locally.
    """)
    
    downloaded_samples = get_downloaded_samples()
    
    # Show downloaded samples first
    if downloaded_samples:
        st.subheader("✅ Downloaded Datasets")
        
        for sample_id in downloaded_samples:
            sample_info = SAMPLE_DATASETS[sample_id]
            
            with st.expander(f"📁 {sample_info['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**{sample_info['description']}**")
                    st.markdown(f"- **Period:** {sample_info['start_date']} to {sample_info['end_date']}")
                    st.markdown(f"- **Region:** {sample_info['region']}")
                    st.markdown(f"- **Variables:** {', '.join(sample_info['variables'])}")
                    st.markdown(f"- **Events:** {', '.join(sample_info['events'])}")
                
                with col2:
                    if st.button(f"🔄 Use This Dataset", key=f"use_{sample_id}", type="primary"):
                        data = load_sample_data(sample_id)
                        if data is not None:
                            st.session_state["era5_data"] = data
                            st.session_state["era5_metadata"] = sample_info
                            st.session_state["active_sample"] = sample_id
                            st.success(f"✅ Now using: {sample_info['name']}")
                            st.rerun()
                    
                    # Show file size if metadata exists
                    meta_file = get_sample_metadata_file(sample_id)
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = json.load(f)
                        st.info(f"Size: {meta.get('file_size_mb', 0):.1f} MB")
    
    st.markdown("---")
    
    # Show available samples for download
    st.subheader("📥 Available for Download")
    
    not_downloaded = [sid for sid in SAMPLE_DATASETS.keys() if sid not in downloaded_samples]
    
    if not_downloaded:
        for sample_id in not_downloaded:
            sample_info = SAMPLE_DATASETS[sample_id]
            
            with st.expander(f"🌐 {sample_info['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**{sample_info['description']}**")
                    st.markdown(f"- **Period:** {sample_info['start_date']} to {sample_info['end_date']}")
                    st.markdown(f"- **Region:** {sample_info['region']}")
                    st.markdown(f"- **Variables:** {', '.join(sample_info['variables'])}")
                    st.markdown("- **Pressure Levels:** " + ", ".join(str(l) for l in sample_info['pressure_levels']) + " hPa")
                    st.markdown(f"- **Notable Events:** {', '.join(sample_info['events'])}")
                
                with col2:
                    download_key = f"download_{sample_id}"
                    if st.button(f"📥 Download", key=download_key):
                        # Store download request in session state
                        st.session_state[f"pending_download_{sample_id}"] = True
                        st.rerun()
                    
                    # Handle pending download
                    if st.session_state.get(f"pending_download_{sample_id}", False):
                        st.session_state[f"pending_download_{sample_id}"] = False
                        
                        # Connect to WeatherBench2
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(pct, msg):
                            progress_bar.progress(pct / 100)
                            status_text.text(msg)
                        
                        update_progress(0, "Connecting to WeatherBench2...")
                        
                        ds, metadata, error = connect_to_weatherbench2()
                        
                        if ds is None:
                            st.error(f"❌ Could not connect to WeatherBench2: {error}")
                        else:
                            try:
                                download_sample_dataset(sample_id, ds, update_progress)
                                st.success(f"✅ Downloaded: {sample_info['name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Download failed: {e}")
    else:
        st.success("✅ All sample datasets are downloaded!")


# Tab 2: Custom Download
with tab2:
    st.header("Download Custom Data")
    
    st.markdown("""
    Download a custom time period and variable selection from ERA5.
    
    ⚠️ **Note:** This requires connecting to WeatherBench2 (Google Cloud Storage).
    Large downloads may take several minutes.
    """)
    
    # Connection status
    if not st.session_state.get("wb2_connected", False):
        if st.button("🌐 Connect to WeatherBench2", type="primary"):
            with st.spinner("Connecting..."):
                ds, metadata, error = connect_to_weatherbench2()
                
                if ds is not None:
                    st.session_state["wb2_connected"] = True
                    st.session_state["wb2_dataset"] = ds
                    st.session_state["wb2_metadata"] = metadata
                    st.success("✅ Connected to WeatherBench2!")
                    st.rerun()
                else:
                    st.error(f"""
                    ❌ Could not connect to WeatherBench2.
                    
                    Error: {error}
                    
                    This may be due to network restrictions. Try using the pre-downloaded sample datasets instead.
                    """)
    else:
        st.success("✅ Connected to WeatherBench2")
        ds = st.session_state["wb2_dataset"]
        metadata = st.session_state["wb2_metadata"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time Range")
            
            # Parse time range from metadata
            try:
                min_time = pd.Timestamp(metadata["time_range"]["start"])
                max_time = pd.Timestamp(metadata["time_range"]["end"])
            except:
                min_time = pd.Timestamp("1959-01-01")
                max_time = pd.Timestamp("2023-12-31")
            
            default_start = pd.Timestamp("2023-01-01")
            if default_start < min_time:
                default_start = min_time
            if default_start > max_time:
                default_start = max_time - timedelta(days=30)
            
            start_date = st.date_input(
                "Start Date",
                value=default_start.date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                key="custom_start"
            )
            
            end_date = st.date_input(
                "End Date",
                value=(default_start + timedelta(days=30)).date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                key="custom_end"
            )
            
            days = (end_date - start_date).days
            time_steps = max(0, days * 4)  # 6-hourly
            st.info(f"≈ {time_steps} time steps ({days} days)")
        
        with col2:
            st.subheader("Variables & Levels")
            
            available_vars = list(ds.data_vars)[:20]
            default_vars = ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"]
            default_vars = [v for v in default_vars if v in available_vars]
            
            selected_vars = st.multiselect(
                "Variables",
                options=available_vars,
                default=default_vars,
                key="custom_vars"
            )
            
            if "level" in ds.coords:
                all_levels = sorted([int(l) for l in ds.level.values])
                default_levels = [l for l in [1000, 850, 500, 200] if l in all_levels]
                
                selected_levels = st.multiselect(
                    "Pressure Levels (hPa)",
                    options=all_levels,
                    default=default_levels,
                    key="custom_levels"
                )
            else:
                selected_levels = []
        
        st.markdown("---")
        
        custom_name = st.text_input(
            "Dataset Name (optional)",
            value=f"custom_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            key="custom_name"
        )
        
        if st.button("📥 Download Custom Dataset", type="primary", disabled=not selected_vars):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct / 100)
                status_text.text(msg)
            
            try:
                update_progress(10, "Selecting data...")
                
                subset = ds[selected_vars].sel(
                    time=slice(str(start_date), str(end_date))
                )
                
                if selected_levels and "level" in ds.coords:
                    subset = subset.sel(level=selected_levels)
                
                update_progress(30, "Downloading data (this may take a while)...")
                
                loaded_data = subset.load()
                
                update_progress(80, "Saving to disk...")
                
                # Save as custom sample
                custom_id = custom_name.replace(" ", "_").lower()
                data_file = SAMPLES_DIR / f"{custom_id}.nc"
                loaded_data.to_netcdf(data_file)
                
                # Add to sample datasets temporarily
                custom_meta = {
                    "name": custom_name,
                    "description": f"Custom download: {start_date} to {end_date}",
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "region": "Global",
                    "variables": selected_vars,
                    "pressure_levels": selected_levels,
                    "events": [],
                    "downloaded_at": datetime.now().isoformat(),
                    "source": "WeatherBench2 ERA5",
                    "is_synthetic": False,
                    "file_size_mb": data_file.stat().st_size / (1024 * 1024),
                    "time_steps": len(loaded_data.time),
                }
                
                with open(SAMPLES_DIR / f"{custom_id}_metadata.json", "w") as f:
                    json.dump(custom_meta, f, indent=2)
                
                update_progress(100, "Complete!")
                
                st.session_state["era5_data"] = loaded_data
                st.session_state["era5_metadata"] = custom_meta
                st.session_state["active_sample"] = custom_id
                
                st.success(f"✅ Downloaded and activated: {custom_name}")
                
            except Exception as e:
                st.error(f"❌ Download failed: {e}")


# Tab 3: Preview Data
with tab3:
    st.header("Preview Active Dataset")
    
    active_data = st.session_state.get("era5_data")
    active_meta = st.session_state.get("era5_metadata")
    
    if active_data is None:
        st.warning("""
        ⚠️ No dataset is currently active.
        
        Please select a sample dataset from the **Sample Datasets** tab or download custom data.
        """)
    else:
        st.success(f"�� Active Dataset: **{active_meta.get('name', 'Unknown')}**")
        
        col1, col2, col3 = st.columns(3)
        
        available_vars = list(active_data.data_vars)
        
        with col1:
            preview_var = st.selectbox(
                "Variable",
                options=available_vars,
                index=0,
                key="preview_var"
            )
        
        with col2:
            if "level" in active_data.coords:
                preview_level = st.selectbox(
                    "Pressure Level (hPa)",
                    options=sorted([int(l) for l in active_data.level.values]),
                    index=min(2, len(active_data.level) - 1),
                    key="preview_level"
                )
            else:
                preview_level = None
        
        with col3:
            time_values = active_data.time.values
            time_options = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in time_values]
            preview_time_idx = st.selectbox(
                "Time",
                options=range(len(time_options)),
                format_func=lambda x: time_options[x],
                index=0,
                key="preview_time"
            )
        
        # Load and display the selected slice
        try:
            if preview_level is not None:
                data_slice = active_data[preview_var].isel(time=preview_time_idx).sel(level=preview_level)
            else:
                data_slice = active_data[preview_var].isel(time=preview_time_idx)
            
            # Get coordinates
            if "latitude" in active_data.coords:
                lat_coord = "latitude"
                lon_coord = "longitude"
            else:
                lat_coord = "lat"
                lon_coord = "lon"
            
            lats = active_data[lat_coord].values
            lons = active_data[lon_coord].values
            
            # Create plot
            fig = go.Figure(
                data=go.Heatmap(
                    z=data_slice.values,
                    x=lons,
                    y=lats,
                    colorscale="RdBu_r" if "temperature" in preview_var else "Viridis",
                    colorbar=dict(title=preview_var),
                )
            )
            
            title = f"REAL ERA5: {preview_var}"
            if preview_level is not None:
                title += f" at {preview_level} hPa"
            title += f" - {time_options[preview_time_idx]}"
            
            fig.update_layout(
                title=title,
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=500,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 This is REAL ERA5 data - {active_meta.get('source', 'atmospheric observations')}")
            
        except Exception as e:
            st.error(f"Error displaying data: {e}")


# Tab 4: Statistics
with tab4:
    st.header("Data Statistics")
    
    active_data = st.session_state.get("era5_data")
    active_meta = st.session_state.get("era5_metadata")
    
    if active_data is None:
        st.warning("⚠️ No dataset is currently active. Please select one from the Sample Datasets tab.")
    else:
        st.success(f"📊 Statistics for: **{active_meta.get('name', 'Unknown')}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stat_var = st.selectbox(
                "Variable",
                options=list(active_data.data_vars),
                key="stat_var"
            )
        
        with col2:
            if "level" in active_data.coords:
                stat_level = st.selectbox(
                    "Level (hPa)",
                    options=sorted([int(l) for l in active_data.level.values]),
                    key="stat_level"
                )
            else:
                stat_level = None
        
        if st.button("Compute Statistics", type="primary"):
            with st.spinner("Computing statistics..."):
                try:
                    if stat_level is not None:
                        data = active_data[stat_var].sel(level=stat_level).values
                    else:
                        data = active_data[stat_var].values
                    
                    values = data.flatten()
                    values = values[~np.isnan(values)]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{np.mean(values):.2f}")
                    col2.metric("Std Dev", f"{np.std(values):.2f}")
                    col3.metric("Min", f"{np.min(values):.2f}")
                    col4.metric("Max", f"{np.max(values):.2f}")
                    
                    # Histogram
                    sample_size = min(50000, len(values))
                    sample_values = np.random.choice(values, sample_size, replace=False)
                    
                    fig = go.Figure(data=go.Histogram(x=sample_values, nbinsx=50))
                    
                    title = f"Distribution of {stat_var}"
                    if stat_level:
                        title += f" at {stat_level} hPa"
                    title += " (Real ERA5 Data)"
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title="Value",
                        yaxis_title="Count",
                        height=400,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error computing statistics: {e}")


# Tab 5: About ERA5
with tab5:
    st.header("About ERA5 Reanalysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is ERA5?")
        st.markdown("""
        **ERA5** is the fifth generation ECMWF atmospheric reanalysis of the global climate.
        
        - Produced by the **Copernicus Climate Change Service (C3S)**
        - Hourly data from **1940 to present**
        - Original resolution: **0.25° × 0.25°** (~31 km)
        - **137 vertical levels** to 0.01 hPa
        - Uses advanced **4D-Var data assimilation**
        
        **This is NOT synthetic data** - these are actual atmospheric observations
        combined with state-of-the-art numerical weather prediction models.
        """)
        
        st.subheader("WeatherBench2")
        st.markdown("""
        This app uses ERA5 data from **WeatherBench2**, a benchmark dataset
        for data-driven weather forecasting.
        
        - Regridded to **5.625° × 2.8125°** (64×32 grid)
        - **6-hourly** temporal resolution
        - Optimized for **machine learning** applications
        - Free and public access via **Google Cloud Storage**
        """)
    
    with col2:
        st.subheader("Variables Available")
        
        st.markdown("""
        | Variable | Description |
        |----------|-------------|
        | `temperature` | Air temperature at pressure levels (K) |
        | `geopotential` | Geopotential height (m²/s²) |
        | `u_component_of_wind` | Eastward wind (m/s) |
        | `v_component_of_wind` | Northward wind (m/s) |
        | `specific_humidity` | Water vapor mixing ratio (kg/kg) |
        """)
        
        st.subheader("Citation")
        st.markdown("""
        > Hersbach, H., Bell, B., Berrisford, P., et al. (2020). 
        > The ERA5 global reanalysis. 
        > *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
        """)
        
        st.subheader("Data Source Verification")
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | **Source** | WeatherBench2 / ECMWF ERA5 |
        | **Type** | REAL atmospheric reanalysis |
        | **Synthetic** | ❌ NO |
        | **License** | Copernicus Climate Data Store |
        """)


# Sidebar: Current Data Status
st.sidebar.header("📊 Current Data Status")

active_data = st.session_state.get("era5_data")
active_meta = st.session_state.get("era5_metadata")

if active_data is not None:
    st.sidebar.success("✅ Data Active")
    st.sidebar.markdown(f"**Dataset:** {active_meta.get('name', 'Unknown')}")
    st.sidebar.markdown(f"**Period:** {active_meta.get('start_date', '?')} to {active_meta.get('end_date', '?')}")
    st.sidebar.markdown(f"**Source:** {active_meta.get('source', 'ERA5')}")
    st.sidebar.markdown(f"**Synthetic:** ❌ NO")
    
    if "time_steps" in active_meta:
        st.sidebar.markdown(f"**Time Steps:** {active_meta['time_steps']}")
else:
    st.sidebar.warning("⚠️ No Data Active")
    st.sidebar.markdown("Select a dataset from the Sample Datasets tab.")

st.sidebar.markdown("---")

# Downloaded samples count
downloaded = get_downloaded_samples()
st.sidebar.markdown(f"**Downloaded Samples:** {len(downloaded)}/{len(SAMPLE_DATASETS)}")

if downloaded:
    st.sidebar.markdown("Available:")
    for sid in downloaded:
        st.sidebar.markdown(f"• {SAMPLE_DATASETS[sid]['name']}")

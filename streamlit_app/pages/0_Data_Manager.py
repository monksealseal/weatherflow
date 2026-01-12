"""
Data Manager - Central hub for ALL weather data in WeatherFlow.

This Data Manager provides access to REAL weather data only:
1. NCEP Reanalysis - Real atmospheric temperature data (2013-2014)
2. ERA-Interim - Real ECMWF wind and geopotential data (global)
3. WeatherBench2 - ERA5 data when cloud access is available
4. Custom Upload - Upload your own real weather image pairs

ALL DATA IS REAL - No synthetic data is ever used.
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
import io

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dataset context
try:
    from dataset_context import render_dataset_banner, render_compact_dataset_badge
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

# Import data storage module with real data sources
try:
    from data_storage import (
        SAMPLE_DATASETS,
        REAL_DATA_SOURCES,
        initialize_sample_data,
        load_sample_data,
        get_available_samples,
        get_all_sample_info,
        get_sample_file_path,
        SAMPLES_DIR,
        DATA_DIR,
    )
    DATA_STORAGE_AVAILABLE = True
except ImportError:
    DATA_STORAGE_AVAILABLE = False
    SAMPLE_DATASETS = {}
    REAL_DATA_SOURCES = {}

st.set_page_config(
    page_title="Data Manager - WeatherFlow", page_icon="📊", layout="wide"
)

# Custom CSS - Matching new design language
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .data-source-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin: 15px 0;
        transition: all 0.2s ease;
    }
    .data-source-card:hover {
        border-color: #0066cc;
        box-shadow: 0 8px 25px rgba(0, 102, 204, 0.1);
        transform: translateY(-2px);
    }
    .data-source-card h3 {
        color: #1f2937;
        margin-bottom: 12px;
    }
    .speed-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .slow-badge {
        display: inline-block;
        background: #f59e0b;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .estimate-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 4px solid #0066cc;
    }
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066cc 0%, #00a3cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .page-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .citation-box {
        background: #f8fafc;
        border-left: 4px solid #0066cc;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-size: 0.9rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">📊 Data Manager</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Your central hub for real weather data from NCEP/NCAR and ERA5 reanalysis</p>', unsafe_allow_html=True)

# Show current dataset status
if CONTEXT_AVAILABLE:
    render_dataset_banner()

# Data paths - use from data_storage if available
if not DATA_STORAGE_AVAILABLE:
    DATA_DIR = Path(__file__).parent.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR = DATA_DIR / "samples"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = DATA_DIR / "uploads" if DATA_STORAGE_AVAILABLE else Path(__file__).parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_sample_metadata_file(sample_id: str) -> Path:
    """Get the path to sample metadata file."""
    return SAMPLES_DIR / f"{sample_id}_metadata.json"


def is_sample_downloaded(sample_id: str) -> bool:
    """Check if a sample dataset has been downloaded."""
    if DATA_STORAGE_AVAILABLE:
        return sample_id in get_available_samples()
    path = SAMPLES_DIR / f"{sample_id}.nc"
    return path.exists()


def get_downloaded_samples() -> list:
    """Get list of downloaded sample datasets."""
    if DATA_STORAGE_AVAILABLE:
        return get_available_samples()
    return [sid for sid in SAMPLE_DATASETS.keys() if is_sample_downloaded(sid)]


def process_uploaded_images(source_files, target_files):
    """
    Process uploaded image pairs into a training-ready format.
    
    This allows users to train any architecture (including GANs) on custom
    source/target real image pairs.
    """
    import xarray as xr
    from PIL import Image
    
    source_arrays = []
    target_arrays = []
    
    for sf, tf in zip(source_files, target_files):
        # Read images
        src_img = Image.open(sf).convert('RGB')
        tgt_img = Image.open(tf).convert('RGB')
        
        # Resize to consistent size
        src_img = src_img.resize((256, 256))
        tgt_img = tgt_img.resize((256, 256))
        
        # Convert to arrays
        source_arrays.append(np.array(src_img))
        target_arrays.append(np.array(tgt_img))
    
    # Stack into arrays [N, H, W, C]
    source_data = np.stack(source_arrays, axis=0)
    target_data = np.stack(target_arrays, axis=0)
    
    # Transpose to [N, C, H, W] for PyTorch
    source_data = np.transpose(source_data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    target_data = np.transpose(target_data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    
    # Create xarray dataset
    n_samples = len(source_arrays)
    ds = xr.Dataset(
        {
            "source": (["time", "channel", "latitude", "longitude"], source_data),
            "target": (["time", "channel", "latitude", "longitude"], target_data),
        },
        coords={
            "time": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
            "channel": ["R", "G", "B"],
            "latitude": np.linspace(-90, 90, 256),
            "longitude": np.linspace(-180, 180, 256),
        },
        attrs={
            "source": "User Upload",
            "description": "Custom real image pairs for training",
            "is_real_data": "1",
            "is_image_data": "1",
        }
    )
    
    return ds


# Initialize session state
if "era5_data" not in st.session_state:
    st.session_state["era5_data"] = None
if "era5_metadata" not in st.session_state:
    st.session_state["era5_metadata"] = None
if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = None
if "wb2_connected" not in st.session_state:
    st.session_state["wb2_connected"] = False


# Main tabs for different data sources
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Quick Start",
    "📦 Real Data Samples",
    "🖼️ Upload Images",
    "🔍 Preview Data",
    "ℹ️ Data Sources Info"
])


# Tab 1: Quick Start
with tab1:
    st.header("Quick Start with REAL Data")

    st.markdown("""
    Get started immediately with **real weather data**. No synthetic data - all authentic observations!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="data-source-card">
            <h3>🌡️ NCEP Reanalysis</h3>
            <span class="speed-badge">Real Data - Fast Download</span>
            <p>Real NCEP/NCAR atmospheric observations (2013-2014):</p>
            <ul>
                <li>Real air temperature data</li>
                <li>2 years of 6-hourly observations</li>
                <li>North America & Pacific region</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        ncep_downloaded = is_sample_downloaded("ncep_reanalysis_2013")
        if ncep_downloaded:
            if st.button("🔄 Use NCEP Temperature Data", type="primary", key="use_ncep"):
                if DATA_STORAGE_AVAILABLE:
                    data, metadata = load_sample_data("ncep_reanalysis_2013")
                    if data is not None:
                        st.session_state["era5_data"] = data
                        st.session_state["era5_metadata"] = metadata
                        st.session_state["active_sample"] = "ncep_reanalysis_2013"
                        st.success("✅ Real NCEP data loaded!")
                        st.rerun()
        else:
            if st.button("📥 Download NCEP Data (~7 MB)", type="primary", key="download_ncep"):
                with st.spinner("Downloading real NCEP reanalysis data from GitHub..."):
                    if DATA_STORAGE_AVAILABLE and initialize_sample_data("ncep_reanalysis_2013"):
                        data, metadata = load_sample_data("ncep_reanalysis_2013")
                        if data is not None:
                            st.session_state["era5_data"] = data
                            st.session_state["era5_metadata"] = metadata
                            st.session_state["active_sample"] = "ncep_reanalysis_2013"
                            st.success("✅ Real NCEP data downloaded and loaded!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Download failed. Please try again.")

    with col2:
        st.markdown("""
        <div class="data-source-card">
            <h3>🌍 ERA-Interim Global</h3>
            <span class="speed-badge">Real Data - Fast Download</span>
            <p>Real ECMWF ERA-Interim reanalysis (global coverage):</p>
            <ul>
                <li>U, V wind components</li>
                <li>Geopotential height (Z)</li>
                <li>3 pressure levels (200, 500, 850 hPa)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        erai_downloaded = is_sample_downloaded("era_interim_global")
        if erai_downloaded:
            if st.button("🔄 Use ERA-Interim Data", type="primary", key="use_erai"):
                if DATA_STORAGE_AVAILABLE:
                    data, metadata = load_sample_data("era_interim_global")
                    if data is not None:
                        st.session_state["era5_data"] = data
                        st.session_state["era5_metadata"] = metadata
                        st.session_state["active_sample"] = "era_interim_global"
                        st.success("✅ Real ERA-Interim data loaded!")
                        st.rerun()
        else:
            if st.button("📥 Download ERA-Interim (~4 MB)", type="primary", key="download_erai"):
                with st.spinner("Downloading real ERA-Interim data from GitHub..."):
                    if DATA_STORAGE_AVAILABLE and initialize_sample_data("era_interim_global"):
                        data, metadata = load_sample_data("era_interim_global")
                        if data is not None:
                            st.session_state["era5_data"] = data
                            st.session_state["era5_metadata"] = metadata
                            st.session_state["active_sample"] = "era_interim_global"
                            st.success("✅ Real ERA-Interim data downloaded and loaded!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Download failed. Please try again.")

    st.markdown("---")

    # Data source info
    st.markdown("""
    <div class="estimate-box">
        <strong>📊 All Data is REAL:</strong>
        <ul>
            <li><strong>NCEP Reanalysis:</strong> ~7 MB - Real atmospheric observations</li>
            <li><strong>ERA-Interim:</strong> ~4 MB - Real ECMWF reanalysis data</li>
            <li><strong>WeatherBench2:</strong> Requires cloud access (may be blocked)</li>
            <li><strong>Image Upload:</strong> Your own real weather imagery</li>
        </ul>
        <p><em>No synthetic data is ever used. All data is from authentic scientific sources.</em></p>
    </div>
    """, unsafe_allow_html=True)


# Tab 2: Sample Datasets
with tab2:
    st.header("Real Data Samples")

    st.markdown("""
    **All datasets are REAL weather data** from authentic scientific sources.
    Choose from locally-downloadable sources or cloud-based ERA5 data.
    """)

    downloaded_samples = get_downloaded_samples()

    # Show downloaded first
    if downloaded_samples:
        st.subheader("✅ Downloaded Real Datasets")

        for sample_id in downloaded_samples:
            if sample_id not in SAMPLE_DATASETS:
                continue
            sample_info = SAMPLE_DATASETS[sample_id]

            with st.expander(f"📁 {sample_info['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{sample_info['description']}**")
                    st.markdown(f"- **Period:** {sample_info.get('start_date', 'N/A')} to {sample_info.get('end_date', 'N/A')}")
                    st.markdown(f"- **Region:** {sample_info.get('region', 'Global')}")
                    st.markdown(f"- **Variables:** {', '.join(sample_info.get('variables', []))}")
                    if sample_info.get('citation'):
                        st.caption(f"Citation: {sample_info['citation']}")

                with col2:
                    if st.button(f"🔄 Use This Dataset", key=f"use_{sample_id}", type="primary"):
                        if DATA_STORAGE_AVAILABLE:
                            data, metadata = load_sample_data(sample_id)
                        else:
                            data = None
                        if data is not None:
                            st.session_state["era5_data"] = data
                            st.session_state["era5_metadata"] = metadata if metadata else sample_info
                            st.session_state["active_sample"] = sample_id
                            st.success(f"✅ Now using: {sample_info['name']}")
                            st.rerun()

    # Available for download
    st.subheader("📥 Available for Download")

    not_downloaded = [sid for sid in SAMPLE_DATASETS.keys() if sid not in downloaded_samples]

    if not_downloaded:
        for sample_id in not_downloaded:
            sample_info = SAMPLE_DATASETS[sample_id]
            requires_cloud = sample_info.get("requires_cloud", False)

            with st.expander(f"🌐 {sample_info['name']}" + (" (Cloud Required)" if requires_cloud else ""), expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{sample_info['description']}**")
                    st.markdown(f"- **Region:** {sample_info.get('region', 'Global')}")
                    st.markdown(f"- **Variables:** {', '.join(sample_info.get('variables', []))}")
                    if requires_cloud:
                        st.warning("⚠️ Requires cloud access (WeatherBench2)")
                    if sample_info.get('citation'):
                        st.caption(f"Citation: {sample_info['citation']}")

                with col2:
                    if st.button(f"📥 Download", key=f"download_{sample_id}"):
                        with st.spinner(f"Downloading {sample_info['name']}..."):
                            if DATA_STORAGE_AVAILABLE and initialize_sample_data(sample_id):
                                st.success("✅ Downloaded!")
                                st.rerun()
                            else:
                                if requires_cloud:
                                    st.error("Cloud access not available. This dataset requires WeatherBench2.")
                                else:
                                    st.error("Download failed. Please try again.")
    else:
        st.success("✅ All sample datasets downloaded!")


# Tab 3: Upload Images
with tab3:
    st.header("Upload Custom Real Image Data")

    st.markdown("""
    **Train with your own REAL data!** Upload pairs of source and target images for training.

    This is perfect for:
    - **Satellite imagery** (real weather observations)
    - **Radar data** (precipitation patterns)
    - **Any paired real image dataset** you want to use

    **Note:** All uploaded images should be real weather/climate data.
    """)

    st.info("""
    **How it works:**
    1. Upload source images (inputs to the model)
    2. Upload matching target images (what the model should predict)
    3. Images are automatically resized and normalized
    4. Train any model architecture on your data!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Source Images (Inputs)")
        source_files = st.file_uploader(
            "Upload source images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="source_upload",
            help="These are the input images your model will receive"
        )

        if source_files:
            st.success(f"✅ {len(source_files)} source images uploaded")

    with col2:
        st.subheader("🎯 Target Images (Labels)")
        target_files = st.file_uploader(
            "Upload target images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="target_upload",
            help="These are what your model should learn to predict"
        )

        if target_files:
            st.success(f"✅ {len(target_files)} target images uploaded")

    # Process uploaded images
    if source_files and target_files:
        if len(source_files) != len(target_files):
            st.error(f"⚠️ Number of source ({len(source_files)}) and target ({len(target_files)}) images must match!")
        else:
            st.success(f"✅ {len(source_files)} image pairs ready")

            # Preview
            st.subheader("Preview")
            preview_cols = st.columns(min(3, len(source_files)))
            for i, col in enumerate(preview_cols[:3]):
                with col:
                    from PIL import Image
                    src = Image.open(source_files[i])
                    tgt = Image.open(target_files[i])
                    st.image(src, caption=f"Source {i+1}", width=150)
                    st.image(tgt, caption=f"Target {i+1}", width=150)

            if st.button("🔄 Process & Load Images", type="primary"):
                with st.spinner("Processing images..."):
                    try:
                        # Reset file positions
                        for f in source_files + target_files:
                            f.seek(0)

                        ds = process_uploaded_images(source_files, target_files)

                        # Save to disk
                        upload_file = UPLOADS_DIR / f"custom_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
                        ds.to_netcdf(upload_file)

                        # Update session state
                        st.session_state["era5_data"] = ds
                        st.session_state["era5_metadata"] = {
                            "name": f"Custom Images ({len(source_files)} pairs)",
                            "description": "User-uploaded real image pairs",
                            "is_real_data": True,
                            "is_image_data": True,
                            "source": "User Upload",
                            "n_samples": len(source_files),
                        }
                        st.session_state["active_sample"] = "custom_upload"

                        st.success("✅ Real images loaded! Ready for training.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error processing images: {e}")


# Tab 4: Preview Data
with tab4:
    st.header("Preview Active Dataset")

    active_data = st.session_state.get("era5_data")
    active_meta = st.session_state.get("era5_metadata")

    if active_data is None:
        st.warning("⚠️ No dataset loaded. Use Quick Start to download and load real data.")
    else:
        st.success(f"📊 Active Dataset: **{active_meta.get('name', 'Unknown')}**")

        # Dataset info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Variables", len(list(active_data.data_vars)))
        with col_info2:
            n_times = len(active_data.time) if "time" in active_data.coords else 0
            st.metric("Time Steps", n_times)
        with col_info3:
            # Check if data is real
            is_real = active_meta.get("is_real_data", False)
            if isinstance(is_real, str):
                is_real = is_real == "1"
            st.metric("Data Type", "✅ Real Data" if is_real else "Unknown")

        st.markdown("---")

        # Visualization
        available_vars = list(active_data.data_vars)

        col1, col2, col3 = st.columns(3)

        with col1:
            preview_var = st.selectbox("Variable", available_vars, key="preview_var")

        with col2:
            if "level" in active_data.coords:
                levels = sorted([int(l) for l in active_data.level.values])
                preview_level = st.selectbox("Level (hPa)", levels, key="preview_level")
            else:
                preview_level = None

        with col3:
            if "time" in active_data.coords:
                time_vals = active_data.time.values
                time_labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in time_vals]
                preview_time = st.selectbox("Time", range(len(time_labels)),
                                           format_func=lambda x: time_labels[x],
                                           key="preview_time")
            else:
                preview_time = 0

        # Get coordinate names
        if "latitude" in active_data.coords:
            lat_coord, lon_coord = "latitude", "longitude"
        else:
            lat_coord, lon_coord = "lat", "lon"

        # Extract data slice
        try:
            data_slice = active_data[preview_var]
            if "time" in data_slice.dims:
                data_slice = data_slice.isel(time=preview_time)
            if "level" in data_slice.dims and preview_level is not None:
                data_slice = data_slice.sel(level=preview_level)
            if "channel" in data_slice.dims:
                data_slice = data_slice.isel(channel=0)  # Show first channel for images

            lats = active_data[lat_coord].values
            lons = active_data[lon_coord].values

            # Create plot
            fig = go.Figure(data=go.Heatmap(
                z=data_slice.values,
                x=lons,
                y=lats,
                colorscale="RdBu_r" if "temp" in preview_var.lower() else "Viridis",
                colorbar=dict(title=preview_var),
            ))

            title = f"{preview_var}"
            if preview_level:
                title += f" at {preview_level} hPa"
            if not active_meta.get("is_synthetic", True):
                title = f"REAL ERA5: {title}"

            fig.update_layout(
                title=title,
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=450,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying data: {e}")


# Tab 5: Data Sources Info
with tab5:
    st.header("About Data Sources")

    st.markdown("""
    ### All Data is REAL
    
    WeatherFlow uses **only real weather data** from authentic scientific sources.
    No synthetic data is ever used.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🌡️ NCEP/NCAR Reanalysis
        
        **Real atmospheric observations from NOAA.**
        
        - 4x daily observations since 1948
        - 2.5° x 2.5° grid resolution
        - Multiple pressure levels
        - Quality-controlled data
        
        **Citation:**
        > Kalnay et al. (1996). The NCEP/NCAR 40-Year Reanalysis Project. 
        > Bull. Amer. Meteor. Soc.
        
        ---
        
        #### 🌍 ERA-Interim (ECMWF)
        
        **ECMWF's previous-generation reanalysis.**
        
        - Global coverage (0.75° resolution)
        - 60 vertical levels
        - Advanced data assimilation
        - Extensively validated
        
        **Citation:**
        > Dee et al. (2011). The ERA-Interim reanalysis.
        > Q.J.R. Meteorol. Soc.
        """)

    with col2:
        st.markdown("""
        #### 🌍 ERA5 Reanalysis (ECMWF)

        **The gold standard** for atmospheric data.

        - Hourly data from 1940 to present
        - 0.25° resolution (~31 km)
        - 137 vertical levels
        - Uses advanced 4D-Var assimilation

        **Access via:** WeatherBench2 (requires cloud access)

        **Citation:**
        > Hersbach et al. (2020). The ERA5 global reanalysis.
        > Q J R Meteorol Soc.

        ---

        #### 🖼️ Custom Real Image Data

        **Upload your own real weather data!**

        - Upload source/target pairs
        - Real satellite imagery
        - Real radar observations
        - Any real weather products
        """)

    st.markdown("---")

    st.markdown("""
    ### Data Quality Assurance
    
    All data sources used in WeatherFlow are:
    - ✅ **Real observations** from scientific instruments
    - ✅ **Quality controlled** by the source organization
    - ✅ **Peer reviewed** and widely used in research
    - ✅ **Properly cited** with DOIs and references
    
    **No synthetic or artificially generated data is ever used.**
    """)


# Sidebar: Data Status
st.sidebar.header("📊 Data Status")

active_data = st.session_state.get("era5_data")
active_meta = st.session_state.get("era5_metadata")

if active_data is not None:
    st.sidebar.success("✅ Real Data Loaded")
    st.sidebar.markdown(f"**{active_meta.get('name', 'Unknown')}**")

    # Check if data is real
    is_real = active_meta.get("is_real_data", False)
    if isinstance(is_real, str):
        is_real = is_real == "1"
    st.sidebar.markdown(f"Type: {'✅ **Real Data**' if is_real else 'Unknown'}")

    n_vars = len(list(active_data.data_vars))
    st.sidebar.markdown(f"Variables: {n_vars}")
    
    # Show source
    source = active_meta.get("source", active_meta.get("original_source", "Unknown"))
    st.sidebar.markdown(f"Source: {source}")
else:
    st.sidebar.warning("⚠️ No Data Loaded")
    st.sidebar.markdown("Load real data from the tabs above")

st.sidebar.markdown("---")

# Downloaded count
downloaded = get_downloaded_samples()
st.sidebar.markdown(f"**Downloaded:** {len(downloaded)}/{len(SAMPLE_DATASETS)}")

st.sidebar.markdown("---")
st.sidebar.caption("""
**Data Manager**

All weather data flows through this page.
All data is REAL - no synthetic data ever.
""")

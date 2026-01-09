"""
Data Manager - Central hub for ALL weather data in WeatherFlow.

This enhanced Data Manager supports multiple data sources:
1. Sample Datasets - Pre-bundled ERA5 samples for key weather events
2. WeatherBench2 - Custom ERA5 downloads from Google Cloud
3. Dynamical.org - GEFS and other ensemble forecasts (fast downloads)
4. Custom Upload - Upload your own image pairs (source/target for GAN training)

All data flows through this single page to ensure consistency.
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

st.set_page_config(
    page_title="Data Manager - WeatherFlow", page_icon="📊", layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .data-source-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #1e88e5;
        margin: 15px 0;
        transition: transform 0.2s;
    }
    .data-source-card:hover {
        transform: translateX(5px);
    }
    .speed-badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .slow-badge {
        display: inline-block;
        background: #ff9800;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .estimate-box {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Data Manager")

st.markdown("""
**Your central hub for weather data.** Choose from multiple sources, upload custom data,
or use our pre-bundled samples to get started quickly.
""")

# Show current dataset status
if CONTEXT_AVAILABLE:
    render_dataset_banner()

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# WeatherBench2 ERA5 URL
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Pre-defined sample datasets
SAMPLE_DATASETS = {
    "tropical_cyclones_2005": {
        "name": "Hurricane Katrina (2005)",
        "description": "Category 5 hurricane, active Atlantic season",
        "start_date": "2005-08-15",
        "end_date": "2005-09-30",
        "region": "Atlantic Basin",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200],
        "events": ["Hurricane Katrina (Aug 23-31)", "Hurricane Rita (Sep 17-26)"],
        "download_time_estimate": "2-5 minutes",
        "size_estimate": "~50 MB",
    },
    "heat_wave_europe_2003": {
        "name": "European Heat Wave (2003)",
        "description": "Record heat wave affecting Western Europe",
        "start_date": "2003-07-20",
        "end_date": "2003-08-20",
        "region": "Europe",
        "variables": ["temperature", "geopotential"],
        "pressure_levels": [1000, 850, 500],
        "events": ["Peak temperatures Aug 4-13"],
        "download_time_estimate": "1-3 minutes",
        "size_estimate": "~30 MB",
    },
    "polar_vortex_2019": {
        "name": "Polar Vortex (2019)",
        "description": "Extreme cold outbreak in North America",
        "start_date": "2019-01-20",
        "end_date": "2019-02-10",
        "region": "North America",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200, 50],
        "events": ["Record lows Jan 27-31"],
        "download_time_estimate": "2-4 minutes",
        "size_estimate": "~45 MB",
    },
    "quick_demo": {
        "name": "Quick Demo (Synthetic)",
        "description": "Small synthetic dataset for fast testing - loads instantly!",
        "start_date": "2023-01-01",
        "end_date": "2023-01-07",
        "region": "Global",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [850, 500],
        "events": ["Synthetic data for demos"],
        "download_time_estimate": "Instant",
        "size_estimate": "~2 MB",
        "is_demo": True,
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


def generate_quick_demo_data():
    """Generate synthetic data for quick demos."""
    import xarray as xr

    # Create realistic synthetic weather data
    np.random.seed(42)

    # Grid dimensions
    n_lat, n_lon = 32, 64
    n_time = 28  # 7 days * 4 (6-hourly)
    n_levels = 2

    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(-180, 180, n_lon)
    times = pd.date_range("2023-01-01", periods=n_time, freq="6H")
    levels = [850, 500]

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Temperature: latitude gradient + diurnal cycle + synoptic waves
    temp_base = 288 - 40 * np.abs(lat_grid) / 90
    temperature = np.zeros((n_time, n_levels, n_lat, n_lon))
    for t in range(n_time):
        for l, level in enumerate(levels):
            lapse = (1000 - level) * 0.0065 * 1000 / 100  # Approximate lapse rate
            diurnal = 3 * np.sin(2 * np.pi * t / 4)
            synoptic = 5 * np.sin(np.radians(lon_grid) * 3 + t * 0.2)
            temperature[t, l] = temp_base - lapse + diurnal + synoptic + np.random.randn(n_lat, n_lon) * 2

    # Geopotential height
    z_base = 5500 + 200 * np.cos(np.radians(lat_grid))
    geopotential = np.zeros((n_time, n_levels, n_lat, n_lon))
    for t in range(n_time):
        for l, level in enumerate(levels):
            height_offset = (1000 - level) * 8  # ~8m per hPa
            wave = 100 * np.sin(np.radians(lon_grid) * 3 + t * 0.15)
            geopotential[t, l] = z_base + height_offset + wave + np.random.randn(n_lat, n_lon) * 20

    # Wind components (geostrophic-like)
    u_wind = 10 * np.sin(np.radians(lat_grid) * 2) + np.random.randn(n_time, n_levels, n_lat, n_lon) * 3
    v_wind = 5 * np.sin(np.radians(lon_grid) * 3) + np.random.randn(n_time, n_levels, n_lat, n_lon) * 2

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "temperature": (["time", "level", "latitude", "longitude"], temperature.astype(np.float32)),
            "geopotential": (["time", "level", "latitude", "longitude"], geopotential.astype(np.float32)),
            "u_component_of_wind": (["time", "level", "latitude", "longitude"], u_wind.astype(np.float32)),
            "v_component_of_wind": (["time", "level", "latitude", "longitude"], v_wind.astype(np.float32)),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "source": "WeatherFlow Synthetic Demo",
            "description": "Quick demo data for testing",
            "is_synthetic": True,
        }
    )

    return ds


def connect_to_weatherbench2():
    """Connect to WeatherBench2."""
    import xarray as xr

    ds = None
    method_used = None
    error_msg = None

    try:
        ds = xr.open_zarr(
            WEATHERBENCH2_URL, storage_options={"anon": True}, consolidated=True
        )
        method_used = "GCS Anonymous"
    except Exception as e1:
        error_msg = str(e1)

        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token="anon")
            mapper = fs.get_mapper(WEATHERBENCH2_URL)
            ds = xr.open_zarr(mapper, consolidated=True)
            method_used = "gcsfs"
        except Exception as e2:
            error_msg = str(e2)

    if ds is not None:
        metadata = {
            "source": "WeatherBench2 ERA5 (ECMWF Reanalysis)",
            "url": WEATHERBENCH2_URL,
            "is_synthetic": False,
            "connection_method": method_used,
        }
        return ds, metadata, None

    return None, None, error_msg


def download_sample_dataset(sample_id: str, ds, progress_callback=None):
    """Download a sample dataset from WeatherBench2."""
    import xarray as xr

    sample_info = SAMPLE_DATASETS[sample_id]

    variables = sample_info["variables"]
    available_vars = [v for v in variables if v in ds.data_vars]

    if not available_vars:
        raise ValueError(f"None of the requested variables are available")

    if progress_callback:
        progress_callback(10, "Selecting data subset...")

    subset = ds[available_vars].sel(
        time=slice(sample_info["start_date"], sample_info["end_date"])
    )

    if "level" in ds.coords:
        available_levels = [l for l in sample_info["pressure_levels"] if l in ds.level.values]
        if available_levels:
            subset = subset.sel(level=available_levels)

    if progress_callback:
        progress_callback(30, "Downloading data (this may take a while)...")

    loaded_data = subset.load()

    if progress_callback:
        progress_callback(80, "Saving to disk...")

    data_file = get_sample_data_file(sample_id)
    loaded_data.to_netcdf(data_file)

    metadata = {
        **sample_info,
        "downloaded_at": datetime.now().isoformat(),
        "source": "WeatherBench2 ERA5 (ECMWF Reanalysis)",
        "is_synthetic": False,
        "file_size_mb": data_file.stat().st_size / (1024 * 1024),
        "time_steps": len(loaded_data.time),
    }

    with open(get_sample_metadata_file(sample_id), "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback(100, "Complete!")

    return loaded_data


def process_uploaded_images(source_files, target_files):
    """
    Process uploaded image pairs into a training-ready format.

    This allows users to train any architecture (including GANs) on custom
    source/target image pairs.
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
            "description": "Custom image pairs for training",
            "is_synthetic": False,
            "is_image_data": True,
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
    "📦 Sample Datasets",
    "🖼️ Upload Images",
    "🔍 Preview Data",
    "ℹ️ Data Sources Info"
])


# Tab 1: Quick Start
with tab1:
    st.header("Quick Start")

    st.markdown("""
    Get started immediately with one of these options. **Recommended for new users!**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="data-source-card">
            <h3>⚡ Instant Demo</h3>
            <span class="speed-badge">Loads Instantly</span>
            <p>Synthetic weather data that loads immediately. Perfect for:</p>
            <ul>
                <li>Testing the platform</li>
                <li>Quick model training demos</li>
                <li>Learning the workflow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Load Demo Data Now", type="primary", key="load_demo"):
            with st.spinner("Generating demo data..."):
                demo_data = generate_quick_demo_data()

                # Save to disk
                demo_file = get_sample_data_file("quick_demo")
                demo_data.to_netcdf(demo_file)

                # Update session state
                st.session_state["era5_data"] = demo_data
                st.session_state["era5_metadata"] = {
                    "name": "Quick Demo (Synthetic)",
                    "description": "Synthetic data for fast testing",
                    "is_synthetic": True,
                    "source": "WeatherFlow Generated",
                    "variables": list(demo_data.data_vars),
                    "time_steps": len(demo_data.time),
                }
                st.session_state["active_sample"] = "quick_demo"

            st.success("✅ Demo data loaded! You can now train models or explore visualizations.")
            st.balloons()
            time.sleep(1)
            st.rerun()

    with col2:
        st.markdown("""
        <div class="data-source-card">
            <h3>🌪️ Real Weather Event</h3>
            <span class="slow-badge">2-5 min download</span>
            <p>Real ERA5 reanalysis data from a famous weather event:</p>
            <ul>
                <li>Hurricane Katrina (2005)</li>
                <li>Actual atmospheric observations</li>
                <li>ECMWF gold standard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        downloaded = is_sample_downloaded("tropical_cyclones_2005")
        if downloaded:
            if st.button("🔄 Use Hurricane Katrina Data", type="primary", key="use_katrina"):
                data = load_sample_data("tropical_cyclones_2005")
                st.session_state["era5_data"] = data
                st.session_state["era5_metadata"] = {
                    **SAMPLE_DATASETS["tropical_cyclones_2005"],
                    "is_synthetic": False,
                    "source": "WeatherBench2 ERA5",
                }
                st.session_state["active_sample"] = "tropical_cyclones_2005"
                st.success("✅ Hurricane Katrina data loaded!")
                st.rerun()
        else:
            if st.button("📥 Download Hurricane Katrina", key="download_katrina"):
                st.session_state["pending_katrina_download"] = True
                st.rerun()

            if st.session_state.get("pending_katrina_download", False):
                st.session_state["pending_katrina_download"] = False

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct / 100)
                    status_text.text(msg)

                update_progress(0, "Connecting to WeatherBench2...")

                ds, metadata, error = connect_to_weatherbench2()

                if ds is None:
                    st.error(f"Could not connect: {error}")
                else:
                    try:
                        download_sample_dataset("tropical_cyclones_2005", ds, update_progress)
                        st.success("✅ Downloaded! Click 'Use Hurricane Katrina Data' to load.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")

    st.markdown("---")

    # Time estimates
    st.markdown("""
    <div class="estimate-box">
        <strong>⏱️ Time Estimates:</strong>
        <ul>
            <li><strong>Demo Data:</strong> Instant (generated locally)</li>
            <li><strong>Sample Datasets:</strong> 1-5 minutes (depends on internet speed)</li>
            <li><strong>Custom Downloads:</strong> 5-30 minutes (large date ranges)</li>
            <li><strong>Image Upload:</strong> Depends on file sizes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# Tab 2: Sample Datasets
with tab2:
    st.header("Pre-Defined Sample Datasets")

    st.markdown("""
    Curated weather events with real ERA5 data. These are designed to showcase
    interesting atmospheric phenomena.
    """)

    downloaded_samples = get_downloaded_samples()

    # Show downloaded first
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

                with col2:
                    if st.button(f"🔄 Use This Dataset", key=f"use_{sample_id}", type="primary"):
                        data = load_sample_data(sample_id)
                        if data is not None:
                            st.session_state["era5_data"] = data
                            st.session_state["era5_metadata"] = {
                                **sample_info,
                                "is_synthetic": sample_info.get("is_demo", False),
                                "source": "WeatherBench2 ERA5" if not sample_info.get("is_demo") else "Synthetic",
                            }
                            st.session_state["active_sample"] = sample_id
                            st.success(f"✅ Now using: {sample_info['name']}")
                            st.rerun()

    # Available for download
    st.subheader("📥 Available for Download")

    not_downloaded = [sid for sid in SAMPLE_DATASETS.keys()
                      if sid not in downloaded_samples and not SAMPLE_DATASETS[sid].get("is_demo")]

    if not_downloaded:
        for sample_id in not_downloaded:
            sample_info = SAMPLE_DATASETS[sample_id]

            with st.expander(f"🌐 {sample_info['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{sample_info['description']}**")
                    st.markdown(f"- **Period:** {sample_info['start_date']} to {sample_info['end_date']}")
                    st.markdown(f"- **Est. Download:** {sample_info.get('download_time_estimate', '2-5 min')}")
                    st.markdown(f"- **Est. Size:** {sample_info.get('size_estimate', '~50 MB')}")

                with col2:
                    if st.button(f"📥 Download", key=f"download_{sample_id}"):
                        st.session_state[f"pending_download_{sample_id}"] = True
                        st.rerun()

                    if st.session_state.get(f"pending_download_{sample_id}", False):
                        st.session_state[f"pending_download_{sample_id}"] = False

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(pct, msg):
                            progress_bar.progress(pct / 100)
                            status_text.text(msg)

                        ds, metadata, error = connect_to_weatherbench2()

                        if ds is None:
                            st.error(f"Could not connect: {error}")
                        else:
                            try:
                                download_sample_dataset(sample_id, ds, update_progress)
                                st.success(f"✅ Downloaded!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Download failed: {e}")
    else:
        st.success("✅ All sample datasets downloaded!")


# Tab 3: Upload Images
with tab3:
    st.header("Upload Custom Image Data")

    st.markdown("""
    **Train with your own data!** Upload pairs of source and target images for training.

    This is perfect for:
    - **GAN-style training** (image-to-image translation)
    - **Custom weather data** from satellites or radar
    - **Any paired image dataset** you want to use

    All WeatherFlow models can be trained on image data - just upload your source/target pairs.
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
                            "description": "User-uploaded image pairs",
                            "is_synthetic": False,
                            "is_image_data": True,
                            "source": "User Upload",
                            "n_samples": len(source_files),
                        }
                        st.session_state["active_sample"] = "custom_upload"

                        st.success("✅ Images loaded! Ready for training.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error processing images: {e}")


# Tab 4: Preview Data
with tab4:
    st.header("Preview Active Dataset")

    active_data = st.session_state.get("era5_data")
    active_meta = st.session_state.get("era5_metadata")

    if active_data is None:
        st.warning("⚠️ No dataset loaded. Use Quick Start or Sample Datasets to load data.")
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
            is_synthetic = active_meta.get("is_synthetic", True)
            st.metric("Data Type", "Synthetic" if is_synthetic else "Real ERA5")

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
    ### Available Data Sources

    WeatherFlow supports multiple data sources to fit different needs:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🌍 ERA5 Reanalysis (ECMWF)

        **The gold standard** for atmospheric data.

        - Hourly data from 1940 to present
        - 0.25° resolution (~31 km)
        - 137 vertical levels
        - Uses advanced 4D-Var assimilation

        **Access via:** WeatherBench2 (Google Cloud)

        **Citation:**
        > Hersbach et al. (2020). The ERA5 global reanalysis.
        > Q J R Meteorol Soc.

        ---

        #### 🌐 Dynamical.org

        **Modern weather data infrastructure.**

        - GEFS ensemble forecasts
        - Fast, developer-friendly access
        - Cloud-native formats (Zarr, Icechunk)

        **Learn more:** [dynamical.org/updates](https://dynamical.org/updates/)
        """)

    with col2:
        st.markdown("""
        #### 📊 WeatherBench2

        **Standardized ML benchmark** from Google Research.

        - Consistent evaluation metrics
        - Preprocessed for ML training
        - Multiple resolutions available
        - Free public access

        **Access:** `gs://weatherbench2/datasets/`

        ---

        #### 🖼️ Custom Image Data

        **Train on your own images!**

        - Upload source/target pairs
        - Supports PNG, JPG, TIFF
        - Automatic preprocessing
        - Works with all architectures

        **Best for:**
        - Satellite imagery
        - Radar data
        - Custom weather products
        """)

    st.markdown("---")

    st.markdown("""
    ### Advanced: ERA5 via Icechunk

    For advanced users who need direct ERA5 access with better performance,
    we recommend using the Icechunk format via Dynamical.org notebooks.

    **Example notebook:** [ECMWF IFS Ensemble via Icechunk](https://github.com/dynamical-org/notebooks/blob/main/ecmwf-ifs-ens-forecast-15-day-0-25-degree-icechunk.ipynb)

    **More notebooks:** [dynamical-org/notebooks](https://github.com/dynamical-org/notebooks)
    """)


# Sidebar: Data Status
st.sidebar.header("📊 Data Status")

active_data = st.session_state.get("era5_data")
active_meta = st.session_state.get("era5_metadata")

if active_data is not None:
    st.sidebar.success("✅ Data Loaded")
    st.sidebar.markdown(f"**{active_meta.get('name', 'Unknown')}**")

    is_synthetic = active_meta.get("is_synthetic", True)
    st.sidebar.markdown(f"Type: {'Synthetic' if is_synthetic else '**Real ERA5**'}")

    n_vars = len(list(active_data.data_vars))
    st.sidebar.markdown(f"Variables: {n_vars}")
else:
    st.sidebar.warning("⚠️ No Data Loaded")
    st.sidebar.markdown("Load data from the tabs above")

st.sidebar.markdown("---")

# Downloaded count
downloaded = get_downloaded_samples()
st.sidebar.markdown(f"**Downloaded:** {len(downloaded)}/{len(SAMPLE_DATASETS)}")

st.sidebar.markdown("---")
st.sidebar.caption("""
**Data Manager**

All weather data flows through this page.
Your active dataset is available across all pages.
""")

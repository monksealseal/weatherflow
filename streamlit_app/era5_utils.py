"""
ERA5 Data Utilities for WeatherFlow Streamlit App

This module provides shared utilities for accessing ERA5 data across all pages.
All pages should use this module to ensure consistent data handling.

Key features:
- Server-side data storage integration
- Automatic sample initialization
- Consistent data access patterns
- Clear data source indicators (REAL vs SYNTHETIC)
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Tuple, List, Dict

# Import data storage module
try:
    from data_storage import (
        load_sample_data,
        get_available_samples,
        get_all_sample_info,
        initialize_sample_data,
        get_data_status,
        SAMPLE_DATASETS,
        get_model_benchmarks,
    )
    DATA_STORAGE_AVAILABLE = True
except ImportError:
    DATA_STORAGE_AVAILABLE = False

# Data paths
DATA_DIR = Path(__file__).parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"


def get_active_era5_data():
    """
    Get the currently active ERA5 dataset.

    Returns:
        tuple: (xarray.Dataset or None, dict or None) - The data and metadata
    """
    data = st.session_state.get("era5_data")
    metadata = st.session_state.get("era5_metadata")
    return data, metadata


def has_era5_data() -> bool:
    """Check if ERA5 data is currently loaded."""
    return st.session_state.get("era5_data") is not None


def is_data_real() -> bool:
    """Check if currently loaded data is real (not synthetic)."""
    _, metadata = get_active_era5_data()
    if metadata is None:
        return False
    return not metadata.get("is_synthetic", True)


def get_era5_variables() -> List[str]:
    """Get list of available variables in the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None:
        return list(data.data_vars)
    return []


def get_era5_levels() -> List[int]:
    """Get list of available pressure levels in the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None and "level" in data.coords:
        return sorted([int(l) for l in data.level.values])
    return []


def get_era5_time_range() -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Get the time range of the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None and "time" in data.coords:
        times = data.time.values
        return pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    return None, None


def get_era5_slice(variable: str, time_idx: int = 0, level: Optional[int] = None):
    """
    Get a 2D slice of ERA5 data.

    Args:
        variable: Variable name
        time_idx: Time index (default 0)
        level: Pressure level (optional)

    Returns:
        tuple: (data_array, lats, lons) or (None, None, None)
    """
    data, _ = get_active_era5_data()
    if data is None:
        return None, None, None

    if variable not in data.data_vars:
        return None, None, None

    try:
        var_data = data[variable].isel(time=time_idx)

        if level is not None and "level" in var_data.dims:
            var_data = var_data.sel(level=level)

        # Get coordinates
        if "latitude" in data.coords:
            lats = data.latitude.values
            lons = data.longitude.values
        else:
            lats = data.lat.values
            lons = data.lon.values

        return var_data.values, lats, lons
    except Exception:
        return None, None, None


def get_era5_time_series(
    variable: str,
    lat_idx: Optional[int] = None,
    lon_idx: Optional[int] = None,
    level: Optional[int] = None
):
    """
    Get a time series from ERA5 data.

    Args:
        variable: Variable name
        lat_idx: Latitude index (if None, returns spatial mean)
        lon_idx: Longitude index (if None, returns spatial mean)
        level: Pressure level (optional)

    Returns:
        tuple: (times, values) or (None, None)
    """
    data, _ = get_active_era5_data()
    if data is None:
        return None, None

    if variable not in data.data_vars:
        return None, None

    try:
        var_data = data[variable]

        if level is not None and "level" in var_data.dims:
            var_data = var_data.sel(level=level)

        if lat_idx is not None and lon_idx is not None:
            # Extract single point
            if "latitude" in var_data.dims:
                values = var_data.isel(latitude=lat_idx, longitude=lon_idx).values
            else:
                values = var_data.isel(lat=lat_idx, lon=lon_idx).values
        else:
            # Spatial mean
            if "latitude" in var_data.dims:
                values = var_data.mean(dim=["latitude", "longitude"]).values
            else:
                values = var_data.mean(dim=["lat", "lon"]).values

        times = pd.to_datetime(data.time.values)

        return times, values
    except Exception:
        return None, None


def get_era5_wind_data(time_idx: int = 0, level: Optional[int] = None):
    """
    Get U and V wind components from ERA5 data.

    Args:
        time_idx: Time index
        level: Pressure level (optional)

    Returns:
        tuple: (u_wind, v_wind, lats, lons) or (None, None, None, None)
    """
    data, _ = get_active_era5_data()
    if data is None:
        return None, None, None, None

    # Try different variable names
    u_names = ["u_component_of_wind", "u", "U"]
    v_names = ["v_component_of_wind", "v", "V"]

    u_var = None
    v_var = None

    for name in u_names:
        if name in data.data_vars:
            u_var = name
            break

    for name in v_names:
        if name in data.data_vars:
            v_var = name
            break

    if u_var is None or v_var is None:
        return None, None, None, None

    u_data, lats, lons = get_era5_slice(u_var, time_idx, level)
    v_data, _, _ = get_era5_slice(v_var, time_idx, level)

    return u_data, v_data, lats, lons


def get_era5_temperature(time_idx: int = 0, level: Optional[int] = None):
    """
    Get temperature data from ERA5.

    Args:
        time_idx: Time index
        level: Pressure level (optional)

    Returns:
        tuple: (temperature, lats, lons) or (None, None, None)
    """
    data, _ = get_active_era5_data()
    if data is None:
        return None, None, None

    # Try different variable names
    temp_names = ["temperature", "t", "T", "air_temperature"]

    for name in temp_names:
        if name in data.data_vars:
            return get_era5_slice(name, time_idx, level)

    return None, None, None


def show_era5_data_warning():
    """Display a warning when ERA5 data is not available."""
    st.warning("""
    **No ERA5 Data Available**

    This feature requires ERA5 data. Please go to the **Data Manager** page and:

    1. Download a sample dataset, or
    2. Use pre-loaded server-side data

    After loading, the data will be available across all app features.
    """)


def show_era5_data_info():
    """Display information about the currently active ERA5 data."""
    data, metadata = get_active_era5_data()

    if data is None:
        st.info("**Data Source:** Not connected to ERA5 data")
        return

    name = metadata.get("name", "Unknown")
    source = metadata.get("source", "ERA5")
    is_synthetic = metadata.get("is_synthetic", False)
    citation = metadata.get("citation", "")

    if is_synthetic:
        st.warning(f"""
        **Data Source:** {name} (SYNTHETIC)

        This is synthetic data for demonstration purposes.
        For research, please download real ERA5 data from WeatherBench2.
        """)
    else:
        st.success(f"""
        **Data Source:** {name}

        **Type:** Real ERA5 Reanalysis
        **Source:** {source}
        {"**Citation:** " + citation if citation else ""}
        """)


def get_era5_data_banner() -> str:
    """
    Get a data source banner for display in pages.

    Returns:
        str: Banner text indicating data source
    """
    data, metadata = get_active_era5_data()

    if data is None:
        return "No ERA5 data loaded"

    name = metadata.get("name", "Unknown")
    is_synthetic = metadata.get("is_synthetic", False)

    if is_synthetic:
        return f"SYNTHETIC Data: {name}"
    else:
        return f"REAL ERA5 Data: {name}"


def get_data_source_badge():
    """
    Get a styled badge indicating data source.

    Returns styled HTML for the data source indicator.
    """
    data, metadata = get_active_era5_data()

    if data is None:
        return """
        <span style="
            background-color: #f0f0f0;
            color: #666;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
        ">No Data</span>
        """

    is_synthetic = metadata.get("is_synthetic", False)
    name = metadata.get("name", "Unknown")

    if is_synthetic:
        return f"""
        <span style="
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
        ">SYNTHETIC: {name}</span>
        """
    else:
        return f"""
        <span style="
            background-color: #d4edda;
            color: #155724;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
        ">REAL ERA5: {name}</span>
        """


def ensure_era5_data_or_demo(page_name: str, use_demo: bool = True) -> bool:
    """
    Ensure ERA5 data is available, or use demo mode.

    Args:
        page_name: Name of the current page
        use_demo: Whether to allow demo mode if data unavailable

    Returns:
        bool: True if ERA5 data is available, False if in demo mode
    """
    if has_era5_data():
        show_era5_data_info()
        return True

    if use_demo:
        st.info(f"""
        **{page_name} - Demo Mode**

        This page is running with synthetic data for demonstration purposes.

        To use **real ERA5 data**:
        1. Go to the **Data Manager** page
        2. Download a sample dataset or use pre-loaded data
        3. Click "Use This Dataset" to activate it
        4. Return to this page

        All calculations and visualizations will then use actual atmospheric observations.
        """)
        return False
    else:
        show_era5_data_warning()
        st.stop()


def auto_load_default_sample():
    """
    Automatically load a default sample if no data is loaded.

    This provides a seamless experience - users see data immediately.
    """
    if has_era5_data():
        return True

    if not DATA_STORAGE_AVAILABLE:
        return False

    # Get available samples
    available = get_available_samples()

    # Prefer real data over synthetic
    preferred_order = [
        "general_sample_2023",
        "european_heatwave_2003",
        "hurricane_katrina_2005",
        "polar_vortex_2019",
    ]

    for sample_id in preferred_order:
        if sample_id in available:
            data, metadata = load_sample_data(sample_id)
            if data is not None:
                st.session_state["era5_data"] = data
                st.session_state["era5_metadata"] = metadata
                st.session_state["active_sample"] = sample_id
                return True

    # Try any available sample
    if available:
        sample_id = available[0]
        data, metadata = load_sample_data(sample_id)
        if data is not None:
            st.session_state["era5_data"] = data
            st.session_state["era5_metadata"] = metadata
            st.session_state["active_sample"] = sample_id
            return True

    # Initialize a sample if none available
    for sample_id in preferred_order:
        if initialize_sample_data(sample_id, force_synthetic=True):
            data, metadata = load_sample_data(sample_id)
            if data is not None:
                st.session_state["era5_data"] = data
                st.session_state["era5_metadata"] = metadata
                st.session_state["active_sample"] = sample_id
                return True

    return False


def generate_synthetic_era5_like_data(
    n_times: int = 10,
    n_lats: int = 32,
    n_lons: int = 64,
    variables: Optional[List[str]] = None,
    seed: int = 42
) -> Dict:
    """
    Generate synthetic data with ERA5-like structure for demo purposes.

    Args:
        n_times: Number of time steps
        n_lats: Number of latitude points
        n_lons: Number of longitude points
        variables: List of variable names (default: temperature, u, v)
        seed: Random seed

    Returns:
        dict: Dictionary with synthetic data arrays

    NOTE: This data is SYNTHETIC and should only be used for demonstration.
    For research and publication, always use real ERA5 data.
    """
    np.random.seed(seed)

    if variables is None:
        variables = ["temperature", "u_wind", "v_wind", "geopotential"]

    lats = np.linspace(-90, 90, n_lats)
    lons = np.linspace(0, 360, n_lons, endpoint=False)
    times = np.arange(n_times)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    result = {
        "lats": lats,
        "lons": lons,
        "times": times,
        "is_synthetic": True,
        "warning": "SYNTHETIC DATA - for demonstration only",
    }

    for var in variables:
        if var == "temperature":
            # Temperature decreases with latitude
            base = 288 - 30 * np.abs(lat_grid) / 90
            data = np.stack([
                base + 5 * np.sin(np.radians(lon_grid) * 2 + t * 0.1) + np.random.randn(n_lats, n_lons) * 2
                for t in range(n_times)
            ])
            result[var] = data

        elif var in ["u_wind", "u_component_of_wind"]:
            # Westerlies in midlatitudes
            base = 20 * np.sin(np.radians(lat_grid) * 2)
            data = np.stack([
                base + np.random.randn(n_lats, n_lons) * 3
                for _ in range(n_times)
            ])
            result[var] = data

        elif var in ["v_wind", "v_component_of_wind"]:
            # Weak meridional flow
            base = 5 * np.sin(np.radians(lon_grid) * 3) * np.cos(np.radians(lat_grid))
            data = np.stack([
                base + np.random.randn(n_lats, n_lons) * 2
                for _ in range(n_times)
            ])
            result[var] = data

        elif var == "geopotential":
            # Height field with wave pattern
            base = 5500 + 100 * np.cos(np.radians(lat_grid - 45) * 3)
            data = np.stack([
                base + 50 * np.sin(np.radians(lon_grid - t * 5) * 3) + np.random.randn(n_lats, n_lons) * 10
                for t in range(n_times)
            ])
            result[var] = data

        else:
            # Generic random field
            data = np.random.randn(n_times, n_lats, n_lons)
            result[var] = data

    return result


def get_available_sample_datasets() -> Dict:
    """
    Get information about available sample datasets.

    Returns:
        Dict with sample info including availability status
    """
    if DATA_STORAGE_AVAILABLE:
        return get_all_sample_info()
    return {}


def load_sample(sample_id: str) -> bool:
    """
    Load a specific sample dataset into session state.

    Args:
        sample_id: ID of the sample to load

    Returns:
        bool: True if successful
    """
    if not DATA_STORAGE_AVAILABLE:
        return False

    data, metadata = load_sample_data(sample_id)
    if data is not None:
        st.session_state["era5_data"] = data
        st.session_state["era5_metadata"] = metadata
        st.session_state["active_sample"] = sample_id
        return True
    return False


def get_benchmark_data() -> Dict:
    """
    Get model benchmark data with proper citations.

    Returns:
        Dict containing benchmark metrics and citations for all models
    """
    if DATA_STORAGE_AVAILABLE:
        return get_model_benchmarks()
    return {}

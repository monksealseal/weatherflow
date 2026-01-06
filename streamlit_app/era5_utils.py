"""
ERA5 Data Utilities for WeatherFlow Streamlit App

This module provides shared utilities for accessing ERA5 data across all pages.
All pages should use this module to ensure consistent data handling.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

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


def has_era5_data():
    """Check if ERA5 data is currently loaded."""
    return st.session_state.get("era5_data") is not None


def get_era5_variables():
    """Get list of available variables in the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None:
        return list(data.data_vars)
    return []


def get_era5_levels():
    """Get list of available pressure levels in the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None and "level" in data.coords:
        return sorted([int(l) for l in data.level.values])
    return []


def get_era5_time_range():
    """Get the time range of the active dataset."""
    data, _ = get_active_era5_data()
    if data is not None and "time" in data.coords:
        times = data.time.values
        return pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    return None, None


def get_era5_slice(variable, time_idx=0, level=None):
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


def get_era5_time_series(variable, lat_idx=None, lon_idx=None, level=None):
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


def get_era5_wind_data(time_idx=0, level=None):
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


def get_era5_temperature(time_idx=0, level=None):
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
    ⚠️ **No ERA5 Data Available**
    
    This feature requires ERA5 data. Please go to the **Data Manager** page and:
    
    1. Download a sample dataset, or
    2. Download custom data from WeatherBench2
    
    After downloading, select "Use This Dataset" to activate it.
    """)


def show_era5_data_info():
    """Display information about the currently active ERA5 data."""
    data, metadata = get_active_era5_data()
    
    if data is None:
        st.info("📊 **Data Source:** Not connected to ERA5 data")
        return
    
    name = metadata.get("name", "Unknown")
    source = metadata.get("source", "ERA5")
    is_synthetic = metadata.get("is_synthetic", False)
    
    if is_synthetic:
        st.warning(f"📊 **Data Source:** {name} (⚠️ Synthetic)")
    else:
        st.success(f"📊 **Data Source:** {name} (✅ Real ERA5 from {source})")


def get_era5_data_banner():
    """
    Get a data source banner for display in pages.
    
    Returns:
        str: Banner text indicating data source
    """
    data, metadata = get_active_era5_data()
    
    if data is None:
        return "⚠️ Demo Mode - No ERA5 data loaded"
    
    name = metadata.get("name", "Unknown")
    is_synthetic = metadata.get("is_synthetic", False)
    
    if is_synthetic:
        return f"⚠️ Using Synthetic Data: {name}"
    else:
        return f"✅ Using Real ERA5 Data: {name}"


def ensure_era5_data_or_demo(page_name: str, use_demo: bool = True):
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
        📊 **{page_name} - Demo Mode**
        
        This page is running with synthetic data for demonstration purposes.
        
        To use **real ERA5 data**:
        1. Go to the **Data Manager** page (📊 in sidebar)
        2. Download a sample dataset or custom data
        3. Click "Use This Dataset" to activate it
        4. Return to this page
        
        All calculations and visualizations will then use actual atmospheric observations.
        """)
        return False
    else:
        show_era5_data_warning()
        st.stop()
        return False


def generate_synthetic_era5_like_data(
    n_times: int = 10,
    n_lats: int = 32,
    n_lons: int = 64,
    variables: list = None,
    seed: int = 42
):
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
    """
    import numpy as np
    
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

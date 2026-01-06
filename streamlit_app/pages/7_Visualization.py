"""
Weather Visualization Tools

Uses the actual WeatherVisualizer class from weatherflow/utils/visualization.py
Supports ERA5 data visualization or demo mode with synthetic data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import visualization utilities
try:
    from weatherflow.utils.visualization import WeatherVisualizer
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False

# Import ERA5 utilities
try:
    from era5_utils import (
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        get_era5_slice,
        get_era5_wind_data,
        get_era5_time_range,
    )
    ERA5_UTILS_AVAILABLE = True
except ImportError:
    ERA5_UTILS_AVAILABLE = False

st.set_page_config(page_title="Weather Visualization", page_icon="📊", layout="wide")

st.title("📊 Weather Visualization")
st.markdown("""
Interactive visualization tools for weather data. Generate maps, flow fields,
error distributions, and animations.
""")

# Show data source banner
if ERA5_UTILS_AVAILABLE:
    banner = get_era5_data_banner()
    if "Demo Mode" in banner or "⚠️" in banner:
        st.info(f"📊 {banner} - Visualizations use synthetic data patterns.")
    else:
        st.success(f"📊 {banner} - Use the Data Manager to preview and visualize real ERA5 data.")

# Variable configurations
VAR_LABELS = {
    'temperature': 'Temperature (K)',
    'geopotential': 'Geopotential (m²/s²)',
    'u_wind': 'U-Wind (m/s)',
    'v_wind': 'V-Wind (m/s)',
    'humidity': 'Specific Humidity (kg/kg)',
    'pressure': 'Surface Pressure (hPa)',
    'precipitation': 'Precipitation (mm/hr)'
}

VAR_CMAPS = {
    'temperature': 'RdBu_r',
    'geopotential': 'viridis',
    'u_wind': 'RdBu_r',
    'v_wind': 'RdBu_r',
    'humidity': 'Blues',
    'pressure': 'viridis',
    'precipitation': 'Blues'
}

# Sidebar
st.sidebar.header("Visualization Settings")

projection = st.sidebar.selectbox(
    "Map Projection",
    ["Equirectangular", "Orthographic", "Mollweide", "Robinson"],
    index=0
)

colormap = st.sidebar.selectbox(
    "Default Colormap",
    ["RdBu_r", "viridis", "plasma", "Blues", "YlOrRd", "coolwarm"]
)

show_coastlines = st.sidebar.checkbox("Show Coastlines", value=True)
show_grid = st.sidebar.checkbox("Show Grid", value=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Global Maps",
    "🌬️ Flow Fields",
    "📈 Error Analysis",
    "🎬 Animation",
    "🌍 ERA5 Visualization"
])

# Tab 1: Global Maps
with tab1:
    st.header("Global Weather Maps")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Generation")

        variable = st.selectbox("Variable", list(VAR_LABELS.keys()))
        grid_lat = st.slider("Latitude Points", 32, 180, 90, key="map_lat")
        grid_lon = st.slider("Longitude Points", 64, 360, 180, key="map_lon")

        # Generate synthetic data
        np.random.seed(42)
        lats = np.linspace(-90, 90, grid_lat)
        lons = np.linspace(-180, 180, grid_lon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        if variable == 'temperature':
            # Temperature decreases with latitude
            data = 300 - 40 * np.abs(lat_grid) / 90
            # Add some zonal variation
            data += 10 * np.cos(np.radians(lon_grid) * 2)
            # Add noise
            data += np.random.randn(grid_lat, grid_lon) * 3
        elif variable == 'geopotential':
            # Jet stream pattern
            data = 5500 + 100 * np.cos(np.radians(lat_grid - 45) * 3)
            data += 50 * np.sin(np.radians(lon_grid) * 3)
        elif variable == 'u_wind':
            # Zonal wind (westerlies in midlatitudes)
            data = 30 * np.sin(np.radians(lat_grid) * 2)
            data += np.random.randn(grid_lat, grid_lon) * 5
        elif variable == 'v_wind':
            # Meridional wind (weaker)
            data = 5 * np.sin(np.radians(lon_grid) * 4) * np.cos(np.radians(lat_grid))
            data += np.random.randn(grid_lat, grid_lon) * 2
        elif variable == 'humidity':
            # Humidity higher in tropics
            data = 0.015 * np.exp(-np.abs(lat_grid) / 30)
            data += np.random.rand(grid_lat, grid_lon) * 0.002
        elif variable == 'pressure':
            # Surface pressure pattern
            data = 1013 - 5 * np.abs(lat_grid) / 90
            data += 10 * np.sin(np.radians(lon_grid) * 3)
        else:  # precipitation
            # Precipitation in ITCZ
            data = 5 * np.exp(-(lat_grid**2) / 200)
            data = np.maximum(0, data + np.random.randn(grid_lat, grid_lon) * 1)

        st.subheader("Statistics")
        st.metric("Min", f"{data.min():.2f}")
        st.metric("Max", f"{data.max():.2f}")
        st.metric("Mean", f"{data.mean():.2f}")

    with col2:
        # Create map
        cmap = VAR_CMAPS.get(variable, colormap)

        if projection == "Orthographic":
            fig = go.Figure(go.Scattergeo(
                lon=lon_grid.flatten(),
                lat=lat_grid.flatten(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=data.flatten(),
                    colorscale=cmap,
                    colorbar=dict(title=VAR_LABELS[variable])
                )
            ))
            fig.update_geos(projection_type="orthographic")
        else:
            fig = go.Figure(go.Heatmap(
                z=data,
                x=lons,
                y=lats,
                colorscale=cmap,
                colorbar=dict(title=VAR_LABELS[variable])
            ))

            if projection == "Mollweide":
                # For non-rectangular projections, use Scattergeo
                pass

        fig.update_layout(
            title=f'{variable.replace("_", " ").title()} Field',
            height=500
        )

        if projection == "Equirectangular":
            fig.update_xaxes(title_text='Longitude')
            fig.update_yaxes(title_text='Latitude')

        st.plotly_chart(fig, use_container_width=True)

        # Zonal mean
        st.subheader("Zonal Mean Profile")
        zonal_mean = data.mean(axis=1)

        fig_zonal = go.Figure()
        fig_zonal.add_trace(go.Scatter(
            x=lats, y=zonal_mean,
            mode='lines',
            line=dict(color='#1e88e5', width=2)
        ))
        fig_zonal.update_layout(
            xaxis_title='Latitude',
            yaxis_title=VAR_LABELS[variable],
            height=250
        )
        st.plotly_chart(fig_zonal, use_container_width=True)

# Tab 2: Flow Fields
with tab2:
    st.header("Flow Field Visualization")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Wind Field Parameters")

        flow_type = st.selectbox(
            "Flow Pattern",
            ["Westerlies", "Trade Winds", "Jet Stream", "Cyclone", "Random"]
        )

        vector_density = st.slider("Vector Density", 1, 5, 2)
        arrow_scale = st.slider("Arrow Scale", 0.5, 3.0, 1.0)

        show_speed = st.checkbox("Show Wind Speed", value=True)

    with col2:
        # Generate wind field
        np.random.seed(123)
        nlat, nlon = 45, 90
        lats = np.linspace(-90, 90, nlat)
        lons = np.linspace(-180, 180, nlon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        if flow_type == "Westerlies":
            u = 20 * np.sin(np.radians(lat_grid) * 2)
            v = np.zeros_like(u)
        elif flow_type == "Trade Winds":
            u = -10 * np.cos(np.radians(lat_grid))
            v = -5 * np.sign(lat_grid) * np.exp(-np.abs(lat_grid) / 20)
        elif flow_type == "Jet Stream":
            u = 50 * np.exp(-((lat_grid - 45)**2) / 200)
            u += 50 * np.exp(-((lat_grid + 45)**2) / 200)
            v = 10 * np.sin(np.radians(lon_grid) * 3) * np.exp(-np.abs(lat_grid - 45) / 15)
        elif flow_type == "Cyclone":
            cx, cy = 0, 45
            r = np.sqrt((lon_grid - cx)**2 + (lat_grid - cy)**2)
            theta = np.arctan2(lat_grid - cy, lon_grid - cx)
            speed = 30 * np.exp(-r / 20)
            u = -speed * np.sin(theta)
            v = speed * np.cos(theta)
        else:  # Random
            u = np.random.randn(nlat, nlon) * 10
            v = np.random.randn(nlat, nlon) * 10

        speed = np.sqrt(u**2 + v**2)

        # Downsample for vectors
        skip = 5 - vector_density + 1

        fig = go.Figure()

        # Background speed field
        if show_speed:
            fig.add_trace(go.Heatmap(
                z=speed,
                x=lons,
                y=lats,
                colorscale='YlOrRd',
                opacity=0.7,
                colorbar=dict(title='Speed (m/s)')
            ))

        # Quiver plot using annotations
        for i in range(0, nlat, skip):
            for j in range(0, nlon, skip):
                if speed[i, j] > 1:  # Only show significant vectors
                    fig.add_annotation(
                        x=lons[j],
                        y=lats[i],
                        ax=lons[j] + u[i, j] * arrow_scale * 0.3,
                        ay=lats[i] + v[i, j] * arrow_scale * 0.3,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor='black'
                    )

        fig.update_layout(
            title=f'{flow_type} Wind Pattern',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Wind rose
        st.subheader("Wind Direction Distribution")

        # Flatten and compute directions
        directions = np.degrees(np.arctan2(v, u)) % 360
        speeds = speed.flatten()
        directions = directions.flatten()

        # Bin directions
        dir_bins = np.linspace(0, 360, 17)
        dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2

        mean_speeds = []
        for i in range(len(dir_bins) - 1):
            mask = (directions >= dir_bins[i]) & (directions < dir_bins[i+1])
            if mask.sum() > 0:
                mean_speeds.append(speeds[mask].mean())
            else:
                mean_speeds.append(0)

        fig_rose = go.Figure()
        fig_rose.add_trace(go.Barpolar(
            r=mean_speeds,
            theta=dir_centers,
            marker_color='#1e88e5',
            opacity=0.8
        ))
        fig_rose.update_layout(
            polar=dict(radialaxis=dict(title='Speed (m/s)')),
            height=350,
            title='Wind Rose'
        )
        st.plotly_chart(fig_rose, use_container_width=True)

# Tab 3: Error Analysis
with tab3:
    st.header("Prediction Error Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Generate Test Data")

        error_type = st.selectbox(
            "Error Pattern",
            ["Random", "Systematic Bias", "Scale-Dependent", "Region-Specific"]
        )

        error_magnitude = st.slider("Error Magnitude", 0.1, 2.0, 0.5)

        n_samples = st.slider("Number of Samples", 1000, 10000, 5000)

    with col2:
        np.random.seed(456)

        # Generate "true" values
        true_values = np.random.randn(n_samples) * 10 + 280  # Temperature-like

        # Generate predictions with different error patterns
        if error_type == "Random":
            predictions = true_values + np.random.randn(n_samples) * error_magnitude * 5
        elif error_type == "Systematic Bias":
            predictions = true_values + 2 + np.random.randn(n_samples) * error_magnitude * 3
        elif error_type == "Scale-Dependent":
            # Error proportional to value
            predictions = true_values * (1 + np.random.randn(n_samples) * error_magnitude * 0.02)
        else:  # Region-specific
            predictions = true_values.copy()
            predictions[:n_samples//3] += 3  # Bias in one region
            predictions += np.random.randn(n_samples) * error_magnitude * 2

        errors = predictions - true_values

        # Compute metrics
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        corr = np.corrcoef(true_values, predictions)[0, 1]

        st.subheader("Error Metrics")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("RMSE", f"{rmse:.3f}")
        with metric_cols[1]:
            st.metric("MAE", f"{mae:.3f}")
        with metric_cols[2]:
            st.metric("Bias", f"{bias:.3f}")
        with metric_cols[3]:
            st.metric("Correlation", f"{corr:.3f}")

        # Plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Distribution', 'True vs Predicted',
                          'Error vs True Value', 'Q-Q Plot')
        )

        # Error histogram
        fig.add_trace(
            go.Histogram(x=errors, nbinsx=50, name='Errors',
                        marker_color='#1e88e5'),
            row=1, col=1
        )

        # Scatter plot
        sample_idx = np.random.choice(n_samples, min(1000, n_samples), replace=False)
        fig.add_trace(
            go.Scatter(x=true_values[sample_idx], y=predictions[sample_idx],
                      mode='markers', marker=dict(size=3, opacity=0.5),
                      name='Data'),
            row=1, col=2
        )
        # Perfect prediction line
        fig.add_trace(
            go.Scatter(x=[true_values.min(), true_values.max()],
                      y=[true_values.min(), true_values.max()],
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='Perfect'),
            row=1, col=2
        )

        # Error vs true value
        fig.add_trace(
            go.Scatter(x=true_values[sample_idx], y=errors[sample_idx],
                      mode='markers', marker=dict(size=3, opacity=0.5),
                      name='Error'),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        # Q-Q plot
        sorted_errors = np.sort(errors)
        theoretical_quantiles = np.random.randn(n_samples)
        theoretical_quantiles = np.sort(theoretical_quantiles) * errors.std()

        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_errors,
                      mode='markers', marker=dict(size=3),
                      name='Q-Q'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                      y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                      mode='lines', line=dict(color='red', dash='dash')),
            row=2, col=2
        )

        fig.update_xaxes(title_text='Error', row=1, col=1)
        fig.update_xaxes(title_text='True Value', row=1, col=2)
        fig.update_xaxes(title_text='True Value', row=2, col=1)
        fig.update_xaxes(title_text='Theoretical Quantiles', row=2, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Predicted', row=1, col=2)
        fig.update_yaxes(title_text='Error', row=2, col=1)
        fig.update_yaxes(title_text='Sample Quantiles', row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Animation
with tab4:
    st.header("Weather Animation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Animation Settings")

        anim_variable = st.selectbox(
            "Variable to Animate",
            ["Temperature", "Pressure", "Wind Speed"],
            key="anim_var"
        )

        n_frames = st.slider("Number of Frames", 10, 50, 20)
        evolution_type = st.selectbox(
            "Evolution Type",
            ["Eastward Propagation", "Growing Disturbance", "Oscillation"]
        )

    with col2:
        # Generate animation data
        np.random.seed(789)
        nlat, nlon = 45, 90
        lats = np.linspace(-90, 90, nlat)
        lons = np.linspace(-180, 180, nlon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        frames_data = []

        for t in range(n_frames):
            if anim_variable == "Temperature":
                base = 300 - 40 * np.abs(lat_grid) / 90

                if evolution_type == "Eastward Propagation":
                    pattern = 10 * np.sin(np.radians(lon_grid - t * 360 / n_frames) * 3)
                elif evolution_type == "Growing Disturbance":
                    pattern = (t / n_frames) * 20 * np.exp(-((lon_grid)**2 + (lat_grid - 30)**2) / 1000)
                else:
                    pattern = 10 * np.sin(2 * np.pi * t / n_frames) * np.cos(np.radians(lon_grid) * 2)

                data = base + pattern
            elif anim_variable == "Pressure":
                base = 1013 * np.ones_like(lat_grid)

                if evolution_type == "Eastward Propagation":
                    # Moving low pressure
                    cx = -90 + t * 180 / n_frames
                    r = np.sqrt((lon_grid - cx)**2 + (lat_grid - 45)**2)
                    pattern = -20 * np.exp(-r**2 / 500)
                else:
                    pattern = 10 * np.sin(np.radians(lon_grid - t * 360 / n_frames) * 2)

                data = base + pattern
            else:  # Wind Speed
                if evolution_type == "Growing Disturbance":
                    speed = (t / n_frames) * 30 * np.exp(-((lat_grid - 45)**2) / 200)
                else:
                    speed = 20 * np.abs(np.sin(np.radians(lat_grid) * 2))
                    speed *= 1 + 0.3 * np.sin(2 * np.pi * t / n_frames)
                data = speed

            frames_data.append(data)

        # Animation slider
        st.subheader("Animation Control")
        frame_idx = st.slider("Frame", 0, n_frames - 1, 0)

        # Display current frame
        fig = go.Figure()

        if anim_variable == "Temperature":
            cmap = 'RdBu_r'
            title = 'Temperature (K)'
        elif anim_variable == "Pressure":
            cmap = 'viridis'
            title = 'Pressure (hPa)'
        else:
            cmap = 'YlOrRd'
            title = 'Wind Speed (m/s)'

        fig.add_trace(go.Heatmap(
            z=frames_data[frame_idx],
            x=lons,
            y=lats,
            colorscale=cmap,
            colorbar=dict(title=title)
        ))

        fig.update_layout(
            title=f'{anim_variable} - Frame {frame_idx + 1}/{n_frames}',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # Time series at a point
        st.subheader("Time Series at Selected Point")

        point_lat = st.slider("Select Latitude", -90, 90, 45)
        point_lon = st.slider("Select Longitude", -180, 180, 0)

        lat_idx = np.argmin(np.abs(lats - point_lat))
        lon_idx = np.argmin(np.abs(lons - point_lon))

        time_series = [frames_data[t][lat_idx, lon_idx] for t in range(n_frames)]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=list(range(n_frames)),
            y=time_series,
            mode='lines+markers',
            line=dict(color='#1e88e5')
        ))
        fig_ts.add_vline(x=frame_idx, line_dash="dash", line_color="red")
        fig_ts.update_layout(
            title=f'Time Series at ({point_lat}°, {point_lon}°)',
            xaxis_title='Frame',
            yaxis_title=title,
            height=250
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# Tab 5: ERA5 Visualization
with tab5:
    st.header("🌍 ERA5 Data Visualization")
    
    st.markdown("""
    Visualize **real ERA5 reanalysis data** from the Data Manager.
    All visualizations use actual atmospheric observations - no synthetic data.
    """)
    
    if ERA5_UTILS_AVAILABLE and has_era5_data():
        data, metadata = get_active_era5_data()
        
        st.success(f"✅ Visualizing Real ERA5 Data: **{metadata.get('name', 'Unknown')}**")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"**Period:** {metadata.get('start_date', '?')} to {metadata.get('end_date', '?')}")
        with col_info2:
            st.markdown(f"**Source:** {metadata.get('source', 'ECMWF ERA5 Reanalysis')}")
        
        st.markdown("---")
        
        # ERA5 visualization options
        era5_viz_type = st.radio(
            "Visualization Type",
            ["🗺️ Spatial Map", "🌬️ Wind Vectors", "📈 Time Series", "📊 Multi-Variable Comparison"],
            horizontal=True
        )
        
        available_vars = list(data.data_vars)
        
        if era5_viz_type == "🗺️ Spatial Map":
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Map Settings")
                
                era5_var = st.selectbox(
                    "Variable",
                    options=available_vars,
                    key="era5_viz_var"
                )
                
                # Time selection
                time_values = data.time.values
                time_options = [str(t)[:19] for t in time_values]
                era5_time_idx = st.selectbox(
                    "Time",
                    options=range(len(time_options)),
                    format_func=lambda x: time_options[x],
                    key="era5_viz_time"
                )
                
                # Level selection
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    era5_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=min(1, len(levels) - 1),
                        key="era5_viz_level"
                    )
                else:
                    era5_level = None
                
                era5_cmap = st.selectbox(
                    "Colormap",
                    options=["RdBu_r", "viridis", "plasma", "Blues", "YlOrRd"],
                    key="era5_viz_cmap"
                )
            
            with col2:
                # Get data slice
                var_data = data[era5_var].isel(time=era5_time_idx)
                if era5_level is not None and "level" in var_data.dims:
                    var_data = var_data.sel(level=era5_level)
                
                # Get coordinates
                if "latitude" in data.coords:
                    lats = data.latitude.values
                    lons = data.longitude.values
                else:
                    lats = data.lat.values
                    lons = data.lon.values
                
                fig = go.Figure(data=go.Heatmap(
                    z=var_data.values,
                    x=lons,
                    y=lats,
                    colorscale=era5_cmap,
                    colorbar=dict(title=era5_var)
                ))
                
                title = f"ERA5 {era5_var}"
                if era5_level:
                    title += f" at {era5_level} hPa"
                title += f" - {time_options[era5_time_idx]}"
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                values = var_data.values.flatten()
                values = values[~np.isnan(values)]
                col_s1.metric("Min", f"{np.min(values):.2f}")
                col_s2.metric("Max", f"{np.max(values):.2f}")
                col_s3.metric("Mean", f"{np.mean(values):.2f}")
                col_s4.metric("Std Dev", f"{np.std(values):.2f}")
        
        elif era5_viz_type == "🌬️ Wind Vectors":
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Wind Vector Settings")
                
                # Time selection
                time_values = data.time.values
                time_options = [str(t)[:19] for t in time_values]
                wind_time_idx = st.selectbox(
                    "Time",
                    options=range(len(time_options)),
                    format_func=lambda x: time_options[x],
                    key="era5_wind_time"
                )
                
                # Level selection
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    wind_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=min(1, len(levels) - 1),
                        key="era5_wind_level"
                    )
                else:
                    wind_level = None
                
                vector_density = st.slider("Vector Density", 1, 5, 2, key="era5_vec_density")
            
            with col2:
                # Get wind data
                u_data, v_data, lats, lons = get_era5_wind_data(wind_time_idx, wind_level)
                
                if u_data is not None and v_data is not None:
                    wind_speed = np.sqrt(u_data**2 + v_data**2)
                    
                    fig = go.Figure()
                    
                    # Background wind speed
                    fig.add_trace(go.Heatmap(
                        z=wind_speed,
                        x=lons,
                        y=lats,
                        colorscale='YlOrRd',
                        opacity=0.7,
                        colorbar=dict(title='Speed (m/s)')
                    ))
                    
                    # Add wind vectors
                    skip = 6 - vector_density
                    nlat, nlon = u_data.shape
                    
                    for i in range(0, nlat, skip):
                        for j in range(0, nlon, skip):
                            if wind_speed[i, j] > 1:
                                scale = 0.3 * (1 + vector_density * 0.2)
                                fig.add_annotation(
                                    x=lons[j],
                                    y=lats[i],
                                    ax=lons[j] + u_data[i, j] * scale,
                                    ay=lats[i] + v_data[i, j] * scale,
                                    xref='x',
                                    yref='y',
                                    axref='x',
                                    ayref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor='black'
                                )
                    
                    title = "ERA5 Wind Field"
                    if wind_level:
                        title += f" at {wind_level} hPa"
                    title += f" - {time_options[wind_time_idx]}"
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Wind statistics
                    col_w1, col_w2, col_w3 = st.columns(3)
                    col_w1.metric("Mean Speed", f"{np.mean(wind_speed):.1f} m/s")
                    col_w2.metric("Max Speed", f"{np.max(wind_speed):.1f} m/s")
                    
                    # Dominant direction
                    mean_u = np.mean(u_data)
                    mean_v = np.mean(v_data)
                    direction = np.degrees(np.arctan2(mean_v, mean_u)) % 360
                    col_w3.metric("Mean Direction", f"{direction:.0f}°")
                else:
                    st.warning("Wind components (u, v) not found in the dataset.")
        
        elif era5_viz_type == "📈 Time Series":
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Time Series Settings")
                
                ts_var = st.selectbox(
                    "Variable",
                    options=available_vars,
                    key="era5_ts_var"
                )
                
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    ts_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=min(1, len(levels) - 1),
                        key="era5_ts_level"
                    )
                else:
                    ts_level = None
                
                ts_type = st.radio(
                    "Aggregation",
                    ["Domain Mean", "Domain Max", "Domain Min", "Specific Location"],
                    key="era5_ts_type"
                )
                
                if ts_type == "Specific Location":
                    if "latitude" in data.coords:
                        lats = data.latitude.values
                        lons = data.longitude.values
                    else:
                        lats = data.lat.values
                        lons = data.lon.values
                    
                    ts_lat = st.select_slider("Latitude", options=lats.tolist(), value=lats[len(lats)//2], key="era5_ts_lat")
                    ts_lon = st.select_slider("Longitude", options=lons.tolist(), value=lons[len(lons)//2], key="era5_ts_lon")
            
            with col2:
                # Get time series data
                var_data = data[ts_var]
                if ts_level is not None and "level" in var_data.dims:
                    var_data = var_data.sel(level=ts_level)
                
                times = pd.to_datetime(data.time.values)
                
                if "latitude" in data.coords:
                    lat_dim = "latitude"
                    lon_dim = "longitude"
                else:
                    lat_dim = "lat"
                    lon_dim = "lon"
                
                if ts_type == "Domain Mean":
                    values = var_data.mean(dim=[lat_dim, lon_dim]).values
                    title = f"ERA5 {ts_var} - Domain Mean"
                elif ts_type == "Domain Max":
                    values = var_data.max(dim=[lat_dim, lon_dim]).values
                    title = f"ERA5 {ts_var} - Domain Maximum"
                elif ts_type == "Domain Min":
                    values = var_data.min(dim=[lat_dim, lon_dim]).values
                    title = f"ERA5 {ts_var} - Domain Minimum"
                else:
                    lat_idx = np.argmin(np.abs(lats - ts_lat))
                    lon_idx = np.argmin(np.abs(lons - ts_lon))
                    values = var_data.isel(**{lat_dim: lat_idx, lon_dim: lon_idx}).values
                    title = f"ERA5 {ts_var} at ({ts_lat:.1f}°, {ts_lon:.1f}°)"
                
                if ts_level:
                    title += f" at {ts_level} hPa"
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    line=dict(color='#1e88e5', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title=ts_var,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
                col_ts1.metric("Min", f"{np.min(values):.2f}")
                col_ts2.metric("Max", f"{np.max(values):.2f}")
                col_ts3.metric("Mean", f"{np.mean(values):.2f}")
                col_ts4.metric("Range", f"{np.max(values) - np.min(values):.2f}")
        
        else:  # Multi-Variable Comparison
            st.subheader("Multi-Variable Comparison")
            
            # Select multiple variables
            compare_vars = st.multiselect(
                "Select Variables to Compare",
                options=available_vars,
                default=available_vars[:min(4, len(available_vars))],
                key="era5_compare_vars"
            )
            
            if compare_vars:
                # Time selection
                time_values = data.time.values
                time_options = [str(t)[:19] for t in time_values]
                compare_time_idx = st.selectbox(
                    "Time",
                    options=range(len(time_options)),
                    format_func=lambda x: time_options[x],
                    key="era5_compare_time"
                )
                
                # Level selection
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    compare_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=min(1, len(levels) - 1),
                        key="era5_compare_level"
                    )
                else:
                    compare_level = None
                
                # Get coordinates
                if "latitude" in data.coords:
                    lats = data.latitude.values
                    lons = data.longitude.values
                else:
                    lats = data.lat.values
                    lons = data.lon.values
                
                # Create subplot grid
                n_vars = len(compare_vars)
                n_cols = min(2, n_vars)
                n_rows = (n_vars + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=compare_vars
                )
                
                for idx, var in enumerate(compare_vars):
                    row = idx // n_cols + 1
                    col = idx % n_cols + 1
                    
                    var_data = data[var].isel(time=compare_time_idx)
                    if compare_level is not None and "level" in var_data.dims:
                        var_data = var_data.sel(level=compare_level)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=var_data.values,
                            x=lons,
                            y=lats,
                            colorscale='RdBu_r' if 'temperature' in var.lower() or 'wind' in var.lower() else 'viridis',
                            showscale=True
                        ),
                        row=row, col=col
                    )
                
                title = f"ERA5 Multi-Variable Comparison - {time_options[compare_time_idx]}"
                if compare_level:
                    title += f" at {compare_level} hPa"
                
                fig.update_layout(
                    title=title,
                    height=300 * n_rows,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one variable to visualize.")
    
    else:
        st.warning("""
        ⚠️ **ERA5 Data Not Available**
        
        To visualize real ERA5 data:
        1. Go to the **📊 Data Manager** page
        2. Download a sample dataset
        3. Click "Use This Dataset" to activate it
        4. Return here to visualize real atmospheric data
        
        The other tabs demonstrate visualization capabilities using synthetic data.
        """)

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page demonstrates visualization capabilities similar to:

    ```python
    # From weatherflow/utils/visualization.py
    from weatherflow.utils.visualization import WeatherVisualizer

    # Create visualizer
    vis = WeatherVisualizer(
        figsize=(15, 10),
        projection='PlateCarree'
    )

    # Plot a single field
    fig, ax = vis.plot_field(
        data, title="Temperature",
        var_name='temperature',
        coastlines=True
    )

    # Plot comparison (true vs predicted)
    fig, axes = vis.plot_comparison(
        true_data, pred_data,
        var_name='temperature'
    )

    # Plot error metrics
    fig, axes = vis.plot_error_metrics(
        true_data, pred_data
    )

    # Plot flow vectors
    fig, ax = vis.plot_flow_vectors(
        u, v,
        background=wind_speed,
        title="Wind Field"
    )

    # Create animation
    anim = vis.create_prediction_animation(
        predictions,
        var_name='temperature',
        interval=200
    )
    ```
    """)

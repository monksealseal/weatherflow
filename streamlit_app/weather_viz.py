"""
Professional Weather Visualization Utilities

Creates visualizations that look like professional weather platforms (weather.com,
TV weather forecasts, etc.) with correct axes, labels, and units.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, List
from datetime import datetime, timedelta


# Standard colormaps for weather variables
WEATHER_COLORMAPS = {
    "temperature": [
        [0.0, "#313695"],   # -40°C: Dark blue
        [0.15, "#4575b4"],  # -20°C: Blue
        [0.25, "#74add1"],  # -5°C: Light blue
        [0.35, "#abd9e9"],  # 5°C: Pale blue
        [0.45, "#e0f3f8"],  # 15°C: Very pale blue
        [0.5, "#ffffbf"],   # 20°C: Yellow
        [0.55, "#fee090"],  # 25°C: Light orange
        [0.65, "#fdae61"],  # 30°C: Orange
        [0.75, "#f46d43"],  # 35°C: Red-orange
        [0.85, "#d73027"],  # 40°C: Red
        [1.0, "#a50026"],   # 45°C+: Dark red
    ],
    "precipitation": [
        [0.0, "#ffffff"],   # 0 mm: White
        [0.1, "#c7e9c0"],   # Light green
        [0.2, "#a1d99b"],   # Green
        [0.3, "#74c476"],   # Darker green
        [0.4, "#41ab5d"],   # Even darker
        [0.5, "#238b45"],   # Dark green
        [0.6, "#006d2c"],   # Very dark green
        [0.7, "#00441b"],   # Almost black-green
        [0.8, "#08306b"],   # Blue
        [0.9, "#41b6c4"],   # Cyan
        [1.0, "#225ea8"],   # Dark blue
    ],
    "wind": [
        [0.0, "#ffffff"],   # Calm
        [0.15, "#deebf7"],  # Light
        [0.3, "#9ecae1"],   # Moderate
        [0.45, "#6baed6"],  # Fresh
        [0.6, "#3182bd"],   # Strong
        [0.75, "#08519c"],  # Gale
        [0.9, "#810f7c"],   # Storm
        [1.0, "#4d004b"],   # Hurricane
    ],
    "geopotential": "Viridis",
    "pressure": "RdYlBu_r",
}


def kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - 273.15


def kelvin_to_fahrenheit(k: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (k - 273.15) * 9/5 + 32


def create_temperature_map(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str = "Temperature",
    units: str = "celsius",
    show_contours: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Create a professional temperature map.

    Args:
        data: 2D temperature data (in Kelvin)
        lats: Latitude values
        lons: Longitude values
        title: Plot title
        units: "celsius", "fahrenheit", or "kelvin"
        show_contours: Whether to show contour lines
        height: Figure height

    Returns:
        Plotly Figure
    """
    # Convert units
    if units == "celsius":
        display_data = kelvin_to_celsius(data)
        unit_label = "°C"
        zmin, zmax = -40, 45
    elif units == "fahrenheit":
        display_data = kelvin_to_fahrenheit(data)
        unit_label = "°F"
        zmin, zmax = -40, 115
    else:
        display_data = data
        unit_label = "K"
        zmin, zmax = 230, 320

    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=display_data,
        x=lons,
        y=lats,
        colorscale=WEATHER_COLORMAPS["temperature"],
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(text=f"Temperature ({unit_label})", side="right"),
            ticksuffix=unit_label,
        ),
        hovertemplate=(
            f"Lon: %{{x:.1f}}°<br>"
            f"Lat: %{{y:.1f}}°<br>"
            f"Temp: %{{z:.1f}}{unit_label}<extra></extra>"
        ),
    ))

    # Add contours if requested
    if show_contours:
        fig.add_trace(go.Contour(
            z=display_data,
            x=lons,
            y=lats,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='black'),
            ),
            line=dict(width=1, color='black'),
            showscale=False,
            opacity=0.3,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=height,
        xaxis=dict(
            ticksuffix="°",
            dtick=30,
        ),
        yaxis=dict(
            ticksuffix="°",
            dtick=30,
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    return fig


def create_wind_map(
    u: np.ndarray,
    v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str = "Wind",
    show_barbs: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Create a professional wind map with speed colors and optional wind barbs.

    Args:
        u: U-component of wind (m/s)
        v: V-component of wind (m/s)
        lats: Latitude values
        lons: Longitude values
        title: Plot title
        show_barbs: Whether to show wind barbs/arrows
        height: Figure height

    Returns:
        Plotly Figure
    """
    # Calculate wind speed
    speed = np.sqrt(u**2 + v**2)

    fig = go.Figure()

    # Add wind speed heatmap
    fig.add_trace(go.Heatmap(
        z=speed,
        x=lons,
        y=lats,
        colorscale=WEATHER_COLORMAPS["wind"],
        zmin=0,
        zmax=30,
        colorbar=dict(
            title=dict(text="Wind Speed (m/s)", side="right"),
            ticksuffix=" m/s",
        ),
        hovertemplate=(
            "Lon: %{x:.1f}°<br>"
            "Lat: %{y:.1f}°<br>"
            "Speed: %{z:.1f} m/s<extra></extra>"
        ),
    ))

    # Add wind vectors if requested
    if show_barbs:
        # Subsample for clarity
        skip = max(1, len(lats) // 15)
        lat_sub = lats[::skip]
        lon_sub = lons[::skip]
        u_sub = u[::skip, ::skip]
        v_sub = v[::skip, ::skip]

        # Normalize vectors for display
        speed_sub = np.sqrt(u_sub**2 + v_sub**2)
        scale = 5.0  # Arrow scale factor
        u_norm = u_sub / (speed_sub + 1e-6) * scale
        v_norm = v_sub / (speed_sub + 1e-6) * scale

        # Create arrow annotations
        for i, lat in enumerate(lat_sub):
            for j, lon in enumerate(lon_sub):
                if speed_sub[i, j] > 2:  # Only show arrows for non-calm wind
                    fig.add_annotation(
                        x=lon,
                        y=lat,
                        ax=lon + u_norm[i, j],
                        ay=lat + v_norm[i, j],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor="black",
                    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=height,
        xaxis=dict(ticksuffix="°", dtick=30),
        yaxis=dict(ticksuffix="°", dtick=30, scaleanchor="x", scaleratio=1),
    )

    return fig


def create_geopotential_map(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    level: int = 500,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    Create a professional geopotential height map.

    Args:
        data: 2D geopotential data (m² s⁻² or m)
        lats: Latitude values
        lons: Longitude values
        level: Pressure level (hPa) for title
        title: Plot title (auto-generated if None)
        height: Figure height

    Returns:
        Plotly Figure
    """
    if title is None:
        title = f"{level} hPa Geopotential Height"

    # Convert geopotential to height if needed (divide by g=9.81)
    if np.mean(data) > 10000:
        display_data = data / 9.81
    else:
        display_data = data

    # Set appropriate range based on level
    if level == 500:
        zmin, zmax = 5000, 6000
    elif level == 850:
        zmin, zmax = 1200, 1600
    elif level == 200:
        zmin, zmax = 11000, 13000
    else:
        zmin, zmax = np.percentile(display_data, 5), np.percentile(display_data, 95)

    fig = go.Figure()

    # Add filled contours
    fig.add_trace(go.Contour(
        z=display_data,
        x=lons,
        y=lats,
        colorscale="Viridis",
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white'),
            start=zmin,
            end=zmax,
            size=(zmax - zmin) / 10,
        ),
        colorbar=dict(
            title=dict(text="Height (m)", side="right"),
            ticksuffix=" m",
        ),
        hovertemplate=(
            "Lon: %{x:.1f}°<br>"
            "Lat: %{y:.1f}°<br>"
            "Height: %{z:.0f} m<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=height,
        xaxis=dict(ticksuffix="°", dtick=30),
        yaxis=dict(ticksuffix="°", dtick=30, scaleanchor="x", scaleratio=1),
    )

    return fig


def create_forecast_comparison(
    prediction: np.ndarray,
    truth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: str = "Temperature",
    units: str = "°C",
    height: int = 400,
) -> go.Figure:
    """
    Create a side-by-side comparison of prediction vs truth.

    Args:
        prediction: 2D prediction data
        truth: 2D ground truth data
        lats: Latitude values
        lons: Longitude values
        variable: Variable name for title
        units: Units for colorbar
        height: Figure height

    Returns:
        Plotly Figure with 3 panels (prediction, truth, error)
    """
    error = prediction - truth

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Model Prediction",
            "Ground Truth",
            "Error (Pred - Truth)"
        ),
        horizontal_spacing=0.05,
    )

    # Common range
    vmin = min(prediction.min(), truth.min())
    vmax = max(prediction.max(), truth.max())

    # Prediction
    fig.add_trace(go.Heatmap(
        z=prediction,
        x=lons,
        y=lats,
        colorscale="RdBu_r",
        zmin=vmin,
        zmax=vmax,
        showscale=False,
    ), row=1, col=1)

    # Truth
    fig.add_trace(go.Heatmap(
        z=truth,
        x=lons,
        y=lats,
        colorscale="RdBu_r",
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title=f"{variable} ({units})", x=0.62),
    ), row=1, col=2)

    # Error
    err_max = np.abs(error).max()
    fig.add_trace(go.Heatmap(
        z=error,
        x=lons,
        y=lats,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-err_max,
        zmax=err_max,
        colorbar=dict(title=f"Error ({units})", x=1.0),
    ), row=1, col=3)

    fig.update_layout(
        title=dict(text=f"{variable} Verification", x=0.5, xanchor='center'),
        height=height,
    )

    # Update all axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="Longitude", ticksuffix="°", row=1, col=i)
        fig.update_yaxes(title_text="Latitude", ticksuffix="°", row=1, col=i)

    return fig


def create_meteogram(
    times: List[datetime],
    temperature: List[float],
    precipitation: Optional[List[float]] = None,
    wind_speed: Optional[List[float]] = None,
    location_name: str = "Location",
    height: int = 500,
) -> go.Figure:
    """
    Create a meteogram (time series weather display).

    Args:
        times: List of datetime objects
        temperature: Temperature values (in display units)
        precipitation: Optional precipitation values (mm)
        wind_speed: Optional wind speed values (m/s)
        location_name: Name of location for title
        height: Figure height

    Returns:
        Plotly Figure
    """
    n_rows = 1 + (1 if precipitation is not None else 0) + (1 if wind_speed is not None else 0)

    titles = ["Temperature"]
    if precipitation is not None:
        titles.append("Precipitation")
    if wind_speed is not None:
        titles.append("Wind Speed")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.1,
        shared_xaxes=True,
    )

    row = 1

    # Temperature
    fig.add_trace(go.Scatter(
        x=times,
        y=temperature,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#e53935', width=2),
        marker=dict(size=6),
    ), row=row, col=1)
    fig.update_yaxes(title_text="°F", row=row, col=1)
    row += 1

    # Precipitation
    if precipitation is not None:
        fig.add_trace(go.Bar(
            x=times,
            y=precipitation,
            name='Precipitation',
            marker_color='#1e88e5',
        ), row=row, col=1)
        fig.update_yaxes(title_text="mm", row=row, col=1)
        row += 1

    # Wind
    if wind_speed is not None:
        fig.add_trace(go.Scatter(
            x=times,
            y=wind_speed,
            mode='lines+markers',
            name='Wind Speed',
            line=dict(color='#43a047', width=2),
            marker=dict(size=6),
        ), row=row, col=1)
        fig.update_yaxes(title_text="m/s", row=row, col=1)

    fig.update_layout(
        title=dict(text=f"Weather Forecast - {location_name}", x=0.5, xanchor='center'),
        height=height,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Time", row=n_rows, col=1)

    return fig


def create_skill_scorecard(
    models: List[str],
    metrics: dict,
    title: str = "Model Skill Scorecard",
    height: int = 400,
) -> go.Figure:
    """
    Create a skill scorecard comparing multiple models.

    Args:
        models: List of model names
        metrics: Dict with metric names as keys and lists of values as values
        title: Plot title
        height: Figure height

    Returns:
        Plotly Figure
    """
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)

    # Create heatmap data
    z_data = np.array([metrics[m] for m in metric_names])

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=models,
        y=metric_names,
        colorscale="RdYlGn",
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Model: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        xaxis_title="Model",
        yaxis_title="Metric",
    )

    return fig

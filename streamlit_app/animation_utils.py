"""
WeatherFlow Animation Utilities

Create and export weather animations as GIFs.
Designed for publication-quality outputs and easy sharing.

Features:
- GIF creation from frame sequences
- Multiple colormap options
- Customizable frame rate and size
- Direct download support in Streamlit
"""

import numpy as np
import io
from typing import List, Optional, Tuple, Union
from pathlib import Path
import base64

# Try to import imaging libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def create_weather_animation(
    frames: List[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    variable_name: str = "Temperature",
    units: str = "K",
    colormap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title_template: str = "{var} - Frame {frame}",
    fps: int = 4,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    add_colorbar: bool = True,
    add_coastlines: bool = True,
    show_timestamp: bool = True,
    timestamps: Optional[List[str]] = None,
) -> Optional[bytes]:
    """
    Create an animated GIF from weather data frames.

    Args:
        frames: List of 2D numpy arrays (lat x lon) for each frame
        lats: Latitude coordinates
        lons: Longitude coordinates
        variable_name: Name of the variable for labeling
        units: Units string for colorbar
        colormap: Matplotlib colormap name
        vmin: Minimum value for color scale (auto if None)
        vmax: Maximum value for color scale (auto if None)
        title_template: Title template with {var}, {frame}, {time} placeholders
        fps: Frames per second
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
        add_colorbar: Whether to add colorbar
        add_coastlines: Whether to add coastline approximation
        show_timestamp: Whether to show timestamp
        timestamps: Optional list of timestamp strings for each frame

    Returns:
        bytes: GIF image data as bytes, or None if creation failed
    """
    if not MPL_AVAILABLE or not PIL_AVAILABLE:
        return None

    if len(frames) == 0:
        return None

    # Determine color scale
    if vmin is None:
        vmin = min(np.nanmin(f) for f in frames)
    if vmax is None:
        vmax = max(np.nanmax(f) for f in frames)

    # Create images for each frame
    pil_frames = []

    for i, data in enumerate(frames):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        im = ax.imshow(
            data,
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        # Add coastlines approximation (simple for demo)
        if add_coastlines:
            # Draw approximate coastlines using simple shapes
            # This is a simplified version - real coastlines would use cartopy
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)  # Equator

        # Add colorbar
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax, label=f"{variable_name} ({units})")

        # Title
        time_str = timestamps[i] if timestamps and i < len(timestamps) else ""
        title = title_template.format(var=variable_name, frame=i+1, time=time_str)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        pil_frames.append(pil_img)

        plt.close(fig)

    # Create GIF
    gif_buffer = io.BytesIO()
    duration = int(1000 / fps)  # milliseconds per frame

    pil_frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def create_comparison_animation(
    frames_left: List[np.ndarray],
    frames_right: List[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    title_left: str = "Model",
    title_right: str = "Ground Truth",
    variable_name: str = "Temperature",
    units: str = "K",
    colormap: str = "RdBu_r",
    fps: int = 4,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 100,
    timestamps: Optional[List[str]] = None,
) -> Optional[bytes]:
    """
    Create a side-by-side comparison animation.

    Args:
        frames_left: List of 2D arrays for left panel
        frames_right: List of 2D arrays for right panel
        lats: Latitude coordinates
        lons: Longitude coordinates
        title_left: Title for left panel
        title_right: Title for right panel
        variable_name: Name of the variable
        units: Units string
        colormap: Matplotlib colormap
        fps: Frames per second
        figsize: Figure size
        dpi: Resolution
        timestamps: Optional timestamp strings

    Returns:
        bytes: GIF image data
    """
    if not MPL_AVAILABLE or not PIL_AVAILABLE:
        return None

    if len(frames_left) == 0 or len(frames_right) == 0:
        return None

    # Determine shared color scale
    all_data = frames_left + frames_right
    vmin = min(np.nanmin(f) for f in all_data)
    vmax = max(np.nanmax(f) for f in all_data)

    pil_frames = []
    n_frames = min(len(frames_left), len(frames_right))

    for i in range(n_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left panel
        im1 = ax1.imshow(
            frames_left[i],
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax1.set_title(title_left)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

        # Right panel
        im2 = ax2.imshow(
            frames_right[i],
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax2.set_title(title_right)
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        # Shared colorbar
        fig.colorbar(im2, ax=[ax1, ax2], label=f"{variable_name} ({units})", shrink=0.8)

        # Timestamp
        time_str = timestamps[i] if timestamps and i < len(timestamps) else f"Frame {i+1}"
        fig.suptitle(time_str, fontsize=12)

        # Convert to PIL
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        pil_frames.append(pil_img)

        plt.close(fig)

    # Create GIF
    gif_buffer = io.BytesIO()
    duration = int(1000 / fps)

    pil_frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def create_error_evolution_animation(
    prediction_frames: List[np.ndarray],
    truth_frames: List[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    variable_name: str = "Temperature",
    units: str = "K",
    fps: int = 4,
    figsize: Tuple[int, int] = (14, 4),
    dpi: int = 100,
    timestamps: Optional[List[str]] = None,
) -> Optional[bytes]:
    """
    Create animation showing prediction, truth, and error evolution.

    Args:
        prediction_frames: Model prediction frames
        truth_frames: Ground truth frames
        lats, lons: Coordinates
        variable_name: Variable name
        units: Units string
        fps: Frames per second
        figsize: Figure size
        dpi: Resolution
        timestamps: Optional timestamps

    Returns:
        bytes: GIF image data
    """
    if not MPL_AVAILABLE or not PIL_AVAILABLE:
        return None

    pil_frames = []
    n_frames = min(len(prediction_frames), len(truth_frames))

    # Determine scales
    all_data = prediction_frames + truth_frames
    vmin = min(np.nanmin(f) for f in all_data)
    vmax = max(np.nanmax(f) for f in all_data)

    errors = [p - t for p, t in zip(prediction_frames[:n_frames], truth_frames[:n_frames])]
    err_max = max(np.abs(e).max() for e in errors)

    for i in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        error = prediction_frames[i] - truth_frames[i]

        # Prediction
        im1 = axes[0].imshow(
            prediction_frames[i],
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            aspect='auto'
        )
        axes[0].set_title("Prediction")
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Truth
        im2 = axes[1].imshow(
            truth_frames[i],
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            aspect='auto'
        )
        axes[1].set_title("Ground Truth")
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Error
        im3 = axes[2].imshow(
            error,
            origin='upper' if lats[0] > lats[-1] else 'lower',
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap='RdBu_r',
            vmin=-err_max, vmax=err_max,
            aspect='auto'
        )
        axes[2].set_title(f"Error (RMSE: {np.sqrt(np.mean(error**2)):.2f})")
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        time_str = timestamps[i] if timestamps and i < len(timestamps) else f"Frame {i+1}"
        fig.suptitle(f"{variable_name} - {time_str}", fontsize=12)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        pil_frames.append(pil_img)

        plt.close(fig)

    gif_buffer = io.BytesIO()
    duration = int(1000 / fps)

    pil_frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def get_gif_download_link(gif_bytes: bytes, filename: str = "animation.gif") -> str:
    """
    Create an HTML download link for a GIF.

    Args:
        gif_bytes: GIF image data
        filename: Download filename

    Returns:
        str: HTML anchor tag for download
    """
    b64 = base64.b64encode(gif_bytes).decode()
    return f'<a href="data:image/gif;base64,{b64}" download="{filename}">Download GIF</a>'


def streamlit_gif_download(gif_bytes: bytes, filename: str = "animation.gif", label: str = "Download Animation"):
    """
    Create a Streamlit download button for a GIF.

    Use this in Streamlit apps:
        streamlit_gif_download(gif_bytes, "my_animation.gif")
    """
    import streamlit as st
    st.download_button(
        label=label,
        data=gif_bytes,
        file_name=filename,
        mime="image/gif"
    )


def generate_sample_animation_data(
    n_frames: int = 20,
    n_lat: int = 32,
    n_lon: int = 64,
    variable: str = "temperature"
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Generate sample animation data for testing.

    Args:
        n_frames: Number of frames
        n_lat: Number of latitude points
        n_lon: Number of longitude points
        variable: Variable type

    Returns:
        tuple: (frames, lats, lons, timestamps)
    """
    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(-180, 180, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    frames = []
    timestamps = []

    for t in range(n_frames):
        if variable == "temperature":
            base = 288 - 35 * np.abs(lat_grid) / 90
            wave = 10 * np.sin(np.radians(lon_grid - t * 360/n_frames) * 3)
            noise = np.random.randn(n_lat, n_lon) * 2
            data = base + wave + noise
        elif variable == "geopotential":
            base = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3)
            wave = 80 * np.sin(np.radians(lon_grid - t * 360/n_frames) * 3)
            data = base + wave + np.random.randn(n_lat, n_lon) * 20
        else:
            data = np.random.randn(n_lat, n_lon)

        frames.append(data)
        timestamps.append(f"T+{t*6}h")

    return frames, lats, lons, timestamps


# Check availability
def is_animation_available() -> bool:
    """Check if animation creation is available."""
    return PIL_AVAILABLE and MPL_AVAILABLE

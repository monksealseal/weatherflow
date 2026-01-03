"""Utilities for transforming SKEW-T/Log-P soundings into 3D visualizations.

This module provides two main components:

``SkewTImageParser``
    Extracts temperature and dewpoint profiles from a bitmap SKEW-T diagram
    using simple colour-based heuristics.  The parser maps the pixel positions
    of the sounding traces onto physical coordinates (pressure, altitude and
    temperature) and derives additional thermodynamic quantities.

``SkewT3DVisualizer``
    Turns the extracted profile into an interactive 3D Plotly figure that
    highlights the thermal and moisture structure of the atmosphere.

The implementation is intentionally modular so that more advanced parsing or
visualisation strategies can be added without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import interp1d

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - plotly is an optional dependency.
    go = None


ArrayLike = Union[np.ndarray, Iterable[float]]


@dataclass
class RGBThreshold:
    """Simple RGB range selector used to extract coloured traces from an image.

    The default values target the colour scheme commonly used in SKEW-T plots
    produced by the U.S. National Weather Service (red temperature trace and
    green dewpoint trace).  Users can provide custom thresholds if their
    sounding images use different colours.
    """

    min_r: int = 0
    max_r: int = 255
    min_g: int = 0
    max_g: int = 255
    min_b: int = 0
    max_b: int = 255

    def create_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a boolean mask selecting pixels inside the RGB range."""

        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        mask = (
            (red >= self.min_r)
            & (red <= self.max_r)
            & (green >= self.min_g)
            & (green <= self.max_g)
            & (blue >= self.min_b)
            & (blue <= self.max_b)
        )
        return mask


@dataclass
class SkewTCalibration:
    """Configuration describing how to convert pixels to physical units."""

    pressure_surface_hpa: float = 1000.0
    pressure_top_hpa: float = 100.0
    temperature_range_c: Tuple[float, float] = (-60.0, 40.0)
    bounding_box: Optional[Tuple[int, int, int, int]] = (
        None  # x_min, y_min, x_max, y_max
    )
    interpolation_levels: int = 80
    smoothing_sigma: float = 1.0
    minimum_points: int = 30
    skew_correction: float = 0.0

    def validate(self) -> None:
        if self.pressure_surface_hpa <= 0 or self.pressure_top_hpa <= 0:
            raise ValueError("Pressures must be positive.")
        if self.pressure_surface_hpa <= self.pressure_top_hpa:
            raise ValueError(
                "Surface pressure must be larger than top pressure in hPa."
            )
        if self.interpolation_levels < 3:
            raise ValueError("At least three interpolation levels are required.")


def _load_image(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Load an image from various input types and return an RGB numpy array."""

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Image array must have three colour channels.")
        if image.shape[2] > 3:
            image = image[..., :3]
        return image.astype(np.uint8)

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot find SKEW-T image: {image_path}")
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"))


def _pressure_from_fraction(
    fraction: np.ndarray, surface: float, top: float
) -> np.ndarray:
    """Convert fractional height (0 at top, 1 at bottom) to pressure (hPa)."""

    fraction = np.clip(fraction, 0.0, 1.0)
    log_top = np.log(top)
    log_surface = np.log(surface)
    log_p = log_top + fraction * (log_surface - log_top)
    return np.exp(log_p)


def _altitude_from_pressure(pressure_hpa: ArrayLike) -> np.ndarray:
    """Estimate altitude from pressure using the U.S. Standard Atmosphere."""

    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    return 44307.69396 * (1.0 - (pressure_hpa / 1013.25) ** 0.190284)


def _saturation_vapour_pressure(temp_c: ArrayLike) -> np.ndarray:
    """Return saturation vapour pressure over liquid water in hPa."""

    temp_c = np.asarray(temp_c, dtype=float)
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def _mixing_ratio(pressure_hpa: ArrayLike, dewpoint_c: ArrayLike) -> np.ndarray:
    """Compute the water vapour mixing ratio (kg/kg)."""

    pressure = np.asarray(pressure_hpa, dtype=float)
    e = _saturation_vapour_pressure(dewpoint_c)
    ratio = 0.622 * e / np.maximum(pressure - e, 1e-6)
    return ratio


def _lcl_temperature_k(temp_c: ArrayLike, dewpoint_c: ArrayLike) -> np.ndarray:
    """Approximate the temperature at the lifted condensation level (K)."""

    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    dewpoint_k = np.asarray(dewpoint_c, dtype=float) + 273.15
    return (
        1.0 / (1.0 / (dewpoint_k - 56.0) + np.log(temp_k / dewpoint_k) / 800.0) + 56.0
    )


class SkewTImageParser:
    """Parse temperature and dewpoint profiles from a SKEW-T image."""

    def __init__(
        self,
        calibration: Optional[SkewTCalibration] = None,
        temperature_threshold: Optional[RGBThreshold] = None,
        dewpoint_threshold: Optional[RGBThreshold] = None,
    ) -> None:
        self.calibration = calibration or SkewTCalibration()
        self.calibration.validate()
        # Thresholds tuned for red (temperature) and green (dewpoint) traces.
        self.temperature_threshold = temperature_threshold or RGBThreshold(
            min_r=180, min_g=0, max_g=140, min_b=0, max_b=140
        )
        self.dewpoint_threshold = dewpoint_threshold or RGBThreshold(
            min_g=150, max_r=160, min_r=0, max_b=160
        )

    @staticmethod
    def _extract_line(
        mask: np.ndarray, minimum_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse a boolean mask to mean x-positions for each y pixel."""

        if mask.ndim != 2:
            raise ValueError("Mask must be a 2-D boolean array.")
        y_indices, x_indices = np.where(mask)
        if y_indices.size < minimum_points:
            raise ValueError(
                "Unable to locate sufficient pixels for the sounding trace."
            )

        # Group by y coordinate
        unique_y = np.unique(y_indices)
        x_means = []
        for y in unique_y:
            xs = x_indices[y_indices == y]
            x_means.append(xs.mean())
        return unique_y.astype(float), np.asarray(x_means, dtype=float)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if self.calibration.bounding_box is not None:
            x0, y0, x1, y1 = self.calibration.bounding_box
            return image[y0:y1, x0:x1]
        return image

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        sigma = self.calibration.smoothing_sigma
        if sigma and sigma > 0:
            return ndimage.gaussian_filter1d(values, sigma=sigma)
        return values

    def _pixel_to_temperature(
        self, x_values: np.ndarray, y_values: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        temp_min, temp_max = self.calibration.temperature_range_c
        span = temp_max - temp_min
        if span <= 0:
            raise ValueError("Temperature range must have positive span.")
        x_norm = x_values / max(width - 1, 1)
        temperatures = temp_min + x_norm * span

        if self.calibration.skew_correction:
            y_norm = y_values / max(height - 1, 1)
            temperatures -= self.calibration.skew_correction * (y_norm - 0.5) * span
        return temperatures

    def parse(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Parse the SKEW-T image and return derived atmospheric quantities."""

        raw_image = _load_image(image)
        working_image = self._prepare_image(raw_image)
        height, width, _ = working_image.shape

        temp_mask = self.temperature_threshold.create_mask(working_image)
        dew_mask = self.dewpoint_threshold.create_mask(working_image)

        temp_y, temp_x = self._extract_line(temp_mask, self.calibration.minimum_points)
        dew_y, dew_x = self._extract_line(dew_mask, self.calibration.minimum_points)

        temp_x = self._smooth(temp_x)
        dew_x = self._smooth(dew_x)

        temp_pressures = _pressure_from_fraction(
            temp_y / max(height - 1, 1),
            self.calibration.pressure_surface_hpa,
            self.calibration.pressure_top_hpa,
        )
        dew_pressures = _pressure_from_fraction(
            dew_y / max(height - 1, 1),
            self.calibration.pressure_surface_hpa,
            self.calibration.pressure_top_hpa,
        )

        temperature_c = self._pixel_to_temperature(temp_x, temp_y, width, height)
        dewpoint_c = self._pixel_to_temperature(dew_x, dew_y, width, height)

        pressure_grid = np.linspace(
            self.calibration.pressure_surface_hpa,
            self.calibration.pressure_top_hpa,
            self.calibration.interpolation_levels,
        )

        temp_interp = interp1d(
            temp_pressures,
            temperature_c,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
            assume_sorted=False,
        )
        dew_interp = interp1d(
            dew_pressures,
            dewpoint_c,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
            assume_sorted=False,
        )

        temperature_grid = temp_interp(pressure_grid)
        dewpoint_grid = dew_interp(pressure_grid)

        altitude_m = _altitude_from_pressure(pressure_grid)
        relative_humidity = _saturation_vapour_pressure(dewpoint_grid) / np.maximum(
            _saturation_vapour_pressure(temperature_grid), 1e-6
        )
        relative_humidity = np.clip(relative_humidity * 100.0, 0.0, 100.0)

        mixing_ratio = _mixing_ratio(pressure_grid, dewpoint_grid)
        temp_k = temperature_grid + 273.15
        theta_k = temp_k * (1000.0 / pressure_grid) ** 0.2854
        lcl_temp_k = _lcl_temperature_k(temperature_grid, dewpoint_grid)
        theta_l = temp_k * (1000.0 / pressure_grid) ** (
            0.2854 * (1 - 0.28 * mixing_ratio)
        )
        theta_e = theta_l * np.exp(
            (3376.0 / np.maximum(lcl_temp_k, 1.0) - 2.54)
            * mixing_ratio
            * (1 + 0.81 * mixing_ratio)
        )
        virtual_temperature_k = temp_k * (1 + 0.61 * mixing_ratio)

        return {
            "pressure_hpa": pressure_grid,
            "altitude_m": altitude_m,
            "altitude_km": altitude_m / 1000.0,
            "temperature_c": temperature_grid,
            "temperature_k": temp_k,
            "dewpoint_c": dewpoint_grid,
            "dewpoint_k": dewpoint_grid + 273.15,
            "relative_humidity_percent": relative_humidity,
            "mixing_ratio_kgkg": mixing_ratio,
            "mixing_ratio_gkg": mixing_ratio * 1000.0,
            "theta_k": theta_k,
            "theta_e_k": theta_e,
            "virtual_temperature_k": virtual_temperature_k,
        }


class SkewT3DVisualizer:
    """Create an interactive 3D Plotly representation of a sounding profile."""

    def __init__(
        self,
        curtain_steps: int = 40,
        surface_opacity: float = 0.85,
        colorscale: str = "RdYlBu_r",
    ) -> None:
        if go is None:  # pragma: no cover - handled by dependency checks.
            raise ImportError(
                "Plotly is required for 3D visualisation. Install weatherflow with"
                " the optional plotting dependencies."
            )
        self.curtain_steps = max(3, curtain_steps)
        self.surface_opacity = np.clip(surface_opacity, 0.1, 1.0)
        self.colorscale = colorscale

    def create_figure(
        self,
        profile: Dict[str, np.ndarray],
        title: str = "3D Atmospheric Structure from SKEW-T",
    ) -> "go.Figure":
        """Return a configured Plotly figure visualising the sounding.

        Raises:
            ImportError: If plotly is not installed
        """
        if go is None:
            raise ImportError(
                "Plotly is required for 3D visualization. "
                "Install it with: pip install plotly"
            )

        altitude_km = np.asarray(profile["altitude_km"], dtype=float)
        temperature_c = np.asarray(profile["temperature_c"], dtype=float)
        dewpoint_c = np.asarray(profile["dewpoint_c"], dtype=float)
        humidity = np.asarray(profile["relative_humidity_percent"], dtype=float)

        steps = self.curtain_steps
        interp = np.linspace(0.0, 1.0, steps)[:, None]
        temp_surface = (
            dewpoint_c[None, :] + (temperature_c - dewpoint_c)[None, :] * interp
        )
        humidity_surface = np.tile(humidity[None, :], (steps, 1))
        altitude_surface = np.tile(altitude_km[None, :], (steps, 1))

        fig = go.Figure()
        fig.add_surface(
            x=altitude_surface,
            y=temp_surface,
            z=humidity_surface,
            surfacecolor=temp_surface,
            colorscale=self.colorscale,
            opacity=self.surface_opacity,
            colorbar=dict(title="Temperature (°C)"),
            showscale=True,
            name="Thermal-Moisture Curtain",
        )

        hover_template = (
            "Altitude: %{x:.2f} km<br>"
            "Temperature: %{y:.1f} °C<br>"
            "Relative Humidity: %{z:.0f}%"
        )

        fig.add_trace(
            go.Scatter3d(
                x=altitude_km,
                y=temperature_c,
                z=humidity,
                mode="lines+markers",
                name="Temperature",
                line=dict(color="#d7191c", width=6),
                marker=dict(size=5, color="#d7191c"),
                hovertemplate=hover_template,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=altitude_km,
                y=dewpoint_c,
                z=humidity,
                mode="lines+markers",
                name="Dewpoint",
                line=dict(color="#1a9641", width=6),
                marker=dict(size=5, color="#1a9641"),
                hovertemplate=hover_template,
            )
        )

        if "theta_k" in profile:
            theta_c = np.asarray(profile["theta_k"], dtype=float) - 273.15
            fig.add_trace(
                go.Scatter3d(
                    x=altitude_km,
                    y=theta_c,
                    z=humidity,
                    mode="lines",
                    name="Potential Temp",
                    line=dict(color="#ffa600", width=3, dash="dash"),
                    hovertemplate=hover_template,
                )
            )

        if "theta_e_k" in profile:
            thetae_c = np.asarray(profile["theta_e_k"], dtype=float) - 273.15
            fig.add_trace(
                go.Scatter3d(
                    x=altitude_km,
                    y=thetae_c,
                    z=humidity,
                    mode="lines",
                    name="Equivalent Potential Temp",
                    line=dict(color="#2c7bb6", width=3, dash="dot"),
                    hovertemplate=hover_template,
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="Altitude (km)", backgroundcolor="#1b1e23"),
                yaxis=dict(
                    title="Temperature / Dewpoint (°C)", backgroundcolor="#1b1e23"
                ),
                zaxis=dict(title="Relative Humidity (%)", backgroundcolor="#1b1e23"),
                bgcolor="#0f1116",
            ),
            paper_bgcolor="#0f1116",
            plot_bgcolor="#0f1116",
            legend=dict(bgcolor="#0f1116", font=dict(color="#f0f0f0")),
        )

        return fig

    def create_and_save(
        self,
        profile: Dict[str, np.ndarray],
        output_html: Union[str, Path],
        title: str = "3D Atmospheric Structure from SKEW-T",
    ) -> Path:
        """Create the Plotly figure and save it as an interactive HTML file."""

        figure = self.create_figure(profile, title=title)
        output_path = Path(output_html)
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

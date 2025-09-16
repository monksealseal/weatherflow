"""Tests for the SKEW-T parsing and visualisation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weatherflow.utils import SkewT3DVisualizer, SkewTCalibration, SkewTImageParser


def _create_synthetic_skewt_image(tmp_path: Path) -> Path:
    width, height = 220, 360
    image = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)

    for y in range(height):
        temp_x = 45 + 0.32 * y
        dew_x = 30 + 0.18 * y
        draw.rectangle((temp_x - 1, y - 1, temp_x + 1, y + 1), fill=(220, 50, 50))
        draw.rectangle((dew_x - 1, y - 1, dew_x + 1, y + 1), fill=(40, 190, 90))

    path = tmp_path / "synthetic_skewt.png"
    image.save(path)
    return path


def _expected_profile(
    pressure: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    temp_min, temp_max = -55.0, 45.0
    span = temp_max - temp_min

    log_top = np.log(100.0)
    log_surface = np.log(1000.0)
    fraction = (np.log(pressure) - log_top) / (log_surface - log_top)
    y_pixels = fraction * (height - 1)

    temp_x = 45 + 0.32 * y_pixels
    dew_x = 30 + 0.18 * y_pixels

    temp = temp_min + (temp_x / (width - 1)) * span
    dew = temp_min + (dew_x / (width - 1)) * span
    return temp, dew


def test_skewt_parser_extracts_reasonable_profile(tmp_path: Path) -> None:
    image_path = _create_synthetic_skewt_image(tmp_path)
    calibration = SkewTCalibration(
        pressure_surface_hpa=1000.0,
        pressure_top_hpa=100.0,
        temperature_range_c=(-55.0, 45.0),
        interpolation_levels=60,
        smoothing_sigma=0.0,
    )

    parser = SkewTImageParser(calibration=calibration)
    profile = parser.parse(image_path)

    assert set(profile) >= {
        "pressure_hpa",
        "temperature_c",
        "dewpoint_c",
        "relative_humidity_percent",
        "mixing_ratio_gkg",
    }

    expected_temp, expected_dew = _expected_profile(
        profile["pressure_hpa"],
        width=220,
        height=360,
    )

    assert np.allclose(profile["temperature_c"], expected_temp, atol=1.5)
    assert np.allclose(profile["dewpoint_c"], expected_dew, atol=1.5)
    assert np.all(profile["relative_humidity_percent"] <= 100.0 + 1e-6)
    assert np.all(profile["relative_humidity_percent"] >= -1e-6)
    assert profile["mixing_ratio_gkg"].shape == profile["pressure_hpa"].shape


def test_skewt_visualizer_returns_plotly_figure(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    image_path = _create_synthetic_skewt_image(tmp_path)
    calibration = SkewTCalibration(
        temperature_range_c=(-55.0, 45.0),
        interpolation_levels=20,
        smoothing_sigma=0.0,
    )

    profile = SkewTImageParser(calibration=calibration).parse(image_path)

    visualizer = SkewT3DVisualizer(curtain_steps=10, surface_opacity=0.9)
    figure = visualizer.create_figure(profile, title="Synthetic Sounding")

    assert figure.layout.title.text == "Synthetic Sounding"
    assert figure.layout.scene.xaxis.title.text == "Altitude (km)"
    assert figure.layout.scene.yaxis.title.text.startswith("Temperature")
    assert figure.layout.scene.zaxis.title.text == "Relative Humidity (%)"
    assert len(figure.data) >= 3

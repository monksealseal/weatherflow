"""Command line utility to turn SKEW-T soundings into 3D interactive plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from weatherflow.utils import (
    RGBThreshold,
    SkewT3DVisualizer,
    SkewTCalibration,
    SkewTImageParser,
)


def _threshold_from_args(values: Optional[list[int]]) -> Optional[RGBThreshold]:
    if values is None:
        return None
    min_r, max_r, min_g, max_g, min_b, max_b = values
    return RGBThreshold(
        min_r=min_r,
        max_r=max_r,
        min_g=min_g,
        max_g=max_g,
        min_b=min_b,
        max_b=max_b,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a SKEW-T/Log-P image into an interactive 3D HTML visualisation.",
    )
    parser.add_argument("image", type=Path, help="Path to the SKEW-T image file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("skewt_visualization.html"),
        help="Location of the generated interactive HTML file.",
    )
    parser.add_argument(
        "--surface-pressure",
        type=float,
        default=1000.0,
        help="Surface pressure in hPa represented by the bottom of the diagram.",
    )
    parser.add_argument(
        "--top-pressure",
        type=float,
        default=100.0,
        help="Top pressure in hPa represented by the top of the diagram.",
    )
    parser.add_argument(
        "--temperature-range",
        type=float,
        nargs=2,
        default=(-60.0, 40.0),
        metavar=("MIN", "MAX"),
        help="Temperature range (°C) covered by the SKEW-T background grid.",
    )
    parser.add_argument(
        "--bounding-box",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional bounding box isolating the plotted area of the SKEW-T image.",
    )
    parser.add_argument(
        "--skew-correction",
        type=float,
        default=0.0,
        help="Fractional skew adjustment applied while converting pixels to temperature.",
    )
    parser.add_argument(
        "--interpolation-levels",
        type=int,
        default=80,
        help="Number of vertical levels used in the derived profile.",
    )
    parser.add_argument(
        "--curtain-steps",
        type=int,
        default=40,
        help="Resolution of the 3D curtain interpolating between temperature and dewpoint.",
    )
    parser.add_argument(
        "--surface-opacity",
        type=float,
        default=0.85,
        help="Opacity of the thermal-moisture curtain (between 0 and 1).",
    )
    parser.add_argument(
        "--colorscale",
        default="RdYlBu_r",
        help="Plotly colourscale used to shade the thermal curtain.",
    )
    parser.add_argument(
        "--temperature-threshold",
        type=int,
        nargs=6,
        metavar=("MIN_R", "MAX_R", "MIN_G", "MAX_G", "MIN_B", "MAX_B"),
        help="Custom RGB bounds used to isolate the temperature trace.",
    )
    parser.add_argument(
        "--dewpoint-threshold",
        type=int,
        nargs=6,
        metavar=("MIN_R", "MAX_R", "MIN_G", "MAX_G", "MIN_B", "MAX_B"),
        help="Custom RGB bounds used to isolate the dewpoint trace.",
    )
    parser.add_argument(
        "--title",
        default="3D Atmospheric Structure from SKEW-T",
        help="Title of the generated figure.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> Path:
    args = build_parser().parse_args(argv)

    calibration = SkewTCalibration(
        pressure_surface_hpa=args.surface_pressure,
        pressure_top_hpa=args.top_pressure,
        temperature_range_c=tuple(args.temperature_range),
        bounding_box=tuple(args.bounding_box) if args.bounding_box else None,
        interpolation_levels=args.interpolation_levels,
        skew_correction=args.skew_correction,
    )

    parser = SkewTImageParser(
        calibration=calibration,
        temperature_threshold=_threshold_from_args(args.temperature_threshold),
        dewpoint_threshold=_threshold_from_args(args.dewpoint_threshold),
    )

    profile = parser.parse(args.image)
    visualizer = SkewT3DVisualizer(
        curtain_steps=args.curtain_steps,
        surface_opacity=args.surface_opacity,
        colorscale=args.colorscale,
    )

    return visualizer.create_and_save(profile, args.output, title=args.title)


if __name__ == "__main__":  # pragma: no cover - manual execution helper.
    main()

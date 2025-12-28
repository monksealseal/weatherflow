"""Command-line entry point for running WeatherFlow simulations."""
from __future__ import annotations

import argparse
from pathlib import Path

from weatherflow.simulation.orchestrator import SimulationOrchestrator


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the WeatherFlow CLI."""
    parser = argparse.ArgumentParser(description="WeatherFlow Game CLI")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario identifier or description to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output path for results (plots, dumps, etc.).",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="Directory for screenshots or clips (defaults to --output).",
    )
    parser.add_argument(
        "--capture-modes",
        type=str,
        default="",
        help="Comma-separated capture modes (pilot,satellite,lat-height,lon-height).",
    )
    parser.add_argument(
        "--randall-source",
        type=str,
        default="",
        help="Optional Randall reference URL for traceability.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the simulation (default is dry-run preview).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    orchestrator = SimulationOrchestrator()
    grid_lat, grid_lon, _ = orchestrator.resolve_grid_size(32, 64, "global-low")
    capture_dir = args.capture_dir or args.output
    capture_modes = [mode for mode in args.capture_modes.split(",") if mode]

    # Placeholder for future integration with the full simulation stack.
    print(f"Running scenario: {args.scenario}")
    print(f"Seed: {args.seed}")
    print(f"Grid: {grid_lat}x{grid_lon}")
    print(f"Output directory: {capture_dir}")
    if capture_modes:
        print(f"Capture modes: {', '.join(capture_modes)}")
    if args.randall_source:
        print(f"Randall source: {args.randall_source}")
    print("Mode: run" if args.run else "Mode: dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

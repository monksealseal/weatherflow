#!/usr/bin/env python3
"""
Interactive 3D Tropic World Simulation

This script runs a Tropic World simulation with interactive 3D visualizations
that update after every simulated day. The simulation pauses after each day
to render the visualization, then continues.

The visualizations are displayed in your web browser and include:
- 3D globe showing Sea Surface Temperature (SST)
- 3D atmospheric slice showing temperature structure
- 2D SST map with detailed coloring
- Temperature cross-section sorted by SST area fraction

Usage:
    python run_tropic_world_interactive_3d.py

    # Or with custom settings:
    python run_tropic_world_interactive_3d.py --days 30 --output-dir ./my_viz

Requirements:
    - plotly (for interactive 3D plots)
    - numpy
    - webbrowser (standard library, opens plots in browser)
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcm import GCM
from gcm.visualization import Interactive3DVisualizer


def run_interactive_3d_simulation(
    duration_days=10,
    output_dir=None,
    auto_open=True,
    base_sst=300.0,
    sst_perturbation=0.5,
    nlon=64,
    nlat=32,
    nlev=20,
    dt=600.0
):
    """
    Run a Tropic World simulation with interactive 3D visualization.

    The simulation pauses after each day to render and display
    an interactive 3D visualization in your browser.

    Parameters
    ----------
    duration_days : int
        Number of days to simulate
    output_dir : str, optional
        Directory to save visualization HTML files.
        If None, uses a temporary directory.
    auto_open : bool
        Whether to automatically open visualizations in browser
    base_sst : float
        Base sea surface temperature (K)
    sst_perturbation : float
        Initial SST perturbation amplitude (K)
    nlon, nlat, nlev : int
        Grid resolution
    dt : float
        Time step in seconds

    Returns
    -------
    model : GCM
        The GCM model instance after simulation
    visualizer : Interactive3DVisualizer
        The visualizer instance with history
    """
    print("=" * 70)
    print("TROPIC WORLD - INTERACTIVE 3D VISUALIZATION")
    print("=" * 70)
    print()
    print("This simulation will pause after each day to render")
    print("interactive 3D visualizations in your web browser.")
    print()
    print("Configuration:")
    print(f"  - Duration: {duration_days} days")
    print(f"  - Grid: {nlon}x{nlat}x{nlev}")
    print(f"  - Time step: {dt} seconds")
    print(f"  - Base SST: {base_sst} K")
    print(f"  - SST perturbation: {sst_perturbation} K")
    print()

    # Create GCM in Tropic World mode
    print("Initializing GCM...")
    model = GCM(
        nlon=nlon,
        nlat=nlat,
        nlev=nlev,
        dt=dt,
        integration_method='rk3',
        co2_ppmv=400.0,
        tropic_world=True,
        tropic_world_sst=base_sst,
        sst_perturbation=sst_perturbation,
        mixed_layer_depth=50.0
    )

    # Initialize
    model.initialize(profile='tropical')

    # Create visualizer
    print("Creating 3D visualizer...")
    visualizer = Interactive3DVisualizer(
        output_dir=output_dir,
        auto_open=auto_open
    )

    print(f"Visualizations will be saved to: {visualizer.output_dir}")
    print()

    # Define visualization callback
    def viz_callback(model, day):
        """Called after each simulated day"""
        html_path = visualizer.update(model, day)
        print(f"    Saved: {html_path}")

    # Run simulation with visualization
    print("Starting simulation...")
    print("-" * 70)

    model.run_with_visualization(
        duration_days=duration_days,
        day_callback=viz_callback,
        output_interval_hours=12
    )

    print("-" * 70)
    print()

    # Create final animation
    print("Creating animation from simulation history...")
    anim_path = visualizer.create_animation(
        visualizer.history,
        filename='tropic_world_animation.html'
    )
    print(f"Animation saved to: {anim_path}")

    # Print final statistics
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    import numpy as np
    sst = model.ocean.sst
    sst_mean = model.grid.global_mean(sst)
    sst_contrast = np.max(sst) - np.min(sst)
    u_surf = model.state.u[-1]
    v_surf = model.state.v[-1]
    wind_speed = np.sqrt(u_surf**2 + v_surf**2)

    print(f"\nFinal Statistics:")
    print(f"  Global mean SST: {sst_mean:.2f} K")
    print(f"  SST contrast: {sst_contrast:.2f} K")
    print(f"  Max surface wind: {np.max(wind_speed):.2f} m/s")

    print(f"\nVisualization files saved to: {visualizer.output_dir}")
    print(f"  - latest.html: Most recent visualization")
    print(f"  - tropic_world_day_*.html: Each day's snapshot")
    print(f"  - tropic_world_animation.html: Animated visualization")

    return model, visualizer


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run Tropic World simulation with interactive 3D visualization'
    )
    parser.add_argument(
        '--days', type=int, default=10,
        help='Number of days to simulate (default: 10)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save visualizations (default: temp directory)'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not automatically open browser'
    )
    parser.add_argument(
        '--base-sst', type=float, default=300.0,
        help='Base SST in Kelvin (default: 300)'
    )
    parser.add_argument(
        '--sst-perturbation', type=float, default=0.5,
        help='SST perturbation amplitude in K (default: 0.5)'
    )
    parser.add_argument(
        '--resolution', type=str, default='medium',
        choices=['low', 'medium', 'high'],
        help='Grid resolution (default: medium)'
    )

    args = parser.parse_args()

    # Set resolution
    resolution_map = {
        'low': (32, 16, 10),
        'medium': (64, 32, 20),
        'high': (96, 48, 30)
    }
    nlon, nlat, nlev = resolution_map[args.resolution]

    # Run simulation
    model, viz = run_interactive_3d_simulation(
        duration_days=args.days,
        output_dir=args.output_dir,
        auto_open=not args.no_browser,
        base_sst=args.base_sst,
        sst_perturbation=args.sst_perturbation,
        nlon=nlon,
        nlat=nlat,
        nlev=nlev
    )

    return model, viz


if __name__ == '__main__':
    main()

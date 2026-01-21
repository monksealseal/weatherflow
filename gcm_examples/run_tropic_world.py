"""
Tropic World Simulation Example

This script demonstrates the Tropic World configuration of the GCM,
based on Section 2.4 "Tropic World: Convection on a Planetary Scale"
from "Heuristic Models of the General Circulation".

Tropic World is an idealized planet that is:
- Non-rotating (no Coriolis force)
- Uniform solar radiation over entire surface
- Covered by a 50m deep slab ocean
- No horizontal heat transport within the ocean

Key phenomena to observe:
1. Spontaneous development of warm and cold SST regions
2. SST contrast oscillation with ~2-3 year period (in model time)
3. Stronger surface winds where SST contrasts are strongest
4. Convection concentrated over warm regions
5. Higher relative humidity in warm regions (greenhouse feedback)

The simulation demonstrates the fundamental instability where:
- Warm regions attract convection
- Convection moistens the atmosphere
- Moisture warms the surface through greenhouse effect
- Cool regions are drier and lose more OLR to space

Run time: This example runs for 100 days by default to show initial
development. For full 2-3 year cycles, run for 1000+ days.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcm import GCM


def run_tropic_world(duration_days=100, output_interval_hours=12):
    """
    Run a Tropic World simulation

    Parameters
    ----------
    duration_days : int
        Simulation duration in days
    output_interval_hours : float
        Output interval for diagnostics

    Returns
    -------
    model : GCM
        The GCM model instance after simulation
    """
    print("=" * 60)
    print("TROPIC WORLD SIMULATION")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - Non-rotating planet")
    print("  - Uniform solar radiation")
    print("  - 50m slab ocean")
    print("  - Initial SST: 300 K with small perturbations")
    print()

    # Create GCM in Tropic World mode
    model = GCM(
        nlon=64,          # Longitude points
        nlat=32,          # Latitude points
        nlev=20,          # Vertical levels
        dt=600.0,         # Time step (seconds)
        integration_method='rk3',
        co2_ppmv=400.0,
        # Tropic World configuration
        tropic_world=True,
        tropic_world_sst=300.0,     # Base SST (K)
        sst_perturbation=0.5,       # Initial perturbation amplitude (K)
        mixed_layer_depth=50.0      # Slab ocean depth (m)
    )

    # Initialize
    model.initialize(profile='tropical')

    # Run simulation
    model.run(duration_days=duration_days, output_interval_hours=output_interval_hours)

    return model


def analyze_results(model):
    """
    Analyze and display Tropic World results

    Parameters
    ----------
    model : GCM
        GCM model instance after simulation
    """
    import numpy as np

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    # Get final state statistics
    sst = model.ocean.sst
    sst_mean = model.grid.global_mean(sst)
    sst_max = np.max(sst)
    sst_min = np.min(sst)
    sst_contrast = sst_max - sst_min

    print(f"\nFinal SST Statistics:")
    print(f"  Global mean SST: {sst_mean:.2f} K ({sst_mean - 273.15:.2f} C)")
    print(f"  Maximum SST: {sst_max:.2f} K")
    print(f"  Minimum SST: {sst_min:.2f} K")
    print(f"  SST Contrast: {sst_contrast:.2f} K")

    # Wind statistics
    u_surf = model.state.u[-1]
    v_surf = model.state.v[-1]
    wind_speed = np.sqrt(u_surf**2 + v_surf**2)
    print(f"\nSurface Wind Statistics:")
    print(f"  Maximum wind speed: {np.max(wind_speed):.2f} m/s")
    print(f"  Mean wind speed: {np.mean(wind_speed):.2f} m/s")

    # Diagnostics over time
    if 'sst_contrast' in model.diagnostics:
        contrasts = model.diagnostics['sst_contrast']
        print(f"\nSST Contrast Evolution:")
        print(f"  Initial contrast: {contrasts[0]:.2f} K")
        print(f"  Final contrast: {contrasts[-1]:.2f} K")
        print(f"  Maximum contrast: {max(contrasts):.2f} K")


def main():
    """Main function"""
    # Run simulation
    model = run_tropic_world(duration_days=100, output_interval_hours=12)

    # Analyze results
    analyze_results(model)

    # Plot results
    print("\nGenerating plots...")
    try:
        # Plot Tropic World state (SST + winds)
        model.plot_tropic_world(filename='tropic_world_state.png')

        # Plot Tropic World diagnostics
        model.plot_tropic_world_diagnostics(filename='tropic_world_diagnostics.png')

        # Standard diagnostics
        model.plot_diagnostics(filename='tropic_world_standard_diagnostics.png')

        print("\nPlots saved!")
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("Continuing without plots...")

    print("\nSimulation complete!")
    return model


if __name__ == '__main__':
    main()

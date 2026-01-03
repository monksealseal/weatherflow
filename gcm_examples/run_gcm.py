#!/usr/bin/env python
"""
Example script to run the GCM

This demonstrates how to initialize and run a simulation
"""

import sys
sys.path.insert(0, '..')

from gcm import GCM
import numpy as np


def main():
    """Run a simple GCM simulation"""

    print("=" * 60)
    print("General Circulation Model - Example Run")
    print("=" * 60)

    # Create GCM instance
    # Resolution: 64 longitude x 32 latitude x 20 vertical levels
    model = GCM(
        nlon=64,
        nlat=32,
        nlev=20,
        dt=600,  # 10-minute time step
        integration_method='rk3',
        co2_ppmv=400.0
    )

    # Initialize with tropical atmosphere
    model.initialize(profile='tropical', sst_pattern='realistic')

    # Print initial state
    print("\nInitial conditions:")
    print(f"  Global mean surface temperature: {np.mean(model.state.tsurf):.2f} K")
    print(f"  Global mean surface pressure: {np.mean(model.state.ps):.2f} Pa")
    print(f"  Max zonal wind: {np.max(model.state.u):.2f} m/s")

    # Run simulation
    print("\n" + "=" * 60)
    model.run(duration_days=10, output_interval_hours=6)

    # Print final state
    print("\nFinal conditions:")
    print(f"  Global mean surface temperature: {np.mean(model.state.tsurf):.2f} K")
    print(f"  Global mean surface pressure: {np.mean(model.state.ps):.2f} Pa")
    print(f"  Max zonal wind: {np.max(model.state.u):.2f} m/s")

    # Plot diagnostics
    print("\n" + "=" * 60)
    print("Generating diagnostic plots...")
    model.plot_diagnostics('gcm_diagnostics.png')
    model.plot_state('gcm_state.png')

    print("\nSimulation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

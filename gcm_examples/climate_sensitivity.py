#!/usr/bin/env python
"""
Climate sensitivity experiment

Tests the model's response to different CO2 concentrations
"""

import sys
sys.path.insert(0, '..')

from gcm import GCM
import numpy as np


def run_equilibrium_simulation(co2_ppmv, duration_days=100):
    """
    Run model to equilibrium at specified CO2 level

    Parameters
    ----------
    co2_ppmv : float
        CO2 concentration in ppmv
    duration_days : float
        Simulation duration

    Returns
    -------
    mean_temp : float
        Equilibrium global mean temperature
    """
    print(f"\nRunning simulation with CO2 = {co2_ppmv} ppmv...")

    model = GCM(
        nlon=48,
        nlat=24,
        nlev=16,
        dt=600,
        integration_method='rk3',
        co2_ppmv=co2_ppmv
    )

    model.initialize(profile='midlatitude')
    model.run(duration_days=duration_days, output_interval_hours=24)

    # Get equilibrium temperature (average of last 10 days)
    last_temps = model.diagnostics['global_mean_T'][-10:]
    mean_temp = np.mean(last_temps)

    print(f"  Equilibrium temperature: {mean_temp:.2f} K")

    return mean_temp


def main():
    """Run climate sensitivity experiment"""

    print("=" * 60)
    print("Climate Sensitivity Experiment")
    print("=" * 60)

    # Test different CO2 levels
    co2_levels = [280, 400, 560, 800]  # Pre-industrial, current, 2x, higher
    temperatures = []

    for co2 in co2_levels:
        temp = run_equilibrium_simulation(co2, duration_days=50)
        temperatures.append(temp)

    # Compute climate sensitivity
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    for co2, temp in zip(co2_levels, temperatures):
        delta_T = temp - temperatures[0]  # Relative to pre-industrial
        print(f"CO2 = {co2:4d} ppmv: T = {temp:.2f} K (ΔT = {delta_T:+.2f} K)")

    # 2x CO2 sensitivity
    idx_2x = co2_levels.index(560)
    sensitivity = temperatures[idx_2x] - temperatures[0]
    print(f"\nClimate sensitivity (2xCO2): {sensitivity:.2f} K")

    print("=" * 60)


if __name__ == '__main__':
    main()

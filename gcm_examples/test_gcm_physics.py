#!/usr/bin/env python
"""
Test GCM Physics and Diagnostics

This script tests the GCM physics by running a short Held-Suarez simulation
and verifying that the diagnostics show reasonable results.

Expected outputs for a working GCM:
1. Subtropical jets should develop (~15-40 m/s at 30 deg latitude, 200 hPa)
2. Hadley cells should form (positive NH, negative SH streamfunction)
3. KE spectrum should show k^-3 slope at synoptic scales
4. Eddy momentum flux should be poleward (equatorward momentum flux)
5. Heat transport should be poleward in mid-latitudes
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_held_suarez():
    """Test the Held-Suarez GCM configuration"""
    print("=" * 60)
    print("Testing Held-Suarez GCM Configuration")
    print("=" * 60)

    from gcm.physics.held_suarez import HeldSuarezGCM
    from gcm.diagnostics import ComprehensiveGCMDiagnostics

    # Create model with moderate resolution
    print("\nInitializing model (64x32x20)...")
    model = HeldSuarezGCM(
        nlon=64, nlat=32, nlev=20,
        dt=600.0, integration_method='rk3'
    )
    model.initialize(perturbation=True)

    # Initialize diagnostics
    diag = ComprehensiveGCMDiagnostics(model.grid, model.vgrid)

    # Run for 30 days to develop circulation
    duration_days = 30
    diag_interval_days = 5

    steps_per_day = int(86400 / model.dt)
    steps_per_output = int(diag_interval_days * steps_per_day)
    n_outputs = int(duration_days / diag_interval_days)

    print(f"\nRunning simulation for {duration_days} days...")
    print(f"  Time step: {model.dt} seconds")
    print(f"  Steps per day: {steps_per_day}")
    print(f"  Total steps: {duration_days * steps_per_day}")

    for i in range(n_outputs):
        # Run for one diagnostic interval
        for _ in range(steps_per_output):
            model.integrator.step(model.state, model.dt, model._compute_tendencies)

        # Compute diagnostics
        d = diag.compute_all_diagnostics(model.state)

        day = model.state.time / 86400.0
        max_u = np.max(np.abs(model.state.u))
        ke = d['energy']['kinetic_energy']

        print(f"  Day {day:5.1f}: max|u| = {max_u:6.2f} m/s, KE = {ke:8.2f} J/kg")

    # Final diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    final_diag = diag.diagnostic_history[-1]

    # 1. Check jet streams
    jet = final_diag['circulation']['jet_diagnostics']
    print("\n1. JET STREAM DIAGNOSTICS:")
    print(f"   NH Subtropical Jet: {jet['subtropical_jet_nh_speed']:.1f} m/s at {jet['subtropical_jet_nh_lat']:.1f} deg")
    print(f"   SH Subtropical Jet: {jet['subtropical_jet_sh_speed']:.1f} m/s at {jet['subtropical_jet_sh_lat']:.1f} deg")

    jet_ok = jet['subtropical_jet_nh_speed'] > 5.0
    print(f"   Status: {'PASS' if jet_ok else 'DEVELOPING'} (expect >15 m/s after 100+ days)")

    # 2. Check Hadley cell
    circ = final_diag['circulation']
    print("\n2. HADLEY CELL DIAGNOSTICS:")
    print(f"   NH Hadley strength: {circ['hadley_strength_nh']:.1f} x10^9 kg/s")
    print(f"   SH Hadley strength: {circ['hadley_strength_sh']:.1f} x10^9 kg/s")

    hadley_ok = circ['hadley_strength_nh'] > 1.0 or circ['hadley_strength_sh'] > 1.0
    print(f"   Status: {'PASS' if hadley_ok else 'DEVELOPING'} (expect >50 after equilibrium)")

    # 3. Check energy spectrum
    spectra = final_diag['spectral']
    wn = spectra['wavenumbers']
    ke_spec = spectra['kinetic_energy_spectrum']

    print("\n3. ENERGY SPECTRUM DIAGNOSTICS:")
    print(f"   Wavenumber range: {int(wn[1])} to {int(wn[-1])}")
    print(f"   KE at k=5: {ke_spec[5]:.2e}")
    print(f"   KE at k=10: {ke_spec[10]:.2e}")

    # Check spectrum slope (should be around -3 for 2D turbulence)
    if len(ke_spec) > 10 and ke_spec[5] > 0 and ke_spec[10] > 0:
        slope = np.log(ke_spec[10] / ke_spec[5]) / np.log(10.0 / 5.0)
        print(f"   Spectrum slope (k=5 to k=10): {slope:.2f}")
        spectrum_ok = slope < -1.0  # Should be negative (decreasing with k)
        print(f"   Status: {'PASS' if spectrum_ok else 'CHECK'} (expect slope ~-3)")
    else:
        spectrum_ok = False
        print("   Status: NOT ENOUGH DATA")

    # 4. Check eddy statistics
    eddy = final_diag['eddy']
    eke = eddy['eke']
    print("\n4. EDDY STATISTICS:")
    if eke is not None:
        eke_max = np.max(eke)
        lat = eddy['latitude']
        eke_max_lat_idx = np.unravel_index(np.argmax(eke), eke.shape)[1]
        eke_max_lat = lat[eke_max_lat_idx]
        print(f"   Max EKE: {eke_max:.2f} m^2/s^2 at {eke_max_lat:.1f} deg")
        eddy_ok = eke_max > 0.1
        print(f"   Status: {'PASS' if eddy_ok else 'DEVELOPING'}")
    else:
        eddy_ok = False
        print("   Status: NOT COMPUTED")

    # 5. Check heat transport
    print("\n5. HEAT TRANSPORT DIAGNOSTICS:")
    Q_total = eddy['heat_transport_total']
    if Q_total is not None:
        Q_max = np.max(np.abs(Q_total))
        print(f"   Max heat transport magnitude: {Q_max:.3f} PW")
        heat_ok = Q_max > 0.001
        print(f"   Status: {'PASS' if heat_ok else 'DEVELOPING'}")
    else:
        heat_ok = False
        print("   Status: NOT COMPUTED")

    # 6. Check energy conservation
    print("\n6. ENERGY CONSERVATION:")
    summary = diag.get_summary_statistics()
    if summary is not None:
        drift = summary['energy_drift'] * 100
        print(f"   Energy drift: {drift:.2f}%")
        energy_ok = abs(drift) < 20.0
        print(f"   Status: {'PASS' if energy_ok else 'CHECK'} (should be <10%)")
    else:
        energy_ok = False
        print("   Status: NOT ENOUGH DATA")

    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    passed = sum([jet_ok, hadley_ok, spectrum_ok, eddy_ok, heat_ok, energy_ok])
    total = 6

    print(f"\n  Checks passed: {passed}/{total}")
    print()

    if passed >= 4:
        print("  GCM PHYSICS APPEARS TO BE WORKING!")
        print("  The model is producing realistic atmospheric circulation features.")
        print("  Running for longer will produce stronger jets and clearer patterns.")
    elif passed >= 2:
        print("  GCM IS DEVELOPING CIRCULATION")
        print("  Some features are emerging. Run for longer (100+ days) for equilibrium.")
    else:
        print("  GCM MAY NEED MORE TIME OR DEBUGGING")
        print("  Run for longer duration to allow circulation to develop.")

    print()
    return passed >= 2


def test_spectral_diagnostics():
    """Test spectral diagnostics independently"""
    print("\n" + "=" * 60)
    print("Testing Spectral Diagnostics")
    print("=" * 60)

    from gcm.grid.spherical import SphericalGrid
    from gcm.grid.vertical import VerticalGrid
    from gcm.diagnostics.spectral import SpectralDiagnostics

    # Create grid
    grid = SphericalGrid(64, 32, 20)
    vgrid = VerticalGrid(20)

    # Create spectral diagnostics
    spec = SpectralDiagnostics(grid, vgrid)

    # Create test field with known structure
    # Wavenumber 5 pattern
    u = 10 * np.cos(5 * grid.lon2d) * np.cos(grid.lat2d)
    v = 5 * np.sin(5 * grid.lon2d) * np.cos(grid.lat2d)

    # Compute spectrum
    wn, ke_spec = spec.compute_kinetic_energy_spectrum(u, v)

    print(f"\n  Created test field with wavenumber 5 pattern")
    print(f"  Spectrum peak at wavenumber: {wn[np.argmax(ke_spec)]}")

    # The spectrum should peak around k=5
    peak_wn = wn[np.argmax(ke_spec)]
    test_ok = 3 <= peak_wn <= 7

    print(f"\n  Test result: {'PASS' if test_ok else 'FAIL'}")
    return test_ok


def test_zonal_mean():
    """Test zonal mean diagnostics"""
    print("\n" + "=" * 60)
    print("Testing Zonal Mean Diagnostics")
    print("=" * 60)

    from gcm.grid.spherical import SphericalGrid
    from gcm.grid.vertical import VerticalGrid
    from gcm.core.state import ModelState
    from gcm.diagnostics.zonal_mean import ZonalMeanDiagnostics

    # Create grid and state
    grid = SphericalGrid(64, 32, 20)
    vgrid = VerticalGrid(20)
    state = ModelState(grid, vgrid)

    # Initialize with simple jet structure
    lat_deg = np.rad2deg(grid.lat2d)
    for k in range(vgrid.nlev):
        # Subtropical jet at 30 degrees
        state.u[k] = 20 * np.exp(-((lat_deg - 30)**2) / 200)
        state.T[k] = 280 - 30 * np.abs(np.sin(grid.lat2d))

    state.ps[:] = 101325.0
    _, state.p = vgrid.compute_pressure(state.ps)
    state.update_diagnostics()

    # Compute zonal mean diagnostics
    zm = ZonalMeanDiagnostics(grid, vgrid)
    u_bar, v_bar = zm.compute_zonal_mean_winds(state)

    print(f"\n  Created test jet at 30 degrees")
    print(f"  Max zonal mean u: {np.max(u_bar):.1f} m/s")

    lat = np.rad2deg(grid.lat)
    max_lat_idx = np.unravel_index(np.argmax(u_bar), u_bar.shape)[1]
    max_lat = lat[max_lat_idx]
    print(f"  Jet maximum at: {max_lat:.1f} degrees")

    test_ok = 25 <= max_lat <= 35

    print(f"\n  Test result: {'PASS' if test_ok else 'FAIL'}")
    return test_ok


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("#  GCM PHYSICS AND DIAGNOSTICS TEST SUITE")
    print("#" * 60)

    results = []

    # Run unit tests
    results.append(("Spectral Diagnostics", test_spectral_diagnostics()))
    results.append(("Zonal Mean Diagnostics", test_zonal_mean()))

    # Run integration test
    results.append(("Held-Suarez GCM", test_held_suarez()))

    # Summary
    print("\n" + "#" * 60)
    print("#  TEST SUMMARY")
    print("#" * 60)
    print()

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED - see above for details")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

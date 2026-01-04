#!/usr/bin/env python
"""
Demonstration of the gradient calculation fix.

This script shows the impact of fixing the metric conversion bug
in atmospheric gradient calculations. It compares the old (buggy)
and new (corrected) gradient calculations.
"""

import numpy as np
import torch

# Constants
R_EARTH = 6.371e6  # Earth radius in meters
OMEGA = 7.292e-5   # Earth's angular velocity


def compute_gradient_old_buggy(field, dlon, dlat, lat_grid):
    """Compute gradients using the OLD BUGGY method."""
    cos_lat = torch.cos(lat_grid).view(1, 1, -1, 1).clamp(min=1e-8)
    
    # BUGGY: includes dlon/dlat in metric factor
    dx = dlon * R_EARTH * cos_lat
    dy = dlat * R_EARTH
    
    dfield_dx = torch.gradient(field, spacing=(dlon,), dim=3)[0] / dx
    dfield_dy = torch.gradient(field, spacing=(dlat,), dim=2)[0] / dy
    
    return dfield_dx, dfield_dy


def compute_gradient_new_corrected(field, dlon, dlat, lat_grid):
    """Compute gradients using the NEW CORRECTED method."""
    cos_lat = torch.cos(lat_grid).view(1, 1, -1, 1).clamp(min=1e-8)
    
    # CORRECTED: meters per radian (no dlon/dlat)
    dx = R_EARTH * cos_lat
    dy = R_EARTH
    
    dfield_dx = torch.gradient(field, spacing=(dlon,), dim=3)[0] / dx
    dfield_dy = torch.gradient(field, spacing=(dlat,), dim=2)[0] / dy
    
    return dfield_dx, dfield_dy


def main():
    print("=" * 70)
    print("DEMONSTRATION: Gradient Calculation Fix")
    print("=" * 70)
    print()
    
    # Setup test field
    lat_dim, lon_dim = 32, 64
    batch, n_levels = 1, 1
    
    lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim)
    lon_grid = torch.linspace(0, 2*np.pi, lon_dim)
    
    dlat = np.pi / (lat_dim - 1)
    dlon = 2 * np.pi / lon_dim
    
    print(f"Grid: {lat_dim} x {lon_dim}")
    print(f"dlat = {dlat:.6f} rad ({np.degrees(dlat):.2f}°)")
    print(f"dlon = {dlon:.6f} rad ({np.degrees(dlon):.2f}°)")
    print()
    
    # Create a test field with known gradient
    # z = 10 * x (where x is physical distance)
    cos_lat = torch.cos(lat_grid).view(-1, 1).clamp(min=1e-8)
    lon_mesh = lon_grid.view(1, -1).expand(lat_dim, lon_dim)
    x_phys = R_EARTH * cos_lat * lon_mesh
    z_field = 10.0 * x_phys
    z_field = z_field.view(batch, n_levels, lat_dim, lon_dim)
    
    print("Test field: z = 10 * x (where x is in meters)")
    print("Expected gradient: dz/dx = 10.0 everywhere")
    print()
    
    # Compute using old buggy method
    dz_dx_old, dz_dy_old = compute_gradient_old_buggy(z_field, dlon, dlat, lat_grid)
    
    # Compute using new corrected method
    dz_dx_new, dz_dy_new = compute_gradient_new_corrected(z_field, dlon, dlat, lat_grid)
    
    # Analyze results (avoid polar regions)
    central_region = slice(8, 24)
    
    mean_grad_old = dz_dx_old[0, 0, central_region, :].mean().item()
    mean_grad_new = dz_dx_new[0, 0, central_region, :].mean().item()
    
    print("-" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"Expected gradient:          dz/dx = 10.0")
    print(f"OLD (BUGGY) gradient:       dz/dx = {mean_grad_old:.2f}")
    print(f"NEW (CORRECTED) gradient:   dz/dx = {mean_grad_new:.2f}")
    print()
    
    # Calculate the error factor
    error_factor_old = dlon  # The bug introduced a factor of dlon
    error_factor_new = 1.0
    
    print(f"Error factor in OLD method: {error_factor_old:.6f} (= dlon)")
    print(f"Error factor in NEW method: {error_factor_new:.6f}")
    print()
    
    print(f"Ratio (OLD/expected):       {mean_grad_old / 10.0:.6f}")
    print(f"Ratio (NEW/expected):       {mean_grad_new / 10.0:.6f}")
    print()
    
    # Show impact on derived quantities
    print("-" * 70)
    print("IMPACT ON DERIVED QUANTITIES:")
    print("-" * 70)
    
    # Example: vorticity calculation
    # Create a simple shear flow: u = y, v = 0
    y = torch.linspace(-1, 1, lat_dim).view(1, 1, lat_dim, 1)
    u = y.expand(batch, n_levels, lat_dim, lon_dim) * 10.0  # m/s
    v = torch.zeros_like(u)
    
    # Compute vorticity (∂v/∂x - ∂u/∂y)
    dvdx_old, dudy_old = compute_gradient_old_buggy(v, dlon, dlat, lat_grid)
    _, dudy_old = compute_gradient_old_buggy(u, dlon, dlat, lat_grid)
    vort_old = dvdx_old - dudy_old
    
    dvdx_new, _ = compute_gradient_new_corrected(v, dlon, dlat, lat_grid)
    _, dudy_new = compute_gradient_new_corrected(u, dlon, dlat, lat_grid)
    vort_new = dvdx_new - dudy_new
    
    mean_vort_old = vort_old[0, 0, central_region, :].mean().item()
    mean_vort_new = vort_new[0, 0, central_region, :].mean().item()
    
    print(f"Vorticity of shear flow (u=y, v=0):")
    print(f"  OLD (BUGGY):      {mean_vort_old:.6e} s⁻¹")
    print(f"  NEW (CORRECTED):  {mean_vort_new:.6e} s⁻¹")
    print(f"  Ratio OLD/NEW:    {mean_vort_old / mean_vort_new:.2f}x")
    print()
    
    print("-" * 70)
    print("CONCLUSION:")
    print("-" * 70)
    print("The OLD method overestimated gradients by a factor of ~1/dlon (or 1/dlat)")
    print(f"For this grid, that's a factor of {1/dlon:.1f}x error!")
    print()
    print("This affected ALL atmospheric calculations:")
    print("  • Vorticity (for cyclone detection)")
    print("  • Divergence (for mass conservation)")
    print("  • Geostrophic balance (for wind-pressure relationships)")
    print("  • Potential vorticity gradients")
    print()
    print("The NEW method correctly accounts for metric conversion.")
    print("=" * 70)


if __name__ == "__main__":
    main()

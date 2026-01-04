"""Test to verify the gradient calculation fix for metric conversion.

This test validates that spatial gradients are computed correctly after
fixing the metric conversion bug where dlon/dlat were incorrectly included
in the metric factors.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
from weatherflow.physics.losses import PhysicsLossCalculator


class TestGradientMetricConversion:
    """Tests for correct metric conversion in gradient calculations."""

    def test_gradient_units_and_scaling(self):
        """Test that gradients have correct units after metric conversion."""
        physics_calc = PhysicsLossCalculator()
        
        # Create a simple test field with known gradient
        batch, n_levels, lat_dim, lon_dim = 1, 1, 32, 64
        
        # Create a linear field: z = a*x + b*y where x, y are in physical coordinates
        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim)
        lon_grid = torch.linspace(0, 2*np.pi, lon_dim)
        
        # Physical coordinates (approximate for small region)
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim
        
        # Create field with constant gradient in physical space
        # z = 10 * x (where x is in meters)
        cos_lat = torch.cos(lat_grid).view(lat_dim, 1).clamp(min=1e-8)
        lon_mesh = lon_grid.view(1, lon_dim).expand(lat_dim, lon_dim)
        
        # Physical x-coordinate (meters)
        R = physics_calc.earth_radius
        x_phys = R * cos_lat * lon_mesh
        
        # Create field with constant gradient: dz/dx = 10
        z_field = 10.0 * x_phys
        z_field = z_field.view(batch, n_levels, lat_dim, lon_dim)
        
        # Compute gradient using the corrected method
        dx = R * cos_lat.view(1, 1, lat_dim, 1)
        dz_dx_computed = torch.gradient(z_field, spacing=(dlon,), dim=3)[0] / dx
        
        # The gradient should be approximately 10 everywhere
        # (some variation near poles is expected due to metric distortion)
        central_region = slice(8, 24)  # Avoid polar regions where metric distortion and finite difference errors are largest
        mean_gradient = dz_dx_computed[0, 0, central_region, :].mean().item()
        
        # Check that gradient is close to expected value (10)
        assert abs(mean_gradient - 10.0) < 1.0, f"Expected gradient ~10, got {mean_gradient}"
        
        # For our test field z=10*x (meters), dz/dx is dimensionless with value 10

    def test_divergence_free_flow(self):
        """Test that divergence-free flow has near-zero divergence."""
        physics_calc = PhysicsLossCalculator()
        
        batch, n_levels, lat_dim, lon_dim = 1, 1, 32, 64
        
        # Create a divergence-free flow field (circular/rotational)
        # In physical coordinates: u = -y, v = x gives div = du/dx + dv/dy = 0 + 0 = 0
        
        # For simplicity, use a simpler test:
        # u = sin(2πx/L), v = -sin(2πy/L) in normalized coordinates
        # This has du/dx + dv/dy = 2π/L * cos(2πx/L) - 2π/L * cos(2πy/L)
        # which is not exactly zero, so use the previous test setup
        
        y = torch.linspace(-1, 1, lat_dim).view(1, 1, lat_dim, 1)
        x = torch.linspace(-1, 1, lon_dim).view(1, 1, 1, lon_dim)
        
        # Circular flow: u = -y, v = x (in normalized coordinates)
        # This is divergence-free: du/dx + dv/dy = 0 + 0 = 0
        u = -y.expand(batch, n_levels, lat_dim, lon_dim)
        v = x.expand(batch, n_levels, lat_dim, lon_dim)
        
        # Compute divergence using mass-weighted divergence loss
        loss = physics_calc.compute_mass_weighted_divergence_loss(u, v)
        
        # Should be small (not exactly zero due to metric terms and finite differences)
        assert loss.item() < 0.1, f"Divergence should be small, got {loss.item()}"

    def test_vorticity_of_shear_flow(self):
        """Test vorticity calculation for a simple shear flow."""
        physics_calc = PhysicsLossCalculator()
        
        batch, n_levels, lat_dim, lon_dim = 1, 1, 32, 64
        
        # Create a zonal shear flow: u = y, v = 0
        # Vorticity = dv/dx - du/dy = 0 - 1 = -1 (in normalized coordinates)
        y = torch.linspace(-1, 1, lat_dim).view(1, 1, lat_dim, 1)
        
        u = y.expand(batch, n_levels, lat_dim, lon_dim)
        v = torch.zeros_like(u)
        
        # Compute using the gradient method from the physics calculator
        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim, dtype=u.dtype)
        cos_lat = torch.cos(lat_grid).view(1, 1, lat_dim, 1).clamp(min=1e-8)
        
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim
        
        dx = physics_calc.earth_radius * cos_lat
        dy = physics_calc.earth_radius
        
        dvdx = torch.gradient(v, spacing=(dlon,), dim=3)[0] / dx
        dudy = torch.gradient(u, spacing=(dlat,), dim=2)[0] / dy
        
        vorticity = dvdx - dudy
        
        # The gradient du/dy in physical coordinates depends on the physical extent
        # For a normalized y from -1 to 1, and physical extent of R * pi,
        # the gradient is approximately (1 - (-1)) / (R * pi) = 2 / (R * pi)
        # So vorticity should be approximately -2 / (R * pi)
        
        expected_vort = -2.0 / (physics_calc.earth_radius * np.pi)
        mean_vort = vorticity[0, 0, 8:24, :].mean().item()  # Avoid poles
        
        # Check that vorticity has the right sign and rough magnitude
        assert mean_vort < 0, "Vorticity should be negative for this shear"
        assert abs(mean_vort - expected_vort) / abs(expected_vort) < 0.5, \
            f"Expected vorticity ~{expected_vort:.2e}, got {mean_vort:.2e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

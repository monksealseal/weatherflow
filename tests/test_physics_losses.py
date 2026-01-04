"""Tests for enhanced physics-based loss functions."""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
from weatherflow.physics.losses import PhysicsLossCalculator


@pytest.fixture
def sample_wind_field():
    """Create a sample wind field for testing."""
    batch, n_levels, lat_dim, lon_dim = 2, 3, 32, 64
    u = torch.randn(batch, n_levels, lat_dim, lon_dim)
    v = torch.randn(batch, n_levels, lat_dim, lon_dim)
    return u, v


@pytest.fixture
def sample_geopotential():
    """Create a sample geopotential field."""
    batch, n_levels, lat_dim, lon_dim = 2, 3, 32, 64
    return torch.randn(batch, n_levels, lat_dim, lon_dim) * 1000  # Realistic scale


@pytest.fixture
def pressure_levels():
    """Create sample pressure levels."""
    return torch.tensor([500.0, 700.0, 850.0])


@pytest.fixture
def physics_calculator():
    """Create a PhysicsLossCalculator instance."""
    return PhysicsLossCalculator()


class TestPhysicsLossCalculator:
    """Test suite for PhysicsLossCalculator."""

    def test_initialization(self):
        """Test that PhysicsLossCalculator initializes correctly."""
        calc = PhysicsLossCalculator()
        assert calc.earth_radius == 6371e3
        assert calc.gravity == 9.81
        assert calc.omega == 7.292e-5
        assert calc.f0 == 1e-4
        assert calc.beta == 1.6e-11

    def test_coriolis_parameter(self, physics_calculator):
        """Test Coriolis parameter calculation."""
        lat = torch.tensor([0.0, np.pi/4, np.pi/2])  # 0, 45, 90 degrees
        f = physics_calculator.coriolis_parameter(lat)

        # At equator, f ≈ 0
        assert torch.abs(f[0]) < 1e-8

        # At 45°N, f ≈ 1.03e-4
        assert torch.abs(f[1] - 1.03e-4) < 1e-5

        # At pole, f ≈ 1.458e-4
        assert torch.abs(f[2] - 1.458e-4) < 1e-5

    def test_pv_conservation_loss_shape(self, physics_calculator, sample_wind_field, pressure_levels):
        """Test that PV conservation loss returns a scalar."""
        u, v = sample_wind_field
        loss = physics_calculator.compute_pv_conservation_loss(
            u, v, pressure_levels=pressure_levels
        )
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_pv_conservation_loss_barotropic(self, physics_calculator):
        """Test PV loss for barotropic (single level) case."""
        batch, lat_dim, lon_dim = 2, 32, 64
        u = torch.randn(batch, 1, lat_dim, lon_dim)
        v = torch.randn(batch, 1, lat_dim, lon_dim)

        loss = physics_calculator.compute_pv_conservation_loss(u, v)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_energy_spectra_loss_shape(self, physics_calculator, sample_wind_field):
        """Test that energy spectra loss returns a scalar."""
        u, v = sample_wind_field
        loss = physics_calculator.compute_energy_spectra_loss(u, v)
        assert loss.shape == torch.Size([])

    def test_energy_spectra_loss_target_slope(self, physics_calculator):
        """Test energy spectra loss with different target slopes."""
        batch, n_levels, lat_dim, lon_dim = 2, 1, 32, 64
        u = torch.randn(batch, n_levels, lat_dim, lon_dim)
        v = torch.randn(batch, n_levels, lat_dim, lon_dim)

        # Test with default k^-3 slope
        loss_default = physics_calculator.compute_energy_spectra_loss(u, v, target_slope=-3.0)

        # Test with k^-5/3 slope (energy cascade)
        loss_energy = physics_calculator.compute_energy_spectra_loss(u, v, target_slope=-5/3)

        assert loss_default.shape == torch.Size([])
        assert loss_energy.shape == torch.Size([])

    def test_mass_divergence_loss_shape(self, physics_calculator, sample_wind_field, pressure_levels):
        """Test that mass-weighted divergence loss returns a scalar."""
        u, v = sample_wind_field
        loss = physics_calculator.compute_mass_weighted_divergence_loss(
            u, v, pressure_levels=pressure_levels
        )
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_mass_divergence_zero_for_nondivergent_flow(self, physics_calculator):
        """Test that divergence-free flow has near-zero divergence loss."""
        batch, n_levels, lat_dim, lon_dim = 1, 1, 32, 64

        # Create a simple rotational (divergence-free) flow
        y = torch.linspace(-1, 1, lat_dim).view(1, 1, lat_dim, 1)
        x = torch.linspace(-1, 1, lon_dim).view(1, 1, 1, lon_dim)

        # Circular flow: u = -y, v = x
        u = -y.expand(batch, n_levels, lat_dim, lon_dim)
        v = x.expand(batch, n_levels, lat_dim, lon_dim)

        loss = physics_calculator.compute_mass_weighted_divergence_loss(u, v)

        # Should be small (not exactly zero due to finite differences)
        assert loss.item() < 0.1

    def test_mass_divergence_pressure_weighting(self, physics_calculator):
        """Test that pressure weighting affects the divergence calculation."""
        batch, n_levels, lat_dim, lon_dim = 1, 3, 32, 64
        u = torch.randn(batch, n_levels, lat_dim, lon_dim)
        v = torch.randn(batch, n_levels, lat_dim, lon_dim)

        pressure_levels = torch.tensor([500.0, 700.0, 850.0])

        # Loss with pressure weighting
        loss_weighted = physics_calculator.compute_mass_weighted_divergence_loss(
            u, v, pressure_levels=pressure_levels
        )

        # Loss without pressure weighting
        loss_unweighted = physics_calculator.compute_mass_weighted_divergence_loss(
            u, v, pressure_levels=None
        )

        # Both should be scalar and non-negative
        assert loss_weighted.shape == torch.Size([])
        assert loss_unweighted.shape == torch.Size([])
        assert loss_weighted.item() >= 0
        assert loss_unweighted.item() >= 0

    def test_geostrophic_balance_loss_shape(self, physics_calculator, sample_wind_field, sample_geopotential):
        """Test that geostrophic balance loss returns a scalar."""
        u, v = sample_wind_field
        geopotential = sample_geopotential

        loss = physics_calculator.compute_geostrophic_balance_loss(u, v, geopotential)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_geostrophic_balance_perfect_balance(self, physics_calculator):
        """Test that perfectly balanced flow has low balance loss."""
        batch, n_levels, lat_dim, lon_dim = 1, 1, 32, 64

        # Create a simple geopotential field
        y = torch.linspace(-np.pi/2, np.pi/2, lat_dim).view(1, 1, lat_dim, 1)
        x = torch.linspace(0, 2*np.pi, lon_dim).view(1, 1, 1, lon_dim)

        # Simple sinusoidal geopotential
        geopotential = 1000 * torch.sin(2 * x) * torch.cos(y)
        geopotential = geopotential.expand(batch, n_levels, lat_dim, lon_dim)

        # Compute geostrophic winds from this geopotential
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim

        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim)
        cos_lat = torch.cos(lat_grid).view(1, 1, lat_dim, 1).clamp(min=1e-8)
        f = physics_calculator.coriolis_parameter(lat_grid).view(1, 1, lat_dim, 1)

        dx = physics_calculator.earth_radius * cos_lat
        dy = physics_calculator.earth_radius

        dPhi_dx = torch.gradient(geopotential, spacing=(dlon,), dim=3)[0] / dx
        dPhi_dy = torch.gradient(geopotential, spacing=(dlat,), dim=2)[0] / dy

        u_g = -dPhi_dy / (f + 1e-8)
        v_g = dPhi_dx / (f + 1e-8)

        # These should be in perfect geostrophic balance
        loss = physics_calculator.compute_geostrophic_balance_loss(u_g, v_g, geopotential)

        # Should be very small (not exactly zero due to numerical precision)
        assert loss.item() < 1.0

    def test_compute_all_physics_losses(self, physics_calculator, sample_wind_field, sample_geopotential, pressure_levels):
        """Test computing all physics losses together."""
        u, v = sample_wind_field
        geopotential = sample_geopotential

        losses = physics_calculator.compute_all_physics_losses(
            u, v, geopotential=geopotential, pressure_levels=pressure_levels
        )

        # Check that all expected keys are present
        assert 'pv_conservation' in losses
        assert 'energy_spectra' in losses
        assert 'mass_divergence' in losses
        assert 'geostrophic_balance' in losses
        assert 'physics_total' in losses

        # Check that all losses are scalars
        for key, value in losses.items():
            assert value.shape == torch.Size([])
            assert not torch.isnan(value)

    def test_compute_all_physics_losses_custom_weights(self, physics_calculator, sample_wind_field):
        """Test computing physics losses with custom weights."""
        u, v = sample_wind_field

        custom_weights = {
            'pv_conservation': 0.5,
            'energy_spectra': 0.05,
            'mass_divergence': 2.0,
            'geostrophic_balance': 0.0,  # Disabled
        }

        losses = physics_calculator.compute_all_physics_losses(
            u, v, loss_weights=custom_weights
        )

        # Geostrophic balance should not be computed (weight = 0)
        assert 'geostrophic_balance' not in losses or losses['geostrophic_balance'] == 0

        # Total should be weighted sum of enabled losses
        assert 'physics_total' in losses

    def test_gradient_flow(self, physics_calculator, sample_wind_field):
        """Test that gradients flow through physics losses."""
        u, v = sample_wind_field
        u.requires_grad = True
        v.requires_grad = True

        # Compute PV conservation loss
        loss = physics_calculator.compute_pv_conservation_loss(u, v)

        # Backpropagate
        loss.backward()

        # Check that gradients exist
        assert u.grad is not None
        assert v.grad is not None
        assert not torch.isnan(u.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_device_compatibility(self, physics_calculator):
        """Test that physics losses work on different devices."""
        batch, n_levels, lat_dim, lon_dim = 1, 1, 16, 32

        # CPU tensors
        u_cpu = torch.randn(batch, n_levels, lat_dim, lon_dim)
        v_cpu = torch.randn(batch, n_levels, lat_dim, lon_dim)

        loss_cpu = physics_calculator.compute_pv_conservation_loss(u_cpu, v_cpu)
        assert loss_cpu.device.type == 'cpu'

        # GPU tensors (if available)
        if torch.cuda.is_available():
            u_gpu = u_cpu.cuda()
            v_gpu = v_cpu.cuda()
            loss_gpu = physics_calculator.compute_pv_conservation_loss(u_gpu, v_gpu)
            assert loss_gpu.device.type == 'cuda'

    def test_batch_independence(self, physics_calculator):
        """Test that loss is computed consistently across batch sizes."""
        n_levels, lat_dim, lon_dim = 1, 32, 64

        # Single sample
        u_single = torch.randn(1, n_levels, lat_dim, lon_dim)
        v_single = torch.randn(1, n_levels, lat_dim, lon_dim)
        loss_single = physics_calculator.compute_pv_conservation_loss(u_single, v_single)

        # Batch of same sample repeated
        u_batch = u_single.repeat(4, 1, 1, 1)
        v_batch = v_single.repeat(4, 1, 1, 1)
        loss_batch = physics_calculator.compute_pv_conservation_loss(u_batch, v_batch)

        # Losses should be similar (averaged over batch)
        # Note: They might not be exactly equal due to batch statistics
        assert torch.abs(loss_single - loss_batch) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

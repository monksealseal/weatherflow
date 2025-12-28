"""Demonstration of enhanced physics-based losses for WeatherFlow.

This script shows how to use the new physics constraints:
- PV (Potential Vorticity) conservation
- Energy spectra regularization (k^-3 enstrophy cascade)
- Mass-weighted column divergence
- Geostrophic balance

These constraints address the research roadmap Phase 2:
"Pushing the Physics Constraints"
"""

import torch
import torch.nn as nn
from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
from weatherflow.physics.losses import PhysicsLossCalculator
import numpy as np


def create_synthetic_weather_state(
    batch_size: int = 4,
    n_levels: int = 3,
    lat_dim: int = 32,
    lon_dim: int = 64,
):
    """Create synthetic weather state for demonstration.

    Returns:
        tuple: (state, pressure_levels)
            state: [batch, channels, lat, lon] where channels = [u, v, z, t]
            pressure_levels: [n_levels] in hPa
    """
    # Generate realistic-looking atmospheric fields
    # In practice, these would come from ERA5 data

    # Wind components (m/s)
    u = torch.randn(batch_size, 1, lat_dim, lon_dim) * 20  # ~20 m/s std
    v = torch.randn(batch_size, 1, lat_dim, lon_dim) * 20

    # Geopotential height (m)
    z = torch.randn(batch_size, 1, lat_dim, lon_dim) * 100 + 5500  # 500 hPa level

    # Temperature (K)
    t = torch.randn(batch_size, 1, lat_dim, lon_dim) * 10 + 250  # ~250K mean

    # Stack channels
    state = torch.cat([u, v, z, t], dim=1)  # [batch, 4, lat, lon]

    # Pressure levels
    pressure_levels = torch.tensor([500.0, 700.0, 850.0])  # hPa

    return state, pressure_levels


def demonstrate_basic_physics_losses():
    """Demonstrate basic usage of physics losses."""
    print("=" * 80)
    print("DEMONSTRATION: Basic Physics Losses")
    print("=" * 80)

    # Create physics calculator
    physics_calc = PhysicsLossCalculator()

    # Create sample wind field
    batch, n_levels, lat_dim, lon_dim = 2, 3, 32, 64
    u = torch.randn(batch, n_levels, lat_dim, lon_dim)
    v = torch.randn(batch, n_levels, lat_dim, lon_dim)
    geopotential = torch.randn(batch, n_levels, lat_dim, lon_dim) * 1000
    pressure_levels = torch.tensor([500.0, 700.0, 850.0])

    print(f"\nInput shapes:")
    print(f"  u: {u.shape}")
    print(f"  v: {v.shape}")
    print(f"  geopotential: {geopotential.shape}")
    print(f"  pressure_levels: {pressure_levels.shape}")

    # Compute individual losses
    print("\n--- Individual Physics Losses ---")

    pv_loss = physics_calc.compute_pv_conservation_loss(
        u, v, geopotential, pressure_levels
    )
    print(f"PV Conservation Loss: {pv_loss.item():.6f}")

    spectra_loss = physics_calc.compute_energy_spectra_loss(u, v)
    print(f"Energy Spectra Loss (k^-3 target): {spectra_loss.item():.6f}")

    mass_loss = physics_calc.compute_mass_weighted_divergence_loss(
        u, v, pressure_levels
    )
    print(f"Mass-Weighted Divergence Loss: {mass_loss.item():.6f}")

    balance_loss = physics_calc.compute_geostrophic_balance_loss(
        u, v, geopotential
    )
    print(f"Geostrophic Balance Loss: {balance_loss.item():.6f}")

    # Compute all losses together
    print("\n--- Combined Physics Losses ---")
    all_losses = physics_calc.compute_all_physics_losses(
        u, v, geopotential, pressure_levels
    )

    for name, value in all_losses.items():
        print(f"{name:25s}: {value.item():.6f}")


def demonstrate_flow_model_with_physics():
    """Demonstrate WeatherFlowMatch with enhanced physics losses."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Flow Matching Model with Enhanced Physics")
    print("=" * 80)

    # Model configuration
    config = {
        'input_channels': 4,  # u, v, z, t
        'hidden_dim': 128,
        'n_layers': 4,
        'grid_size': (32, 64),  # lat, lon
        'physics_informed': True,
        'enhanced_physics_losses': True,  # Enable new physics losses!
        'physics_loss_weights': {
            'pv_conservation': 0.1,
            'energy_spectra': 0.01,
            'mass_divergence': 1.0,
            'geostrophic_balance': 0.1,
        },
    }

    print(f"\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model
    model = WeatherFlowMatch(**config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create synthetic data
    state0, pressure_levels = create_synthetic_weather_state(batch_size=4)
    state1, _ = create_synthetic_weather_state(batch_size=4)

    # Random time points
    t = torch.rand(4)

    print(f"\nData shapes:")
    print(f"  state0: {state0.shape}")
    print(f"  state1: {state1.shape}")
    print(f"  t: {t.shape}")

    # Compute flow loss (includes all physics losses)
    losses = model.compute_flow_loss(
        state0, state1, t,
        pressure_levels=pressure_levels
    )

    print("\n--- Training Losses ---")
    for name, value in losses.items():
        print(f"{name:25s}: {value.item():.6f}")

    # Demonstrate gradient flow
    print("\n--- Gradient Check ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    losses['total_loss'].backward()

    # Check that gradients exist
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2

    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print("✓ Gradients computed successfully!")


def demonstrate_training_loop():
    """Demonstrate a simple training loop with physics losses."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Training Loop with Physics Constraints")
    print("=" * 80)

    # Create model
    model = WeatherFlowMatch(
        input_channels=4,
        hidden_dim=64,
        n_layers=2,
        grid_size=(32, 64),
        enhanced_physics_losses=True,
        physics_loss_weights={
            'pv_conservation': 0.1,
            'energy_spectra': 0.01,
            'mass_divergence': 1.0,
            'geostrophic_balance': 0.05,
        },
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining for 5 iterations...")
    print(f"{'Iter':>4} {'Flow':>10} {'PV':>10} {'Spectra':>10} {'Mass':>10} {'Balance':>10} {'Total':>10}")
    print("-" * 80)

    for iteration in range(5):
        # Generate random batch
        state0, pressure_levels = create_synthetic_weather_state(batch_size=2)
        state1, _ = create_synthetic_weather_state(batch_size=2)
        t = torch.rand(2)

        # Forward pass
        losses = model.compute_flow_loss(state0, state1, t, pressure_levels=pressure_levels)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Print losses
        print(
            f"{iteration:4d} "
            f"{losses.get('flow_loss', 0).item():10.4f} "
            f"{losses.get('pv_conservation', 0).item():10.4f} "
            f"{losses.get('energy_spectra', 0).item():10.4f} "
            f"{losses.get('mass_divergence', 0).item():10.4f} "
            f"{losses.get('geostrophic_balance', 0).item():10.4f} "
            f"{losses['total_loss'].item():10.4f}"
        )

    print("\n✓ Training demonstration complete!")


def demonstrate_ablation_study():
    """Demonstrate ablation study: comparing models with/without physics losses."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Ablation Study - Physics Losses Impact")
    print("=" * 80)

    # Create two models: with and without enhanced physics
    model_baseline = WeatherFlowMatch(
        input_channels=4,
        hidden_dim=64,
        n_layers=2,
        grid_size=(32, 64),
        enhanced_physics_losses=False,  # Baseline
    )

    model_physics = WeatherFlowMatch(
        input_channels=4,
        hidden_dim=64,
        n_layers=2,
        grid_size=(32, 64),
        enhanced_physics_losses=True,  # With physics
        physics_loss_weights={
            'pv_conservation': 0.1,
            'energy_spectra': 0.01,
            'mass_divergence': 1.0,
            'geostrophic_balance': 0.1,
        },
    )

    # Generate test data
    state0, pressure_levels = create_synthetic_weather_state(batch_size=4)
    state1, _ = create_synthetic_weather_state(batch_size=4)
    t = torch.rand(4)

    print("\nComparing models on same data:")

    # Baseline model losses
    with torch.no_grad():
        losses_baseline = model_baseline.compute_flow_loss(state0, state1, t)

    print("\n--- Baseline Model (no enhanced physics) ---")
    for name, value in losses_baseline.items():
        print(f"{name:25s}: {value.item():.6f}")

    # Physics-enhanced model losses
    with torch.no_grad():
        losses_physics = model_physics.compute_flow_loss(
            state0, state1, t, pressure_levels=pressure_levels
        )

    print("\n--- Physics-Enhanced Model ---")
    for name, value in losses_physics.items():
        print(f"{name:25s}: {value.item():.6f}")

    print("\n--- Comparison ---")
    print(f"Baseline has {len(losses_baseline)} loss terms")
    print(f"Physics-enhanced has {len(losses_physics)} loss terms")
    print(f"\nAdditional constraints in physics-enhanced model:")
    for key in losses_physics:
        if key not in losses_baseline:
            print(f"  + {key}")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("ENHANCED PHYSICS LOSSES FOR WEATHERFLOW")
    print("Research Roadmap Phase 2: Pushing the Physics Constraints")
    print("=" * 80)

    # Run demonstrations
    demonstrate_basic_physics_losses()
    demonstrate_flow_model_with_physics()
    demonstrate_training_loop()
    demonstrate_ablation_study()

    print("\n" + "=" * 80)
    print("All demonstrations complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Integrate with ERA5 data pipeline")
    print("2. Run WeatherBench2 evaluation (Phase 1 validation)")
    print("3. Train models with different physics loss weights")
    print("4. Analyze impact on forecast skill scores")
    print("5. Implement Phase 3: Uncertainty Quantification")
    print("=" * 80)

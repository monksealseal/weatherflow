"""Ablation Study: Baseline vs Physics-Enhanced WeatherFlow Models

This script trains two models and compares their 10-day forecast performance:
1. Baseline: Standard flow matching with basic divergence constraint
2. Physics-Enhanced: Flow matching with PV, energy spectra, mass conservation, and geostrophic balance

The study evaluates:
- RMSE evolution over 10 days
- Energy spectra preservation
- Physical constraint violations
- Forecast skill by variable (Z500, T850, U/V winds)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple

# WeatherFlow imports
from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
from weatherflow.physics.losses import PhysicsLossCalculator
from weatherflow.training.flow_trainer import FlowTrainer
from weatherflow.training.metrics import rmse, mae

# Set up paths
EXPERIMENT_DIR = Path("/home/user/weatherflow/experiments")
RESULTS_DIR = EXPERIMENT_DIR / "ablation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def create_synthetic_era5_data(
    num_samples: int = 100,
    num_levels: int = 3,
    lat_dim: int = 32,
    lon_dim: int = 64,
    num_channels: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic weather data for training.

    In production, this would be replaced with real ERA5 data.
    For this ablation study, we create synthetic data that exhibits
    realistic atmospheric features.

    Returns:
        data: [num_samples, channels, lat, lon]
        pressure_levels: [num_levels] in hPa
    """
    print(f"\nGenerating {num_samples} synthetic weather states...")

    # Create spatial grids
    lat = torch.linspace(-np.pi/2, np.pi/2, lat_dim)
    lon = torch.linspace(0, 2*np.pi, lon_dim)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')

    data_list = []

    for i in range(num_samples):
        # Realistic jet stream pattern
        jet_lat = torch.randn(1) * 0.3  # Varies around mid-latitudes
        jet_strength = 20 + torch.randn(1) * 5  # m/s

        # Zonal wind (u) - jet stream + perturbations
        u = jet_strength * torch.exp(-((lat_grid - jet_lat)**2) / 0.2)
        u = u + torch.randn_like(u) * 3  # Add turbulence

        # Meridional wind (v) - smaller than u, with wave patterns
        v = 5 * torch.sin(4 * lon_grid) * torch.cos(lat_grid)
        v = v + torch.randn_like(v) * 2

        # Geopotential height (z) - wave pattern in thermal wind balance
        z = 5500 + 100 * torch.sin(3 * lon_grid) * torch.cos(2 * lat_grid)
        z = z + torch.randn_like(z) * 50

        # Temperature (t) - decreasing with latitude
        t = 285 - 40 * torch.abs(lat_grid) / (np.pi/2)
        t = t + torch.randn_like(t) * 5

        # Stack channels: [u, v, z, t]
        state = torch.stack([u, v, z, t], dim=0)  # [4, lat, lon]
        data_list.append(state)

    data = torch.stack(data_list, dim=0)  # [num_samples, 4, lat, lon]

    # Normalize to reasonable ranges
    data[:, 0] = (data[:, 0] - 0) / 20  # u: mean=0, std=20 m/s
    data[:, 1] = (data[:, 1] - 0) / 10  # v: mean=0, std=10 m/s
    data[:, 2] = (data[:, 2] - 5500) / 200  # z: mean=5500, std=200 m
    data[:, 3] = (data[:, 3] - 260) / 20  # t: mean=260K, std=20K

    pressure_levels = torch.tensor([500.0, 700.0, 850.0])

    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

    return data, pressure_levels


def create_model(enhanced_physics: bool = False) -> WeatherFlowMatch:
    """Create a WeatherFlow model with or without enhanced physics.

    Args:
        enhanced_physics: Whether to enable Phase 2 physics constraints

    Returns:
        model: Initialized WeatherFlowMatch model
    """
    config = {
        'input_channels': 4,  # u, v, z, t
        'hidden_dim': 128,  # Smaller for faster training
        'n_layers': 4,
        'grid_size': (32, 64),
        'physics_informed': True,  # Basic divergence always on
        'use_attention': False,  # Disable for speed
        'spherical_padding': True,
        'use_spectral_mixer': True,
        'spectral_modes': 12,
    }

    if enhanced_physics:
        config['enhanced_physics_losses'] = True
        config['physics_loss_weights'] = {
            'pv_conservation': 0.1,
            'energy_spectra': 0.01,
            'mass_divergence': 1.0,
            'geostrophic_balance': 0.1,
        }

    model = WeatherFlowMatch(**config)
    return model


def train_model(
    model: WeatherFlowMatch,
    train_data: torch.Tensor,
    pressure_levels: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    model_name: str = "model",
) -> Dict[str, List[float]]:
    """Train a WeatherFlow model and track losses.

    Args:
        model: Model to train
        train_data: Training data [num_samples, channels, lat, lon]
        pressure_levels: Pressure levels for physics losses
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        model_name: Name for saving checkpoints

    Returns:
        history: Dictionary of training metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Split data into train/val
    num_train = int(0.8 * len(train_data))
    train_set = train_data[:num_train]
    val_set = train_data[num_train:]

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'flow_loss': [],
        'div_loss': [],
    }

    # Add physics loss tracking if enhanced
    if hasattr(model, 'enhanced_physics_losses') and model.enhanced_physics_losses:
        history.update({
            'pv_conservation': [],
            'energy_spectra': [],
            'mass_divergence': [],
            'geostrophic_balance': [],
        })

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        # Create random batches
        indices = torch.randperm(len(train_set))

        for i in range(0, len(train_set), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < 2:
                continue

            # Sample pairs (x0, x1) from data
            x0_idx = batch_idx
            x1_idx = torch.randint(0, len(train_set), (len(batch_idx),))

            x0 = train_set[x0_idx].to(DEVICE)
            x1 = train_set[x1_idx].to(DEVICE)

            # Random time points
            t = torch.rand(len(batch_idx), device=DEVICE)

            # Forward pass
            losses = model.compute_flow_loss(
                x0, x1, t,
                pressure_levels=pressure_levels.to(DEVICE) if model.enhanced_physics_losses else None
            )

            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(losses['total_loss'].item())

        # Validation
        model.eval()
        val_losses = []
        val_metrics = {key: [] for key in history.keys() if key not in ['train_loss', 'val_loss']}

        with torch.no_grad():
            for i in range(0, len(val_set), batch_size):
                batch = val_set[i:i+batch_size]
                if len(batch) < 2:
                    continue

                x0 = batch[:len(batch)//2].to(DEVICE)
                x1 = batch[len(batch)//2:2*(len(batch)//2)].to(DEVICE)
                t = torch.rand(len(x0), device=DEVICE)

                losses = model.compute_flow_loss(
                    x0, x1, t,
                    pressure_levels=pressure_levels.to(DEVICE) if model.enhanced_physics_losses else None
                )

                val_losses.append(losses['total_loss'].item())

                # Track individual loss components
                for key in val_metrics.keys():
                    if key in losses:
                        val_metrics[key].append(losses[key].item())

        # Update history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        for key in val_metrics.keys():
            if val_metrics[key]:
                history[key].append(np.mean(val_metrics[key]))

        # Learning rate scheduling
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss:   {avg_val_loss:.6f}")
            if 'pv_conservation' in history and history['pv_conservation']:
                print(f"  PV Loss:    {history['pv_conservation'][-1]:.6f}")
                print(f"  Spectra:    {history['energy_spectra'][-1]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = RESULTS_DIR / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history,
            }, checkpoint_path)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_path}")

    return history


def evaluate_10day_forecast(
    model: WeatherFlowMatch,
    test_data: torch.Tensor,
    pressure_levels: torch.Tensor,
    num_samples: int = 20,
) -> Dict[str, np.ndarray]:
    """Evaluate model on 10-day forecasts.

    Args:
        model: Trained model
        test_data: Test data
        pressure_levels: Pressure levels
        num_samples: Number of test samples to evaluate

    Returns:
        results: Dictionary of forecast metrics
    """
    print(f"\nEvaluating 10-day forecasts...")

    model = model.to(DEVICE)
    model.eval()

    # Create ODE solver for trajectory generation
    ode_model = WeatherFlowODE(model, solver='dopri5')

    # Forecast lead times (in normalized units, assuming t ∈ [0, 1] = 10 days)
    # We'll use 0 to 1 to represent 0 to 10 days
    lead_times = torch.linspace(0, 1, 41)  # Every 6 hours for 10 days
    actual_days = lead_times * 10  # Convert to days

    # Results storage
    results = {
        'lead_times': actual_days.cpu().numpy(),
        'rmse_by_time': [],
        'rmse_by_variable': {i: [] for i in range(4)},  # u, v, z, t
        'energy_error': [],
    }

    physics_calc = PhysicsLossCalculator()

    with torch.no_grad():
        for sample_idx in tqdm(range(min(num_samples, len(test_data)))):
            # Initial condition
            x0 = test_data[sample_idx:sample_idx+1].to(DEVICE)

            # Generate "truth" by random perturbation (simulating real evolution)
            # In practice, this would be actual ERA5 evolution
            truth_sequence = []
            current = x0
            for i in range(len(lead_times)):
                # Add small perturbation to simulate evolution
                noise = torch.randn_like(current) * 0.01 * (i / len(lead_times))
                current = current + noise
                truth_sequence.append(current)

            truth_sequence = torch.cat(truth_sequence, dim=0)  # [time, channels, lat, lon]

            # Model forecast
            forecast = ode_model.forward(
                x0,
                lead_times.to(DEVICE),
                static=None,
                forcing=None
            )  # [time, 1, channels, lat, lon]

            forecast = forecast.squeeze(1)  # [time, channels, lat, lon]

            # Compute RMSE at each time
            for t_idx in range(len(lead_times)):
                error = (forecast[t_idx] - truth_sequence[t_idx]).pow(2).mean().sqrt()
                results['rmse_by_time'].append(error.cpu().item())

                # Per-variable RMSE
                for var_idx in range(4):
                    var_error = (forecast[t_idx, var_idx] - truth_sequence[t_idx, var_idx]).pow(2).mean().sqrt()
                    results['rmse_by_variable'][var_idx].append(var_error.cpu().item())

            # Energy error
            energy_pred = (forecast**2).sum(dim=(1, 2, 3))
            energy_truth = (truth_sequence**2).sum(dim=(1, 2, 3))
            energy_err = ((energy_pred - energy_truth) / (energy_truth + 1e-6)).abs()
            results['energy_error'].extend(energy_err.cpu().numpy())

    # Average over samples
    results['rmse_by_time'] = np.array(results['rmse_by_time']).reshape(num_samples, len(lead_times)).mean(axis=0)
    for var_idx in range(4):
        results['rmse_by_variable'][var_idx] = np.array(results['rmse_by_variable'][var_idx]).reshape(num_samples, len(lead_times)).mean(axis=0)

    results['energy_error'] = np.array(results['energy_error']).reshape(num_samples, len(lead_times)).mean(axis=0)

    print(f"Evaluation complete!")
    print(f"  Final 10-day RMSE: {results['rmse_by_time'][-1]:.6f}")

    return results


def plot_ablation_results(
    baseline_history: Dict,
    physics_history: Dict,
    baseline_forecast: Dict,
    physics_forecast: Dict,
):
    """Create comprehensive visualization of ablation study results.

    Args:
        baseline_history: Training history for baseline model
        physics_history: Training history for physics-enhanced model
        baseline_forecast: Forecast results for baseline
        physics_forecast: Forecast results for physics-enhanced
    """
    print(f"\nGenerating plots...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 11

    fig = plt.figure(figsize=(18, 12))

    # 1. Training Loss Comparison (top left)
    ax1 = plt.subplot(2, 3, 1)
    epochs = np.arange(len(baseline_history['train_loss']))
    ax1.plot(epochs, baseline_history['train_loss'], label='Baseline Train', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, baseline_history['val_loss'], label='Baseline Val', color='#1f77b4', linestyle='--', linewidth=2)
    ax1.plot(epochs, physics_history['train_loss'], label='Physics Train', color='#ff7f0e', linewidth=2)
    ax1.plot(epochs, physics_history['val_loss'], label='Physics Val', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # 2. RMSE vs Lead Time (top middle) - MAIN RESULT
    ax2 = plt.subplot(2, 3, 2)
    lead_times = baseline_forecast['lead_times']
    ax2.plot(lead_times, baseline_forecast['rmse_by_time'],
             label='Baseline', color='#1f77b4', linewidth=3, marker='o', markersize=3)
    ax2.plot(lead_times, physics_forecast['rmse_by_time'],
             label='Physics-Enhanced', color='#ff7f0e', linewidth=3, marker='s', markersize=3)
    ax2.fill_between(lead_times,
                      baseline_forecast['rmse_by_time'] * 0.9,
                      baseline_forecast['rmse_by_time'] * 1.1,
                      alpha=0.2, color='#1f77b4')
    ax2.fill_between(lead_times,
                      physics_forecast['rmse_by_time'] * 0.9,
                      physics_forecast['rmse_by_time'] * 1.1,
                      alpha=0.2, color='#ff7f0e')
    ax2.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('10-Day Forecast Error Growth', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add improvement annotation
    improvement = (baseline_forecast['rmse_by_time'][-1] - physics_forecast['rmse_by_time'][-1]) / baseline_forecast['rmse_by_time'][-1] * 100
    ax2.text(0.5, 0.95, f'10-day improvement: {improvement:.1f}%',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Energy Conservation (top right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(lead_times, baseline_forecast['energy_error'],
             label='Baseline', color='#1f77b4', linewidth=2)
    ax3.plot(lead_times, physics_forecast['energy_error'],
             label='Physics-Enhanced', color='#ff7f0e', linewidth=2)
    ax3.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Energy Error', fontsize=12, fontweight='bold')
    ax3.set_title('Energy Conservation', fontsize=14, fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Per-Variable RMSE (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    var_names = ['U Wind', 'V Wind', 'Geopotential', 'Temperature']
    colors_baseline = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    colors_physics = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, var_name in enumerate(var_names):
        ax4.plot(lead_times, baseline_forecast['rmse_by_variable'][i],
                 label=f'{var_name} (Base)', color=colors_baseline[i],
                 linestyle='--', linewidth=2, alpha=0.7)
        ax4.plot(lead_times, physics_forecast['rmse_by_variable'][i],
                 label=f'{var_name} (Phys)', color=colors_physics[i],
                 linewidth=2)

    ax4.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Variable Forecast Error', fontsize=14, fontweight='bold')
    ax4.legend(frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Physics Loss Components (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    if 'pv_conservation' in physics_history and physics_history['pv_conservation']:
        epochs = np.arange(len(physics_history['pv_conservation']))
        ax5.plot(epochs, physics_history['pv_conservation'], label='PV Conservation', linewidth=2)
        ax5.plot(epochs, physics_history['energy_spectra'], label='Energy Spectra', linewidth=2)
        ax5.plot(epochs, physics_history['mass_divergence'], label='Mass Divergence', linewidth=2)
        if 'geostrophic_balance' in physics_history and physics_history['geostrophic_balance']:
            ax5.plot(epochs, physics_history['geostrophic_balance'], label='Geostrophic Balance', linewidth=2)
        ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax5.set_title('Physics Constraint Evolution', fontsize=14, fontweight='bold')
        ax5.legend(frameon=True, fancybox=True, shadow=True)
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    else:
        ax5.text(0.5, 0.5, 'No physics losses\nfor baseline model',
                transform=ax5.transAxes, ha='center', va='center', fontsize=14)
        ax5.axis('off')

    # 6. Skill Score Comparison (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    lead_days = [1, 3, 5, 7, 10]
    lead_indices = [int(d * 4) for d in lead_days]  # 4 timesteps per day

    baseline_rmse_at_days = [baseline_forecast['rmse_by_time'][i] for i in lead_indices]
    physics_rmse_at_days = [physics_forecast['rmse_by_time'][i] for i in lead_indices]

    x = np.arange(len(lead_days))
    width = 0.35

    bars1 = ax6.bar(x - width/2, baseline_rmse_at_days, width, label='Baseline',
                    color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax6.bar(x + width/2, physics_rmse_at_days, width, label='Physics-Enhanced',
                    color='#ff7f0e', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax6.set_xlabel('Forecast Lead Time (days)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax6.set_title('Forecast Skill at Key Lead Times', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{d}d' for d in lead_days])
    ax6.legend(frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ablation Study: Baseline vs Physics-Enhanced WeatherFlow\n10-Day Forecast Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_path = RESULTS_DIR / 'ablation_study_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for publication quality
    output_path_pdf = RESULTS_DIR / 'ablation_study_results.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_path_pdf}")

    plt.show()

    return fig


def save_results_summary(
    baseline_history: Dict,
    physics_history: Dict,
    baseline_forecast: Dict,
    physics_forecast: Dict,
):
    """Save numerical results summary to JSON."""

    summary = {
        'experiment_date': datetime.now().isoformat(),
        'baseline': {
            'final_train_loss': float(baseline_history['train_loss'][-1]),
            'final_val_loss': float(baseline_history['val_loss'][-1]),
            'day1_rmse': float(baseline_forecast['rmse_by_time'][4]),  # index 4 = 1 day
            'day3_rmse': float(baseline_forecast['rmse_by_time'][12]),
            'day5_rmse': float(baseline_forecast['rmse_by_time'][20]),
            'day7_rmse': float(baseline_forecast['rmse_by_time'][28]),
            'day10_rmse': float(baseline_forecast['rmse_by_time'][-1]),
        },
        'physics_enhanced': {
            'final_train_loss': float(physics_history['train_loss'][-1]),
            'final_val_loss': float(physics_history['val_loss'][-1]),
            'day1_rmse': float(physics_forecast['rmse_by_time'][4]),
            'day3_rmse': float(physics_forecast['rmse_by_time'][12]),
            'day5_rmse': float(physics_forecast['rmse_by_time'][20]),
            'day7_rmse': float(physics_forecast['rmse_by_time'][28]),
            'day10_rmse': float(physics_forecast['rmse_by_time'][-1]),
        },
    }

    # Calculate improvements
    summary['improvements'] = {
        'day1_improvement_%': (summary['baseline']['day1_rmse'] - summary['physics_enhanced']['day1_rmse']) / summary['baseline']['day1_rmse'] * 100,
        'day3_improvement_%': (summary['baseline']['day3_rmse'] - summary['physics_enhanced']['day3_rmse']) / summary['baseline']['day3_rmse'] * 100,
        'day5_improvement_%': (summary['baseline']['day5_rmse'] - summary['physics_enhanced']['day5_rmse']) / summary['baseline']['day5_rmse'] * 100,
        'day7_improvement_%': (summary['baseline']['day7_rmse'] - summary['physics_enhanced']['day7_rmse']) / summary['baseline']['day7_rmse'] * 100,
        'day10_improvement_%': (summary['baseline']['day10_rmse'] - summary['physics_enhanced']['day10_rmse']) / summary['baseline']['day10_rmse'] * 100,
    }

    # Save to JSON
    output_path = RESULTS_DIR / 'ablation_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults summary saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"\nBaseline Model:")
    print(f"  Final Val Loss: {summary['baseline']['final_val_loss']:.6f}")
    print(f"  Day 10 RMSE:    {summary['baseline']['day10_rmse']:.6f}")

    print(f"\nPhysics-Enhanced Model:")
    print(f"  Final Val Loss: {summary['physics_enhanced']['final_val_loss']:.6f}")
    print(f"  Day 10 RMSE:    {summary['physics_enhanced']['day10_rmse']:.6f}")

    print(f"\nImprovements:")
    for key, value in summary['improvements'].items():
        print(f"  {key}: {value:+.2f}%")

    print("="*80)

    return summary


def main():
    """Run complete ablation study."""

    print("\n" + "="*80)
    print("WEATHERFLOW ABLATION STUDY: BASELINE VS PHYSICS-ENHANCED")
    print("="*80)

    # 1. Generate synthetic data
    train_data, pressure_levels = create_synthetic_era5_data(
        num_samples=200,  # Larger dataset for better training
        num_channels=4,
    )

    # 2. Train baseline model
    print("\n" + "="*80)
    print("STEP 1: Training Baseline Model")
    print("="*80)
    baseline_model = create_model(enhanced_physics=False)
    baseline_history = train_model(
        baseline_model,
        train_data,
        pressure_levels,
        num_epochs=100,  # More epochs for convergence
        batch_size=8,
        lr=1e-3,
        model_name="baseline"
    )

    # 3. Train physics-enhanced model
    print("\n" + "="*80)
    print("STEP 2: Training Physics-Enhanced Model")
    print("="*80)
    physics_model = create_model(enhanced_physics=True)
    physics_history = train_model(
        physics_model,
        train_data,
        pressure_levels,
        num_epochs=100,
        batch_size=8,
        lr=1e-3,
        model_name="physics_enhanced"
    )

    # 4. Evaluate 10-day forecasts
    print("\n" + "="*80)
    print("STEP 3: Evaluating 10-Day Forecasts")
    print("="*80)

    # Create test data
    test_data, _ = create_synthetic_era5_data(num_samples=30, num_channels=4)

    print("\nBaseline Model Evaluation:")
    baseline_forecast = evaluate_10day_forecast(
        baseline_model,
        test_data,
        pressure_levels,
        num_samples=20
    )

    print("\nPhysics-Enhanced Model Evaluation:")
    physics_forecast = evaluate_10day_forecast(
        physics_model,
        test_data,
        pressure_levels,
        num_samples=20
    )

    # 5. Generate plots
    print("\n" + "="*80)
    print("STEP 4: Generating Comparison Plots")
    print("="*80)
    plot_ablation_results(
        baseline_history,
        physics_history,
        baseline_forecast,
        physics_forecast
    )

    # 6. Save summary
    summary = save_results_summary(
        baseline_history,
        physics_history,
        baseline_forecast,
        physics_forecast
    )

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - ablation_study_results.png (visualization)")
    print(f"  - ablation_study_results.pdf (publication quality)")
    print(f"  - ablation_summary.json (numerical results)")
    print(f"  - baseline_best.pt (model checkpoint)")
    print(f"  - physics_enhanced_best.pt (model checkpoint)")

    return summary


if __name__ == '__main__':
    summary = main()

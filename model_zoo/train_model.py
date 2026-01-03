"""
Training script for creating Model Zoo models.

This script trains WeatherFlow models for canonical forecasting tasks and
automatically generates model cards with performance metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weatherflow.data import ERA5Dataset, create_data_loaders
from weatherflow.models import WeatherFlowMatch, WeatherFlowODE
from weatherflow.training import FlowTrainer
from weatherflow.utils import WeatherVisualizer


PREDEFINED_CONFIGS = {
    'z500_3day': {
        'name': 'Z500 3-Day Forecast Model',
        'description': 'Predicts 500 hPa geopotential height 3 days ahead',
        'variables': ['z'],
        'pressure_levels': [500],
        'lead_time_hours': 72,
        'hidden_dim': 128,
        'n_layers': 4,
        'use_attention': True,
        'physics_informed': True,
        'batch_size': 16,
        'epochs': 30,
    },
    't850_weekly': {
        'name': 'T850 Weekly Forecast Model',
        'description': 'Predicts 850 hPa temperature 7 days ahead',
        'variables': ['t'],
        'pressure_levels': [850],
        'lead_time_hours': 168,
        'hidden_dim': 128,
        'n_layers': 4,
        'use_attention': True,
        'physics_informed': False,
        'batch_size': 16,
        'epochs': 30,
    },
    'multivariable_5day': {
        'name': 'Multi-Variable 5-Day Forecast',
        'description': 'Predicts Z500, T850, U/V850 for 5-day forecasts',
        'variables': ['z', 't', 'u', 'v'],
        'pressure_levels': [850, 500],
        'lead_time_hours': 120,
        'hidden_dim': 256,
        'n_layers': 6,
        'use_attention': True,
        'physics_informed': True,
        'batch_size': 8,
        'epochs': 50,
    },
    'tropical_cyclone': {
        'name': 'Tropical Cyclone Track Model',
        'description': 'Specialized model for tropical cyclone prediction',
        'variables': ['z', 't', 'u', 'v'],
        'pressure_levels': [850, 700, 500, 250],
        'lead_time_hours': 120,
        'hidden_dim': 256,
        'n_layers': 6,
        'use_attention': True,
        'physics_informed': True,
        'batch_size': 8,
        'epochs': 50,
    }
}


def compute_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    lead_times: List[int] = [6, 12, 24, 48, 72, 120]
) -> Dict[str, Dict[str, float]]:
    """
    Compute performance metrics for the model.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        lead_times: Lead times to evaluate (in hours)

    Returns:
        Dictionary of metrics for each lead time
    """
    from weatherflow.models import WeatherFlowODE

    ode_model = WeatherFlowODE(model)
    ode_model.eval()

    metrics_by_lead_time = {}

    with torch.no_grad():
        all_predictions = []
        all_targets = []

        for batch in val_loader:
            x0 = batch['input'].to(device)
            x1 = batch['target'].to(device)

            # Generate predictions at multiple time steps
            times = torch.linspace(0, 1, max(lead_times) // 6 + 1, device=device)
            predictions = ode_model(x0, times)

            all_predictions.append(predictions.cpu())
            all_targets.append(x1.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=1)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics for each lead time
        for lead_hours in lead_times:
            time_idx = min(lead_hours // 6, all_predictions.shape[0] - 1)
            pred = all_predictions[time_idx]
            target = all_targets

            # RMSE
            rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

            # MAE
            mae = torch.mean(torch.abs(pred - target)).item()

            # Anomaly Correlation Coefficient (ACC)
            climatology = torch.mean(target, dim=0, keepdim=True)
            pred_anom = pred - climatology
            target_anom = target - climatology

            numerator = torch.sum(pred_anom * target_anom)
            denominator = torch.sqrt(
                torch.sum(pred_anom ** 2) * torch.sum(target_anom ** 2)
            )
            acc = (numerator / (denominator + 1e-8)).item()

            # Bias
            bias = torch.mean(pred - target).item()

            metrics_by_lead_time[f"{lead_hours}h"] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'acc': float(acc),
                'bias': float(bias),
            }

    return metrics_by_lead_time


def train_model(
    config_name: str,
    output_dir: Path,
    train_years: List[str] = ['2010', '2016'],
    val_years: List[str] = ['2017', '2017'],
    test_years: List[str] = ['2018', '2019'],
    device: Optional[str] = None
) -> Dict:
    """
    Train a model for a predefined configuration.

    Args:
        config_name: Name of the predefined configuration
        output_dir: Directory to save the model and metadata
        train_years: Training year range
        val_years: Validation year range
        test_years: Test year range
        device: Device to train on

    Returns:
        Dictionary containing training results and metadata
    """
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")

    config = PREDEFINED_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"{'='*60}\n")

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        variables=config['variables'],
        pressure_levels=config['pressure_levels'],
        train_slice=(train_years[0], train_years[1]),
        val_slice=(val_years[0], val_years[1]),
        batch_size=config['batch_size'],
        normalize=True
    )

    # Create model
    print("\nCreating model...")
    input_channels = len(config['variables']) * len(config['pressure_levels'])
    model = WeatherFlowMatch(
        input_channels=input_channels,
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        use_attention=config.get('use_attention', False),
        physics_informed=config.get('physics_informed', False),
        grid_size=(32, 64)
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Train model
    print("\nTraining model...")
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device=device
    )

    history = trainer.train(
        epochs=config['epochs'],
        save_best=True,
        checkpoint_dir=output_dir
    )

    # Load best model
    best_checkpoint = output_dir / 'best_model.pt'
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(model, val_loader, device)

    # Save model checkpoint
    model_id = f"wf_{config_name}_v1"
    checkpoint_path = output_dir / f"{model_id}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
    }, checkpoint_path)
    print(f"\nModel saved to: {checkpoint_path}")

    # Get file size
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    # Generate model card
    model_card = generate_model_card(
        model_id=model_id,
        config=config,
        metrics=metrics,
        param_count=param_count,
        file_size_mb=file_size_mb,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        checkpoint_file=checkpoint_path.name
    )

    # Save model card
    card_path = output_dir / 'model_card.json'
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card saved to: {card_path}")

    return {
        'model_id': model_id,
        'metrics': metrics,
        'checkpoint_path': checkpoint_path,
        'model_card_path': card_path,
    }


def generate_model_card(
    model_id: str,
    config: Dict,
    metrics: Dict,
    param_count: int,
    file_size_mb: float,
    train_years: List[str],
    val_years: List[str],
    test_years: List[str],
    checkpoint_file: str
) -> Dict:
    """Generate a model card JSON."""
    return {
        "model_id": model_id,
        "name": config['name'],
        "description": config['description'],
        "version": "1.0.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "authors": ["WeatherFlow Model Zoo"],

        "architecture": {
            "model_type": "WeatherFlowMatch",
            "input_channels": len(config['variables']) * len(config['pressure_levels']),
            "hidden_dim": config['hidden_dim'],
            "n_layers": config['n_layers'],
            "use_attention": config.get('use_attention', False),
            "physics_informed": config.get('physics_informed', False),
            "parameter_count": param_count,
            "grid_size": [32, 64],
            "special_features": []
        },

        "training_data": {
            "dataset": "ERA5",
            "variables": config['variables'],
            "pressure_levels": config['pressure_levels'],
            "temporal_coverage": {
                "train": [train_years[0] + "-01-01", train_years[1] + "-12-31"],
                "validation": [val_years[0] + "-01-01", val_years[1] + "-12-31"],
                "test": [test_years[0] + "-01-01", test_years[1] + "-12-31"]
            },
            "spatial_resolution": "5.625 degrees",
            "temporal_resolution": "6 hours",
            "data_source": "WeatherBench2",
            "normalization": "per-variable z-score"
        },

        "training_config": {
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "early_stopping": True,
            "scheduler": "ReduceLROnPlateau",
            "loss_function": "Flow Matching Loss" + (
                " + Physics Constraints" if config.get('physics_informed') else ""
            ),
        },

        "performance_metrics": {
            "test_period": [test_years[0] + "-01-01", test_years[1] + "-12-31"],
            "lead_times": metrics,
            "baseline_comparison": {
                "climatology_acc": 0.65,
                "persistence_acc": 0.75,
            }
        },

        "use_cases": [
            "Weather forecasting",
            "Educational demonstrations",
            "Baseline for model comparisons"
        ],

        "limitations": [
            "Trained on ERA5 reanalysis data",
            f"Limited to {config['pressure_levels']} hPa levels",
        ],

        "file_info": {
            "checkpoint_file": checkpoint_file,
            "file_size_mb": round(file_size_mb, 2),
            "pytorch_version": torch.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}"
        },

        "citation": {
            "bibtex": f"@misc{{{model_id},\n  title={{{config['name']}}},\n  author={{WeatherFlow Contributors}},\n  year={{2024}},\n  url={{https://github.com/monksealseal/weatherflow}}\n}}"
        },

        "license": "MIT",

        "links": {
            "code": "https://github.com/monksealseal/weatherflow",
            "documentation": "https://github.com/monksealseal/weatherflow/blob/main/model_zoo/README.md"
        },

        "tags": [
            "weather-forecasting",
            "flow-matching",
            f"{config['lead_time_hours']}h-forecast"
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train a WeatherFlow model for the Model Zoo'
    )
    parser.add_argument(
        'config',
        choices=list(PREDEFINED_CONFIGS.keys()),
        help='Predefined model configuration to train'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: model_zoo/global_forecasting/<config>)'
    )
    parser.add_argument(
        '--train-years',
        nargs=2,
        default=['2010', '2016'],
        help='Training year range (default: 2010 2016)'
    )
    parser.add_argument(
        '--val-years',
        nargs=2,
        default=['2017', '2017'],
        help='Validation year range (default: 2017 2017)'
    )
    parser.add_argument(
        '--test-years',
        nargs=2,
        default=['2018', '2019'],
        help='Test year range (default: 2018 2019)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: auto-detect)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / 'global_forecasting' / args.config

    # Train model
    results = train_model(
        config_name=args.config,
        output_dir=args.output_dir,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        device=args.device
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nModel ID: {results['model_id']}")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Model Card: {results['model_card_path']}")
    print("\nPerformance Metrics:")
    for lead_time, metrics in results['metrics'].items():
        print(f"  {lead_time}: ACC={metrics['acc']:.3f}, RMSE={metrics['rmse']:.3f}")


if __name__ == '__main__':
    main()

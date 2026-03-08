"""Fixed-budget training script for a single autoresearch experiment.

Trains a WeatherFlowMatch model for exactly N minutes (default 5), evaluates
on the validation set, and returns the result. This is the atomic unit of
the autoresearch loop — every experiment is judged on the same wall-clock
budget, making architectural and hyperparameter changes fairly comparable.
"""

import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from weatherflow.models.flow_matching import WeatherFlowMatch
from weatherflow.training.flow_trainer import FlowTrainer, compute_flow_loss
from weatherflow.training.metrics import energy_ratio, mae, rmse

from .experiment_config import ExperimentConfig, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation (runs once, shared across experiments)
# ---------------------------------------------------------------------------

def prepare_synthetic_data(
    num_samples: int = 400,
    lat_dim: int = 32,
    lon_dim: int = 64,
    num_channels: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Create synthetic ERA5-like weather data for training and validation.

    In production, replace this with real ERA5 data from
    ``weatherflow.data.era5.ERA5Dataset``. The synthetic data includes
    realistic jet stream patterns, Rossby waves, and temperature gradients.

    Returns:
        (train_dataset, val_dataset) as TensorDatasets of (x0, x1) pairs.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    lat = torch.linspace(-np.pi / 2, np.pi / 2, lat_dim)
    lon = torch.linspace(0, 2 * np.pi, lon_dim)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

    states = []
    for _ in range(num_samples):
        jet_lat = torch.tensor(rng.randn() * 0.3)
        jet_strength = 20 + torch.tensor(rng.randn() * 5.0)

        u = jet_strength * torch.exp(-((lat_grid - jet_lat) ** 2) / 0.2)
        u = u + torch.randn_like(u) * 3

        v = 5 * torch.sin(4 * lon_grid) * torch.cos(lat_grid)
        v = v + torch.randn_like(v) * 2

        z = 5500 + 100 * torch.sin(3 * lon_grid) * torch.cos(2 * lat_grid)
        z = z + torch.randn_like(z) * 50

        t = 285 - 40 * torch.abs(lat_grid) / (np.pi / 2)
        t = t + torch.randn_like(t) * 5

        state = torch.stack([u, v, z, t], dim=0)
        states.append(state)

    data = torch.stack(states, dim=0)

    # Normalize to zero mean, unit variance per channel
    data[:, 0] = (data[:, 0] - 0) / 20
    data[:, 1] = (data[:, 1] - 0) / 10
    data[:, 2] = (data[:, 2] - 5500) / 200
    data[:, 3] = (data[:, 3] - 260) / 20

    # Create (x0, x1) pairs by pairing sequential states
    n_pairs = num_samples // 2
    x0 = data[:n_pairs]
    x1 = data[n_pairs : 2 * n_pairs]

    # Split into train/val
    n_val = int(n_pairs * val_fraction)
    n_train = n_pairs - n_val

    train_dataset = TensorDataset(x0[:n_train], x1[:n_train])
    val_dataset = TensorDataset(x0[n_train:], x1[n_train:])

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Model & optimizer construction from config
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig) -> WeatherFlowMatch:
    """Instantiate a WeatherFlowMatch model from config."""
    return WeatherFlowMatch(
        input_channels=config.input_channels,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        use_attention=config.use_attention,
        grid_size=config.grid_size,
        physics_informed=config.physics_informed,
        window_size=config.window_size,
        spherical_padding=config.spherical_padding,
        use_spectral_mixer=config.use_spectral_mixer,
        spectral_modes=config.spectral_modes,
    )


def build_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Instantiate an optimizer from config."""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Instantiate a learning rate scheduler from config."""
    warmup_steps = int(total_steps * config.warmup_fraction)

    if config.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
    elif config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=max(config.warmup_fraction, 0.1),
        )
    elif config.scheduler == "linear":
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(total_steps - warmup_steps, 1),
        )
    elif config.scheduler == "constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_steps
        )
    else:
        return None

    if warmup_steps > 0 and config.scheduler != "onecycle":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps],
        )

    return main_scheduler


# ---------------------------------------------------------------------------
# Core: run a single fixed-budget experiment
# ---------------------------------------------------------------------------

class TimeBudgetExceeded(Exception):
    """Raised when the training time budget is exhausted."""


def run_experiment(
    config: ExperimentConfig,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    budget_minutes: float = 5.0,
    device: str = "auto",
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training experiment under a fixed time budget.

    Args:
        config: Full experiment configuration.
        train_dataset: Training data as TensorDataset of (x0, x1) pairs.
        val_dataset: Validation data as TensorDataset of (x0, x1) pairs.
        budget_minutes: Wall-clock time budget in minutes.
        device: Device to train on ('auto', 'cuda', 'cpu').
        checkpoint_dir: Optional directory to save best checkpoint.

    Returns:
        Dictionary with experiment results including val_rmse, val_loss,
        epochs_completed, training_time, and per-epoch metrics.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    budget_seconds = budget_minutes * 60
    start_time = time.time()

    # Build components
    model = build_model(config.model)
    optimizer = build_optimizer(model, config.training)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Estimate total steps for scheduler
    steps_per_epoch = len(train_loader)
    estimated_epochs = max(int(budget_seconds / 10), 5)  # rough estimate
    total_steps = steps_per_epoch * estimated_epochs

    scheduler = build_scheduler(optimizer, config.training, total_steps)

    # Build trainer
    noise_std = config.training.noise_std
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_amp=config.training.use_amp,
        use_wandb=False,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
        physics_regularization=config.model.physics_informed,
        physics_lambda=config.training.physics_lambda,
        loss_type=config.training.loss_type,
        loss_weighting=config.training.loss_weighting,
        grad_clip=config.training.grad_clip,
        ema_decay=config.training.ema_decay,
        seed=config.training.seed,
        noise_std=noise_std,
    )

    # Training loop with time budget
    results = {
        "epochs_completed": 0,
        "train_losses": [],
        "val_losses": [],
        "val_rmses": [],
        "val_maes": [],
        "val_energy_ratios": [],
        "best_val_rmse": float("inf"),
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "final_val_rmse": float("inf"),
        "final_val_loss": float("inf"),
        "training_time_seconds": 0.0,
        "error": None,
        "nan_detected": False,
    }

    try:
        epoch = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed >= budget_seconds:
                break

            # Train one epoch
            trainer.current_epoch = epoch
            train_metrics = trainer.train_epoch(train_loader)

            # Check for NaN
            if np.isnan(train_metrics.get("loss", 0)):
                results["nan_detected"] = True
                results["error"] = f"NaN loss at epoch {epoch}"
                break

            # Validate
            val_metrics = trainer.validate(val_loader)

            val_rmse = val_metrics.get("val_rmse", float("inf"))
            val_loss = val_metrics.get("val_loss", float("inf"))

            # Check for NaN in validation
            if np.isnan(val_rmse) or np.isnan(val_loss):
                results["nan_detected"] = True
                results["error"] = f"NaN in validation at epoch {epoch}"
                break

            # Log
            results["train_losses"].append(train_metrics["loss"])
            results["val_losses"].append(val_loss)
            results["val_rmses"].append(val_rmse)
            results["val_maes"].append(val_metrics.get("val_mae", 0.0))
            results["val_energy_ratios"].append(
                val_metrics.get("val_energy_ratio", 0.0)
            )

            # Track best
            if val_rmse < results["best_val_rmse"]:
                results["best_val_rmse"] = val_rmse
                results["best_val_loss"] = val_loss
                results["best_epoch"] = epoch

                # Save checkpoint if directory provided
                if checkpoint_dir:
                    trainer.save_checkpoint("best_model.pt")

            results["epochs_completed"] = epoch + 1
            epoch += 1

            # Log progress
            elapsed = time.time() - start_time
            remaining = budget_seconds - elapsed
            logger.info(
                f"Epoch {epoch}: val_rmse={val_rmse:.6f} "
                f"(best={results['best_val_rmse']:.6f}) "
                f"[{remaining:.0f}s remaining]"
            )

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Experiment failed: {e}")

    # Final metrics
    results["training_time_seconds"] = time.time() - start_time
    if results["val_rmses"]:
        results["final_val_rmse"] = results["val_rmses"][-1]
        results["final_val_loss"] = results["val_losses"][-1]

    # Count model parameters
    results["num_parameters"] = sum(p.numel() for p in model.parameters())
    results["num_trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return results

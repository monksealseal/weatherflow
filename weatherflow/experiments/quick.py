"""
Quick Experiment Utilities

High-level functions for rapid experimentation with weather AI models.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .mini_datasets import create_mini_era5, MiniDataset
from .manager import ExperimentConfig, ExperimentResult


@dataclass
class QuickExperimentResult:
    """Result from a quick experiment."""
    model_name: str
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    duration_seconds: float
    num_params: int
    samples_per_second: float

    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"{self.model_name}: "
            f"val_loss={self.best_val_loss:.4f} @ epoch {self.best_epoch}, "
            f"params={self.num_params/1e6:.2f}M, "
            f"time={self.duration_seconds:.1f}s"
        )


def quick_experiment(
    model: nn.Module,
    train_data: Union[Dataset, DataLoader, None] = None,
    val_data: Union[Dataset, DataLoader, None] = None,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_category: str = "transformer",
    verbose: bool = True,
    **kwargs,
) -> QuickExperimentResult:
    """
    Run a quick experiment for rapid prototyping.

    This is the main entry point for testing models quickly.

    Args:
        model: Model to train
        train_data: Training data (Dataset, DataLoader, or None for synthetic)
        val_data: Validation data (Dataset, DataLoader, or None for synthetic)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        model_category: Model category for training strategy
        verbose: Print progress
        **kwargs: Additional training config arguments

    Returns:
        QuickExperimentResult with training results

    Example:
        >>> from weatherflow.model_library.components import create_model
        >>> model = create_model("vit_tiny", in_channels=4, out_channels=4)
        >>> result = quick_experiment(model, epochs=3)
        >>> print(result.summary())
    """
    # Import trainer
    from ..training.unified_trainer import UnifiedTrainer, TrainingConfig

    # Create synthetic data if not provided
    if train_data is None or val_data is None:
        # Infer dimensions from model if possible
        in_channels = getattr(model, "in_channels", 4)
        img_size = getattr(model, "img_size", (32, 64))

        train_loader, val_loader = create_mini_era5(
            num_train=100,
            num_val=20,
            in_channels=in_channels,
            img_size=img_size,
            batch_size=batch_size,
        )
    else:
        # Convert to DataLoader if needed
        if isinstance(train_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            train_loader = train_data

        if isinstance(val_data, Dataset):
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        else:
            val_loader = val_data

    # Create training config
    config = TrainingConfig(
        model_name=model.__class__.__name__,
        model_category=model_category,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        mini_mode=True,
        log_every=1 if verbose else epochs,
        **kwargs,
    )

    # Move model to device
    model = model.to(device)

    # Create trainer
    trainer = UnifiedTrainer(model, config)

    # Track time
    start_time = time.time()

    # Training loop with progress
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        train_losses.append(train_metrics["loss"])

        # Validate
        val_metrics = trainer.validate(val_loader)
        val_losses.append(val_metrics["val_loss"])

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['val_loss']:.4f}")

    duration = time.time() - start_time

    # Compute results
    num_params = sum(p.numel() for p in model.parameters())
    total_samples = len(train_loader.dataset) * epochs
    samples_per_second = total_samples / duration

    result = QuickExperimentResult(
        model_name=model.__class__.__name__,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        best_val_loss=min(val_losses),
        best_epoch=val_losses.index(min(val_losses)),
        train_losses=train_losses,
        val_losses=val_losses,
        duration_seconds=duration,
        num_params=num_params,
        samples_per_second=samples_per_second,
    )

    if verbose:
        print(f"\n{result.summary()}")

    return result


def compare_models(
    models: Dict[str, nn.Module],
    train_data: Union[Dataset, DataLoader, None] = None,
    val_data: Union[Dataset, DataLoader, None] = None,
    epochs: int = 5,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, QuickExperimentResult]:
    """
    Compare multiple models on the same data.

    Args:
        models: Dictionary mapping model names to model instances
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary mapping model names to results

    Example:
        >>> from weatherflow.model_library.components import create_model
        >>> models = {
        ...     "ViT-Tiny": create_model("vit_tiny"),
        ...     "Flow-Tiny": create_model("flow_matching_tiny"),
        ... }
        >>> results = compare_models(models, epochs=3)
        >>> for name, result in results.items():
        ...     print(result.summary())
    """
    # Create shared data if not provided
    if train_data is None:
        # Use first model to infer dimensions
        first_model = next(iter(models.values()))
        in_channels = getattr(first_model, "in_channels", 4)
        img_size = getattr(first_model, "img_size", (32, 64))

        train_loader, val_loader = create_mini_era5(
            num_train=100,
            num_val=20,
            in_channels=in_channels,
            img_size=img_size,
            batch_size=batch_size,
        )
    else:
        if isinstance(train_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            train_loader = train_data

        if isinstance(val_data, Dataset):
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        else:
            val_loader = val_data

    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print('='*50)

        result = quick_experiment(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        results[name] = result

    # Print comparison summary
    if verbose:
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print('='*50)

        # Sort by best val loss
        sorted_results = sorted(results.items(), key=lambda x: x[1].best_val_loss)

        for i, (name, result) in enumerate(sorted_results):
            rank = i + 1
            print(f"{rank}. {result.summary()}")

    return results


def benchmark_architectures(
    architecture_names: Optional[List[str]] = None,
    in_channels: int = 4,
    out_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
    epochs: int = 3,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, QuickExperimentResult]:
    """
    Benchmark multiple preset architectures.

    Args:
        architecture_names: List of preset names (None = all tiny presets)
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Image size
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary mapping architecture names to results

    Example:
        >>> results = benchmark_architectures(
        ...     ["vit_tiny", "flow_matching_tiny", "fourcastnet_tiny"],
        ...     epochs=3
        ... )
    """
    from ..model_library.components import create_model, list_presets

    # Default to all tiny presets
    if architecture_names is None:
        architecture_names = [p for p in list_presets() if "tiny" in p]

    # Create models
    models = {}
    for name in architecture_names:
        try:
            model = create_model(
                preset=name,
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=img_size,
            )
            models[name] = model
        except Exception as e:
            if verbose:
                print(f"Failed to create {name}: {e}")

    if not models:
        raise ValueError("No models could be created")

    # Create data
    train_loader, val_loader = create_mini_era5(
        num_train=100,
        num_val=20,
        in_channels=in_channels,
        img_size=img_size,
        batch_size=batch_size,
    )

    # Run comparison
    return compare_models(
        models,
        train_loader,
        val_loader,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )


def test_model_forward(
    model: nn.Module,
    in_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
    batch_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Test that a model's forward pass works correctly.

    Args:
        model: Model to test
        in_channels: Number of input channels
        img_size: Image size
        batch_size: Batch size
        device: Device to test on

    Returns:
        Dictionary with test results

    Example:
        >>> model = create_model("vit_tiny")
        >>> results = test_model_forward(model)
        >>> print(results["success"])
    """
    model = model.to(device)
    model.eval()

    results = {
        "success": False,
        "input_shape": None,
        "output_shape": None,
        "num_params": sum(p.numel() for p in model.parameters()),
        "forward_time_ms": 0,
        "error": None,
    }

    # Create test input
    x = torch.randn(batch_size, in_channels, *img_size, device=device)
    results["input_shape"] = tuple(x.shape)

    try:
        # Test forward pass
        start = time.time()
        with torch.no_grad():
            # Check if model needs time input (flow matching / diffusion)
            if hasattr(model, "is_flow_matching") and model.is_flow_matching:
                t = torch.rand(batch_size, device=device)
                output = model(x, t)
            else:
                output = model(x)

        results["forward_time_ms"] = (time.time() - start) * 1000
        results["output_shape"] = tuple(output.shape)
        results["success"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def profile_model(
    model: nn.Module,
    in_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
    batch_size: int = 2,
    num_iterations: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Profile a model's performance.

    Args:
        model: Model to profile
        in_channels: Number of input channels
        img_size: Image size
        batch_size: Batch size
        num_iterations: Number of iterations to average
        device: Device to profile on

    Returns:
        Dictionary with profiling results
    """
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, in_channels, *img_size, device=device)

    # Check if flow matching model
    is_flow = hasattr(model, "is_flow_matching") and model.is_flow_matching
    t = torch.rand(batch_size, device=device) if is_flow else None

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            if is_flow:
                _ = model(x, t)
            else:
                _ = model(x)

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward passes
    times = []
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            if is_flow:
                _ = model(x, t)
            else:
                _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    # Memory usage
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    return {
        "mean_forward_ms": np.mean(times) * 1000,
        "std_forward_ms": np.std(times) * 1000,
        "min_forward_ms": np.min(times) * 1000,
        "max_forward_ms": np.max(times) * 1000,
        "throughput_samples_per_sec": batch_size / np.mean(times),
        "gpu_memory_mb": memory_mb,
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }

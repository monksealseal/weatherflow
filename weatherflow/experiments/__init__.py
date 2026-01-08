"""
Experiment Management System for Weather AI Research

Provides tools for:
- Running multiple experiments with different hyperparameters
- Tracking and comparing experiment results
- Hyperparameter sweeps (grid, random, bayesian)
- Experiment persistence and resumption
- Results visualization and analysis

Example:
    >>> from weatherflow.experiments import (
    ...     ExperimentManager, ExperimentConfig,
    ...     HyperparameterSweep, quick_experiment
    ... )
    >>>
    >>> # Run a single experiment
    >>> result = quick_experiment(model, train_data, val_data, epochs=5)
    >>>
    >>> # Run hyperparameter sweep
    >>> sweep = HyperparameterSweep({
    ...     "learning_rate": [1e-3, 1e-4, 1e-5],
    ...     "batch_size": [4, 8, 16],
    ... })
    >>> manager = ExperimentManager("my_experiments")
    >>> results = manager.run_sweep(sweep, model_fn, train_data, val_data)
"""

from .manager import (
    ExperimentManager,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
)

from .sweeps import (
    HyperparameterSweep,
    GridSweep,
    RandomSweep,
    SweepResult,
)

from .tracking import (
    ExperimentTracker,
    MetricsLogger,
)

from .mini_datasets import (
    MiniDataset,
    create_mini_era5,
    create_synthetic_weather_data,
    get_sample_data,
)

from .quick import (
    quick_experiment,
    compare_models,
    benchmark_architectures,
)

__all__ = [
    # Manager
    "ExperimentManager",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    # Sweeps
    "HyperparameterSweep",
    "GridSweep",
    "RandomSweep",
    "SweepResult",
    # Tracking
    "ExperimentTracker",
    "MetricsLogger",
    # Mini datasets
    "MiniDataset",
    "create_mini_era5",
    "create_synthetic_weather_data",
    "get_sample_data",
    # Quick experiments
    "quick_experiment",
    "compare_models",
    "benchmark_architectures",
]

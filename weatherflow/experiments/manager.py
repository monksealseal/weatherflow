"""
Experiment Manager for Weather AI Research

Orchestrates experiment execution, tracking, and comparison.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Experiment identity
    name: str = "experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Model configuration
    model_name: str = "custom"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Loss configuration
    loss_type: str = "mse"
    use_physics_loss: bool = False
    physics_weight: float = 0.1

    # Data configuration
    data_source: str = "era5"
    train_samples: Optional[int] = None  # None = use all
    val_samples: Optional[int] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    # Mini mode for quick experiments
    mini_mode: bool = False
    mini_train_samples: int = 100
    mini_val_samples: int = 20

    # Random seed
    seed: int = 42

    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """Results from an experiment."""

    # Identity
    experiment_id: str = ""
    config: Optional[ExperimentConfig] = None

    # Status
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # History
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Model info
    num_params: int = 0
    gpu_memory_mb: float = 0.0

    # Paths
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict() if self.config else None,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "metrics": self.metrics,
            "num_params": self.num_params,
            "gpu_memory_mb": self.gpu_memory_mb,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
            "error_message": self.error_message,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        data = data.copy()
        if data.get("config"):
            data["config"] = ExperimentConfig.from_dict(data["config"])
        if data.get("status"):
            data["status"] = ExperimentStatus(data["status"])
        if data.get("start_time"):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of results."""
        return {
            "id": self.experiment_id,
            "status": self.status.value,
            "duration": f"{self.duration_seconds:.1f}s",
            "best_val_loss": f"{self.best_val_loss:.4f}",
            "best_epoch": self.best_epoch,
            "params": f"{self.num_params / 1e6:.2f}M",
        }


class ExperimentManager:
    """
    Manager for running and tracking experiments.

    Provides:
    - Experiment execution with automatic tracking
    - Hyperparameter sweep support
    - Result persistence and comparison
    - Checkpoint management

    Example:
        >>> manager = ExperimentManager("./experiments")
        >>> config = ExperimentConfig(name="test", epochs=5)
        >>> result = manager.run_experiment(model, train_loader, val_loader, config)
        >>> print(result.best_val_loss)
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = "./experiments",
        auto_save: bool = True,
    ):
        """
        Initialize experiment manager.

        Args:
            base_dir: Directory to store experiment data
            auto_save: Automatically save results after each experiment
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # Create subdirectories
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"

        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Load existing results
        self.results: Dict[str, ExperimentResult] = {}
        self._load_results()

    def _load_results(self) -> None:
        """Load existing experiment results."""
        results_file = self.results_dir / "all_results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                for exp_id, result_data in data.items():
                    self.results[exp_id] = ExperimentResult.from_dict(result_data)
                logger.info(f"Loaded {len(self.results)} existing experiments")
            except Exception as e:
                logger.warning(f"Failed to load results: {e}")

    def _save_results(self) -> None:
        """Save all experiment results."""
        results_file = self.results_dir / "all_results.json"
        data = {exp_id: result.to_dict() for exp_id, result in self.results.items()}
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.get_hash()
        return f"{config.name}_{timestamp}_{config_hash}"

    def run_experiment(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[ExperimentConfig] = None,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            config: Experiment configuration
            progress_callback: Optional callback(epoch, metrics) for progress

        Returns:
            ExperimentResult with training results
        """
        if config is None:
            config = ExperimentConfig()

        # Import trainer here to avoid circular imports
        from ..training.unified_trainer import UnifiedTrainer, TrainingConfig

        # Generate experiment ID
        experiment_id = self._generate_experiment_id(config)
        logger.info(f"Starting experiment: {experiment_id}")

        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
        )

        # Store in results
        self.results[experiment_id] = result

        try:
            # Set seed
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

            # Create training config
            training_config = TrainingConfig(
                model_name=config.model_name,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                optimizer=config.optimizer,
                scheduler=config.scheduler,
                weight_decay=config.weight_decay,
                grad_clip=config.grad_clip,
                loss_type=config.loss_type,
                use_physics_loss=config.use_physics_loss,
                physics_weight=config.physics_weight,
                device=config.device,
                use_amp=config.use_amp,
                mini_mode=config.mini_mode,
                mini_samples=config.mini_train_samples,
                mini_val_samples=config.mini_val_samples,
                checkpoint_dir=str(self.checkpoints_dir / experiment_id),
                experiment_name=experiment_id,
                seed=config.seed,
            )

            # Create trainer
            trainer = UnifiedTrainer(model, training_config)

            # Track model info
            result.num_params = sum(p.numel() for p in model.parameters())

            # Run training
            history = trainer.train(train_loader, val_loader, progress_callback)

            # Extract results
            result.train_losses = [m.train_loss for m in history]
            result.val_losses = [m.val_loss for m in history]
            result.learning_rates = [m.learning_rate for m in history]

            if history:
                result.final_train_loss = history[-1].train_loss
                result.final_val_loss = history[-1].val_loss
                result.best_val_loss = min(m.val_loss for m in history) if any(m.val_loss > 0 for m in history) else 0
                result.best_epoch = min(range(len(history)), key=lambda i: history[i].val_loss if history[i].val_loss > 0 else float("inf"))
                result.gpu_memory_mb = max(m.gpu_memory_mb for m in history)

            # Additional metrics
            result.metrics = trainer.get_training_summary()
            result.checkpoint_path = str(self.checkpoints_dir / experiment_id / "best_model.pt")

            # Mark completed
            result.status = ExperimentStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            logger.info(f"Experiment completed: {experiment_id}")
            logger.info(f"Best val loss: {result.best_val_loss:.4f} at epoch {result.best_epoch}")

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        # Save results
        if self.auto_save:
            self._save_results()

        return result

    def run_sweep(
        self,
        sweep,  # HyperparameterSweep
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        base_config: Optional[ExperimentConfig] = None,
        max_experiments: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, ExperimentResult], None]] = None,
    ) -> List[ExperimentResult]:
        """
        Run a hyperparameter sweep.

        Args:
            sweep: HyperparameterSweep defining parameters to search
            model_factory: Function that creates model from config dict
            train_loader: Training data loader
            val_loader: Optional validation data loader
            base_config: Base experiment configuration
            max_experiments: Maximum number of experiments to run
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of ExperimentResult for each configuration
        """
        if base_config is None:
            base_config = ExperimentConfig()

        # Generate configurations
        configs = sweep.generate_configs()
        if max_experiments is not None:
            configs = configs[:max_experiments]

        logger.info(f"Running sweep with {len(configs)} configurations")

        results = []
        for i, param_config in enumerate(configs):
            # Create experiment config
            exp_config = ExperimentConfig(
                name=f"{base_config.name}_sweep_{i}",
                description=f"Sweep config {i}: {param_config}",
                tags=base_config.tags + ["sweep"],
                epochs=param_config.get("epochs", base_config.epochs),
                batch_size=param_config.get("batch_size", base_config.batch_size),
                learning_rate=param_config.get("learning_rate", base_config.learning_rate),
                optimizer=param_config.get("optimizer", base_config.optimizer),
                model_config=param_config,
                mini_mode=base_config.mini_mode,
                mini_train_samples=base_config.mini_train_samples,
                mini_val_samples=base_config.mini_val_samples,
                seed=base_config.seed + i,  # Different seed per experiment
            )

            # Create model
            model = model_factory(param_config)

            # Run experiment
            result = self.run_experiment(model, train_loader, val_loader, exp_config)
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(configs), result)

        return results

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result by ID."""
        return self.results.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """List experiments with optional filtering."""
        results = list(self.results.values())

        if status:
            results = [r for r in results if r.status == status]

        if tags:
            results = [r for r in results if r.config and all(t in r.config.tags for t in tags)]

        return sorted(results, key=lambda r: r.start_time or datetime.min, reverse=True)

    def compare_experiments(
        self,
        experiment_ids: List[str],
    ) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = [self.results[eid] for eid in experiment_ids if eid in self.results]

        if not experiments:
            return {}

        comparison = {
            "experiments": [e.get_summary() for e in experiments],
            "best_experiment": min(experiments, key=lambda e: e.best_val_loss).experiment_id,
            "metrics_comparison": {
                "best_val_loss": {e.experiment_id: e.best_val_loss for e in experiments},
                "final_val_loss": {e.experiment_id: e.final_val_loss for e in experiments},
                "duration": {e.experiment_id: e.duration_seconds for e in experiments},
                "params": {e.experiment_id: e.num_params for e in experiments},
            }
        }

        return comparison

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its artifacts."""
        if experiment_id not in self.results:
            return False

        result = self.results[experiment_id]

        # Delete checkpoint directory
        ckpt_dir = self.checkpoints_dir / experiment_id
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

        # Remove from results
        del self.results[experiment_id]
        self._save_results()

        return True

    def cleanup_failed(self) -> int:
        """Clean up failed experiments."""
        failed = [eid for eid, r in self.results.items() if r.status == ExperimentStatus.FAILED]
        for eid in failed:
            self.delete_experiment(eid)
        return len(failed)

    def get_best_experiment(
        self,
        metric: str = "best_val_loss",
        tags: Optional[List[str]] = None,
    ) -> Optional[ExperimentResult]:
        """Get the best experiment by a metric."""
        experiments = self.list_experiments(status=ExperimentStatus.COMPLETED, tags=tags)

        if not experiments:
            return None

        if metric == "best_val_loss":
            return min(experiments, key=lambda e: e.best_val_loss)
        elif metric == "final_val_loss":
            return min(experiments, key=lambda e: e.final_val_loss)
        elif metric == "duration":
            return min(experiments, key=lambda e: e.duration_seconds)
        else:
            return experiments[0]

    def export_results(
        self,
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Export all results to a file."""
        output_path = Path(output_path)

        if format == "json":
            data = {eid: r.to_dict() for eid, r in self.results.items()}
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "experiment_id", "status", "duration_seconds",
                    "best_val_loss", "best_epoch", "num_params"
                ])
                writer.writeheader()
                for eid, result in self.results.items():
                    writer.writerow({
                        "experiment_id": eid,
                        "status": result.status.value,
                        "duration_seconds": result.duration_seconds,
                        "best_val_loss": result.best_val_loss,
                        "best_epoch": result.best_epoch,
                        "num_params": result.num_params,
                    })

        logger.info(f"Results exported to {output_path}")

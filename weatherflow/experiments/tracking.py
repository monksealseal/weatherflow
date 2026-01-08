"""
Experiment Tracking Utilities

Provides logging and metric tracking for experiments.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Single metric measurement."""
    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp,
        }


class MetricsLogger:
    """
    Simple metrics logger for experiment tracking.

    Logs metrics to files and provides retrieval methods.

    Example:
        >>> logger = MetricsLogger("./logs/exp1")
        >>> logger.log("train_loss", 0.5, step=0)
        >>> logger.log("val_loss", 0.6, step=0)
        >>> losses = logger.get_metric("train_loss")
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        flush_every: int = 10,
    ):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to store logs
            flush_every: Flush to disk every N entries
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every

        self.metrics: Dict[str, List[MetricEntry]] = {}
        self.entry_count = 0

    def log(
        self,
        name: str,
        value: float,
        step: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step/epoch
            timestamp: Optional timestamp (defaults to current time)
        """
        if name not in self.metrics:
            self.metrics[name] = []

        entry = MetricEntry(
            name=name,
            value=value,
            step=step,
            timestamp=timestamp or time.time(),
        )
        self.metrics[name].append(entry)
        self.entry_count += 1

        # Flush periodically
        if self.entry_count % self.flush_every == 0:
            self.flush()

    def log_dict(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple metrics at once."""
        timestamp = time.time()
        for name, value in metrics.items():
            self.log(name, value, step, timestamp)

    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric."""
        if name not in self.metrics:
            return []
        return [e.value for e in self.metrics[name]]

    def get_metric_with_steps(self, name: str) -> List[tuple]:
        """Get (step, value) pairs for a metric."""
        if name not in self.metrics:
            return []
        return [(e.step, e.value) for e in self.metrics[name]]

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1].value

    def get_best(self, name: str, minimize: bool = True) -> Optional[float]:
        """Get the best value for a metric."""
        values = self.get_metric(name)
        if not values:
            return None
        return min(values) if minimize else max(values)

    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics."""
        return {name: self.get_metric(name) for name in self.metrics}

    def flush(self) -> None:
        """Flush metrics to disk."""
        metrics_file = self.log_dir / "metrics.json"

        data = {
            name: [e.to_dict() for e in entries]
            for name, entries in self.metrics.items()
        }

        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load metrics from disk."""
        metrics_file = self.log_dir / "metrics.json"

        if not metrics_file.exists():
            return

        with open(metrics_file, "r") as f:
            data = json.load(f)

        self.metrics = {}
        for name, entries in data.items():
            self.metrics[name] = [
                MetricEntry(**entry) for entry in entries
            ]


class ExperimentTracker:
    """
    High-level experiment tracker.

    Combines metrics logging with experiment metadata tracking.

    Example:
        >>> tracker = ExperimentTracker("my_experiment")
        >>> tracker.set_config({"lr": 1e-4, "batch_size": 8})
        >>> tracker.log_metrics({"loss": 0.5}, step=0)
        >>> tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: Union[str, Path] = "./experiments",
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
            tags: Optional tags for the experiment
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.tags = tags or []

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics_logger = MetricsLogger(self.experiment_dir / "metrics")
        self.config: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.status = "running"

        # Save initial metadata
        self._save_metadata()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set experiment configuration."""
        self.config = config
        self._save_metadata()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._save_metadata()

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
    ) -> None:
        """Log a single metric."""
        self.metrics_logger.log(name, value, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple metrics."""
        self.metrics_logger.log_dict(metrics, step)

    def log_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ) -> Path:
        """
        Log an artifact (model checkpoint, data, etc.).

        Args:
            name: Artifact name
            data: Data to save
            artifact_type: Type of artifact (json, torch, numpy)

        Returns:
            Path to saved artifact
        """
        artifacts_dir = self.experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        if artifact_type == "json":
            path = artifacts_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        elif artifact_type == "torch":
            import torch
            path = artifacts_dir / f"{name}.pt"
            torch.save(data, path)

        elif artifact_type == "numpy":
            path = artifacts_dir / f"{name}.npy"
            np.save(path, data)

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

        return path

    def get_metric(self, name: str) -> List[float]:
        """Get metric values."""
        return self.metrics_logger.get_metric(name)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        metrics = self.metrics_logger.get_all_metrics()

        summary = {
            "experiment_name": self.experiment_name,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "tags": self.tags,
            "config": self.config,
            "metrics_summary": {},
        }

        for name, values in metrics.items():
            if values:
                summary["metrics_summary"][name] = {
                    "final": values[-1],
                    "best": min(values) if "loss" in name else max(values),
                    "mean": np.mean(values),
                }

        return summary

    def finish(self, status: str = "completed") -> None:
        """Mark experiment as finished."""
        self.status = status
        self.end_time = datetime.now()
        self.metrics_logger.flush()
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": getattr(self, "end_time", None),
            "status": self.status,
            "tags": self.tags,
            "config": self.config,
        }

        if hasattr(self, "end_time") and self.end_time:
            metadata["end_time"] = self.end_time.isoformat()

        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def create_progress_callback(
    tracker: Optional[ExperimentTracker] = None,
    print_every: int = 1,
) -> Callable[[int, Dict], None]:
    """
    Create a progress callback for training.

    Args:
        tracker: Optional ExperimentTracker
        print_every: Print progress every N epochs

    Returns:
        Callback function
    """
    def callback(epoch: int, metrics: Dict[str, float]) -> None:
        # Log to tracker
        if tracker is not None:
            tracker.log_metrics(metrics, epoch)

        # Print progress
        if epoch % print_every == 0:
            loss_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
                if "loss" in k.lower()
            )
            print(f"Epoch {epoch}: {loss_str}")

    return callback

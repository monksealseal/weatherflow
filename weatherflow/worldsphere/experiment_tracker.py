"""
Worldsphere Experiment Tracker

Comprehensive experiment tracking for understanding what changes
lead to better RMSE and other performance metrics.

Features:
- Hyperparameter tracking
- RMSE trend analysis
- A/B experiment comparison
- Automatic correlation analysis
- Visualization support
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSet:
    """Set of hyperparameters for an experiment."""

    # Model architecture
    model_type: str = "pix2pix"
    generator_features: int = 64
    discriminator_features: int = 64
    num_residual_blocks: int = 9

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 100
    optimizer: str = "adam"

    # Loss weights
    lambda_l1: float = 100.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5

    # Data
    image_size: int = 256
    augmentation: bool = True
    normalize_wind: bool = True
    max_wind_speed: float = 80.0

    # For diffusion models
    num_timesteps: int = 1000
    num_inference_steps: int = 50
    cfg_scale: float = 7.5
    num_frames: int = 25

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterSet":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def diff(self, other: "HyperparameterSet") -> Dict[str, Tuple[Any, Any]]:
        """Get differences between two hyperparameter sets."""
        diffs = {}
        for key in self.__dataclass_fields__:
            val1 = getattr(self, key)
            val2 = getattr(other, key)
            if val1 != val2:
                diffs[key] = (val1, val2)
        return diffs


@dataclass
class ExperimentRun:
    """A single experiment run."""

    # Identity
    run_id: str = ""
    experiment_name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Configuration
    hyperparameters: Optional[HyperparameterSet] = None
    model_id: Optional[str] = None

    # Status
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    # Metrics - Training
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    # Metrics - Evaluation
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # RMSE tracking (the key metric)
    rmse_history: List[float] = field(default_factory=list)
    best_rmse: float = float("inf")
    final_rmse: float = 0.0

    # Additional metrics
    mae: float = 0.0
    wind_speed_rmse: float = 0.0
    wind_direction_rmse: float = 0.0

    # For sequence models
    frame_rmses: List[float] = field(default_factory=list)
    temporal_consistency: float = 0.0

    # Resources
    gpu_memory_peak_mb: float = 0.0
    training_samples_per_second: float = 0.0

    # Paths
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None

    # Error info
    error_message: Optional[str] = None

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.hyperparameters:
            result["hyperparameters"] = self.hyperparameters.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        data = data.copy()
        if "hyperparameters" in data and data["hyperparameters"]:
            data["hyperparameters"] = HyperparameterSet.from_dict(data["hyperparameters"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WorldsphereExperimentTracker:
    """
    Experiment tracker for Worldsphere AI models.

    Tracks experiments and provides analysis tools to understand
    what hyperparameter changes lead to better RMSE.

    Example:
        >>> tracker = WorldsphereExperimentTracker("./experiments")
        >>> run = tracker.start_run("cyclegan_v1", HyperparameterSet(learning_rate=1e-4))
        >>> tracker.log_metrics(run.run_id, epoch=10, rmse=0.15)
        >>> tracker.end_run(run.run_id)
        >>> analysis = tracker.analyze_rmse_correlations()
    """

    def __init__(self, base_dir: Union[str, Path] = "./worldsphere_experiments"):
        """
        Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiment data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.runs_dir = self.base_dir / "runs"
        self.analysis_dir = self.base_dir / "analysis"

        self.runs_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)

        # Load existing runs
        self.runs: Dict[str, ExperimentRun] = {}
        self._load_runs()

    def _load_runs(self) -> None:
        """Load existing experiment runs."""
        registry_file = self.base_dir / "runs_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                for run_id, run_data in data.items():
                    self.runs[run_id] = ExperimentRun.from_dict(run_data)
                logger.info(f"Loaded {len(self.runs)} experiment runs")
            except Exception as e:
                logger.warning(f"Failed to load runs: {e}")

    def _save_runs(self) -> None:
        """Save runs registry."""
        registry_file = self.base_dir / "runs_registry.json"
        data = {run_id: run.to_dict() for run_id, run in self.runs.items()}
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_run_id(self, experiment_name: str) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = len([r for r in self.runs.values() if r.experiment_name == experiment_name])
        return f"{experiment_name}_run{count + 1}_{timestamp}"

    def start_run(
        self,
        experiment_name: str,
        hyperparameters: Optional[HyperparameterSet] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            experiment_name: Name of the experiment
            hyperparameters: Hyperparameter configuration
            description: Run description
            tags: Tags for filtering

        Returns:
            ExperimentRun object
        """
        run = ExperimentRun(
            run_id=self._generate_run_id(experiment_name),
            experiment_name=experiment_name,
            description=description,
            tags=tags or [],
            hyperparameters=hyperparameters or HyperparameterSet(),
            status="running",
            start_time=datetime.now().isoformat(),
        )

        self.runs[run.run_id] = run
        self._save_runs()

        logger.info(f"Started experiment run: {run.run_id}")
        return run

    def log_metrics(
        self,
        run_id: str,
        epoch: Optional[int] = None,
        **metrics: float,
    ) -> None:
        """
        Log metrics for an experiment run.

        Args:
            run_id: Run ID
            epoch: Current epoch (optional)
            **metrics: Metrics to log (e.g., rmse=0.15, loss=0.05)
        """
        if run_id not in self.runs:
            raise ValueError(f"Run not found: {run_id}")

        run = self.runs[run_id]

        # Update history metrics
        if "train_loss" in metrics:
            run.train_losses.append(metrics["train_loss"])
        if "val_loss" in metrics:
            run.val_losses.append(metrics["val_loss"])
        if "learning_rate" in metrics:
            run.learning_rates.append(metrics["learning_rate"])
        if "rmse" in metrics:
            run.rmse_history.append(metrics["rmse"])
            if metrics["rmse"] < run.best_rmse:
                run.best_rmse = metrics["rmse"]
                run.best_epoch = epoch or len(run.rmse_history)

        # Update scalar metrics
        for key, value in metrics.items():
            if hasattr(run, key):
                setattr(run, key, value)

        self._save_runs()

    def end_run(
        self,
        run_id: str,
        status: str = "completed",
        final_metrics: Optional[Dict[str, float]] = None,
        notes: str = "",
    ) -> ExperimentRun:
        """
        End an experiment run.

        Args:
            run_id: Run ID
            status: Final status (completed, failed)
            final_metrics: Final metrics to record
            notes: Notes about the run

        Returns:
            Updated ExperimentRun
        """
        if run_id not in self.runs:
            raise ValueError(f"Run not found: {run_id}")

        run = self.runs[run_id]
        run.status = status
        run.end_time = datetime.now().isoformat()
        run.notes = notes

        if run.start_time:
            start = datetime.fromisoformat(run.start_time)
            end = datetime.fromisoformat(run.end_time)
            run.duration_seconds = (end - start).total_seconds()

        if final_metrics:
            for key, value in final_metrics.items():
                if hasattr(run, key):
                    setattr(run, key, value)

        # Set final RMSE
        if run.rmse_history:
            run.final_rmse = run.rmse_history[-1]

        self._save_runs()
        logger.info(f"Ended experiment run: {run_id} with status: {status}")

        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by ID."""
        return self.runs.get(run_id)

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "start_time",
        ascending: bool = False,
    ) -> List[ExperimentRun]:
        """
        List experiment runs with optional filtering.

        Args:
            experiment_name: Filter by experiment name
            status: Filter by status
            tags: Filter by tags
            sort_by: Field to sort by
            ascending: Sort order

        Returns:
            List of matching runs
        """
        runs = list(self.runs.values())

        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]
        if status:
            runs = [r for r in runs if r.status == status]
        if tags:
            runs = [r for r in runs if any(t in r.tags for t in tags)]

        runs.sort(
            key=lambda r: getattr(r, sort_by, "") or "",
            reverse=not ascending,
        )

        return runs

    def compare_runs(
        self,
        run_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple experiment runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison dictionary
        """
        runs = [self.runs.get(rid) for rid in run_ids if rid in self.runs]

        if not runs:
            return {}

        comparison = {
            "runs": [],
            "hyperparameter_diffs": {},
            "metric_comparison": {},
            "best_run": None,
        }

        # Collect run summaries
        for run in runs:
            comparison["runs"].append({
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "status": run.status,
                "best_rmse": run.best_rmse,
                "final_rmse": run.final_rmse,
                "duration_hours": run.duration_seconds / 3600,
            })

        # Find hyperparameter differences
        if len(runs) >= 2 and runs[0].hyperparameters and runs[1].hyperparameters:
            base_hp = runs[0].hyperparameters
            for run in runs[1:]:
                if run.hyperparameters:
                    diff = base_hp.diff(run.hyperparameters)
                    comparison["hyperparameter_diffs"][run.run_id] = diff

        # Metric comparison
        comparison["metric_comparison"] = {
            "best_rmse": {r.run_id: r.best_rmse for r in runs},
            "final_rmse": {r.run_id: r.final_rmse for r in runs},
            "wind_speed_rmse": {r.run_id: r.wind_speed_rmse for r in runs},
            "duration_seconds": {r.run_id: r.duration_seconds for r in runs},
        }

        # Find best run
        completed_runs = [r for r in runs if r.status == "completed"]
        if completed_runs:
            best = min(completed_runs, key=lambda r: r.best_rmse)
            comparison["best_run"] = best.run_id

        return comparison

    def analyze_rmse_correlations(
        self,
        experiment_name: Optional[str] = None,
        min_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between hyperparameters and RMSE.

        This helps understand what changes lead to better RMSE.

        Args:
            experiment_name: Filter to specific experiment
            min_runs: Minimum runs needed for analysis

        Returns:
            Analysis results including correlations
        """
        runs = self.list_runs(experiment_name=experiment_name, status="completed")

        if len(runs) < min_runs:
            return {"error": f"Need at least {min_runs} completed runs for analysis"}

        # Collect data
        hp_values: Dict[str, List[float]] = {}
        rmse_values: List[float] = []

        for run in runs:
            if run.hyperparameters and run.best_rmse < float("inf"):
                rmse_values.append(run.best_rmse)
                hp_dict = run.hyperparameters.to_dict()
                for key, value in hp_dict.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key not in hp_values:
                            hp_values[key] = []
                        hp_values[key].append(float(value))

        if not rmse_values:
            return {"error": "No valid RMSE data found"}

        # Compute correlations
        correlations = {}
        for hp_name, values in hp_values.items():
            if len(values) == len(rmse_values):
                corr = np.corrcoef(values, rmse_values)[0, 1]
                if not np.isnan(corr):
                    correlations[hp_name] = corr

        # Sort by absolute correlation
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Find optimal values
        optimal_values = {}
        best_run = min(runs, key=lambda r: r.best_rmse)
        if best_run.hyperparameters:
            optimal_values = best_run.hyperparameters.to_dict()

        return {
            "num_runs_analyzed": len(runs),
            "rmse_stats": {
                "mean": np.mean(rmse_values),
                "std": np.std(rmse_values),
                "min": np.min(rmse_values),
                "max": np.max(rmse_values),
            },
            "correlations": sorted_correlations,
            "most_impactful": list(sorted_correlations.keys())[:5],
            "optimal_hyperparameters": optimal_values,
            "recommendations": self._generate_recommendations(sorted_correlations, optimal_values),
        }

    def _generate_recommendations(
        self,
        correlations: Dict[str, float],
        optimal_values: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []

        for hp_name, corr in list(correlations.items())[:5]:
            if abs(corr) > 0.3:  # Meaningful correlation
                direction = "decrease" if corr > 0 else "increase"
                optimal = optimal_values.get(hp_name, "N/A")
                recommendations.append(
                    f"{hp_name}: {direction} may improve RMSE "
                    f"(correlation: {corr:.3f}, best value: {optimal})"
                )

        return recommendations

    def get_rmse_trend(
        self,
        experiment_name: Optional[str] = None,
        window: int = 5,
    ) -> Dict[str, Any]:
        """
        Get RMSE trend over time for an experiment.

        Args:
            experiment_name: Filter to specific experiment
            window: Moving average window

        Returns:
            Trend analysis
        """
        runs = self.list_runs(
            experiment_name=experiment_name,
            status="completed",
            sort_by="start_time",
            ascending=True,
        )

        if not runs:
            return {"error": "No completed runs found"}

        dates = []
        rmses = []

        for run in runs:
            if run.start_time and run.best_rmse < float("inf"):
                dates.append(run.start_time)
                rmses.append(run.best_rmse)

        if len(rmses) < 2:
            return {"error": "Need at least 2 runs for trend analysis"}

        # Compute moving average
        if len(rmses) >= window:
            moving_avg = [
                np.mean(rmses[i:i + window])
                for i in range(len(rmses) - window + 1)
            ]
        else:
            moving_avg = rmses

        # Trend direction
        if len(rmses) >= 2:
            trend = "improving" if rmses[-1] < rmses[0] else "degrading"
            improvement = (rmses[0] - rmses[-1]) / rmses[0] * 100
        else:
            trend = "stable"
            improvement = 0

        return {
            "num_runs": len(runs),
            "dates": dates,
            "rmses": rmses,
            "moving_average": moving_avg,
            "trend": trend,
            "improvement_percent": improvement,
            "best_rmse": min(rmses),
            "worst_rmse": max(rmses),
            "current_rmse": rmses[-1] if rmses else None,
        }

    def get_best_hyperparameters(
        self,
        experiment_name: Optional[str] = None,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get hyperparameters from top performing runs.

        Args:
            experiment_name: Filter to specific experiment
            top_n: Number of top runs to analyze

        Returns:
            List of hyperparameter configurations from top runs
        """
        runs = self.list_runs(experiment_name=experiment_name, status="completed")

        # Sort by best RMSE
        runs.sort(key=lambda r: r.best_rmse)

        results = []
        for run in runs[:top_n]:
            if run.hyperparameters:
                results.append({
                    "run_id": run.run_id,
                    "best_rmse": run.best_rmse,
                    "hyperparameters": run.hyperparameters.to_dict(),
                })

        return results

    def export_analysis(
        self,
        output_path: Union[str, Path],
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Export comprehensive analysis to file.

        Args:
            output_path: Path to save analysis
            experiment_name: Filter to specific experiment
        """
        output_path = Path(output_path)

        analysis = {
            "generated_at": datetime.now().isoformat(),
            "experiment_name": experiment_name or "all",
            "rmse_correlations": self.analyze_rmse_correlations(experiment_name),
            "rmse_trend": self.get_rmse_trend(experiment_name),
            "best_hyperparameters": self.get_best_hyperparameters(experiment_name),
            "all_runs": [r.to_dict() for r in self.list_runs(experiment_name)],
        }

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Exported analysis to: {output_path}")

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment run."""
        if run_id not in self.runs:
            return False

        del self.runs[run_id]
        self._save_runs()

        logger.info(f"Deleted run: {run_id}")
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary."""
        runs = list(self.runs.values())
        completed = [r for r in runs if r.status == "completed"]

        return {
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "running_runs": len([r for r in runs if r.status == "running"]),
            "failed_runs": len([r for r in runs if r.status == "failed"]),
            "best_rmse": min((r.best_rmse for r in completed), default=None),
            "avg_rmse": np.mean([r.best_rmse for r in completed if r.best_rmse < float("inf")]) if completed else None,
            "total_training_hours": sum(r.duration_seconds for r in runs) / 3600,
            "experiments": list(set(r.experiment_name for r in runs)),
        }

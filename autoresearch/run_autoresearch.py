#!/usr/bin/env python3
"""WeatherFlow AutoResearch: Autonomous ML experiment runner.

Runs ~100 ML experiments overnight on a single GPU. The human writes
program.md, the agent runs experiments until morning.

Usage:
    # Run overnight (default 100 experiments, 5 min each)
    python -m autoresearch.run_autoresearch

    # Quick test (10 experiments, 1 min each)
    python -m autoresearch.run_autoresearch --max-experiments 10 --budget-minutes 1

    # Generate report from previous run
    python -m autoresearch.run_autoresearch --report

    # Resume from a previous session
    python -m autoresearch.run_autoresearch --resume
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_config import (
    ExperimentConfig,
    generate_phase_experiment,
    generate_random_mutation,
)
from .train_experiment import prepare_synthetic_data, run_experiment

# Optional: Worldsphere experiment tracker integration
try:
    from weatherflow.worldsphere.experiment_tracker import (
        HyperparameterSet,
        WorldsphereExperimentTracker,
    )

    HAS_TRACKER = True
except ImportError:
    HAS_TRACKER = False

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


class ExperimentLog:
    """Persistent log of all experiments run in this autoresearch session."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "experiment_log.json"
        self.best_config_path = self.results_dir / "best_config.json"
        self.experiments: List[Dict[str, Any]] = []
        self.best_val_rmse: float = float("inf")
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_experiment_number: int = 0
        self._load()

    def _load(self) -> None:
        """Load existing log if resuming."""
        if self.log_path.exists():
            with open(self.log_path) as f:
                data = json.load(f)
            self.experiments = data.get("experiments", [])
            self.best_val_rmse = data.get("best_val_rmse", float("inf"))
            self.best_config = data.get("best_config")
            self.best_experiment_number = data.get("best_experiment_number", 0)
            logger.info(
                f"Resumed from {len(self.experiments)} previous experiments. "
                f"Best val_rmse: {self.best_val_rmse:.6f}"
            )

    def _save(self) -> None:
        """Persist the log to disk."""
        data = {
            "session_start": self.experiments[0]["timestamp"] if self.experiments else None,
            "session_last_update": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "kept_improvements": sum(1 for e in self.experiments if e.get("kept")),
            "discarded": sum(1 for e in self.experiments if not e.get("kept")),
            "failed": sum(1 for e in self.experiments if e.get("error")),
            "best_val_rmse": self.best_val_rmse,
            "best_experiment_number": self.best_experiment_number,
            "best_config": self.best_config,
            "experiments": self.experiments,
        }
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

        if self.best_config:
            with open(self.best_config_path, "w") as f:
                json.dump(self.best_config, f, indent=2)

    def add(
        self,
        experiment_number: int,
        description: str,
        config: ExperimentConfig,
        results: Dict[str, Any],
    ) -> bool:
        """Add an experiment result. Returns True if it's a new best."""
        val_rmse = results.get("best_val_rmse", float("inf"))
        kept = val_rmse < self.best_val_rmse

        entry = {
            "experiment_number": experiment_number,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "config": config.to_dict(),
            "val_rmse": val_rmse,
            "val_loss": results.get("best_val_loss", float("inf")),
            "epochs_completed": results.get("epochs_completed", 0),
            "training_time_seconds": results.get("training_time_seconds", 0),
            "num_parameters": results.get("num_parameters", 0),
            "kept": kept,
            "error": results.get("error"),
        }

        self.experiments.append(entry)

        if kept:
            self.best_val_rmse = val_rmse
            self.best_config = config.to_dict()
            self.best_experiment_number = experiment_number
            logger.info(
                f"NEW BEST! Experiment #{experiment_number}: "
                f"val_rmse={val_rmse:.6f} ({description})"
            )
        else:
            logger.info(
                f"Discarded experiment #{experiment_number}: "
                f"val_rmse={val_rmse:.6f} (best={self.best_val_rmse:.6f})"
            )

        self._save()
        return kept


# ---------------------------------------------------------------------------
# Git integration (optional)
# ---------------------------------------------------------------------------


def git_commit_improvement(experiment_number: int, description: str, val_rmse: float) -> None:
    """Commit the current best config to git, accumulating improvements."""
    try:
        subprocess.run(
            ["git", "add", "autoresearch/results/"],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent),
        )
        msg = (
            f"autoresearch #{experiment_number}: {description}\n\n"
            f"val_rmse: {val_rmse:.6f}"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent),
        )
        logger.info(f"Git commit for experiment #{experiment_number}")
    except Exception as e:
        logger.warning(f"Git commit failed: {e}")


# ---------------------------------------------------------------------------
# Worldsphere tracker integration
# ---------------------------------------------------------------------------


def log_to_worldsphere(
    tracker: "WorldsphereExperimentTracker",
    experiment_number: int,
    description: str,
    config: ExperimentConfig,
    results: Dict[str, Any],
) -> None:
    """Log an experiment to the Worldsphere tracker for correlation analysis."""
    hp = HyperparameterSet(
        model_type="weatherflow_autoresearch",
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        epochs=results.get("epochs_completed", 0),
        optimizer=config.training.optimizer,
        image_size=config.model.grid_size[0],
    )

    run = tracker.start_run(
        experiment_name="autoresearch",
        hyperparameters=hp,
        description=f"#{experiment_number}: {description}",
        tags=["autoresearch", f"phase_{_get_phase(experiment_number)}"],
    )

    # Log metrics
    for epoch, (rmse_val, loss_val) in enumerate(
        zip(
            results.get("val_rmses", []),
            results.get("val_losses", []),
        )
    ):
        tracker.log_metrics(
            run.run_id,
            epoch=epoch,
            rmse=rmse_val,
            val_loss=loss_val,
        )

    status = "failed" if results.get("error") else "completed"
    tracker.end_run(
        run.run_id,
        status=status,
        final_metrics={
            "best_rmse": results.get("best_val_rmse", float("inf")),
            "final_rmse": results.get("final_val_rmse", float("inf")),
            "mae": results.get("val_maes", [0])[-1] if results.get("val_maes") else 0,
        },
        notes=description,
    )


def _get_phase(experiment_number: int) -> int:
    if experiment_number <= 30:
        return 1
    elif experiment_number <= 60:
        return 2
    elif experiment_number <= 80:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Progress visualization
# ---------------------------------------------------------------------------


def generate_progress_plot(log: ExperimentLog) -> None:
    """Generate a progress visualization similar to Karpathy's autoresearch."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    experiments = log.experiments
    if not experiments:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. val_rmse over time (top left)
    ax = axes[0, 0]
    numbers = [e["experiment_number"] for e in experiments if not e.get("error")]
    rmses = [e["val_rmse"] for e in experiments if not e.get("error")]
    kept = [e["kept"] for e in experiments if not e.get("error")]

    colors = ["#2ecc71" if k else "#95a5a6" for k in kept]
    ax.scatter(numbers, rmses, c=colors, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Best trajectory
    best_so_far = []
    current_best = float("inf")
    for r in rmses:
        current_best = min(current_best, r)
        best_so_far.append(current_best)
    ax.plot(numbers, best_so_far, color="#e74c3c", linewidth=2, label="Best so far")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_rmse")
    ax.set_title("AutoResearch Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training time per experiment (top right)
    ax = axes[0, 1]
    times = [e["training_time_seconds"] / 60 for e in experiments]
    ax.bar(range(len(times)), times, color="#3498db", alpha=0.7)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Training time (min)")
    ax.set_title("Time Budget Usage")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Kept vs discarded pie (bottom left)
    ax = axes[1, 0]
    n_kept = sum(1 for e in experiments if e.get("kept"))
    n_discarded = sum(1 for e in experiments if not e.get("kept") and not e.get("error"))
    n_failed = sum(1 for e in experiments if e.get("error"))
    labels = ["Kept", "Discarded", "Failed"]
    sizes = [n_kept, n_discarded, n_failed]
    colors_pie = ["#2ecc71", "#95a5a6", "#e74c3c"]
    # Only plot non-zero slices
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if nonzero:
        ax.pie(
            [s for _, s, _ in nonzero],
            labels=[f"{l} ({s})" for l, s, _ in nonzero],
            colors=[c for _, _, c in nonzero],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax.set_title(
        f"Experiment Outcomes ({len(experiments)} total)"
    )

    # 4. Phase breakdown (bottom right)
    ax = axes[1, 1]
    phase_data = {1: [], 2: [], 3: [], 4: []}
    for e in experiments:
        if not e.get("error"):
            phase = _get_phase(e["experiment_number"])
            phase_data[phase].append(e["val_rmse"])

    phase_labels = []
    phase_means = []
    phase_bests = []
    for phase, values in phase_data.items():
        if values:
            phase_labels.append(f"Phase {phase}")
            phase_means.append(sum(values) / len(values))
            phase_bests.append(min(values))

    if phase_labels:
        x = range(len(phase_labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], phase_means, width, label="Mean", color="#3498db", alpha=0.7)
        ax.bar([i + width / 2 for i in x], phase_bests, width, label="Best", color="#2ecc71", alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(phase_labels)
        ax.set_ylabel("val_rmse")
        ax.set_title("Performance by Phase")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"WeatherFlow AutoResearch: {len(experiments)} experiments, "
        f"{n_kept} kept improvements\n"
        f"Best val_rmse: {log.best_val_rmse:.6f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_path = log.results_dir / "progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Progress plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_report(log: ExperimentLog) -> None:
    """Print a human-readable summary of the autoresearch session."""
    experiments = log.experiments

    if not experiments:
        print("No experiments found. Run autoresearch first.")
        return

    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("kept"))
    failed = sum(1 for e in experiments if e.get("error"))
    total_time_hours = sum(e.get("training_time_seconds", 0) for e in experiments) / 3600

    print("\n" + "=" * 72)
    print("  WEATHERFLOW AUTORESEARCH REPORT")
    print("=" * 72)
    print(f"\n  Total experiments:     {total}")
    print(f"  Kept improvements:     {kept} ({kept/max(total,1)*100:.0f}%)")
    print(f"  Discarded:             {total - kept - failed}")
    print(f"  Failed:                {failed}")
    print(f"  Total training time:   {total_time_hours:.1f} hours")
    print(f"  Best val_rmse:         {log.best_val_rmse:.6f}")
    print(f"  Best experiment:       #{log.best_experiment_number}")
    print()

    # Timeline of improvements
    print("  IMPROVEMENT TIMELINE:")
    print("  " + "-" * 68)
    for e in experiments:
        if e.get("kept"):
            marker = ">>>"
            print(
                f"  {marker} #{e['experiment_number']:3d}  "
                f"val_rmse={e['val_rmse']:.6f}  "
                f"{e['description']}"
            )

    print()

    # Best configuration
    if log.best_config:
        print("  BEST CONFIGURATION:")
        print("  " + "-" * 68)
        print(f"  {json.dumps(log.best_config, indent=4)}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_autoresearch(
    max_experiments: int = 100,
    budget_minutes: float = 5.0,
    device: str = "auto",
    resume: bool = False,
    use_git: bool = True,
    results_dir: Optional[Path] = None,
    seed: int = 42,
) -> ExperimentLog:
    """Run the autonomous experiment loop.

    Args:
        max_experiments: Maximum number of experiments to run.
        budget_minutes: Wall-clock training budget per experiment (minutes).
        device: Device to train on.
        resume: If True, resume from previous session's best config.
        use_git: If True, commit improvements to git.
        results_dir: Directory for experiment logs and artifacts.
        seed: Random seed for experiment generation.

    Returns:
        ExperimentLog with all results.
    """
    results_dir = results_dir or RESULTS_DIR
    log = ExperimentLog(results_dir)
    rng = random.Random(seed)

    # Prepare data once (shared across all experiments)
    print("\nPreparing training data...")
    train_dataset, val_dataset = prepare_synthetic_data(seed=seed)
    print(f"Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

    # Initialize Worldsphere tracker if available
    tracker = None
    if HAS_TRACKER:
        tracker_dir = results_dir / "worldsphere_tracking"
        tracker = WorldsphereExperimentTracker(str(tracker_dir))
        print("Worldsphere ExperimentTracker initialized")

    # Determine starting config
    if resume and log.best_config:
        base_config = ExperimentConfig.from_dict(log.best_config)
        start_number = len(log.experiments) + 1
        print(f"Resuming from experiment #{start_number}, best val_rmse={log.best_val_rmse:.6f}")
    else:
        base_config = ExperimentConfig()
        start_number = 1

    # Checkpoint directory
    checkpoint_dir = str(results_dir / "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    session_start = time.time()

    print(f"\n{'='*72}")
    print(f"  WEATHERFLOW AUTORESEARCH")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Budget per experiment: {budget_minutes} min")
    print(f"  Device: {device}")
    print(f"  Results: {results_dir}")
    print(f"{'='*72}\n")

    # --- Run experiment 1 as baseline if starting fresh ---
    if start_number == 1:
        print(f"--- Experiment #1: BASELINE ---")
        results = run_experiment(
            config=base_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            budget_minutes=budget_minutes,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        log.add(1, "baseline", base_config, results)

        if tracker:
            log_to_worldsphere(tracker, 1, "baseline", base_config, results)

        start_number = 2

    # --- Main experiment loop ---
    for exp_num in range(start_number, start_number + max_experiments - 1):
        elapsed_total = (time.time() - session_start) / 3600
        print(f"\n--- Experiment #{exp_num} (session: {elapsed_total:.1f}h) ---")

        # Generate experiment config from current best
        current_best_config = (
            ExperimentConfig.from_dict(log.best_config)
            if log.best_config
            else base_config
        )

        config, description = generate_phase_experiment(
            current_best_config, exp_num, rng=rng
        )
        print(f"  Change: {description}")

        # Run the experiment
        try:
            results = run_experiment(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                budget_minutes=budget_minutes,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )
        except torch.cuda.OutOfMemoryError:
            # OOM: retry with halved batch size
            print("  OOM! Retrying with smaller batch size...")
            config.training.batch_size = max(config.training.batch_size // 2, 1)
            description += f" (OOM retry, batch_size={config.training.batch_size})"
            try:
                results = run_experiment(
                    config=config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    budget_minutes=budget_minutes,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
            except Exception as e:
                results = {"error": str(e), "best_val_rmse": float("inf")}
        except Exception as e:
            results = {"error": str(e), "best_val_rmse": float("inf")}
            print(f"  FAILED: {e}")

        # NaN retry: reduce LR by 10x
        if results.get("nan_detected"):
            print("  NaN detected! Retrying with lower learning rate...")
            config.training.learning_rate *= 0.1
            description += f" (NaN retry, lr={config.training.learning_rate:.1e})"
            try:
                results = run_experiment(
                    config=config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    budget_minutes=budget_minutes,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
            except Exception as e:
                results = {"error": str(e), "best_val_rmse": float("inf")}

        # Log results
        kept = log.add(exp_num, description, config, results)

        val_rmse = results.get("best_val_rmse", float("inf"))
        epochs = results.get("epochs_completed", 0)
        train_time = results.get("training_time_seconds", 0)
        status = "KEPT" if kept else "discarded"
        print(
            f"  Result: val_rmse={val_rmse:.6f} | "
            f"epochs={epochs} | time={train_time:.0f}s | {status}"
        )

        # Worldsphere tracking
        if tracker:
            log_to_worldsphere(tracker, exp_num, description, config, results)

        # Git commit if improvement
        if kept and use_git:
            git_commit_improvement(exp_num, description, val_rmse)

        # Generate progress plot every 10 experiments
        if exp_num % 10 == 0:
            generate_progress_plot(log)

    # Final report
    generate_progress_plot(log)
    print_report(log)

    # Generate Worldsphere analysis if available
    if tracker:
        analysis = tracker.analyze_rmse_correlations("autoresearch", min_runs=3)
        if "error" not in analysis:
            analysis_path = results_dir / "worldsphere_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(
                    {k: v if not hasattr(v, "item") else float(v) for k, v in analysis.items()},
                    f,
                    indent=2,
                    default=str,
                )
            print(f"\nWorldsphere analysis saved to {analysis_path}")
            if analysis.get("recommendations"):
                print("\nWorldsphere Recommendations:")
                for rec in analysis["recommendations"]:
                    print(f"  - {rec}")

    return log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="WeatherFlow AutoResearch: Autonomous ML experiment runner"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=100,
        help="Maximum number of experiments to run (default: 100)",
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=5.0,
        help="Wall-clock training budget per experiment in minutes (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous session's best config",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Disable git commits for improvements",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print report from previous run and exit",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for results (default: autoresearch/results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / "autoresearch.log"),
        ],
    )

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    if args.report:
        log = ExperimentLog(results_dir)
        print_report(log)
        generate_progress_plot(log)
        return

    run_autoresearch(
        max_experiments=args.max_experiments,
        budget_minutes=args.budget_minutes,
        device=args.device,
        resume=args.resume,
        use_git=not args.no_git,
        results_dir=results_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

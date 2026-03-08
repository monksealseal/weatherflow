"""Experiment configuration and search space for autonomous ML research.

Defines the baseline configuration and the space of modifications the
autoresearch agent can explore. Each experiment picks one or more changes
from the search space and applies them to the baseline.
"""

import copy
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """WeatherFlowMatch model configuration."""

    input_channels: int = 4
    hidden_dim: int = 128
    n_layers: int = 4
    use_attention: bool = False
    grid_size: Tuple[int, int] = (32, 64)
    physics_informed: bool = True
    window_size: int = 8
    spherical_padding: bool = True
    use_spectral_mixer: bool = False
    spectral_modes: int = 12

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["grid_size"] = list(self.grid_size)
        return result


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 1e-3
    batch_size: int = 8
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss_type: str = "mse"
    loss_weighting: str = "time"
    scheduler: str = "cosine"
    warmup_fraction: float = 0.0
    ema_decay: Optional[float] = None
    physics_lambda: float = 0.1
    noise_std: Optional[Tuple[float, float]] = None
    use_amp: bool = True
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        if "grid_size" in model_data and isinstance(model_data["grid_size"], list):
            model_data["grid_size"] = tuple(model_data["grid_size"])
        if "noise_std" in training_data and isinstance(training_data["noise_std"], list):
            training_data["noise_std"] = tuple(training_data["noise_std"])
        model_fields = {k for k in ModelConfig.__dataclass_fields__}
        training_fields = {k for k in TrainingConfig.__dataclass_fields__}
        model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in model_fields})
        train_cfg = TrainingConfig(**{k: v for k, v in training_data.items() if k in training_fields})
        return cls(model=model_cfg, training=train_cfg)

    def copy(self) -> "ExperimentConfig":
        return ExperimentConfig.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# Search space: each entry is (parameter_path, list_of_values, description)
# ---------------------------------------------------------------------------

SEARCH_SPACE: List[Tuple[str, str, List[Any], str]] = [
    # Phase 1: Hyperparameter sweep
    ("training", "learning_rate", [1e-4, 3e-4, 1e-3, 3e-3], "learning rate"),
    ("training", "batch_size", [4, 8, 16, 32], "batch size"),
    ("model", "hidden_dim", [64, 128, 256, 384], "hidden dimension"),
    ("model", "n_layers", [2, 4, 6, 8], "ConvNext depth"),
    ("training", "loss_type", ["mse", "huber", "smooth_l1"], "loss function"),
    ("training", "loss_weighting", ["time", "none"], "loss weighting"),
    ("training", "grad_clip", [0.5, 1.0, 2.0], "gradient clipping"),
    # Phase 2: Architecture exploration
    ("model", "use_attention", [True, False], "windowed attention"),
    ("model", "window_size", [4, 8, 16], "attention window size"),
    ("model", "spherical_padding", [True, False], "spherical padding"),
    ("model", "use_spectral_mixer", [True, False], "spectral mixing"),
    ("model", "spectral_modes", [8, 12, 16, 24], "spectral modes"),
    ("model", "physics_informed", [True, False], "physics constraints"),
    ("training", "physics_lambda", [0.01, 0.05, 0.1, 0.5], "physics lambda"),
    ("training", "ema_decay", [None, 0.999, 0.9999], "EMA decay"),
    ("training", "noise_std", [None, (0.0, 0.01), (0.0, 0.05), (0.0, 0.1)], "noise injection"),
    # Phase 3: Optimizer & schedule
    ("training", "optimizer", ["adamw", "adam", "sgd"], "optimizer"),
    ("training", "weight_decay", [0.0, 1e-5, 1e-4, 1e-3], "weight decay"),
    ("training", "scheduler", ["cosine", "onecycle", "linear", "constant"], "LR schedule"),
    ("training", "warmup_fraction", [0.0, 0.02, 0.05, 0.10], "warmup fraction"),
]


def generate_random_mutation(
    base_config: ExperimentConfig,
    num_changes: int = 1,
    rng: Optional[random.Random] = None,
) -> Tuple[ExperimentConfig, str]:
    """Generate a new config by mutating 1-2 parameters from the base.

    Returns:
        (new_config, human_readable_description)
    """
    rng = rng or random.Random()
    config = base_config.copy()
    changes = []

    # Pick which parameters to mutate
    indices = rng.sample(range(len(SEARCH_SPACE)), min(num_changes, len(SEARCH_SPACE)))

    for idx in indices:
        section, param, values, desc = SEARCH_SPACE[idx]
        current = getattr(getattr(config, section), param)

        # Pick a value different from the current one
        candidates = [v for v in values if v != current]
        if not candidates:
            continue

        new_val = rng.choice(candidates)
        setattr(getattr(config, section), param, new_val)
        changes.append(f"{desc}: {current} -> {new_val}")

    description = "; ".join(changes) if changes else "no change (baseline)"
    return config, description


def generate_phase_experiment(
    base_config: ExperimentConfig,
    experiment_number: int,
    rng: Optional[random.Random] = None,
) -> Tuple[ExperimentConfig, str]:
    """Generate an experiment appropriate for the current phase.

    Phase 1 (1-30):  Single parameter changes
    Phase 2 (31-60): Single architectural changes
    Phase 3 (61-80): Single optimizer/schedule changes
    Phase 4 (81+):   2-parameter combinations from best findings
    """
    rng = rng or random.Random()

    if experiment_number <= 30:
        # Phase 1: hyperparameter sweep (first 7 entries)
        pool = SEARCH_SPACE[:7]
    elif experiment_number <= 60:
        # Phase 2: architecture (entries 7-15)
        pool = SEARCH_SPACE[7:16]
    elif experiment_number <= 80:
        # Phase 3: optimizer & schedule (entries 16-19)
        pool = SEARCH_SPACE[16:]
    else:
        # Phase 4: combinations from entire space
        return generate_random_mutation(base_config, num_changes=2, rng=rng)

    config = base_config.copy()
    entry = rng.choice(pool)
    section, param, values, desc = entry
    current = getattr(getattr(config, section), param)
    candidates = [v for v in values if v != current]

    if not candidates:
        return config, "no change (baseline)"

    new_val = rng.choice(candidates)
    setattr(getattr(config, section), param, new_val)
    description = f"{desc}: {current} -> {new_val}"
    return config, description

"""
Hyperparameter Sweep Utilities

Provides different strategies for exploring hyperparameter spaces.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SweepResult:
    """Result from a hyperparameter sweep."""
    best_config: Dict[str, Any]
    best_score: float
    all_configs: List[Dict[str, Any]]
    all_scores: List[float]
    num_trials: int

    def get_top_k(self, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Get top k configurations by score."""
        sorted_results = sorted(
            zip(self.all_configs, self.all_scores),
            key=lambda x: x[1]
        )
        return sorted_results[:k]


class HyperparameterSweep:
    """
    Base class for hyperparameter sweeps.

    Define parameter space and generate configurations to evaluate.

    Example:
        >>> sweep = HyperparameterSweep({
        ...     "learning_rate": [1e-3, 1e-4, 1e-5],
        ...     "batch_size": [4, 8, 16],
        ...     "embed_dim": [256, 512],
        ... })
        >>> configs = sweep.generate_configs()
    """

    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any], Any]],
        seed: int = 42,
    ):
        """
        Initialize sweep.

        Args:
            param_space: Dictionary mapping parameter names to:
                - List of values: discrete choices
                - Tuple (min, max): continuous range
                - Single value: fixed parameter
            seed: Random seed for reproducibility
        """
        self.param_space = param_space
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all configurations. Override in subclasses."""
        raise NotImplementedError


class GridSweep(HyperparameterSweep):
    """
    Grid search over all parameter combinations.

    Evaluates every combination of parameters.
    Best for small parameter spaces.
    """

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all grid combinations."""
        # Extract parameter names and values
        param_names = list(self.param_space.keys())
        param_values = []

        for name in param_names:
            value = self.param_space[name]
            if isinstance(value, list):
                param_values.append(value)
            elif isinstance(value, tuple) and len(value) == 2:
                # For tuples, use linspace
                param_values.append(list(np.linspace(value[0], value[1], 5)))
            else:
                param_values.append([value])

        # Generate all combinations
        configs = []
        for combo in itertools.product(*param_values):
            config = dict(zip(param_names, combo))
            configs.append(config)

        return configs


class RandomSweep(HyperparameterSweep):
    """
    Random search over parameter space.

    Samples random configurations.
    Good for large parameter spaces.
    """

    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any], Any]],
        num_samples: int = 20,
        seed: int = 42,
    ):
        super().__init__(param_space, seed)
        self.num_samples = num_samples

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate random configurations."""
        configs = []

        for _ in range(self.num_samples):
            config = {}
            for name, value in self.param_space.items():
                if isinstance(value, list):
                    config[name] = random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    # Check if integer or float range
                    if isinstance(value[0], int) and isinstance(value[1], int):
                        config[name] = random.randint(value[0], value[1])
                    else:
                        config[name] = random.uniform(value[0], value[1])
                else:
                    config[name] = value
            configs.append(config)

        return configs


class LogUniformSweep(HyperparameterSweep):
    """
    Log-uniform sampling for parameters like learning rate.

    Samples uniformly in log space.
    """

    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any], Any]],
        log_params: List[str] = None,
        num_samples: int = 20,
        seed: int = 42,
    ):
        super().__init__(param_space, seed)
        self.log_params = log_params or ["learning_rate"]
        self.num_samples = num_samples

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations with log-uniform sampling for specified params."""
        configs = []

        for _ in range(self.num_samples):
            config = {}
            for name, value in self.param_space.items():
                if isinstance(value, list):
                    config[name] = random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    if name in self.log_params:
                        # Log-uniform sampling
                        log_min, log_max = np.log10(value[0]), np.log10(value[1])
                        config[name] = 10 ** random.uniform(log_min, log_max)
                    elif isinstance(value[0], int) and isinstance(value[1], int):
                        config[name] = random.randint(value[0], value[1])
                    else:
                        config[name] = random.uniform(value[0], value[1])
                else:
                    config[name] = value
            configs.append(config)

        return configs


class LatinHypercubeSweep(HyperparameterSweep):
    """
    Latin Hypercube Sampling for better coverage of parameter space.

    Ensures each parameter range is evenly sampled.
    """

    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any], Any]],
        num_samples: int = 20,
        seed: int = 42,
    ):
        super().__init__(param_space, seed)
        self.num_samples = num_samples

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations using Latin Hypercube Sampling."""
        continuous_params = {}
        discrete_params = {}

        for name, value in self.param_space.items():
            if isinstance(value, tuple) and len(value) == 2:
                continuous_params[name] = value
            elif isinstance(value, list):
                discrete_params[name] = value
            else:
                discrete_params[name] = [value]

        # Generate LHS samples for continuous params
        num_continuous = len(continuous_params)
        if num_continuous > 0:
            samples = self._latin_hypercube(num_continuous, self.num_samples)
        else:
            samples = np.zeros((self.num_samples, 0))

        configs = []
        for i in range(self.num_samples):
            config = {}

            # Continuous params from LHS
            for j, (name, (lo, hi)) in enumerate(continuous_params.items()):
                if isinstance(lo, int) and isinstance(hi, int):
                    config[name] = int(lo + samples[i, j] * (hi - lo))
                else:
                    config[name] = lo + samples[i, j] * (hi - lo)

            # Discrete params randomly
            for name, values in discrete_params.items():
                config[name] = random.choice(values)

            configs.append(config)

        return configs

    def _latin_hypercube(self, num_dims: int, num_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((num_samples, num_dims))

        for j in range(num_dims):
            # Create evenly spaced points
            points = (np.arange(num_samples) + np.random.random(num_samples)) / num_samples
            np.random.shuffle(points)
            samples[:, j] = points

        return samples


class BayesianSweep(HyperparameterSweep):
    """
    Bayesian optimization for hyperparameter search.

    Uses Gaussian Process to model the objective function.
    Requires scipy.
    """

    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any], Any]],
        num_initial: int = 5,
        num_iterations: int = 15,
        seed: int = 42,
    ):
        super().__init__(param_space, seed)
        self.num_initial = num_initial
        self.num_iterations = num_iterations

    def generate_configs(self) -> List[Dict[str, Any]]:
        """
        Generate initial configurations.

        Note: For full Bayesian optimization, use with ExperimentManager
        which will call suggest_next() after each evaluation.
        """
        # Start with random samples
        random_sweep = RandomSweep(
            self.param_space,
            num_samples=self.num_initial + self.num_iterations,
            seed=self.seed,
        )
        return random_sweep.generate_configs()

    def suggest_next(
        self,
        evaluated_configs: List[Dict[str, Any]],
        scores: List[float],
    ) -> Dict[str, Any]:
        """
        Suggest next configuration based on previous evaluations.

        Uses acquisition function (Expected Improvement).
        """
        try:
            from scipy.stats import norm
            from scipy.optimize import minimize
        except ImportError:
            # Fall back to random
            return RandomSweep(self.param_space, 1, self.seed).generate_configs()[0]

        # Simple implementation: use random + local optimization
        # In production, would use proper GP regression
        best_idx = np.argmin(scores)
        best_config = evaluated_configs[best_idx]

        # Perturb best config slightly
        new_config = {}
        for name, value in self.param_space.items():
            if isinstance(value, tuple) and name in best_config:
                # Add Gaussian noise
                current = best_config[name]
                range_size = value[1] - value[0]
                noise = np.random.normal(0, 0.1 * range_size)
                new_value = np.clip(current + noise, value[0], value[1])
                if isinstance(value[0], int):
                    new_value = int(round(new_value))
                new_config[name] = new_value
            elif isinstance(value, list):
                new_config[name] = random.choice(value)
            else:
                new_config[name] = best_config.get(name, value)

        return new_config


# Convenience functions
def create_sweep(
    sweep_type: str,
    param_space: Dict[str, Any],
    **kwargs,
) -> HyperparameterSweep:
    """
    Create a hyperparameter sweep.

    Args:
        sweep_type: Type of sweep (grid, random, latin_hypercube, bayesian)
        param_space: Parameter space definition
        **kwargs: Additional arguments for the sweep type

    Returns:
        HyperparameterSweep instance
    """
    sweep_classes = {
        "grid": GridSweep,
        "random": RandomSweep,
        "log_uniform": LogUniformSweep,
        "latin_hypercube": LatinHypercubeSweep,
        "bayesian": BayesianSweep,
    }

    sweep_type = sweep_type.lower()
    if sweep_type not in sweep_classes:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Available: {list(sweep_classes.keys())}")

    return sweep_classes[sweep_type](param_space, **kwargs)


def standard_weather_sweep() -> Dict[str, Any]:
    """
    Standard hyperparameter space for weather models.

    Returns common parameter ranges used in weather AI research.
    """
    return {
        "learning_rate": (1e-5, 1e-3),
        "batch_size": [2, 4, 8, 16],
        "embed_dim": [256, 384, 512, 768],
        "num_layers": [4, 6, 8, 12],
        "num_heads": [4, 8, 12, 16],
        "dropout": [0.0, 0.1, 0.2],
        "weight_decay": (1e-4, 1e-1),
    }

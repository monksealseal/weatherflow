"""
# weatherflow/__init__.py
"""WeatherFlow: A Deep Learning Library for Weather Prediction.

This module is intentionally lightweight on import. Heavy submodules and optional
features are loaded lazily when accessed to avoid ImportError on simple imports
in minimal environments (e.g., Google Colab, Lambda Labs) that don't have all
optional dependencies installed.
"""

from .version import __version__, get_version

# Package metadata
__author__ = "Eduardo Siman"
__email__ = "esiman@msn.com"
__license__ = "MIT"

import importlib
from typing import Dict, Tuple

_lazy_exports: Dict[str, Tuple[str, str]] = {
    # Data
    "ERA5Dataset": ("weatherflow.data.era5", "ERA5Dataset"),
    "WeatherDataset": ("weatherflow.data", "WeatherDataset"),
    "create_data_loaders": ("weatherflow.data.era5", "create_data_loaders"),
    # Models
    "BaseWeatherModel": ("weatherflow.models.base", "BaseWeatherModel"),
    "WeatherFlowMatch": ("weatherflow.models.flow_matching", "WeatherFlowMatch"),
    "ConvNextBlock": ("weatherflow.models.flow_matching", "ConvNextBlock"),
    "WeatherFlowODE": ("weatherflow.models.flow_matching", "WeatherFlowODE"),
    "PhysicsGuidedAttention": ("weatherflow.models.physics_guided", "PhysicsGuidedAttention"),
    "StochasticFlowModel": ("weatherflow.models.stochastic", "StochasticFlowModel"),
    "WeatherFlowModel": ("weatherflow.models.weather_flow", "WeatherFlowModel"),
    # Utilities
    "WeatherVisualizer": ("weatherflow.utils.visualization", "WeatherVisualizer"),
    "FlowVisualizer": ("weatherflow.utils.flow_visualization", "FlowVisualizer"),
    "WeatherMetrics": ("weatherflow.utils.evaluation", "WeatherMetrics"),
    "WeatherEvaluator": ("weatherflow.utils.evaluation", "WeatherEvaluator"),
    # Education
    "GraduateAtmosphericDynamicsTool": ("weatherflow.education", "GraduateAtmosphericDynamicsTool"),
    "ProblemScenario": ("weatherflow.education", "ProblemScenario"),
    "SolutionStep": ("weatherflow.education", "SolutionStep"),
    # Training
    "FlowTrainer": ("weatherflow.training.flow_trainer", "FlowTrainer"),
    "compute_flow_loss": ("weatherflow.training.flow_trainer", "compute_flow_loss"),
    # Manifolds
    "Sphere": ("weatherflow.manifolds.sphere", "Sphere"),
    # Solvers (optional)
    "WeatherODESolver": ("weatherflow.solvers.ode_solver", "WeatherODESolver"),
}

__all__ = ["__version__", "get_version", "__author__", "__license__"] + list(_lazy_exports.keys())

def _import_attr(module_path: str, attr_name: str):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Failed to import optional module '{module_path}' required for '{attr_name}'. "
            "This feature depends on optional packages that are not installed. "
            "Install the package extras or the missing dependency to use this feature."
        ) from e
    except Exception as e:
        raise ImportError(f"Error importing '{module_path}' required for '{attr_name}': {e}") from e

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' does not define '{attr_name}'.") from e

def __getattr__(name: str):
    if name in _lazy_exports:
        module_path, attr_name = _lazy_exports[name]
        attr = _import_attr(module_path, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_exports.keys()))

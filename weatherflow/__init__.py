# weatherflow/__init__.py
"""WeatherFlow: A Deep Learning Library for Weather Prediction."""

from .version import __version__, get_version

# Data loading
from .data.era5 import ERA5Dataset, create_data_loaders
from .data import WeatherDataset  # Import from the data package directly

# Models
from .models.base import BaseWeatherModel
from .models.flow_matching import WeatherFlowMatch, ConvNextBlock, WeatherFlowODE
from .models.physics_guided import PhysicsGuidedAttention
from .models.stochastic import StochasticFlowModel
from .models.weather_flow import WeatherFlowModel

# Utilities
from .utils.visualization import WeatherVisualizer
from .utils.flow_visualization import FlowVisualizer
from .utils.evaluation import WeatherMetrics, WeatherEvaluator

# Education
from .education import (
    GraduateAtmosphericDynamicsTool,
    ProblemScenario,
    SolutionStep,
)

# Training
from .training.flow_trainer import FlowTrainer, compute_flow_loss

# Manifolds
from .manifolds.sphere import Sphere

# Optional imports that may fail
_HAS_ODE_SOLVER = False
try:
    from .solvers.ode_solver import WeatherODESolver
    _HAS_ODE_SOLVER = True
except ImportError:
    # torchdiffeq might not be installed
    WeatherODESolver = None

# Package metadata
__author__ = "Eduardo Siman"
__email__ = "esiman@msn.com"
__license__ = "MIT"

# Define public API
__all__ = [
    # Version
    "__version__",
    "get_version",

    # Data
    "ERA5Dataset",
    "WeatherDataset",
    "create_data_loaders",

    # Models
    "BaseWeatherModel",
    "WeatherFlowMatch",
    "WeatherFlowODE",
    "PhysicsGuidedAttention",
    "StochasticFlowModel",
    "WeatherFlowModel",
    "ConvNextBlock",

    # Utilities
    "WeatherVisualizer",
    "FlowVisualizer",
    "WeatherMetrics",
    "WeatherEvaluator",

    # Education
    "GraduateAtmosphericDynamicsTool",
    "ProblemScenario",
    "SolutionStep",

    # Training
    "FlowTrainer",
    "compute_flow_loss",

    # Manifolds
    "Sphere",
]

# Add optional modules to __all__ only if they're available
if _HAS_ODE_SOLVER:
    __all__.append("WeatherODESolver")
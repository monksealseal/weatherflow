# weatherflow/__init__.py
"""WeatherFlow: A Deep Learning Library for Weather Prediction."""

from .version import __version__, get_version

# Data loading
from .data.era5 import ERA5Dataset, create_data_loaders, WeatherDataset

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

# Training
from .training.flow_trainer import FlowTrainer, compute_flow_loss

# Manifolds
from .manifolds.sphere import Sphere

# Optional imports that may fail
try:
    from .solvers.ode_solver import WeatherODESolver
except ImportError:
    # torchdiffeq might not be installed
    pass

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
    
    # Training
    "FlowTrainer",
    "compute_flow_loss",
    
    # Manifolds
    "Sphere",
    
    # Solvers
    "WeatherODESolver"
]
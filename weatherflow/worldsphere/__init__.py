"""
Worldsphere AI Model Command Center

Comprehensive infrastructure for managing Worldsphere's AI models:
- CycleGAN models for satellite to wind field translation
- Video diffusion models for atmospheric sequence prediction
- Data management and preprocessing
- Training infrastructure
- Experiment tracking with RMSE analysis
- Model repository
- Inference pipeline
"""

from .model_registry import (
    WorldsphereModelRegistry,
    ModelType,
    ModelMetadata,
    get_registry,
)
from .experiment_tracker import (
    WorldsphereExperimentTracker,
    ExperimentRun,
    HyperparameterSet,
)
from .data_manager import (
    WorldsphereDataManager,
    DatasetConfig,
    PreprocessingPipeline,
)

__all__ = [
    # Registry
    "WorldsphereModelRegistry",
    "ModelType",
    "ModelMetadata",
    "get_registry",
    # Experiments
    "WorldsphereExperimentTracker",
    "ExperimentRun",
    "HyperparameterSet",
    # Data
    "WorldsphereDataManager",
    "DatasetConfig",
    "PreprocessingPipeline",
]

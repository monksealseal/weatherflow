"""
Model Registry for WeatherFlow

Central registry for all weather AI model architectures with metadata,
configuration, and instantiation capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn


class ModelCategory(Enum):
    """Categories of weather AI models."""

    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    VISION_TRANSFORMER = "vision_transformer"
    TRANSFORMER_3D = "transformer_3d"
    DIFFUSION = "diffusion"
    IMAGE_TO_IMAGE = "image_to_image"
    FOUNDATION_MODEL = "foundation_model"
    HYBRID_PHYSICS_ML = "hybrid_physics_ml"
    FLOW_MATCHING = "flow_matching"
    RECURRENT = "recurrent"
    CONVOLUTIONAL = "convolutional"
    CLASSICAL = "classical"
    ENSEMBLE = "ensemble"


class ModelScale(Enum):
    """Scale/size of model for resource planning."""

    TINY = "tiny"          # <1M params, runs on CPU
    SMALL = "small"        # 1-10M params
    MEDIUM = "medium"      # 10-100M params
    LARGE = "large"        # 100M-1B params
    XLARGE = "xlarge"      # >1B params


@dataclass
class ModelInfo:
    """Metadata about a registered model."""

    name: str
    category: ModelCategory
    scale: ModelScale
    description: str
    paper_title: str
    paper_url: str
    paper_year: int
    authors: List[str]
    organization: str

    # Technical details
    input_variables: List[str]
    output_variables: List[str]
    supported_resolutions: List[str]  # e.g., ["0.25deg", "1deg"]
    forecast_range: str  # e.g., "0-10 days"
    temporal_resolution: str  # e.g., "6h"

    # Capabilities
    is_probabilistic: bool = False
    supports_ensemble: bool = False
    has_pretrained_weights: bool = False
    pretrained_weight_url: Optional[str] = None

    # Resource requirements
    min_gpu_memory_gb: float = 8.0
    typical_training_time: str = "N/A"
    inference_time_per_step: str = "N/A"

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    related_models: List[str] = field(default_factory=list)


class ModelRegistry:
    """
    Central registry for all weather AI models.

    Provides model discovery, instantiation, and metadata access.

    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.get_model("graphcast", resolution="0.25deg")
        >>> info = registry.get_info("graphcast")
        >>> print(info.paper_title)
    """

    _instance = None
    _models: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[nn.Module],
        info: ModelInfo,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model architecture."""
        cls._models[name.lower()] = {
            "class": model_class,
            "info": info,
            "default_config": default_config or {},
        }

    @classmethod
    def get_model(
        cls,
        name: str,
        **kwargs,
    ) -> nn.Module:
        """Instantiate a registered model."""
        name = name.lower()
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        entry = cls._models[name]
        config = {**entry["default_config"], **kwargs}
        return entry["class"](**config)

    @classmethod
    def get_info(cls, name: str) -> ModelInfo:
        """Get metadata about a model."""
        name = name.lower()
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found.")
        return cls._models[name]["info"]

    @classmethod
    def list_models(
        cls,
        category: Optional[ModelCategory] = None,
        scale: Optional[ModelScale] = None,
        probabilistic_only: bool = False,
    ) -> List[str]:
        """List registered models with optional filtering."""
        models = []
        for name, entry in cls._models.items():
            info = entry["info"]
            if category and info.category != category:
                continue
            if scale and info.scale != scale:
                continue
            if probabilistic_only and not info.is_probabilistic:
                continue
            models.append(name)
        return sorted(models)

    @classmethod
    def get_all_info(cls) -> Dict[str, ModelInfo]:
        """Get metadata for all registered models."""
        return {name: entry["info"] for name, entry in cls._models.items()}

    @classmethod
    def search(cls, query: str) -> List[str]:
        """Search models by name, description, or tags."""
        query = query.lower()
        results = []
        for name, entry in cls._models.items():
            info = entry["info"]
            if (
                query in name.lower()
                or query in info.description.lower()
                or any(query in tag.lower() for tag in info.tags)
                or query in info.organization.lower()
            ):
                results.append(name)
        return sorted(results)


# Convenience functions
_registry = ModelRegistry()


def register_model(
    name: str,
    model_class: Type[nn.Module],
    info: ModelInfo,
    default_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a model in the global registry."""
    _registry.register(name, model_class, info, default_config)


def get_model(name: str, **kwargs) -> nn.Module:
    """Get a model instance from the global registry."""
    return _registry.get_model(name, **kwargs)


def list_models(**kwargs) -> List[str]:
    """List models in the global registry."""
    return _registry.list_models(**kwargs)


def model_decorator(
    name: str,
    info: ModelInfo,
    default_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator to register a model class."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        register_model(name, cls, info, default_config)
        return cls
    return decorator

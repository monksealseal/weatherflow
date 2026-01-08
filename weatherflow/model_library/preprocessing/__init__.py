"""
Preprocessing Pipelines for Weather AI Models

Each state-of-the-art weather model has specific preprocessing
requirements. This module provides standardized pipelines that
match the preprocessing used in the original implementations.

Available pipelines:
    - GraphCast: Multi-scale mesh preparation, normalization
    - FourCastNet: Patch-based preprocessing, AFNO-specific
    - Pangu-Weather: 3D tensor preparation, surface/upper split
    - GenCast: Diffusion-specific normalization
    - ClimaX: Variable tokenization preprocessing

References:
    - GraphCast preprocessing: https://github.com/google-deepmind/graphcast
    - FourCastNet preprocessing: https://github.com/NVlabs/FourCastNet
    - Pangu-Weather preprocessing: https://github.com/198808xc/Pangu-Weather
"""

from weatherflow.model_library.preprocessing.graphcast_prep import (
    GraphCastPreprocessor,
    normalize_graphcast,
    create_mesh_features,
)
from weatherflow.model_library.preprocessing.fourcastnet_prep import (
    FourCastNetPreprocessor,
    normalize_fourcastnet,
)
from weatherflow.model_library.preprocessing.pangu_prep import (
    PanguPreprocessor,
    split_surface_upper,
)
from weatherflow.model_library.preprocessing.gencast_prep import (
    GenCastPreprocessor,
)
from weatherflow.model_library.preprocessing.climax_prep import (
    ClimaXPreprocessor,
)
from weatherflow.model_library.preprocessing.common import (
    Normalizer,
    ERA5Normalizer,
    WeatherBenchNormalizer,
)

__all__ = [
    # GraphCast
    "GraphCastPreprocessor",
    "normalize_graphcast",
    "create_mesh_features",
    # FourCastNet
    "FourCastNetPreprocessor",
    "normalize_fourcastnet",
    # Pangu-Weather
    "PanguPreprocessor",
    "split_surface_upper",
    # GenCast
    "GenCastPreprocessor",
    # ClimaX
    "ClimaXPreprocessor",
    # Common
    "Normalizer",
    "ERA5Normalizer",
    "WeatherBenchNormalizer",
]

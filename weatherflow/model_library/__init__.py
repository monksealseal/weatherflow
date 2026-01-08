"""
WeatherFlow Model Library

A comprehensive collection of all major weather AI model architectures,
from classical methods to state-of-the-art deep learning approaches.

Model Categories:
    - Graph Neural Networks (GraphCast, etc.)
    - Vision Transformers (FourCastNet, AFNO, etc.)
    - 3D Transformers (Pangu-Weather, etc.)
    - Diffusion Models (GenCast, etc.)
    - Image-to-Image (Pix2Pix, CycleGAN for satellite)
    - Foundation Models (ClimaX, Aurora, etc.)
    - Hybrid Physics-ML (NeuralGCM, etc.)
    - Classical Methods (Persistence, Climatology, etc.)

References:
    - GraphCast: Lam et al. 2023, "Learning skillful medium-range global weather forecasting"
    - FourCastNet: Pathak et al. 2022, "FourCastNet: A Global Data-driven High-resolution Weather Model"
    - Pangu-Weather: Bi et al. 2023, "Pangu-Weather: A 3D High-Resolution Model"
    - GenCast: Price et al. 2023, "GenCast: Diffusion-based ensemble forecasting"
    - Pix2Pix: Isola et al. 2017, "Image-to-Image Translation with Conditional Adversarial Networks"
    - CycleGAN: Zhu et al. 2017, "Unpaired Image-to-Image Translation using Cycle-Consistent Networks"
    - ClimaX: Nguyen et al. 2023, "ClimaX: A Foundation Model for Weather and Climate"
    - NeuralGCM: Kochkov et al. 2024, "Neural General Circulation Models"
"""

from weatherflow.model_library.registry import (
    ModelRegistry,
    get_model,
    list_models,
    register_model,
)

__all__ = [
    "ModelRegistry",
    "get_model",
    "list_models",
    "register_model",
]

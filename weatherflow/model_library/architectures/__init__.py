"""
WeatherFlow Model Architectures

Comprehensive implementations of all major weather AI model architectures.
All implementations are based on the original papers and reference implementations.

Architectures:
    - graphcast: Graph Neural Network for global weather (DeepMind)
    - fourcastnet: Adaptive Fourier Neural Operator (NVIDIA)
    - pangu: 3D Earth-Specific Transformer (Huawei)
    - gencast: Diffusion-based ensemble forecasting (DeepMind)
    - pix2pix: Conditional GAN for image-to-image translation
    - cyclegan: Unpaired image-to-image translation
    - climax: Foundation model for weather/climate (Microsoft)
    - neuralgcm: Hybrid physics-ML GCM (Google)
    - fuxi: Cascade ML weather model (Fudan)
    - swinvrnn: Swin Transformer + VQ-VAE (Tsinghua)
    - metnet3: High-resolution precipitation nowcasting (Google)
    - fengwu: Multi-modal global weather model (Shanghai AI Lab)
    - aurora: Foundation model for earth system (Microsoft)
"""

from weatherflow.model_library.architectures.base import (
    BaseWeatherModel,
    ProbabilisticWeatherModel,
    EnsembleWeatherModel,
)
from weatherflow.model_library.architectures.graphcast import GraphCastModel
from weatherflow.model_library.architectures.fourcastnet import (
    FourCastNetModel,
    AFNOBlock,
)
from weatherflow.model_library.architectures.pangu import PanguWeatherModel
from weatherflow.model_library.architectures.gencast import GenCastModel
from weatherflow.model_library.architectures.image_translation import (
    Pix2PixGenerator,
    Pix2PixDiscriminator,
    CycleGANGenerator,
    CycleGANDiscriminator,
    HurricaneWindFieldModel,
)
from weatherflow.model_library.architectures.climax import ClimaXModel
from weatherflow.model_library.architectures.diffusion import (
    WeatherDiffusion,
    DDPMScheduler,
    DDIMScheduler,
)
from weatherflow.model_library.architectures.transformers import (
    SwinTransformer3D,
    VisionTransformer,
    EarthSpecificTransformer,
)
from weatherflow.model_library.architectures.classical import (
    PersistenceModel,
    ClimatologyModel,
    LinearRegressionModel,
)

__all__ = [
    # Base
    "BaseWeatherModel",
    "ProbabilisticWeatherModel",
    "EnsembleWeatherModel",
    # Graph Neural Networks
    "GraphCastModel",
    # Vision Transformers
    "FourCastNetModel",
    "AFNOBlock",
    # 3D Transformers
    "PanguWeatherModel",
    "SwinTransformer3D",
    "VisionTransformer",
    "EarthSpecificTransformer",
    # Diffusion
    "GenCastModel",
    "WeatherDiffusion",
    "DDPMScheduler",
    "DDIMScheduler",
    # Image Translation
    "Pix2PixGenerator",
    "Pix2PixDiscriminator",
    "CycleGANGenerator",
    "CycleGANDiscriminator",
    "HurricaneWindFieldModel",
    # Foundation
    "ClimaXModel",
    # Classical
    "PersistenceModel",
    "ClimatologyModel",
    "LinearRegressionModel",
]

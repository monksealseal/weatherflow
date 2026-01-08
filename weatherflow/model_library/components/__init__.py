"""
Modular Architecture Components for Weather AI

This module provides mix-and-match building blocks for weather ML research:

Encoders:
- CNNEncoder: Convolutional encoder with configurable depth
- ViTEncoder: Vision Transformer patch embedding
- GraphEncoder: Graph neural network encoder for mesh data
- IcosahedralEncoder: Icosahedral mesh encoder (like GAIA)
- FourierEncoder: Fourier/spectral encoder (like FourCastNet AFNO)

Processors:
- AttentionProcessor: Standard multi-head attention
- AFNOProcessor: Adaptive Fourier Neural Operator
- MessagePassingProcessor: Graph message passing
- ConvNextProcessor: ConvNext-style blocks
- UNetProcessor: UNet encoder-decoder structure

Decoders:
- GridDecoder: Decode to regular lat-lon grid
- IcosahedralDecoder: Decode to icosahedral mesh
- PatchDecoder: Reverse patch embedding

Embeddings:
- TimeEmbedding: Sinusoidal time/diffusion step embedding
- LeadTimeEmbedding: Forecast lead time embedding
- PositionalEmbedding2D: Lat-lon positional encoding
- SphericalEmbedding: Spherical harmonics embedding
- Earth3DEmbedding: 3D Earth-specific embedding (like Pangu)

Normalization:
- LayerNorm, BatchNorm, GroupNorm, InstanceNorm, RMSNorm

Activations:
- GELU, ReLU, SiLU, Swish, LeakyReLU, PReLU, Mish

Example:
    >>> from weatherflow.model_library.components import (
    ...     ViTEncoder, AFNOProcessor, GridDecoder,
    ...     LeadTimeEmbedding, ComponentRegistry
    ... )
    >>> # Build custom model from components
    >>> encoder = ViTEncoder(in_channels=69, embed_dim=768, patch_size=4)
    >>> processor = AFNOProcessor(dim=768, num_blocks=8)
    >>> decoder = GridDecoder(embed_dim=768, out_channels=69)
"""

from .encoders import (
    BaseEncoder,
    CNNEncoder,
    ViTEncoder,
    GraphEncoder,
    IcosahedralEncoder,
    FourierEncoder,
)

from .processors import (
    BaseProcessor,
    AttentionProcessor,
    AFNOProcessor,
    MessagePassingProcessor,
    ConvNextProcessor,
    UNetProcessor,
)

from .decoders import (
    BaseDecoder,
    GridDecoder,
    IcosahedralDecoder,
    PatchDecoder,
)

from .embeddings import (
    TimeEmbedding,
    LeadTimeEmbedding,
    PositionalEmbedding2D,
    SphericalEmbedding,
    Earth3DEmbedding,
    VariableEmbedding,
)

from .activations import (
    get_activation,
    GELU,
    ReLU,
    SiLU,
    Swish,
    LeakyReLU,
    PReLU,
    Mish,
)

from .normalization import (
    get_normalization,
    LayerNorm,
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    RMSNorm,
)

from .registry import ComponentRegistry

from .modular_model import ModularWeatherModel, build_model_from_config

__all__ = [
    # Encoders
    "BaseEncoder",
    "CNNEncoder",
    "ViTEncoder",
    "GraphEncoder",
    "IcosahedralEncoder",
    "FourierEncoder",
    # Processors
    "BaseProcessor",
    "AttentionProcessor",
    "AFNOProcessor",
    "MessagePassingProcessor",
    "ConvNextProcessor",
    "UNetProcessor",
    # Decoders
    "BaseDecoder",
    "GridDecoder",
    "IcosahedralDecoder",
    "PatchDecoder",
    # Embeddings
    "TimeEmbedding",
    "LeadTimeEmbedding",
    "PositionalEmbedding2D",
    "SphericalEmbedding",
    "Earth3DEmbedding",
    "VariableEmbedding",
    # Activations
    "get_activation",
    "GELU",
    "ReLU",
    "SiLU",
    "Swish",
    "LeakyReLU",
    "PReLU",
    "Mish",
    # Normalization
    "get_normalization",
    "LayerNorm",
    "BatchNorm",
    "GroupNorm",
    "InstanceNorm",
    "RMSNorm",
    # Registry & Model Builder
    "ComponentRegistry",
    "ModularWeatherModel",
    "build_model_from_config",
]

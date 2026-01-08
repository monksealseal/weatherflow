"""
Modular Weather Model Builder

Build custom weather AI models by combining components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .registry import ComponentRegistry, ComponentType, get_component
from .embeddings import TimeEmbedding, LeadTimeEmbedding, CombinedEmbedding


@dataclass
class ModelConfig:
    """Configuration for building a modular weather model."""

    # Model identity
    name: str = "custom_weather_model"
    category: str = "transformer"  # transformer, flow_matching, diffusion, gnn, gan

    # Input/output
    in_channels: int = 69
    out_channels: int = 69
    img_size: Tuple[int, int] = (721, 1440)

    # Components
    encoder: str = "vit_encoder"
    encoder_config: Dict[str, Any] = field(default_factory=dict)

    processor: str = "attention_processor"
    processor_config: Dict[str, Any] = field(default_factory=dict)

    decoder: str = "grid_decoder"
    decoder_config: Dict[str, Any] = field(default_factory=dict)

    # Embeddings
    use_time_embedding: bool = True
    time_embedding_config: Dict[str, Any] = field(default_factory=dict)

    use_positional_embedding: bool = True
    positional_embedding_config: Dict[str, Any] = field(default_factory=dict)

    use_lead_time_embedding: bool = False
    lead_time_config: Dict[str, Any] = field(default_factory=dict)

    # Model settings
    embed_dim: int = 768
    dropout: float = 0.0

    # Flow matching specific
    flow_weighting: str = "time"

    # Diffusion specific
    diffusion_steps: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "encoder": self.encoder,
            "encoder_config": self.encoder_config,
            "processor": self.processor,
            "processor_config": self.processor_config,
            "decoder": self.decoder,
            "decoder_config": self.decoder_config,
            "embed_dim": self.embed_dim,
        }


class ModularWeatherModel(nn.Module):
    """
    Modular weather model built from components.

    Allows researchers to mix and match:
    - Encoders (ViT, CNN, Graph, Icosahedral, Fourier)
    - Processors (Attention, AFNO, Message Passing, ConvNext, UNet)
    - Decoders (Grid, Patch, Icosahedral)
    - Embeddings (Time, Position, Lead Time, Spherical)

    Example:
        >>> config = ModelConfig(
        ...     encoder="vit_encoder",
        ...     processor="afno_processor",
        ...     decoder="patch_decoder",
        ...     embed_dim=768,
        ... )
        >>> model = ModularWeatherModel(config)
        >>> output = model(input_tensor, t=torch.rand(batch_size))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build encoder
        encoder_cfg = {
            "in_channels": config.in_channels,
            "embed_dim": config.embed_dim,
            "img_size": config.img_size,
            **config.encoder_config,
        }
        self.encoder = get_component(config.encoder, **encoder_cfg)

        # Build processor
        processor_cfg = {
            "dim": config.embed_dim,
            **config.processor_config,
        }
        self.processor = get_component(config.processor, **processor_cfg)

        # Build decoder
        decoder_cfg = {
            "embed_dim": config.embed_dim,
            "out_channels": config.out_channels,
            "output_size": config.img_size,
            **config.decoder_config,
        }
        self.decoder = get_component(config.decoder, **decoder_cfg)

        # Build embeddings
        if config.use_time_embedding:
            time_cfg = {"dim": config.embed_dim, **config.time_embedding_config}
            self.time_embedding = TimeEmbedding(**time_cfg)
        else:
            self.time_embedding = None

        if config.use_lead_time_embedding:
            lead_cfg = {"dim": config.embed_dim, **config.lead_time_config}
            self.lead_time_embedding = LeadTimeEmbedding(**lead_cfg)
        else:
            self.lead_time_embedding = None

        # Track model properties
        self.supports_style_conditioning = False
        self.is_flow_matching = config.category == "flow_matching"
        self.is_diffusion = config.category == "diffusion"

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        lead_hours: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            t: Time/diffusion step [B] (for flow matching/diffusion)
            lead_hours: Forecast lead time in hours [B]
            **kwargs: Additional arguments

        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Compute time embedding
        time_emb = None
        if self.time_embedding is not None and t is not None:
            time_emb = self.time_embedding(t)

        # Add lead time embedding
        if self.lead_time_embedding is not None and lead_hours is not None:
            lead_emb = self.lead_time_embedding(lead_hours)
            if time_emb is not None:
                time_emb = time_emb + lead_emb
            else:
                time_emb = lead_emb

        # Encode
        x = self.encoder(x)

        # Process
        x = self.processor(x, time_emb=time_emb)

        # Decode
        x = self.decoder(x)

        return x

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_mb(self) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024


def build_model_from_config(config: Union[ModelConfig, Dict[str, Any]]) -> ModularWeatherModel:
    """
    Build a modular model from configuration.

    Args:
        config: ModelConfig or dictionary with configuration

    Returns:
        ModularWeatherModel instance

    Example:
        >>> config = {
        ...     "encoder": "fourier_encoder",
        ...     "processor": "afno_processor",
        ...     "decoder": "grid_decoder",
        ...     "embed_dim": 768,
        ...     "category": "transformer",
        ... }
        >>> model = build_model_from_config(config)
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)

    return ModularWeatherModel(config)


# Preset configurations for common architectures
PRESETS = {
    "fourcastnet_tiny": ModelConfig(
        name="FourCastNet-Tiny",
        category="transformer",
        encoder="fourier_encoder",
        encoder_config={"num_blocks": 4},
        processor="afno_processor",
        processor_config={"num_blocks": 4},
        decoder="grid_decoder",
        embed_dim=256,
    ),
    "fourcastnet_small": ModelConfig(
        name="FourCastNet-Small",
        category="transformer",
        encoder="fourier_encoder",
        encoder_config={"num_blocks": 8},
        processor="afno_processor",
        processor_config={"num_blocks": 8},
        decoder="grid_decoder",
        embed_dim=512,
    ),
    "vit_tiny": ModelConfig(
        name="ViT-Weather-Tiny",
        category="transformer",
        encoder="vit_encoder",
        encoder_config={"patch_size": 8},
        processor="attention_processor",
        processor_config={"num_layers": 4, "num_heads": 4},
        decoder="patch_decoder",
        decoder_config={"patch_size": 8},
        embed_dim=256,
    ),
    "vit_small": ModelConfig(
        name="ViT-Weather-Small",
        category="transformer",
        encoder="vit_encoder",
        encoder_config={"patch_size": 4},
        processor="attention_processor",
        processor_config={"num_layers": 6, "num_heads": 8},
        decoder="patch_decoder",
        decoder_config={"patch_size": 4},
        embed_dim=512,
    ),
    "flow_matching_tiny": ModelConfig(
        name="FlowMatch-Tiny",
        category="flow_matching",
        encoder="cnn_encoder",
        encoder_config={"depths": [32, 64, 128]},
        processor="convnext_processor",
        processor_config={"num_blocks": 4},
        decoder="grid_decoder",
        embed_dim=128,
        use_time_embedding=True,
    ),
    "flow_matching_small": ModelConfig(
        name="FlowMatch-Small",
        category="flow_matching",
        encoder="cnn_encoder",
        encoder_config={"depths": [64, 128, 256, 512]},
        processor="convnext_processor",
        processor_config={"num_blocks": 6},
        decoder="grid_decoder",
        embed_dim=256,
        use_time_embedding=True,
    ),
    "diffusion_tiny": ModelConfig(
        name="Diffusion-Tiny",
        category="diffusion",
        encoder="cnn_encoder",
        encoder_config={"depths": [64, 128]},
        processor="unet_processor",
        processor_config={"dim_mults": [1, 2, 4], "num_res_blocks": 1},
        decoder="grid_decoder",
        embed_dim=64,
        use_time_embedding=True,
    ),
    "graphcast_tiny": ModelConfig(
        name="GraphCast-Tiny",
        category="gnn",
        encoder="graph_encoder",
        encoder_config={"num_layers": 2, "hidden_dim": 128},
        processor="message_passing_processor",
        processor_config={"num_layers": 8},
        decoder="grid_decoder",
        embed_dim=256,
    ),
    "gaia_tiny": ModelConfig(
        name="GAIA-Tiny",
        category="gnn",
        encoder="icosahedral_encoder",
        encoder_config={"mesh_resolution": 3},
        processor="message_passing_processor",
        processor_config={"num_layers": 4},
        decoder="icosahedral_decoder",
        decoder_config={"mesh_resolution": 3},
        embed_dim=256,
    ),
}


def get_preset(name: str) -> ModelConfig:
    """
    Get a preset model configuration.

    Args:
        name: Preset name (fourcastnet_tiny, vit_small, flow_matching_tiny, etc.)

    Returns:
        ModelConfig for the preset

    Example:
        >>> config = get_preset("fourcastnet_tiny")
        >>> model = ModularWeatherModel(config)
    """
    name = name.lower()
    if name not in PRESETS:
        available = list(PRESETS.keys())
        raise ValueError(f"Preset '{name}' not found. Available: {available}")

    return PRESETS[name]


def list_presets() -> List[str]:
    """List all available presets."""
    return list(PRESETS.keys())


def create_model(
    preset: Optional[str] = None,
    **kwargs,
) -> ModularWeatherModel:
    """
    Create a modular weather model.

    Args:
        preset: Optional preset name
        **kwargs: Override any config parameters

    Returns:
        ModularWeatherModel instance

    Example:
        >>> # Use preset
        >>> model = create_model("fourcastnet_tiny")

        >>> # Custom model
        >>> model = create_model(
        ...     encoder="vit_encoder",
        ...     processor="afno_processor",
        ...     embed_dim=768,
        ... )

        >>> # Modify preset
        >>> model = create_model("vit_small", embed_dim=1024)
    """
    if preset is not None:
        config = get_preset(preset)
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = ModelConfig(**kwargs)

    return ModularWeatherModel(config)

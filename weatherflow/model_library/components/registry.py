"""
Component Registry for Weather AI Models

Central registry for all components enabling easy discovery and instantiation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch.nn as nn


class ComponentType(Enum):
    """Types of components."""
    ENCODER = "encoder"
    PROCESSOR = "processor"
    DECODER = "decoder"
    EMBEDDING = "embedding"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"


@dataclass
class ComponentInfo:
    """Metadata about a registered component."""
    name: str
    type: ComponentType
    description: str
    default_config: Dict[str, Any]
    compatible_with: List[str]  # List of model categories
    paper_reference: Optional[str] = None


class ComponentRegistry:
    """
    Central registry for all model components.

    Provides component discovery, instantiation, and configuration.

    Example:
        >>> registry = ComponentRegistry()
        >>> encoder = registry.get("vit_encoder", embed_dim=768)
        >>> components = registry.list_components(ComponentType.ENCODER)
    """

    _instance = None
    _components: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self):
        """Register all default components."""
        # Import components
        from . import encoders, processors, decoders, embeddings, activations, normalization

        # Register encoders
        self.register(
            "cnn_encoder",
            encoders.CNNEncoder,
            ComponentInfo(
                name="CNN Encoder",
                type=ComponentType.ENCODER,
                description="Convolutional encoder with configurable depth and downsampling",
                default_config={"embed_dim": 256, "depths": [64, 128, 256, 512]},
                compatible_with=["flow_matching", "transformer", "diffusion"],
            )
        )

        self.register(
            "vit_encoder",
            encoders.ViTEncoder,
            ComponentInfo(
                name="Vision Transformer Encoder",
                type=ComponentType.ENCODER,
                description="ViT-style patch embedding encoder",
                default_config={"embed_dim": 768, "patch_size": 4, "img_size": (721, 1440)},
                compatible_with=["transformer", "flow_matching"],
                paper_reference="ViT: An Image is Worth 16x16 Words",
            )
        )

        self.register(
            "graph_encoder",
            encoders.GraphEncoder,
            ComponentInfo(
                name="Graph Encoder",
                type=ComponentType.ENCODER,
                description="Graph neural network encoder for mesh data",
                default_config={"embed_dim": 512, "num_nodes": 40962, "num_layers": 2},
                compatible_with=["gnn"],
                paper_reference="GraphCast",
            )
        )

        self.register(
            "icosahedral_encoder",
            encoders.IcosahedralEncoder,
            ComponentInfo(
                name="Icosahedral Mesh Encoder",
                type=ComponentType.ENCODER,
                description="Icosahedral mesh encoder with KNN attention",
                default_config={"embed_dim": 512, "mesh_resolution": 5, "knn_k": 8},
                compatible_with=["gnn", "transformer"],
                paper_reference="GAIA",
            )
        )

        self.register(
            "fourier_encoder",
            encoders.FourierEncoder,
            ComponentInfo(
                name="Fourier Encoder",
                type=ComponentType.ENCODER,
                description="Spectral encoder using FFT with learnable filters",
                default_config={"embed_dim": 768, "num_blocks": 4},
                compatible_with=["transformer", "flow_matching"],
                paper_reference="FourCastNet (AFNO)",
            )
        )

        # Register processors
        self.register(
            "attention_processor",
            processors.AttentionProcessor,
            ComponentInfo(
                name="Attention Processor",
                type=ComponentType.PROCESSOR,
                description="Standard multi-head self-attention",
                default_config={"dim": 768, "num_heads": 8, "num_layers": 6},
                compatible_with=["transformer", "flow_matching"],
            )
        )

        self.register(
            "afno_processor",
            processors.AFNOProcessor,
            ComponentInfo(
                name="AFNO Processor",
                type=ComponentType.PROCESSOR,
                description="Adaptive Fourier Neural Operator for spectral processing",
                default_config={"dim": 768, "num_blocks": 8, "sparsity_threshold": 0.01},
                compatible_with=["transformer", "flow_matching"],
                paper_reference="FourCastNet",
            )
        )

        self.register(
            "message_passing_processor",
            processors.MessagePassingProcessor,
            ComponentInfo(
                name="Message Passing Processor",
                type=ComponentType.PROCESSOR,
                description="Graph message passing for mesh-based models",
                default_config={"dim": 512, "num_layers": 16, "edge_dim": 3},
                compatible_with=["gnn"],
                paper_reference="GraphCast",
            )
        )

        self.register(
            "convnext_processor",
            processors.ConvNextProcessor,
            ComponentInfo(
                name="ConvNext Processor",
                type=ComponentType.PROCESSOR,
                description="Modern convolutional blocks with depthwise convolutions",
                default_config={"dim": 256, "num_blocks": 6, "kernel_size": 7},
                compatible_with=["flow_matching", "diffusion"],
            )
        )

        self.register(
            "unet_processor",
            processors.UNetProcessor,
            ComponentInfo(
                name="UNet Processor",
                type=ComponentType.PROCESSOR,
                description="UNet encoder-decoder with skip connections",
                default_config={"dim": 128, "dim_mults": [1, 2, 4, 8], "num_res_blocks": 2},
                compatible_with=["diffusion", "gan"],
                paper_reference="UNet, Diffusion models",
            )
        )

        # Register decoders
        self.register(
            "grid_decoder",
            decoders.GridDecoder,
            ComponentInfo(
                name="Grid Decoder",
                type=ComponentType.DECODER,
                description="Decode to regular lat-lon grid",
                default_config={"embed_dim": 256, "out_channels": 69, "output_size": (721, 1440)},
                compatible_with=["transformer", "flow_matching", "diffusion"],
            )
        )

        self.register(
            "patch_decoder",
            decoders.PatchDecoder,
            ComponentInfo(
                name="Patch Decoder",
                type=ComponentType.DECODER,
                description="Reverse patch embedding for ViT models",
                default_config={"embed_dim": 768, "out_channels": 69, "patch_size": 4},
                compatible_with=["transformer"],
                paper_reference="ViT",
            )
        )

        self.register(
            "icosahedral_decoder",
            decoders.IcosahedralDecoder,
            ComponentInfo(
                name="Icosahedral Decoder",
                type=ComponentType.DECODER,
                description="Decode from mesh to lat-lon grid",
                default_config={"embed_dim": 512, "out_channels": 69, "mesh_resolution": 5},
                compatible_with=["gnn"],
                paper_reference="GAIA",
            )
        )

        # Register embeddings
        self.register(
            "time_embedding",
            embeddings.TimeEmbedding,
            ComponentInfo(
                name="Time Embedding",
                type=ComponentType.EMBEDDING,
                description="Sinusoidal time embedding for flow/diffusion",
                default_config={"dim": 256, "max_time": 1000.0},
                compatible_with=["flow_matching", "diffusion"],
            )
        )

        self.register(
            "lead_time_embedding",
            embeddings.LeadTimeEmbedding,
            ComponentInfo(
                name="Lead Time Embedding",
                type=ComponentType.EMBEDDING,
                description="Forecast lead time encoding",
                default_config={"dim": 256, "max_lead_hours": 240},
                compatible_with=["transformer", "flow_matching", "diffusion", "gnn"],
            )
        )

        self.register(
            "positional_embedding_2d",
            embeddings.PositionalEmbedding2D,
            ComponentInfo(
                name="2D Positional Embedding",
                type=ComponentType.EMBEDDING,
                description="Sinusoidal lat-lon positional encoding",
                default_config={"dim": 256, "height": 721, "width": 1440},
                compatible_with=["transformer", "flow_matching", "diffusion"],
            )
        )

        self.register(
            "spherical_embedding",
            embeddings.SphericalEmbedding,
            ComponentInfo(
                name="Spherical Embedding",
                type=ComponentType.EMBEDDING,
                description="Spherical harmonics position embedding",
                default_config={"dim": 256, "max_degree": 20},
                compatible_with=["transformer", "gnn"],
                paper_reference="Spherical CNNs",
            )
        )

        self.register(
            "earth_3d_embedding",
            embeddings.Earth3DEmbedding,
            ComponentInfo(
                name="3D Earth Embedding",
                type=ComponentType.EMBEDDING,
                description="3D Earth-specific position + level embedding",
                default_config={"dim": 256, "num_levels": 13},
                compatible_with=["transformer"],
                paper_reference="Pangu-Weather",
            )
        )

    @classmethod
    def register(
        cls,
        name: str,
        component_class: Type[nn.Module],
        info: ComponentInfo,
    ) -> None:
        """Register a component."""
        cls._components[name.lower()] = {
            "class": component_class,
            "info": info,
        }

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs,
    ) -> nn.Module:
        """Instantiate a registered component."""
        name = name.lower()
        if name not in cls._components:
            available = list(cls._components.keys())
            raise ValueError(f"Component '{name}' not found. Available: {available}")

        entry = cls._components[name]
        config = {**entry["info"].default_config, **kwargs}
        return entry["class"](**config)

    @classmethod
    def get_info(cls, name: str) -> ComponentInfo:
        """Get metadata about a component."""
        name = name.lower()
        if name not in cls._components:
            raise ValueError(f"Component '{name}' not found.")
        return cls._components[name]["info"]

    @classmethod
    def list_components(
        cls,
        component_type: Optional[ComponentType] = None,
        compatible_with: Optional[str] = None,
    ) -> List[str]:
        """List registered components with optional filtering."""
        components = []
        for name, entry in cls._components.items():
            info = entry["info"]
            if component_type and info.type != component_type:
                continue
            if compatible_with and compatible_with not in info.compatible_with:
                continue
            components.append(name)
        return sorted(components)

    @classmethod
    def get_all_info(cls) -> Dict[str, ComponentInfo]:
        """Get metadata for all registered components."""
        return {name: entry["info"] for name, entry in cls._components.items()}

    @classmethod
    def get_components_by_type(cls) -> Dict[ComponentType, List[str]]:
        """Get components organized by type."""
        result = {t: [] for t in ComponentType}
        for name, entry in cls._components.items():
            result[entry["info"].type].append(name)
        return result


# Convenience functions
_registry = ComponentRegistry()


def get_component(name: str, **kwargs) -> nn.Module:
    """Get a component instance from the global registry."""
    return _registry.get(name, **kwargs)


def list_components(
    component_type: Optional[ComponentType] = None,
    compatible_with: Optional[str] = None,
) -> List[str]:
    """List components in the global registry."""
    return _registry.list_components(component_type, compatible_with)


def get_component_info(name: str) -> ComponentInfo:
    """Get component info from the global registry."""
    return _registry.get_info(name)

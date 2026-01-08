"""
ClimaX Foundation Model Architecture

Implementation based on:
    "ClimaX: A foundation model for weather and climate"
    Nguyen et al., ICML 2023
    Microsoft Research

Key innovations:
    - Variable tokenization for flexible input/output
    - Lead time embedding for multi-step forecasting
    - Pre-training on diverse climate data
    - Fine-tuning for specific tasks
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import (
    BaseWeatherModel,
    TimeEmbedding,
)
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class VariableTokenizer(nn.Module):
    """
    Variable-specific tokenization for flexible input handling.

    Each atmospheric variable gets its own embedding, allowing
    the model to handle different variable combinations.
    """

    def __init__(
        self,
        variable_names: List[str],
        embed_dim: int,
        patch_size: int = 4,
        max_vars: int = 100,
    ):
        super().__init__()
        self.variable_names = variable_names
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Variable embeddings (learnable)
        self.var_embed = nn.Embedding(max_vars, embed_dim)

        # Mapping from variable name to index
        self.var_to_idx = {name: i for i, name in enumerate(variable_names)}

        # Patch projection per variable type
        self.patch_proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        variables: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize input with variable-aware embeddings.

        Args:
            x: (batch, num_vars, height, width)
            variables: List of variable names

        Returns:
            tokens: (batch, num_vars * num_patches, embed_dim)
            var_ids: Variable indices for each token
        """
        batch, num_vars, h, w = x.shape
        tokens_list = []
        var_ids_list = []

        for i, var in enumerate(variables):
            # Get variable embedding
            var_idx = self.var_to_idx.get(var, 0)
            var_emb = self.var_embed(torch.tensor([var_idx], device=x.device))

            # Patch projection
            var_data = x[:, i:i+1]  # (batch, 1, h, w)
            patches = self.patch_proj(var_data)  # (batch, embed_dim, h/p, w/p)

            # Flatten patches
            patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

            # Add variable embedding
            patches = patches + var_emb

            tokens_list.append(patches)
            var_ids_list.extend([var_idx] * patches.shape[1])

        tokens = torch.cat(tokens_list, dim=1)
        var_ids = torch.tensor(var_ids_list, device=x.device)

        return tokens, var_ids


class LeadTimeAgnosticAttention(nn.Module):
    """
    Attention mechanism that is aware of forecast lead time.

    Modulates attention based on lead time to handle
    different forecast horizons appropriately.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Lead time modulation
        self.lead_time_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Apply lead time modulation if provided
        if lead_time_emb is not None:
            gate = self.lead_time_gate(lead_time_emb)
            x = x * gate.unsqueeze(1)

        return x


class ClimaXBlock(nn.Module):
    """Transformer block for ClimaX with lead time awareness."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = LeadTimeAgnosticAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), lead_time_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class ClimaXModel(BaseWeatherModel):
    """
    ClimaX: Foundation Model for Weather and Climate.

    Based on Nguyen et al. (2023), Microsoft Research.

    A transformer-based foundation model that can be pre-trained
    on diverse climate data and fine-tuned for specific tasks.

    Key features:
    - Variable tokenization for flexible input/output
    - Lead time embedding for multi-step forecasting
    - Pre-training on CMIP6, ERA5, and other datasets
    - Fine-tuning for forecasting, downscaling, projection

    Args:
        default_variables: Default input/output variables
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        patch_size: Spatial patch size for tokenization
        img_size: Input image size (height, width)
    """

    def __init__(
        self,
        default_variables: Optional[List[str]] = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        img_size: Tuple[int, int] = (64, 128),
        drop_rate: float = 0.0,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if default_variables is None:
            default_variables = self._get_default_variables()
        if input_variables is None:
            input_variables = default_variables
        if output_variables is None:
            output_variables = default_variables

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="1.4deg",
            forecast_hours=6,
        )

        self.default_variables = default_variables
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        # Calculate dimensions
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Variable tokenizer
        self.tokenizer = VariableTokenizer(
            variable_names=default_variables,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # Lead time embedding
        self.lead_time_emb = TimeEmbedding(embed_dim, max_time=240)  # Up to 10 days

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ClimaXBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # Output norm
        self.norm = nn.LayerNorm(embed_dim)

        # Variable-specific output heads
        self.output_heads = nn.ModuleDict()
        for var in default_variables:
            self.output_heads[var] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, patch_size * patch_size),
            )

    def _get_default_variables(self) -> List[str]:
        """Default ClimaX variables (from ERA5)."""
        return [
            "z_500", "z_850", "z_1000",
            "t_500", "t_850", "t_1000",
            "u_500", "u_850", "u_1000",
            "v_500", "v_850", "v_1000",
            "q_500", "q_850", "q_1000",
            "t2m", "u10", "v10",
        ]

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with flexible variable handling.

        Args:
            x: Input tensor (batch, num_vars, height, width)
            lead_time: Forecast lead time in hours
            input_vars: List of input variable names
            output_vars: List of output variable names

        Returns:
            Output tensor (batch, num_output_vars, height, width)
        """
        batch = x.shape[0]

        if input_vars is None:
            input_vars = self.default_variables[:x.shape[1]]
        if output_vars is None:
            output_vars = input_vars

        # Tokenize input
        tokens, var_ids = self.tokenizer(x, input_vars)

        # Add positional embedding (repeat for each variable)
        num_var_tokens = tokens.shape[1] // len(input_vars)
        pos_emb = self.pos_embed.repeat(1, len(input_vars), 1)
        tokens = tokens + pos_emb[:, :tokens.shape[1]]

        # Lead time embedding
        if lead_time is None:
            lead_time = torch.full((batch,), 6, device=x.device)
        lt_emb = self.lead_time_emb(lead_time)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, lt_emb)

        tokens = self.norm(tokens)

        # Decode output variables
        outputs = []
        for var in output_vars:
            # Average tokens across all input variables for this output
            var_tokens = tokens[:, :self.num_patches]  # Use first variable's spatial tokens

            # Apply variable-specific head
            if var in self.output_heads:
                head = self.output_heads[var]
            else:
                head = self.output_heads[self.default_variables[0]]

            var_out = head(var_tokens)  # (batch, num_patches, patch_size^2)

            # Reshape to image
            var_out = var_out.reshape(
                batch, self.grid_h, self.grid_w,
                self.patch_size, self.patch_size
            )
            var_out = var_out.permute(0, 1, 3, 2, 4).contiguous()
            var_out = var_out.reshape(batch, 1, self.img_size[0], self.img_size[1])
            outputs.append(var_out)

        output = torch.cat(outputs, dim=1)

        # Residual connection
        if output.shape == x.shape:
            output = x + output

        return output


# Register model
climax_info = ModelInfo(
    name="ClimaX",
    category=ModelCategory.FOUNDATION_MODEL,
    scale=ModelScale.LARGE,
    description="Foundation model for weather and climate pre-trained on diverse data",
    paper_title="ClimaX: A foundation model for weather and climate",
    paper_url="https://arxiv.org/abs/2301.10343",
    paper_year=2023,
    authors=["Tung Nguyen", "Johannes Brandstetter", "Ashish Kapoor", "et al."],
    organization="Microsoft Research",
    input_variables=["z", "t", "u", "v", "q", "t2m", "u10", "v10"],
    output_variables=["z", "t", "u", "v", "q", "t2m", "u10", "v10"],
    supported_resolutions=["5.625deg", "1.40625deg"],
    forecast_range="0-14 days",
    temporal_resolution="6h",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=True,
    pretrained_weight_url="https://huggingface.co/microsoft/climax",
    min_gpu_memory_gb=16.0,
    typical_training_time="Pre-training: ~1 week on 8 A100s",
    inference_time_per_step="~1 second on A100",
    tags=["foundation-model", "pre-training", "fine-tuning", "multi-task"],
    related_models=["aurora", "prithvi"],
)

register_model("climax", ClimaXModel, climax_info, {
    "embed_dim": 256,
    "depth": 8,
    "num_heads": 8,
    "patch_size": 4,
    "img_size": (64, 128),
})

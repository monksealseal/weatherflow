"""
Embedding Components for Weather AI Models

Embeddings encode various types of information:
- Time/diffusion step (for flow matching, diffusion)
- Forecast lead time
- Spatial position (lat-lon, spherical)
- Variable type (for foundation models)
"""

from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion/flow models.

    Good for: Flow matching, diffusion models
    """

    def __init__(
        self,
        dim: int,
        max_time: float = 1000.0,
        learned: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_time = max_time

        # Sinusoidal frequencies
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

        # Optional learned projection
        if learned:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] in range [0, 1] or [0, max_time]

        Returns:
            Time embedding [B, dim]
        """
        # Normalize if needed
        if t.max() > 1.0:
            t = t / self.max_time

        t = t.unsqueeze(-1) * self.emb.unsqueeze(0) * self.max_time
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

        return self.mlp(emb)


class LeadTimeEmbedding(nn.Module):
    """
    Forecast lead time embedding.

    Encodes how far into the future the prediction is.
    Good for: Multi-step forecasting, ClimaX, Pangu-Weather
    """

    def __init__(
        self,
        dim: int,
        max_lead_hours: int = 240,
        temporal_resolution_hours: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.max_lead_hours = max_lead_hours
        self.temporal_resolution = temporal_resolution_hours

        # Number of discrete lead times
        self.num_lead_times = max_lead_hours // temporal_resolution_hours + 1

        # Learnable embeddings for each lead time
        self.lead_embed = nn.Embedding(self.num_lead_times, dim)

        # MLP for continuous interpolation
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, lead_hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lead_hours: Lead time in hours [B]

        Returns:
            Lead time embedding [B, dim]
        """
        # Convert to discrete index
        idx = (lead_hours / self.temporal_resolution).long().clamp(0, self.num_lead_times - 1)

        # Get embedding
        emb = self.lead_embed(idx)

        return self.mlp(emb)


class PositionalEmbedding2D(nn.Module):
    """
    2D sinusoidal positional encoding for lat-lon grids.

    Good for: All grid-based weather models
    """

    def __init__(
        self,
        dim: int,
        height: int = 721,
        width: int = 1440,
        learned: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width

        if learned:
            self.pe = nn.Parameter(torch.randn(1, dim, height, width) * 0.02)
        else:
            pe = self._create_sinusoidal_pe(dim, height, width)
            self.register_buffer("pe", pe)

    def _create_sinusoidal_pe(self, dim: int, height: int, width: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(dim, height, width)

        # Latitude encoding
        y_pos = torch.arange(height).unsqueeze(1).float()
        y_div = torch.exp(
            torch.arange(0, dim // 2, 2).float() *
            (-math.log(10000.0) / (dim // 2))
        )

        pe[0::4, :, :] = torch.sin(y_pos * y_div).unsqueeze(2).repeat(1, 1, width)
        pe[1::4, :, :] = torch.cos(y_pos * y_div).unsqueeze(2).repeat(1, 1, width)

        # Longitude encoding
        x_pos = torch.arange(width).unsqueeze(0).float()
        x_div = torch.exp(
            torch.arange(0, dim // 2, 2).float() *
            (-math.log(10000.0) / (dim // 2))
        )

        pe[2::4, :, :] = torch.sin(x_pos * x_div.unsqueeze(1)).unsqueeze(1).repeat(1, height, 1)
        pe[3::4, :, :] = torch.cos(x_pos * x_div.unsqueeze(1)).unsqueeze(1).repeat(1, height, 1)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Input with positional encoding added [B, C, H, W]
        """
        # Interpolate if sizes don't match
        pe = self.pe
        if x.shape[-2:] != pe.shape[-2:]:
            pe = F.interpolate(pe, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return x + pe


class SphericalEmbedding(nn.Module):
    """
    Spherical harmonics positional embedding.

    Better captures the spherical geometry of Earth.
    Good for: Spherical weather models, GAIA, foundation models
    """

    def __init__(
        self,
        dim: int,
        max_degree: int = 20,
        height: int = 721,
        width: int = 1440,
    ):
        super().__init__()
        self.dim = dim
        self.max_degree = max_degree

        # Compute spherical harmonics basis
        lat = torch.linspace(-90, 90, height) * math.pi / 180
        lon = torch.linspace(0, 360, width) * math.pi / 180

        # Number of spherical harmonic coefficients
        num_coeffs = (max_degree + 1) ** 2

        # Project to embedding dim
        self.proj = nn.Linear(min(num_coeffs, dim * 2), dim)

        # Precompute latitude-dependent terms
        self.register_buffer("cos_lat", torch.cos(lat).view(1, 1, -1, 1))
        self.register_buffer("sin_lat", torch.sin(lat).view(1, 1, -1, 1))

        # Learnable spherical embedding
        sph_embed = torch.randn(1, dim, height, width) * 0.02
        self.sph_embed = nn.Parameter(sph_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Input with spherical embedding added [B, C, H, W]
        """
        # Interpolate if needed
        sph = self.sph_embed
        if x.shape[-2:] != sph.shape[-2:]:
            sph = F.interpolate(sph, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Weight by latitude (more information at tropics, less at poles)
        cos_lat = self.cos_lat
        if x.shape[-2] != cos_lat.shape[-2]:
            cos_lat = F.interpolate(cos_lat, size=(x.shape[-2], 1), mode="linear", align_corners=False)
            cos_lat = cos_lat.expand(-1, -1, -1, x.shape[-1])

        return x + sph * cos_lat


class Earth3DEmbedding(nn.Module):
    """
    3D Earth-specific embedding (like Pangu-Weather).

    Encodes both horizontal position and vertical (pressure) level.
    Good for: Pangu-Weather, 3D weather models
    """

    def __init__(
        self,
        dim: int,
        height: int = 721,
        width: int = 1440,
        num_levels: int = 13,
        learned: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # Horizontal position embedding
        self.pos_embed_h = PositionalEmbedding2D(dim, height, width, learned=learned)

        # Vertical (pressure level) embedding
        self.level_embed = nn.Embedding(num_levels, dim)

        # 3D position MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        level_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, L, H, W] for 3D or [B, C, H, W] for 2D
            level_idx: Pressure level indices [L]

        Returns:
            Input with 3D position embedding [B, C, ...]
        """
        if x.dim() == 5:
            B, C, L, H, W = x.shape

            # Level embedding
            if level_idx is None:
                level_idx = torch.arange(L, device=x.device)
            level_emb = self.level_embed(level_idx)  # [L, dim]

            # Apply to each level
            outputs = []
            for i in range(L):
                x_level = x[:, :, i]  # [B, C, H, W]

                # Add horizontal PE
                x_level = self.pos_embed_h(x_level)

                # Add level embedding (broadcast)
                x_level = x_level + level_emb[i].view(1, -1, 1, 1)

                outputs.append(x_level)

            return torch.stack(outputs, dim=2)

        else:
            # 2D case - just horizontal PE
            return self.pos_embed_h(x)


class VariableEmbedding(nn.Module):
    """
    Variable type embedding (for foundation models).

    Encodes which weather variable the data represents.
    Good for: ClimaX, foundation models with variable inputs
    """

    def __init__(
        self,
        dim: int,
        num_variables: int = 69,
        variable_names: Optional[list] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_variables = num_variables

        # Learnable embeddings per variable
        self.var_embed = nn.Embedding(num_variables, dim)

        # Variable name to index mapping
        if variable_names is not None:
            self.var_to_idx = {name: i for i, name in enumerate(variable_names)}
        else:
            self.var_to_idx = None

        # MLP for combining with spatial features
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        var_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] where C = num_variables
            var_idx: Variable indices [C]

        Returns:
            Input with variable embedding added [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Get variable indices
        if var_idx is None:
            var_idx = torch.arange(min(C, self.num_variables), device=x.device)

        # Get embeddings
        var_emb = self.var_embed(var_idx)  # [C, dim]
        var_emb = self.mlp(var_emb)

        # Broadcast to spatial dimensions
        var_emb = var_emb.view(1, C, self.dim, 1, 1)

        # Add to input (assumes dim == C for proper broadcasting)
        # If dim != C, we need to project
        if self.dim != C:
            var_emb = var_emb.view(1, C, -1, 1, 1)
            var_emb = var_emb.mean(dim=2)  # [1, C, 1, 1]

        return x + var_emb.squeeze(2)


class SeasonalEmbedding(nn.Module):
    """
    Seasonal/temporal embedding for climate patterns.

    Encodes day of year, time of day for seasonal patterns.
    Good for: Long-range forecasting, climate models
    """

    def __init__(
        self,
        dim: int,
        include_day_of_year: bool = True,
        include_hour_of_day: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # Day of year embedding (365 days)
        if include_day_of_year:
            self.doy_embed = nn.Embedding(366, dim)
        else:
            self.doy_embed = None

        # Hour of day embedding (24 hours)
        if include_hour_of_day:
            self.hour_embed = nn.Embedding(24, dim)
        else:
            self.hour_embed = None

        # Combine embeddings
        num_embeds = int(include_day_of_year) + int(include_hour_of_day)
        if num_embeds > 1:
            self.combine = nn.Linear(dim * num_embeds, dim)
        else:
            self.combine = nn.Identity()

    def forward(
        self,
        day_of_year: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            day_of_year: Day of year [B] (1-365)
            hour_of_day: Hour of day [B] (0-23)

        Returns:
            Seasonal embedding [B, dim]
        """
        embeds = []

        if self.doy_embed is not None and day_of_year is not None:
            doy_idx = day_of_year.clamp(0, 365).long()
            embeds.append(self.doy_embed(doy_idx))

        if self.hour_embed is not None and hour_of_day is not None:
            hour_idx = hour_of_day.clamp(0, 23).long()
            embeds.append(self.hour_embed(hour_idx))

        if not embeds:
            raise ValueError("At least one temporal input required")

        if len(embeds) == 1:
            return embeds[0]

        return self.combine(torch.cat(embeds, dim=-1))


class CombinedEmbedding(nn.Module):
    """
    Combines multiple embedding types.

    Good for: Complex models needing multiple embeddings
    """

    def __init__(
        self,
        dim: int,
        height: int = 721,
        width: int = 1440,
        use_positional: bool = True,
        use_spherical: bool = False,
        use_time: bool = True,
        use_lead_time: bool = True,
        use_seasonal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.pos_embed = PositionalEmbedding2D(dim, height, width) if use_positional else None
        self.sph_embed = SphericalEmbedding(dim, height=height, width=width) if use_spherical else None
        self.time_embed = TimeEmbedding(dim) if use_time else None
        self.lead_embed = LeadTimeEmbedding(dim) if use_lead_time else None
        self.seasonal_embed = SeasonalEmbedding(dim) if use_seasonal else None

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        lead_hours: Optional[torch.Tensor] = None,
        day_of_year: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply all enabled embeddings.

        Args:
            x: Input tensor [B, C, H, W]
            t: Time/diffusion step [B]
            lead_hours: Forecast lead time [B]
            day_of_year: Day of year [B]
            hour_of_day: Hour of day [B]

        Returns:
            Tuple of (embedded_x, conditioning_embed)
        """
        # Spatial embeddings (added to x)
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        if self.sph_embed is not None:
            x = self.sph_embed(x)

        # Conditioning embeddings (returned separately)
        cond_embeds = []
        if self.time_embed is not None and t is not None:
            cond_embeds.append(self.time_embed(t))
        if self.lead_embed is not None and lead_hours is not None:
            cond_embeds.append(self.lead_embed(lead_hours))
        if self.seasonal_embed is not None and (day_of_year is not None or hour_of_day is not None):
            cond_embeds.append(self.seasonal_embed(day_of_year, hour_of_day))

        if cond_embeds:
            cond = sum(cond_embeds) / len(cond_embeds)
        else:
            cond = None

        return x, cond

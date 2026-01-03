"""
FlowFormer: Hierarchical Spherical Transformer for global atmospheric modeling.

This module implements a transformer architecture that respects spherical geometry
and operates efficiently on multi-scale global weather data.
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalPositionalEncoding(nn.Module):
    """
    Spherical positional encoding using icosahedral mesh vertices.
    Encodes latitude/longitude positions with spherical harmonics.
    """

    def __init__(self, d_model: int, max_degree: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_degree = max_degree

        # Learnable projection from spherical harmonics to model dim
        num_harmonics = (max_degree + 1) ** 2
        self.harmonic_proj = nn.Linear(num_harmonics, d_model)

    def compute_spherical_harmonics(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spherical harmonics up to max_degree.

        Args:
            lat: Latitude in radians, shape [..., H, W]
            lon: Longitude in radians, shape [..., H, W]

        Returns:
            Spherical harmonic features, shape [..., H, W, num_harmonics]
        """
        # Convert to colatitude (theta) and azimuth (phi)
        theta = torch.pi / 2 - lat  # colatitude
        phi = lon

        harmonics = []

        # Compute Y_l^m for l=0 to max_degree, m=-l to l
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Simplified computation - in practice use scipy.special.sph_harm
                # or precomputed values
                if m == 0:
                    Y = torch.sqrt(torch.tensor((2*l + 1) / (4 * math.pi))) * \
                        self._legendre_polynomial(l, torch.cos(theta))
                else:
                    # Associated Legendre polynomial with azimuthal component
                    P = self._associated_legendre(l, abs(m), torch.cos(theta))
                    norm = torch.sqrt(
                        torch.tensor((2*l + 1) * math.factorial(l - abs(m)) /
                                   (4 * math.pi * math.factorial(l + abs(m))))
                    )
                    if m > 0:
                        Y = norm * P * torch.cos(m * phi)
                    else:
                        Y = norm * P * torch.sin(abs(m) * phi)

                harmonics.append(Y)

        return torch.stack(harmonics, dim=-1)

    def _legendre_polynomial(self, n: int, x: torch.Tensor) -> torch.Tensor:
        """Compute Legendre polynomial P_n(x) using recurrence relation."""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            P_prev = torch.ones_like(x)
            P_curr = x
            for i in range(2, n + 1):
                P_next = ((2*i - 1) * x * P_curr - (i - 1) * P_prev) / i
                P_prev = P_curr
                P_curr = P_next
            return P_curr

    def _associated_legendre(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        """Compute associated Legendre polynomial (simplified)."""
        # Simplified implementation - use PyTorch special functions in production
        P = self._legendre_polynomial(l, x)
        if m > 0:
            # Apply derivative m times (simplified)
            P = P * torch.pow(1 - x**2, m/2)
        return P

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lat: Latitude grid in radians, shape [H, W]
            lon: Longitude grid in radians, shape [H, W]

        Returns:
            Position embeddings, shape [H, W, d_model]
        """
        harmonics = self.compute_spherical_harmonics(lat, lon)
        pos_emb = self.harmonic_proj(harmonics)
        return pos_emb


class SphericalAttention(nn.Module):
    """
    Multi-head attention that respects spherical geometry.
    Uses geodesic distance for attention weighting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.window_size = window_size

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable relative position bias based on geodesic distance
        if window_size is not None:
            self.relative_pos_bias = nn.Parameter(
                torch.zeros(2 * window_size - 1, 2 * window_size - 1, num_heads)
            )
        else:
            self.relative_pos_bias = None

    def compute_geodesic_distance(
        self,
        lat1: torch.Tensor,
        lon1: torch.Tensor,
        lat2: torch.Tensor,
        lon2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute great circle distance on unit sphere.

        Uses Haversine formula for numerical stability.
        """
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2

        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        return c

    def forward(
        self,
        x: torch.Tensor,
        lat_grid: Optional[torch.Tensor] = None,
        lon_grid: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features, shape [B, H, W, d_model]
            lat_grid: Latitude coordinates, shape [H, W]
            lon_grid: Longitude coordinates, shape [H, W]
            mask: Attention mask, shape [B, H, W, H, W]

        Returns:
            Output features, shape [B, H, W, d_model]
        """
        B, H, W, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, H, W, 3*d_model]
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # [3, B, num_heads, H, W, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Windowed attention if window_size is specified
        if self.window_size is not None:
            return self._windowed_attention(q, k, v, lat_grid, lon_grid, mask)

        # Global attention (memory intensive for large grids)
        # Flatten spatial dimensions
        q = q.reshape(B, self.num_heads, H * W, self.d_head)
        k = k.reshape(B, self.num_heads, H * W, self.d_head)
        v = v.reshape(B, self.num_heads, H * W, self.d_head)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Add geodesic distance bias if coordinates provided
        if lat_grid is not None and lon_grid is not None:
            dist_bias = self._compute_distance_bias(lat_grid, lon_grid)
            attn = attn + dist_bias.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, H*W, d_head]
        out = out.reshape(B, self.num_heads, H, W, self.d_head)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, self.d_model)

        out = self.out_proj(out)
        return out

    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lat_grid: Optional[torch.Tensor],
        lon_grid: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply attention within local windows."""
        B, num_heads, H, W, d_head = q.shape
        win = self.window_size

        # Pad if needed
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win

        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h))
            k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
            v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))

        _, _, H_pad, W_pad, _ = q.shape

        # Reshape into windows
        q = q.view(B, num_heads, H_pad // win, win, W_pad // win, win, d_head)
        q = q.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, num_heads, win * win, d_head)

        k = k.view(B, num_heads, H_pad // win, win, W_pad // win, win, d_head)
        k = k.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, num_heads, win * win, d_head)

        v = v.view(B, num_heads, H_pad // win, win, W_pad // win, win, d_head)
        v = v.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, num_heads, win * win, d_head)

        # Attention within windows
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)

        if self.relative_pos_bias is not None:
            # Add relative position bias
            bias = self.relative_pos_bias[:win, :win].reshape(win * win, -1)
            attn = attn + bias.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B*num_windows, num_heads, win*win, d_head]

        # Reshape back
        num_windows = (H_pad // win) * (W_pad // win)
        out = out.view(B, num_windows, num_heads, win, win, d_head)
        out = out.view(B, H_pad // win, W_pad // win, num_heads, win, win, d_head)
        out = out.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, H_pad, W_pad, -1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]

        return out

    def _compute_distance_bias(
        self,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention bias based on geodesic distance."""
        H, W = lat_grid.shape
        lat_flat = lat_grid.reshape(-1)
        lon_flat = lon_grid.reshape(-1)

        # Compute pairwise distances (expensive - consider approximations)
        lat1 = lat_flat.unsqueeze(1)
        lon1 = lon_flat.unsqueeze(1)
        lat2 = lat_flat.unsqueeze(0)
        lon2 = lon_flat.unsqueeze(0)

        dist = self.compute_geodesic_distance(lat1, lon1, lat2, lon2)

        # Convert distance to bias (closer = higher attention)
        bias = -dist / dist.std()

        return bias


class SphericalFeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class HierarchicalSphericalTransformerLayer(nn.Module):
    """
    Single transformer layer with spherical attention.
    Includes both local (windowed) and global attention paths.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        window_size: Optional[int] = 16,
        use_global_attention: bool = False,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.local_attn = SphericalAttention(
            d_model, num_heads, dropout, window_size=window_size
        )

        self.use_global_attention = use_global_attention
        if use_global_attention:
            self.norm1_global = nn.LayerNorm(d_model)
            self.global_attn = SphericalAttention(
                d_model, num_heads, dropout, window_size=None
            )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SphericalFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lat_grid: Optional[torch.Tensor] = None,
        lon_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features, shape [B, H, W, d_model]
            lat_grid: Latitude coordinates, shape [H, W]
            lon_grid: Longitude coordinates, shape [H, W]

        Returns:
            Output features, shape [B, H, W, d_model]
        """
        # Local attention
        residual = x
        x = self.norm1(x)
        x = self.local_attn(x, lat_grid, lon_grid)
        x = residual + self.dropout(x)

        # Global attention (every few layers)
        if self.use_global_attention:
            residual = x
            x = self.norm1_global(x)
            x = self.global_attn(x, lat_grid, lon_grid)
            x = residual + self.dropout(x)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


class HierarchicalSphericalTransformer(nn.Module):
    """
    Multi-scale spherical transformer for global atmospheric modeling.

    Uses a hierarchy of resolutions with cross-scale attention to capture
    both large-scale circulations and mesoscale phenomena.
    """

    def __init__(
        self,
        input_channels: int = 4,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        window_size: int = 16,
        num_scales: int = 3,
        global_attention_freq: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_scales = num_scales

        # Input projection
        self.input_proj = nn.Conv2d(input_channels, d_model, kernel_size=1)

        # Positional encoding
        self.pos_encoding = SphericalPositionalEncoding(d_model)

        # Multi-scale transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_global = (i % global_attention_freq == 0)
            self.layers.append(
                HierarchicalSphericalTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    window_size=window_size,
                    use_global_attention=use_global,
                )
            )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input weather state, shape [B, C, H, W]
            lat_grid: Latitude coordinates in radians, shape [H, W]
            lon_grid: Longitude coordinates in radians, shape [H, W]

        Returns:
            Encoded features, shape [B, H, W, d_model]
        """
        B, C, H, W = x.shape

        # Input projection
        x = self.input_proj(x)  # [B, d_model, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, d_model]

        # Add positional encoding
        pos_emb = self.pos_encoding(lat_grid, lon_grid)
        x = x + pos_emb.unsqueeze(0)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, lat_grid, lon_grid)

        x = self.output_norm(x)

        return x


class FlowFormer(nn.Module):
    """
    FlowFormer: Transformer-based flow matching model for weather prediction.

    Combines hierarchical spherical transformer with flow matching objective
    for large-scale weather foundation models.
    """

    def __init__(
        self,
        input_channels: int = 4,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        window_size: int = 16,
        time_embed_dim: int = 256,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, d_model),
        )

        # Spherical transformer encoder
        self.encoder = HierarchicalSphericalTransformer(
            input_channels=input_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            window_size=window_size,
        )

        # Output projection for velocity field
        self.velocity_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity field for flow matching.

        Args:
            x: Weather state, shape [B, C, H, W]
            t: Time values in [0, 1], shape [B]
            lat_grid: Latitude coordinates in radians, shape [H, W]
            lon_grid: Longitude coordinates in radians, shape [H, W]

        Returns:
            Velocity field, shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Encode with transformer
        h = self.encoder(x, lat_grid, lon_grid)  # [B, H, W, d_model]

        # Add time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, d_model]
        t_emb = t_emb.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d_model]
        h = h + t_emb

        # Project to velocity field
        v = self.velocity_proj(h)  # [B, H, W, C]
        v = v.permute(0, 3, 1, 2)  # [B, C, H, W]

        return v

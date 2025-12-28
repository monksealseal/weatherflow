import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, List
import numpy as np
from ..manifolds.sphere import Sphere

class ConvNextBlock(nn.Module):
    """ConvNext block for efficient spatial processing."""
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6, padding_mode: str = "zeros"):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode=padding_mode
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + residual

class TimeEncoder(nn.Module):
    """Sinusoidal time encoding as used in transformers."""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time values into high-dimensional features.
        
        Args:
            t: Time values, shape [batch_size]
            
        Returns:
            Time embeddings, shape [batch_size, dim]
        """
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeatherFlowMatch(nn.Module):
    """Flow matching model for weather prediction.
    
    This model implements the vector field for continuous normalizing flows
    in the context of weather prediction. It can be used with ODE solvers
    to generate trajectories of weather states.
    """
    def __init__(
        self, 
        input_channels: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        use_attention: bool = True,
        grid_size: Tuple[int, int] = (32, 64),  # lat, lon
        physics_informed: bool = True,
        window_size: int = 8,
        static_channels: int = 0,
        forcing_dim: int = 0,
        spherical_padding: bool = False,
        use_graph_mp: bool = False,
        use_spectral_mixer: bool = False,
        spectral_modes: int = 12,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.grid_size = grid_size
        self.physics_informed = physics_informed
        self.window_size = window_size
        self.static_channels = static_channels
        self.forcing_dim = forcing_dim
        self.use_graph_mp = use_graph_mp
        self.use_spectral_mixer = use_spectral_mixer
        self.spectral_modes = spectral_modes
        padding_mode = "circular" if spherical_padding else "zeros"
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )

        if static_channels > 0:
            self.static_proj = nn.Conv2d(static_channels, hidden_dim, kernel_size=1)
        else:
            self.static_proj = None

        if forcing_dim > 0:
            self.forcing_proj = nn.Linear(forcing_dim, hidden_dim)
        else:
            self.forcing_proj = None

        if use_graph_mp:
            self.register_buffer(
                "graph_neighbors",
                self._build_grid_adjacency(grid_size[0], grid_size[1]),
                persistent=False,
            )
            self.graph_deg = (self.graph_neighbors >= 0).sum(dim=1).view(1, -1, 1)
        else:
            self.graph_neighbors = None
            self.graph_deg = None
        
        # Time encoding
        self.time_encoder = TimeEncoder(hidden_dim)
        
        # Main processing blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(ConvNextBlock(hidden_dim, padding_mode=padding_mode))
            
        # Attention layer if requested
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim, 
                num_heads=8, 
                batch_first=True
            )
            
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2,
                input_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )
        
        # Physics constraints (divergence regularization)
        if physics_informed:
            self.sphere = Sphere()
        else:
            self.sphere = None

        if self.use_spectral_mixer:
            # Real-valued parameters applied in Fourier domain
            self.freq_weight = nn.Parameter(
                torch.randn(2, spectral_modes, spectral_modes)
            )
    
    def _apply_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Apply windowed attention to reduce quadratic blow-up on large grids."""
        if not self.use_attention:
            return h

        batch_size, c, height, width = h.shape
        # Enforce local attention; if unset, pick a safe default window
        win = self.window_size if self.window_size and self.window_size > 0 else min(height, width, 16)
        pad_h = (win - height % win) % win
        pad_w = (win - width % win) % win
        if pad_h or pad_w:
            h = F.pad(h, (0, pad_w, 0, pad_h))
        _, _, padded_h, padded_w = h.shape

        h_windows = h.view(batch_size, c, padded_h // win, win, padded_w // win, win)
        h_windows = h_windows.permute(0, 2, 4, 3, 5, 1)  # [B, nh, nw, win, win, C]
        tokens = h_windows.reshape(-1, win * win, c)  # [B*nh*nw, L, C]

        attn_out, _ = self.attention(tokens, tokens, tokens)
        attn_out = attn_out.view(batch_size, padded_h // win, padded_w // win, win, win, c)
        attn_out = attn_out.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, c, padded_h, padded_w
        )

        if pad_h or pad_w:
            attn_out = attn_out[:, :, :height, :width]
        return h[:, :, :height, :width] + attn_out

    def _build_grid_adjacency(self, lat: int, lon: int) -> torch.Tensor:
        """Build 4-neighbour adjacency with longitude wrap to reduce seam artefacts."""
        neighbors: List[List[int]] = []
        for i in range(lat):
            for j in range(lon):
                idx = i * lon + j
                north = ((i - 1) % lat) * lon + j
                south = ((i + 1) % lat) * lon + j
                west = i * lon + ((j - 1) % lon)
                east = i * lon + ((j + 1) % lon)
                neighbors.append([north, south, west, east])
        return torch.tensor(neighbors, dtype=torch.long)

    def _graph_message_passing(self, h: torch.Tensor) -> torch.Tensor:
        """Lightweight graph aggregation over a wrapped lat/lon grid."""
        if self.graph_neighbors is None:
            return h
        batch_size, channels, height, width = h.shape
        nodes = h.flatten(2).permute(0, 2, 1)  # [B, N, C]
        neigh_idx = self.graph_neighbors  # [N, 4]
        neigh_feat = nodes[:, neigh_idx, :]  # [B, N, 4, C]
        deg = self.graph_deg.to(h.device).clamp(min=1)
        aggregated = neigh_feat.sum(dim=2) / deg
        nodes = nodes + aggregated  # residual
        h_out = nodes.permute(0, 2, 1).view(batch_size, channels, height, width)
        return h_out

    def _spectral_mix(self, h: torch.Tensor) -> torch.Tensor:
        """Apply lightweight spectral mixing on lowest frequencies."""
        if not self.use_spectral_mixer:
            return h
        batch, channels, height, width = h.shape
        # rfft2 returns complex; operate on truncated modes
        freq = torch.fft.rfft2(h, norm="ortho")
        max_h = min(self.spectral_modes, freq.shape[-2])
        max_w = min(self.spectral_modes, freq.shape[-1])
        weight = self.freq_weight[:, :max_h, :max_w]
        # Apply real/imag weights
        freq_real = freq.real
        freq_imag = freq.imag
        freq_real[..., :max_h, :max_w] = (
            freq_real[..., :max_h, :max_w] * weight[0]
        )
        freq_imag[..., :max_h, :max_w] = (
            freq_imag[..., :max_h, :max_w] * weight[1]
        )
        mixed = torch.complex(freq_real, freq_imag)
        out = torch.fft.irfft2(mixed, s=(height, width), norm="ortho")
        return out

    def _spherical_divergence(
        self, u: torch.Tensor, v_comp: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence on a sphere using latitude/longitude grids (wrapped)."""
        _, _, lat_size, lon_size = u.shape
        lat_grid = torch.linspace(
            -torch.pi / 2, torch.pi / 2, steps=lat_size, device=u.device, dtype=u.dtype
        )
        eps = self.sphere._get_eps(u.dtype) if self.sphere is not None else torch.finfo(u.dtype).eps
        cos_lat = torch.cos(lat_grid).clamp(min=eps)
        cos_lat = cos_lat.view(1, 1, lat_size, 1)

        dlon = (2 * torch.pi) / max(lon_size, 1)
        dphi = torch.pi / max(lat_size - 1, 1)

        du_dlambda = torch.gradient(u, spacing=(dlon,), dim=(3,))[0]
        dvcos_dphi = torch.gradient(v_comp * cos_lat, spacing=(dphi,), dim=(2,))[0]

        radius = torch.tensor(self.sphere.radius, device=u.device, dtype=u.dtype)
        return (du_dlambda / (radius * cos_lat)) + (dvcos_dphi / radius)
    
    def _add_time_embedding(
        self, x: torch.Tensor, t: torch.Tensor, forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add time (and optional forcing) embedding to feature maps."""
        time_embed = self.time_encoder(t)  # [batch_size, hidden_dim]
        time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
        out = x + time_embed

        if forcing is not None and self.forcing_proj is not None:
            forcing_proj = self.forcing_proj(forcing).unsqueeze(-1).unsqueeze(-1)
            out = out + forcing_proj
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity field for flow matching.
        
        Args:
            x: Weather state, shape [batch_size, channels, lat, lon]
            t: Time values in [0, 1], shape [batch_size]
            
        Returns:
            Velocity field, same shape as x
        """
        # Input projection
        h = self.input_proj(x)

        if static is not None and self.static_proj is not None:
            h = h + self.static_proj(static)
        
        # Add time embedding
        h = self._add_time_embedding(h, t, forcing)
        
        # Process through main blocks
        for block in self.blocks:
            h = block(h)
        
        # Optional spectral mixing of low-frequency components
        h = self._spectral_mix(h)

        # Apply attention if requested
        h = self._apply_attention(h)

        if self.use_graph_mp:
            h = self._graph_message_passing(h)
        
        # Output projection
        v = self.output_proj(h)
        
        # Apply physics constraints if requested
        if self.physics_informed:
            v = self._apply_physics_constraints(v, x)
            
        return v
    
    def _apply_physics_constraints(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply physics-based constraints to the velocity field.
        
        Currently implements:
        - Approximate divergence-free constraint for mass conservation
        
        Args:
            v: Velocity field, shape [batch_size, channels, lat, lon]
            x: Current state, shape [batch_size, channels, lat, lon]
            
        Returns:
            Constrained velocity field, same shape as v
        """
        # We focus on the first two channels if they represent u,v components
        if v.shape[1] < 2:
            return v

        # Spherical divergence-based correction (wrap-aware)
        u = v[:, 0:1]
        v_comp = v[:, 1:2]
        div = self._spherical_divergence(u, v_comp)

        # Create a correction field to make the flow more divergence-free
        u_corr = torch.gradient(-div, dim=3)[0]
        v_corr = torch.gradient(-div, dim=2)[0]

        alpha = 0.1
        v_new = v.clone()
        v_new[:, 0:1] = u + alpha * u_corr
        v_new[:, 1:2] = v_comp + alpha * v_corr
        return v_new
    
    def compute_flow_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss.
        
        Args:
            x0: Initial state, shape [batch_size, channels, lat, lon]
            x1: Target state, shape [batch_size, channels, lat, lon]
            t: Time values in [0, 1], shape [batch_size]
            
        Returns:
            Dictionary of loss components
        """
        # Compute model's predicted velocity
        v_pred = self(x0, t, static=static, forcing=forcing)
        
        # Compute target velocity (straight-line path)
        # For spherical geometries, this should use geodesics
        v_target = (x1 - x0) / (1 - t).view(-1, 1, 1, 1)
        
        # Main flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Physics-based loss components
        losses = {'flow_loss': flow_loss}
        
        if self.physics_informed and v_pred.shape[1] >= 2:
            div = self._spherical_divergence(v_pred[:, 0:1], v_pred[:, 1:2])
            div_loss = torch.mean(div**2)
            losses['div_loss'] = div_loss
            flow_loss = flow_loss + 0.1 * div_loss

            energy_x0 = torch.sum(x0**2)
            energy_x1 = torch.sum(x1**2)
            energy_diff = (energy_x0 - energy_x1).abs() / (energy_x0 + 1e-6)
            losses['energy_diff'] = energy_diff

        losses['total_loss'] = flow_loss
        return losses


class StyleFlowMatch(WeatherFlowMatch):
    """Style-conditioned flow matching for general style transfer tasks.

    A lightweight style encoder produces FiLM parameters that modulate the
    velocity field, enabling content/style translation without adversarial
    objectives.
    """

    supports_style_conditioning: bool = True

    def __init__(
        self,
        input_channels: int = 3,
        style_channels: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 4,
        use_attention: bool = True,
        grid_size: Tuple[int, int] = (32, 64),
        physics_informed: bool = False,
        window_size: int = 8,
        static_channels: int = 0,
        forcing_dim: int = 0,
        spherical_padding: bool = False,
    ):
        super().__init__(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_attention=use_attention,
            grid_size=grid_size,
            physics_informed=physics_informed,
            window_size=window_size,
            static_channels=static_channels,
            forcing_dim=forcing_dim,
            spherical_padding=spherical_padding,
        )

        self.style_encoder = nn.Sequential(
            nn.Conv2d(style_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.style_to_film = nn.Linear(hidden_dim, hidden_dim * 2)

    def _condition_with_style(self, h: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning derived from the style reference."""

        style_features = self.style_encoder(style)
        gamma, beta = self.style_to_film(style_features).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * h + beta

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input projection
        h = self.input_proj(x)

        # Add time embedding
        h = self._add_time_embedding(h, t)

        # Inject style conditioning if provided
        if style is not None:
            h = self._condition_with_style(h, style)

        # Process through main blocks
        for block in self.blocks:
            h = block(h)

        # Apply attention if requested
        if self.use_attention:
            batch_size, c, height, width = h.shape
            h_flat = h.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            h_att, _ = self.attention(h_flat, h_flat, h_flat)
            h_att = h_att.permute(0, 2, 1).view(batch_size, c, height, width)

            h = h + h_att

        # Output projection
        v = self.output_proj(h)

        # Style transfer typically does not enforce physical constraints
        if self.physics_informed:
            v = self._apply_physics_constraints(v, x)

        return v


class WeatherFlowODE(nn.Module):
    """ODE-based weather prediction using flow matching.
    
    This model wraps a flow matching model and uses it with an ODE solver
    to generate weather predictions over time.
    """
    def __init__(
        self,
        flow_model: nn.Module,
        solver_method: str = 'dopri5',
        rtol: float = 1e-4,
        atol: float = 1e-4,
        fast_mode: bool = False,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.fast_mode = fast_mode
        
    def forward(
        self,
        x0: torch.Tensor,
        times: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate weather predictions by solving the ODE.
        
        Args:
            x0: Initial weather state, shape [batch_size, channels, lat, lon]
            times: Time points for prediction, shape [num_times]
            static: Optional static conditioning features [batch_size, channels, lat, lon]
            forcing: Optional per-sample forcing vector [batch_size, forcing_dim]
            
        Returns:
            Predicted weather states at requested times,
            shape [num_times, batch_size, channels, lat, lon]
        """
        if self.fast_mode:
            # Fixed-step Heun integration for speed; assumes times sorted.
            preds = [x0]
            x = x0
            for i in range(len(times) - 1):
                dt = times[i + 1] - times[i]
                t_mid = times[i] + 0.5 * dt
                t_batch = times[i].expand(x.shape[0])
                k1 = self.flow_model(x, t_batch, static=static, forcing=forcing)
                x_mid = x + dt * 0.5 * k1
                k2 = self.flow_model(x_mid, t_mid.expand(x.shape[0]), static=static, forcing=forcing)
                x = x + dt * (k1 + k2) * 0.5
                preds.append(x)
            return torch.stack(preds, dim=0)
        else:
            from torchdiffeq import odeint
            
            def ode_func(t, x):
                """ODE function for the solver."""
                t_batch = t.expand(x.shape[0])
                return self.flow_model(x, t_batch, static=static, forcing=forcing)
            
            predictions = odeint(
                ode_func,
                x0,
                times,
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol
            )
            return predictions

    def ensemble_forecast(
        self,
        x0: torch.Tensor,
        times: torch.Tensor,
        num_members: int = 4,
        noise_std: float = 0.0,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate an ensemble by perturbing the initial state."""
        members = []
        for _ in range(num_members):
            x0_perturbed = x0
            if noise_std > 0:
                x0_perturbed = x0 + noise_std * torch.randn_like(x0)
            members.append(self.forward(x0_perturbed, times, static=static, forcing=forcing))
        return torch.stack(members, dim=0)

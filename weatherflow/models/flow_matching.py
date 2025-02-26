import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, List
import numpy as np
from ..manifolds.sphere import Sphere

class ConvNextBlock(nn.Module):
    """ConvNext block for efficient spatial processing."""
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
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
        physics_informed: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.grid_size = grid_size
        self.physics_informed = physics_informed
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        )
        
        # Time encoding
        self.time_encoder = TimeEncoder(hidden_dim)
        
        # Main processing blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(ConvNextBlock(hidden_dim))
            
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
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, input_channels, kernel_size=3, padding=1),
        )
        
        # Physics constraints (divergence regularization)
        if physics_informed:
            self.sphere = Sphere()
    
    def _add_time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add time embedding to feature maps.
        
        Args:
            x: Feature maps, shape [batch_size, channels, height, width]
            t: Time values, shape [batch_size]
            
        Returns:
            Features with time embedding, same shape as x
        """
        # Encode time
        time_embed = self.time_encoder(t)  # [batch_size, hidden_dim]
        
        # Reshape for broadcasting
        time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
        
        # Add to all spatial locations
        return x + time_embed
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute velocity field for flow matching.
        
        Args:
            x: Weather state, shape [batch_size, channels, lat, lon]
            t: Time values in [0, 1], shape [batch_size]
            
        Returns:
            Velocity field, same shape as x
        """
        # Input projection
        h = self.input_proj(x)
        
        # Add time embedding
        h = self._add_time_embedding(h, t)
        
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
        if v.shape[1] >= 2:
            # Simple divergence calculation
            u = v[:, 0:1]  # u component
            v_comp = v[:, 1:2]  # v component
            
            # Calculate approximate divergence
            # (This is a simplified version - a proper implementation would 
            # account for spherical geometry)
            du_dx = torch.gradient(u, dim=3)[0]
            dv_dy = torch.gradient(v_comp, dim=2)[0]
            
            div = du_dx + dv_dy
            
            # Create a correction field to make the flow more divergence-free
            u_corr = torch.gradient(-div, dim=3)[0]
            v_corr = torch.gradient(-div, dim=2)[0]
            
            # Apply correction with a small weight
            alpha = 0.1
            v_new = v.clone()
            v_new[:, 0:1] = u + alpha * u_corr
            v_new[:, 1:2] = v_comp + alpha * v_corr
            
            return v_new
        
        return v
    
    def compute_flow_loss(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor
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
        v_pred = self(x0, t)
        
        # Compute target velocity (straight-line path)
        # For spherical geometries, this should use geodesics
        v_target = (x1 - x0) / (1 - t).view(-1, 1, 1, 1)
        
        # Main flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Physics-based loss components
        losses = {'flow_loss': flow_loss}
        
        if self.physics_informed:
            # Add divergence penalty
            u = v_pred[:, 0:1] if v_pred.shape[1] >= 2 else None
            v_comp = v_pred[:, 1:2] if v_pred.shape[1] >= 2 else None
            
            if u is not None and v_comp is not None:
                du_dx = torch.gradient(u, dim=3)[0]
                dv_dy = torch.gradient(v_comp, dim=2)[0]
                div = du_dx + dv_dy
                div_loss = torch.mean(div**2)
                losses['div_loss'] = div_loss
                
                # Add to total loss
                flow_loss = flow_loss + 0.1 * div_loss
            
            # Energy conservation - soft constraint
            energy_x0 = torch.sum(x0**2)
            energy_x1 = torch.sum(x1**2)
            energy_diff = (energy_x0 - energy_x1).abs() / (energy_x0 + 1e-6)
            losses['energy_diff'] = energy_diff
        
        losses['total_loss'] = flow_loss
        return losses


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
        atol: float = 1e-4
    ):
        super().__init__()
        self.flow_model = flow_model
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        
    def forward(
        self, 
        x0: torch.Tensor, 
        times: torch.Tensor
    ) -> torch.Tensor:
        """Generate weather predictions by solving the ODE.
        
        Args:
            x0: Initial weather state, shape [batch_size, channels, lat, lon]
            times: Time points for prediction, shape [num_times]
            
        Returns:
            Predicted weather states at requested times,
            shape [num_times, batch_size, channels, lat, lon]
        """
        from torchdiffeq import odeint
        
        def ode_func(t, x):
            """ODE function for the solver."""
            # Reshape t for the model
            t_batch = t.expand(x.shape[0])
            return self.flow_model(x, t_batch)
        
        # Solve ODE
        predictions = odeint(
            ode_func,
            x0,
            times,
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        return predictions
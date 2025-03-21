
import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..manifolds.sphere import Sphere
from ..solvers.ode_solver import WeatherODESolver

class WeatherFlowModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        physics_constraints: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sphere = Sphere()
        self.solver = WeatherODESolver(physics_constraints=physics_constraints)
        
        # Project to and from hidden dimension
        self.input_proj = nn.Linear(4, hidden_dim)  # 4 is number of features
        self.output_proj = nn.Linear(hidden_dim, 4)
        
        # Neural network for velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ) for _ in range(n_layers)],
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity field at given points and times."""
        batch_size, n_lat, n_lon, features = x.shape
        
        # Project features to hidden dimension (first flatten spatial dims)
        x_flat = x.reshape(batch_size * n_lat * n_lon, features)
        x_hidden = self.input_proj(x_flat)  # [B*lat*lon, hidden_dim]
        x_hidden = x_hidden.reshape(batch_size, n_lat, n_lon, -1)  # [B, lat, lon, hidden_dim]
        
        # Add time dimension
        t = t.view(-1, 1, 1, 1).expand(batch_size, n_lat, n_lon, 1)
        h = torch.cat([x_hidden, t], dim=-1)  # [B, lat, lon, hidden_dim + 1]
        
        # Compute velocity (flatten again for the velocity network)
        h_flat = h.reshape(batch_size * n_lat * n_lon, -1)
        v = self.velocity_net(h_flat)  # [B*lat*lon, hidden_dim]
        v = v.reshape(batch_size, n_lat, n_lon, -1)  # [B, lat, lon, hidden_dim]
        
        # Project back to feature dimension (flatten one more time)
        v_flat = v.reshape(batch_size * n_lat * n_lon, -1)
        v_out = self.output_proj(v_flat)  # [B*lat*lon, features]
        v_out = v_out.reshape(batch_size, n_lat, n_lon, features)
        
        return v_out

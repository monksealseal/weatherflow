
import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..manifolds.sphere import Sphere
from ..solvers.ode_solver import WeatherODESolver

class WeatherFlowModel(nn.Module):
    """Weather prediction model using flow matching on spherical manifold."""
    
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
        
        # Neural network for velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ) for _ in range(n_layers)],
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity field at given points and times."""
        batch_size = x.shape[0]
        
        # Flatten spatial dimensions while keeping batch and feature dims
        x_flat = x.reshape(batch_size, -1)  # [B, (lat*lon*features)]
        
        # Project to hidden dimension
        x_hidden = nn.Linear(x_flat.shape[1], self.hidden_dim).to(x.device)(x_flat)
        
        # Add time dimension
        t = t.view(-1, 1).expand(batch_size, 1)
        h = torch.cat([x_hidden, t], dim=1)
        
        # Compute velocity
        v = self.velocity_net(h)
        
        # Project back to original shape
        v_out = nn.Linear(self.hidden_dim, x_flat.shape[1]).to(x.device)(v)
        
        return v_out.reshape(x.shape)

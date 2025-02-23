
# Copyright (c) 2024 WeatherFlow
# Implementation integrating probability paths, sphere manifold, and ODE solver

import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..path.prob_path import ProbPath
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
        
        # Initialize components
        self.sphere = Sphere()
        self.solver = WeatherODESolver(physics_constraints=physics_constraints)
        
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
        # Project input to tangent space if needed
        if not self._is_on_sphere(x):
            x = self._project_to_sphere(x)
            
        # Combine state and time
        t = t.view(-1, 1).expand(x.size(0), 1)
        h = torch.cat([x, t], dim=-1)
        
        # Compute velocity in tangent space
        v = self.velocity_net(h)
        
        # Ensure velocity is tangent to sphere
        v = self._project_to_tangent_space(x, v)
        
        return v
    
    def _is_on_sphere(self, x: Tensor, tol: float = 1e-5) -> bool:
        """Check if points lie on the sphere."""
        norms = torch.norm(x, dim=-1)
        return torch.all((norms - self.sphere.radius).abs() < tol)
    
    def _project_to_sphere(self, x: Tensor) -> Tensor:
        """Project points onto the sphere."""
        norms = torch.norm(x, dim=-1, keepdim=True)
        return x * (self.sphere.radius / norms)
    
    def _project_to_tangent_space(self, x: Tensor, v: Tensor) -> Tensor:
        """Project vectors to tangent space of sphere."""
        # Remove radial component
        dot_prod = torch.sum(x * v, dim=-1, keepdim=True) / self.sphere.radius
        v_tangent = v - dot_prod * x / self.sphere.radius
        return v_tangent

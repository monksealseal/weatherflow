"""Geodesic paths on manifolds for weather prediction."""

import torch
from torch import Tensor
from typing import Optional, Tuple
from ..manifolds.sphere import Sphere

class GeodesicPath:
    """Computes geodesic paths between points on a manifold."""
    
    def __init__(self, manifold=None):
        """Initialize with a manifold (defaults to sphere)."""
        self.manifold = manifold or Sphere()
        
    def compute_path(
        self, 
        x0: Tensor, 
        x1: Tensor, 
        num_steps: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """Compute geodesic path between two points.
        
        Args:
            x0: Starting point
            x1: Ending point
            num_steps: Number of steps along the path
            
        Returns:
            Tuple of (path points, parameters)
        """
        # Get the vector from x0 to x1 in the tangent space
        v = self.manifold.log_map(x0, x1)
        
        # Create parameter sequence
        ts = torch.linspace(0, 1, num_steps, device=x0.device)
        
        # Create sequence of points along the geodesic
        points = []
        for t in ts:
            # Scale the tangent vector
            v_t = t * v
            # Map to the manifold
            x_t = self.manifold.exp_map(x0, v_t)
            points.append(x_t)
        
        return torch.stack(points, dim=0), ts
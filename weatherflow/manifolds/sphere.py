
# Copyright (c) 2024 WeatherFlow
# Implementation inspired by Meta's flow matching approach

import torch
from torch import Tensor
import math

class Sphere:
    """Represents the spherical manifold for atmospheric dynamics.
    
    Implements operations on the sphere (S²) for weather modeling.
    """
    
    def __init__(self, radius: float = 6371.0):  # Earth's radius in km
        self.radius = radius
        self.eps = {torch.float32: 1e-4, torch.float64: 1e-7}
    
    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map from tangent space to sphere.
        
        Args:
            x: Points on sphere
            v: Tangent vectors
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_norm = torch.where(v_norm < self.eps[x.dtype], 
                           torch.ones_like(v_norm), v_norm)
        
        cos_theta = torch.cos(v_norm / self.radius)
        sin_theta = torch.sin(v_norm / self.radius)
        
        return cos_theta * x + self.radius * sin_theta * v / v_norm
    
    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map from sphere to tangent space.
        
        Args:
            x: Source points on sphere
            y: Target points on sphere
        """
        dot_prod = torch.sum(x * y, dim=-1, keepdim=True) / self.radius**2
        dot_prod = torch.clamp(dot_prod, -1 + self.eps[x.dtype], 1 - self.eps[x.dtype])
        
        theta = torch.arccos(dot_prod)
        sin_theta = torch.sin(theta)
        
        return self.radius * theta * (y - dot_prod * x) / (sin_theta + self.eps[x.dtype])
    
    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport of tangent vector along geodesic.
        
        Args:
            x: Source point
            y: Target point
            v: Vector to transport
        """
        log_xy = self.log_map(x, y)
        dot_prod = torch.sum(x * y, dim=-1, keepdim=True) / self.radius**2
        theta = torch.arccos(torch.clamp(dot_prod, -1 + self.eps[x.dtype], 1 - self.eps[x.dtype]))
        
        return v - (torch.sum(log_xy * v, dim=-1, keepdim=True) / 
                   (theta**2 * self.radius**2 + self.eps[x.dtype])) * (log_xy + theta**2 * x)

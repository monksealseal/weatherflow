# Copyright (c) 2024 WeatherFlow
# Implementation inspired by Meta's flow matching approach

import torch
from torch import Tensor
import math
from typing import Dict, Union, Optional

class Sphere:
    """Represents the spherical manifold for atmospheric dynamics.
    
    This class implements operations on the sphere (S²) for weather modeling,
    including exponential and logarithmic maps, parallel transport, and other
    geometric operations needed for spherical computations in atmospheric science.
    
    The implementation includes numerical stability improvements to handle
    edge cases and prevent division by zero errors.
    """
    
    def __init__(self, radius: float = 6371.0, epsilon: Optional[float] = None):
        """Initialize the sphere manifold.
        
        Args:
            radius: Radius of the sphere in km (default: Earth's radius)
            epsilon: Custom epsilon value for numerical stability
                    (if None, defaults based on dtype will be used)
        """
        self.radius = radius
        # Type-specific epsilon values for numerical stability
        self._eps_by_dtype = {
            torch.float16: 1e-3,
            torch.float32: 1e-6, 
            torch.float64: 1e-12
        }
        self.custom_epsilon = epsilon
    
    def _get_eps(self, dtype: torch.dtype) -> float:
        """Get appropriate epsilon value for the given dtype.
        
        Args:
            dtype: PyTorch data type
            
        Returns:
            Epsilon value for numerical stability
        """
        if self.custom_epsilon is not None:
            return self.custom_epsilon
        return self._eps_by_dtype.get(dtype, 1e-6)
    
    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map from tangent space to sphere."""

        dtype = x.dtype
        eps = self._get_eps(dtype)
        eps64 = torch.tensor(eps, dtype=torch.float64, device=x.device)
        radius = torch.tensor(self.radius, dtype=torch.float64, device=x.device)

        x64 = x.to(torch.float64)
        v64 = v.to(torch.float64)

        v_norm = torch.linalg.norm(v64, dim=-1, keepdim=True)
        zero_mask = v_norm < eps64
        safe_v_norm = torch.where(zero_mask, torch.ones_like(v_norm), v_norm)

        normalized_v = torch.where(
            zero_mask,
            torch.zeros_like(v64),
            v64 / safe_v_norm,
        )

        cos_theta = torch.cos(v_norm / radius)
        sin_theta = torch.sin(v_norm / radius)
        result = cos_theta * x64 + radius * sin_theta * normalized_v

        result_norm = torch.clamp(
            torch.linalg.norm(result, dim=-1, keepdim=True), min=eps64
        )
        projected = result * (radius / result_norm)
        return projected.to(dtype)

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map from sphere to tangent space."""

        dtype = x.dtype
        eps = self._get_eps(dtype)
        eps64 = torch.tensor(eps, dtype=torch.float64, device=x.device)

        x64 = x.to(torch.float64)
        y64 = y.to(torch.float64)

        radius_sq = torch.sum(x64 * x64, dim=-1, keepdim=True)
        radius_sq = torch.clamp(radius_sq, min=eps64)

        dot_prod = torch.sum(x64 * y64, dim=-1, keepdim=True) / radius_sq
        dot_prod = torch.clamp(dot_prod, -1.0 + eps64, 1.0 - eps64)

        theta = torch.arccos(dot_prod)
        sin_theta = torch.sin(theta)

        scale = torch.where(
            sin_theta.abs() < eps64,
            torch.ones_like(sin_theta),
            theta / (sin_theta + eps64),
        )

        base = y64 - dot_prod * x64
        tangent = base * scale

        correction = torch.sum(tangent * x64, dim=-1, keepdim=True) / radius_sq
        tangent = tangent - correction * x64

        # Perform an additional projection step in double precision to reduce
        # numerical drift before returning the vector.
        for _ in range(2):
            correction = torch.sum(tangent * x64, dim=-1, keepdim=True) / radius_sq
            tangent = tangent - correction * x64

        return tangent
    
    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport of tangent vector along geodesic.
        
        Transports a tangent vector v at point x to the corresponding
        tangent vector at point y along the geodesic connecting x and y.
        
        Args:
            x: Source point on sphere of shape [..., 3]
            y: Target point on sphere of shape [..., 3]
            v: Vector to transport of shape [..., 3]
            
        Returns:
            Transported vector at point y
        """
        eps = self._get_eps(x.dtype)
        
        # Compute the logarithmic map from x to y
        log_xy = self.log_map(x, y)
        
        # Compute the angle between x and y
        dot_prod = torch.sum(x * y, dim=-1, keepdim=True) / (self.radius**2)
        dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
        theta = torch.arccos(dot_prod)
        
        # Compute the inner product between log_xy and v
        inner_prod = torch.sum(log_xy * v, dim=-1, keepdim=True)
        
        # Handle small theta case to avoid division by zero
        safe_denominator = torch.where(
            theta < eps,
            torch.ones_like(theta) * eps,
            theta**2 * self.radius**2 + eps
        )
        
        # Compute the parallel transport
        return v - (inner_prod / safe_denominator) * (log_xy + theta**2 * x)
    
    def geodesic(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """Compute points along the geodesic between x and y.
        
        Args:
            x: Starting point on sphere of shape [..., 3]
            y: Ending point on sphere of shape [..., 3]
            t: Parameter values in [0, 1] for interpolation
            
        Returns:
            Points along the geodesic at times t
        """
        eps = self._get_eps(x.dtype)
        
        # Compute the logarithmic map from x to y
        v = self.log_map(x, y)
        
        # Scale the tangent vector by t
        if t.dim() < v.dim():
            t = t.view(*t.shape, *([1] * (v.dim() - t.dim())))
        
        v_t = t * v
        
        # Apply the exponential map
        return self.exp_map(x, v_t)
    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the geodesic distance between points on the sphere.
        
        Args:
            x: Points on sphere of shape [..., 3]
            y: Points on sphere of shape [..., 3]
            
        Returns:
            Geodesic distances between corresponding points
        """
        eps = self._get_eps(x.dtype)
        
        # Compute the cosine of the angle between x and y
        dot_prod = torch.sum(x * y, dim=-1) / (self.radius**2)
        
        # Clamp to valid range to avoid numerical issues
        dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
        
        # Compute the angle and distance
        theta = torch.arccos(dot_prod)
        return self.radius * theta

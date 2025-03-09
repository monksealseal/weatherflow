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
        """Exponential map from tangent space to sphere.
        
        Maps tangent vectors v at points x on the sphere to
        their corresponding points on the sphere.
        
        Args:
            x: Points on sphere of shape [..., 3]
            v: Tangent vectors of shape [..., 3]
            
        Returns:
            Points on sphere resulting from following v from x
        """
        eps = self._get_eps(x.dtype)
        
        # Compute norm of tangent vectors
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        
        # Handle small norm case to avoid division by zero
        safe_v_norm = torch.where(v_norm < eps, 
                               torch.ones_like(v_norm), 
                               v_norm)
        
        # Compute the exponential map
        cos_theta = torch.cos(v_norm / self.radius)
        sin_theta = torch.sin(v_norm / self.radius)
        
        # Use safe division and handle the case where v_norm is close to zero
        normalized_v = torch.where(
            v_norm < eps,
            torch.zeros_like(v),
            v / safe_v_norm
        )
        
        result = cos_theta * x + self.radius * sin_theta * normalized_v
        
        # Normalize to ensure result stays on the sphere
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        return result / (result_norm + eps)
    
    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map from sphere to tangent space.
        
        Maps points y on the sphere to tangent vectors at points x
        that would reach y via the exponential map.
        
        Args:
            x: Source points on sphere of shape [..., 3]
            y: Target points on sphere of shape [..., 3]
            
        Returns:
            Tangent vectors at x pointing toward y
        """
        eps = self._get_eps(x.dtype)
        
        # Compute the cosine of the angle between x and y
        dot_prod = torch.sum(x * y, dim=-1, keepdim=True) / (self.radius**2)
        
        # Clamp to valid range to avoid numerical issues
        dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
        
        # Compute the angle between x and y
        theta = torch.arccos(dot_prod)
        sin_theta = torch.sin(theta)
        
        # Handle small sin_theta case to avoid division by zero
        safe_factor = torch.where(
            sin_theta < eps,
            torch.ones_like(sin_theta) / eps,  # Limit for small angles
            1.0 / (sin_theta + eps)
        )
        
        # Compute the logarithmic map
        return self.radius * theta * (y - dot_prod * x) * safe_factor
    
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

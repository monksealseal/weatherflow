import torch
import numpy as np
from typing import Callable, Optional, Tuple
from .prob_path import ProbPath

class GaussianProbPath(ProbPath):
    """Gaussian conditional probability path as described in MIT lecture notes (p.14, Example 9).
    
    Implements the path: p_t(·|z) = N(α_t z, β_t^2 Id) where α_t, β_t are noise schedulers.
    """
    
    def __init__(
        self, 
        alpha_schedule: Callable[[torch.Tensor], torch.Tensor],
        beta_schedule: Callable[[torch.Tensor], torch.Tensor]
    ):
        """Initialize Gaussian probability path.
        
        Args:
            alpha_schedule: Function that maps time t to α_t
            beta_schedule: Function that maps time t to β_t
        """
        super().__init__()
        self.alpha_schedule = alpha_schedule
        self.beta_schedule = beta_schedule
    
    def sample_conditional(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from conditional path p_t(·|z).
        
        Args:
            z: Data point (ground truth)
            t: Time value in [0, 1]
            
        Returns:
            Sample from p_t(·|z)
        """
        # Reshape t for proper broadcasting
        alpha_t = self.alpha_schedule(t)
        beta_t = self.beta_schedule(t)
        
        # Reshape for broadcasting to match z dimensions
        if len(z.shape) > 1:
            alpha_t = alpha_t.view(-1, *([1] * (len(z.shape) - 1)))
            beta_t = beta_t.view(-1, *([1] * (len(z.shape) - 1)))
        
        # Sample from p_t(·|z) = N(α_t z, β_t^2 Id) as in Eq. 16
        epsilon = torch.randn_like(z)
        return alpha_t * z + beta_t * epsilon
    
    def get_conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional score ∇log p_t(x|z) as in Eq. 28.
        
        Args:
            x: Input point
            z: Data point (ground truth)
            t: Time value in [0, 1]
            
        Returns:
            Conditional score vector
        """
        alpha_t = self.alpha_schedule(t)
        beta_t = self.beta_schedule(t)
        
        # Reshape for broadcasting
        if len(x.shape) > 1:
            alpha_t = alpha_t.view(-1, *([1] * (len(x.shape) - 1)))
            beta_t = beta_t.view(-1, *([1] * (len(x.shape) - 1)))
        
        # Score function for Gaussian is -(x - mean)/variance (p.19, Example 14)
        return -(x - alpha_t * z) / (beta_t**2)
    
    def get_conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional vector field u_t(x|z) as in Eq. 21.
        
        Args:
            x: Input point
            z: Data point (ground truth)
            t: Time value in [0, 1]
            
        Returns:
            Conditional vector field
        """
        alpha_t = self.alpha_schedule(t)
        beta_t = self.beta_schedule(t)
        
        # Compute time derivatives
        with torch.enable_grad():
            t_param = t.detach().requires_grad_(True)
            alpha_t_param = self.alpha_schedule(t_param)
            beta_t_param = self.beta_schedule(t_param)
            
            alpha_dot = torch.autograd.grad(alpha_t_param, t_param, 
                                           torch.ones_like(alpha_t_param))[0]
            beta_dot = torch.autograd.grad(beta_t_param, t_param,
                                          torch.ones_like(beta_t_param))[0]
        
        # Reshape for broadcasting
        if len(x.shape) > 1:
            alpha_t = alpha_t.view(-1, *([1] * (len(x.shape) - 1)))
            beta_t = beta_t.view(-1, *([1] * (len(x.shape) - 1)))
            alpha_dot = alpha_dot.view(-1, *([1] * (len(x.shape) - 1)))
            beta_dot = beta_dot.view(-1, *([1] * (len(x.shape) - 1)))
        
        # Construct vector field from Eq. 21
        return (alpha_dot - (beta_dot / beta_t) * alpha_t) * z + (beta_dot / beta_t) * x
    
    # Implementation of abstract methods from ProbPath
    def sample_path(self, x_1: torch.Tensor, x_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a path between weather states.
        
        Args:
            x_1: Target weather state
            x_0: Optional source state (if None, sampled from prior)
            
        Returns:
            Tuple of (path_sample, time_points)
        """
        batch_size = x_1.size(0)
        device = x_1.device
        
        # Sample time points
        t = torch.rand(batch_size, device=device)
        
        # If x_0 is not provided, sample from a standard normal prior
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        
        # For our test, we'll just interpolate between x_0 and x_1
        x_t = x_0 + t.view(-1, 1, 1, 1) * (x_1 - x_0)
        
        return x_t, t
    
    def get_flow_vector(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get the flow vector field at a given point and time.
        
        Args:
            x: Current weather state
            t: Time point
            
        Returns:
            Flow vector indicating state evolution
        """
        # Simple drift for testing
        return -x

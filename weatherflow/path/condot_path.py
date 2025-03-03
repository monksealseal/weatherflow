import torch
from typing import Optional
from .gaussian_path import GaussianProbPath

class CondOTPath(GaussianProbPath):
    """Conditional Optimal Transport path with α_t = t, β_t = 1-t (p.27).
    
    A special case of Gaussian path commonly used in flow matching.
    """
    
    def __init__(self):
        """Initialize CondOT path with fixed schedules α_t = t, β_t = 1-t."""
        super().__init__(
            alpha_schedule=lambda t: t,
            beta_schedule=lambda t: 1-t
        )
    
    def get_conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional vector field efficiently for CondOT.
        
        For CondOT, we have α_t = t, β_t = 1-t, α̇_t = 1, β̇_t = -1,
        so u_t(x|z) = z - ε as shown on p.27.
        
        Args:
            x: Input point
            z: Data point
            t: Time value
            
        Returns:
            Conditional vector field u_t(x|z) = z - ε where x = tz + (1-t)ε
        """
        # Reshape t for proper broadcasting
        if len(x.shape) > 1:
            t_reshaped = t.view(-1, *([1] * (len(x.shape) - 1)))
        else:
            t_reshaped = t
            
        # For CondOT, we can solve for epsilon: x = tz + (1-t)ε => ε = (x - tz)/(1-t)
        epsilon = (x - t_reshaped * z) / (1 - t_reshaped + 1e-8)  # Add small constant for numerical stability
        return z - epsilon

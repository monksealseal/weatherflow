import torch
from typing import Callable, Optional
from ..path import GaussianProbPath

def vector_field_to_score(
    vector_field_fn: Callable,
    path: GaussianProbPath,
    x: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """Convert vector field to score function using Proposition 1 (p.30, Eq. 55).
    
    Args:
        vector_field_fn: Vector field function
        path: Gaussian probability path
        x: Input point
        t: Time value
        
    Returns:
        Score function
    """
    alpha_t = path.alpha_schedule(t)
    beta_t = path.beta_schedule(t)
    
    # Compute derivatives
    with torch.enable_grad():
        t_param = t.detach().requires_grad_(True)
        alpha_t_param = path.alpha_schedule(t_param)
        
        alpha_dot = torch.autograd.grad(alpha_t_param, t_param, 
                                       torch.ones_like(alpha_t_param))[0]
    
    # Apply conversion formula from p.30, Eq. 55
    u_t = vector_field_fn(x, t)
    
    # Get denominator from p.30
    with torch.enable_grad():
        t_param = t.detach().requires_grad_(True)
        alpha_t_param = path.alpha_schedule(t_param)
        beta_t_param = path.beta_schedule(t_param)
        
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
    
    denominator = beta_t**2 * alpha_dot - alpha_t * beta_dot * beta_t
    
    # Check that denominator isn't too close to zero
    if torch.any(torch.abs(denominator) < 1e-6):
        raise ValueError("Denominator too close to zero in conversion formula")
    
    return (alpha_t * u_t - alpha_dot * x) / denominator
def score_to_vector_field(
    score_fn: Callable,
    path: GaussianProbPath,
    x: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """Convert score function to vector field using Proposition 1 (p.30, Eq. 54).
    
    Args:
        score_fn: Score function
        path: Gaussian probability path
        x: Input point
        t: Time value
        
    Returns:
        Vector field
    """
    alpha_t = path.alpha_schedule(t)
    beta_t = path.beta_schedule(t)
    
    # Compute derivatives
    with torch.enable_grad():
        t_param = t.detach().requires_grad_(True)
        alpha_t_param = path.alpha_schedule(t_param)
        beta_t_param = path.beta_schedule(t_param)
        
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
    
    # Apply conversion formula from p.30, Eq. 54
    coef = (beta_t**2 * alpha_dot/alpha_t - beta_dot*beta_t)
    score = score_fn(x, t)
    return coef * score + (alpha_dot/alpha_t) * x

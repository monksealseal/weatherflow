import torch
from typing import Callable, Optional, Tuple

def langevin_dynamics(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int = 1000,
    step_size: float = 1e-2,
    sigma: float = 0.01,
    clip_grad: bool = True,
    clip_norm: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """Implement Langevin dynamics sampling as described in Remark 16 (p.21).
    
    Args:
        score_fn: Function computing score ∇log p(x)
        x0: Initial state
        n_steps: Number of steps
        step_size: Step size for updates
        sigma: Noise scale
        clip_grad: Whether to clip gradients
        clip_norm: Maximum norm for gradient clipping
        device: Device to use
        
    Returns:
        Final sample
    """
    device = device or x0.device
    x = x0.clone().to(device)
    
    for i in range(n_steps):
        # Compute score
        score = score_fn(x.detach(), torch.tensor([1.0], device=device))
        
        # Clip gradients if needed
        if clip_grad:
            score_norm = torch.norm(score.view(score.size(0), -1), dim=1).mean()
            if score_norm > clip_norm:
                score = score * (clip_norm / score_norm)
        
        # Langevin update as in Eq. 31
        noise = torch.randn_like(x) * sigma
        x = x + step_size * score + torch.sqrt(torch.tensor(2.0) * step_size) * noise
    
    return x

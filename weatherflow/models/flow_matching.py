
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional

class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x

class ConvNextBlock(nn.Module):
    """ConvNext block for spatial processing."""
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + residual

class WeatherFlowMatch(nn.Module):
    """Flow matching model for weather prediction."""
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.main = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        t = t.reshape(-1, 1).float()
        h = torch.cat([x_flat, t], dim=1)
        return self.main(h).reshape(x.shape)

# Keep existing PhysicsGuidedAttention class

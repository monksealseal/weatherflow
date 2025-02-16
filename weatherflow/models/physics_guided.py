
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding using sinusoidal functions as in the Transformer."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PhysicsGuidedLayer(nn.Module):
    """Physics-informed neural network layer."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Physics-informed network
        self.physics_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.physics_net(x)
        return x + residual

class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing different spatial scales."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention at different scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(3)  # 3 different scales
        ])
        
        # Scale-specific convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2**i, padding=2**(i-1))
            for i in range(3)
        ])
        
        self.output_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        outputs = []
        
        # Process each scale
        for attention, conv in zip(self.attention_layers, self.scale_convs):
            # Reshape for convolution
            h = x.view(batch_size, int(np.sqrt(seq_len)), int(np.sqrt(seq_len)), hidden_dim)
            h = h.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Apply scale-specific convolution
            h = conv(h)
            
            # Reshape for attention
            h = h.permute(0, 2, 3, 1).reshape(batch_size, -1, hidden_dim)
            
            # Apply attention
            h, _ = attention(h, h, h)
            outputs.append(h)
        
        # Combine multi-scale features
        combined = torch.cat(outputs, dim=-1)
        output = self.output_projection(combined)
        output = self.dropout(output)
        
        return self.layer_norm(x + output)

class PhysicsGuidedAttention(nn.Module):
    """Enhanced physics-guided attention model for weather prediction."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (32, 64)  # lat, lon
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # Multi-scale attention layers
        self.attention_layers = nn.ModuleList([
            MultiScaleAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Physics-guided layers
        self.physics_layers = nn.ModuleList([
            PhysicsGuidedLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_channels)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Reshape input: [B, C, H, W] -> [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.input_channels)
        
        # Input embedding
        h = self.input_embedding(x)
        
        # Add time embedding if provided
        if timestep is not None:
            time_emb = self.time_embedding(timestep)
            h = h + time_emb.unsqueeze(1)
        
        # Apply attention and physics layers
        for attn, physics in zip(self.attention_layers, self.physics_layers):
            h = attn(h)
            h = physics(h)
        
        # Final layer norm
        h = self.layer_norm(h)
        
        # Output projection
        output = self.output_projection(h)
        
        # Reshape output: [B, H*W, C] -> [B, C, H, W]
        output = output.reshape(batch_size, *self.grid_size, -1).permute(0, 3, 1, 2)
        
        return output

    def compute_physics_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss terms."""
        # Gradient-based physics constraints
        dx = torch.gradient(pred, dim=3)[0]  # Longitude gradient
        dy = torch.gradient(pred, dim=2)[0]  # Latitude gradient
        
        # Conservation of mass (divergence should be small)
        div = dx[..., 0] + dy[..., 1]  # For u,v components
        mass_loss = torch.mean(div ** 2)
        
        # Energy conservation
        energy_pred = torch.sum(pred ** 2, dim=(1, 2, 3))
        energy_target = torch.sum(target ** 2, dim=(1, 2, 3))
        energy_loss = torch.mean((energy_pred - energy_target) ** 2)
        
        return mass_loss + 0.1 * energy_loss

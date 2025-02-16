
import torch
import torch.nn as nn
from ..physics import PhysicsGuidedLayer

class EnhancedPhysicsGuidedAttention(nn.Module):
    """Enhanced version of our successful physics-guided attention model."""
    
    def __init__(self, input_channels, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.physics_layers = nn.ModuleList([
            PhysicsGuidedLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_channels)
        
    def forward(self, x, lat=None):
        h = self.input_proj(x)
        for attn, physics in zip(self.attention_layers, self.physics_layers):
            attn_out, _ = attn(h, h, h)
            h = h + attn_out
            physics_out = physics(h, lat)
            h = h + physics_out
        return self.output_proj(h)

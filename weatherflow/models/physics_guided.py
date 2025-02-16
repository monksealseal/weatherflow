import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGuidedAttention(nn.Module):
    def __init__(self, channels=1, hidden_dim=256, n_heads=8):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv2d(channels, hidden_dim, 1)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.1)
        self.coriolis_scale = nn.Parameter(torch.ones(1))
        self.density_scale = nn.Parameter(torch.ones(1))
        self.output_proj = nn.Conv2d(hidden_dim, channels, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = (x - 54222.0) / 1400.0
        h = self.input_proj(x)
        h = h.flatten(2).permute(2, 0, 1)
        h = self.norm1(h)
        h_attn, _ = self.attention(h, h, h)
        h = h + h_attn
        h = self.norm2(h)
        h = h.permute(1, 2, 0).view(B, -1, H, W)
        x = self.output_proj(h)
        x = x * 1400.0 + 54222.0
        return x

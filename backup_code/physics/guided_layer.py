
import torch
import torch.nn as nn
from .atmospheric import AtmosphericPhysics

class PhysicsGuidedLayer(nn.Module):
    """Physics-guided neural network layer from our experiments."""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.physics = AtmosphericPhysics()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.physics_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, lat=None):
        features = self.encoder(x)
        if lat is not None:
            if hasattr(x, 'temperature') and hasattr(x, 'pressure'):
                stability = self.physics.static_stability(x.temperature, x.pressure)
                features = features * torch.sigmoid(stability)
            if hasattr(x, 'u') and hasattr(x, 'v'):
                vorticity = self.physics.vorticity(x.u, x.v, lat)
                features = features + vorticity
        return self.physics_projection(features)


import torch
import torch.nn as nn
from .base import BaseWeatherModel

class ProbabilisticEncoder(nn.Module):
    """Encodes weather data into probabilistic latent space."""
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # For mean and variance
        )
        
    def forward(self, x):
        h = self.net(x)
        return torch.chunk(h, 2, dim=-1)  # Split into mean and variance

class NormalizingFlow(nn.Module):
    """Normalizing flow for flexible probability distributions."""
    def __init__(self, latent_dim):
        super().__init__()
        self.flows = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(4)
        ])
    
    def forward(self, z_mu, z_var):
        z = z_mu + torch.randn_like(z_var) * torch.sqrt(z_var)
        for flow in self.flows:
            z = flow(z)
        return z

class StochasticFlowModel(BaseWeatherModel):
    """Stochastic flow model for uncertainty quantification."""
    def __init__(self, input_channels, latent_dim=32):
        super().__init__()
        self.encoder = ProbabilisticEncoder(input_channels, latent_dim)
        self.flow = NormalizingFlow(latent_dim)
        self.decoder = nn.Linear(latent_dim, input_channels)
    
    def forward(self, x, num_samples=10):
        z_mu, z_var = self.encoder(x)
        predictions = []
        
        for _ in range(num_samples):
            z = self.flow(z_mu, z_var)
            pred = self.decoder(z)
            predictions.append(pred)
            
        return torch.stack(predictions)

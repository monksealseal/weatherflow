
import torch.nn as nn

class BaseWeatherModel(nn.Module):
    """Base class for weather prediction models."""
    def __init__(self):
        super().__init__()
        
    def compute_physics_loss(self, pred, target):
        """Compute physics-informed loss."""
        mass_conservation_loss = self.mass_conservation_constraint(pred)
        energy_conservation_loss = self.energy_conservation_constraint(pred)
        return mass_conservation_loss + energy_conservation_loss

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, List, Optional, Any

class BaseWeatherModel(nn.Module, ABC):
    """Base class for weather prediction models.
    
    This abstract class defines the interface for weather prediction models
    with physics-informed constraints. All weather models should inherit from
    this class and implement the required methods.
    """
    def __init__(self):
        """Initialize the base weather model."""
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor representing weather state
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Predicted weather state
        """
        pass
    
    def compute_physics_loss(self, 
                            pred: torch.Tensor, 
                            target: Optional[torch.Tensor] = None,
                            weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Compute physics-informed loss.
        
        Args:
            pred: Predicted weather state
            target: Optional target weather state for reference
            weights: Optional dictionary of weights for different physics constraints
            
        Returns:
            Combined physics-based loss
        """
        # Default weights if none provided
        if weights is None:
            weights = {'mass': 1.0, 'energy': 1.0}
            
        # Compute individual constraint losses
        mass_conservation_loss = self.mass_conservation_constraint(pred) * weights.get('mass', 1.0)
        energy_conservation_loss = self.energy_conservation_constraint(pred) * weights.get('energy', 1.0)
        
        # Sum all physics-based losses
        return mass_conservation_loss + energy_conservation_loss
    
    @abstractmethod
    def mass_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mass conservation constraint loss.
        
        This method should implement the specific mass conservation
        physics constraint for the model.
        
        Args:
            x: Weather state tensor
            
        Returns:
            Loss value representing violation of mass conservation
        """
        raise NotImplementedError("Subclasses must implement mass_conservation_constraint")
    
    @abstractmethod
    def energy_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy conservation constraint loss.
        
        This method should implement the specific energy conservation
        physics constraint for the model.
        
        Args:
            x: Weather state tensor
            
        Returns:
            Loss value representing violation of energy conservation
        """
        raise NotImplementedError("Subclasses must implement energy_conservation_constraint")
    
    def configure_optimizer(self, 
                           lr: float = 1e-4, 
                           weight_decay: float = 1e-5) -> torch.optim.Optimizer:
        """Configure the optimizer for the model.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

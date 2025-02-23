
# Copyright (c) 2024 WeatherFlow
# Implementation inspired by Meta's flow matching approach

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Optional, Tuple, Callable

class ProbPath(ABC):
    """Abstract base class for probability paths in weather prediction.
    
    Implements continuous-time flow matching for weather system evolution.
    """
    
    @abstractmethod
    def sample_path(self, x_1: Tensor, x_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Sample a path between weather states.
        
        Args:
            x_1: Target weather state
            x_0: Optional source state (if None, sampled from prior)
            
        Returns:
            Tuple of (path_sample, time_points)
        """
        pass

    @abstractmethod
    def get_flow_vector(self, x: Tensor, t: Tensor) -> Tensor:
        """Get the flow vector field at a given point and time.
        
        Args:
            x: Current weather state
            t: Time point
            
        Returns:
            Flow vector indicating state evolution
        """
        pass


import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
import numpy as np
from ..models import PhysicsGuidedAttention
from ..utils import WeatherVisualizer
import matplotlib.pyplot as plt

class WeatherEnsemble:
    """Advanced ensemble methods from our successful experiments."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.models = [model.to(device) for model in models]
        self.weights = weights or [1/len(models)] * len(models)
        self.device = device
        self.visualizer = WeatherVisualizer()
    
    def predict(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Generate ensemble prediction with uncertainty estimates."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(self.device))
                predictions.append(pred.cpu())
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Weighted mean
        weights = torch.tensor(self.weights).view(-1, 1, 1, 1)
        mean = torch.sum(predictions * weights, dim=0)
        
        # Uncertainty estimates
        std = torch.std(predictions, dim=0)
        
        # Confidence intervals
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        result = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        if return_individual:
            result['individual_predictions'] = predictions
            
        return result
    
    def plot_ensemble_prediction(
        self,
        x: torch.Tensor,
        true_value: Optional[torch.Tensor] = None,
        variable_idx: int = 0
    ):
        """Visualize ensemble prediction with uncertainty."""
        prediction = self.predict(x, return_individual=True)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean prediction
        self.visualizer.plot_weather_field(
            prediction['mean'][variable_idx],
            title="Ensemble Mean",
            ax=axs[0, 0]
        )
        
        # Uncertainty (std)
        self.visualizer.plot_weather_field(
            prediction['std'][variable_idx],
            title="Prediction Uncertainty (Std)",
            ax=axs[0, 1],
            cmap='viridis'
        )
        
        # Individual predictions
        axs[1, 0].boxplot(
            prediction['individual_predictions'][:, variable_idx].numpy().reshape(-1)
        )
        axs[1, 0].set_title("Distribution of Predictions")
        
        # Error if true value provided
        if true_value is not None:
            error = (prediction['mean'] - true_value)[variable_idx]
            self.visualizer.plot_weather_field(
                error,
                title="Prediction Error",
                ax=axs[1, 1],
                cmap='RdBu_r'
            )
        
        plt.tight_layout()
        return fig, axs

class DiverseEnsemble(WeatherEnsemble):
    """Ensemble with diverse model architectures."""
    
    @classmethod
    def create(
        cls,
        input_channels: int,
        n_models: int = 5,
        base_config: Dict = None
    ):
        """Create ensemble with varied architectures."""
        base_config = base_config or {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8
        }
        
        models = []
        for i in range(n_models):
            # Vary architecture
            config = base_config.copy()
            config['hidden_dim'] += i * 32
            config['num_layers'] += i % 2
            config['num_heads'] += i % 4
            
            model = PhysicsGuidedAttention(
                input_channels=input_channels,
                **config
            )
            models.append(model)
        
        return cls(models)

class BaggingEnsemble(WeatherEnsemble):
    """Ensemble trained with bootstrapped data."""
    
    def __init__(
        self,
        base_model: nn.Module,
        n_models: int = 5,
        sample_size: float = 0.8
    ):
        self.base_model = base_model
        self.n_models = n_models
        self.sample_size = sample_size
        super().__init__([])
    
    def fit(
        self,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        **training_kwargs
    ):
        """Train ensemble members on bootstrap samples."""
        from torch.utils.data import DataLoader, SubsetRandomSampler
        
        self.models = []
        dataset_size = len(train_data)
        sample_size = int(dataset_size * self.sample_size)
        
        for i in range(self.n_models):
            # Create bootstrap sample
            indices = np.random.choice(
                dataset_size, size=sample_size, replace=True
            )
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(
                train_data, sampler=sampler,
                batch_size=training_kwargs.get('batch_size', 32)
            )
            
            # Initialize and train model
            model = self.base_model.__class__(
                *self.base_model.args,
                **self.base_model.kwargs
            )
            
            # Train model (implement your training logic here)
            self._train_model(
                model, train_loader, val_data, **training_kwargs
            )
            
            self.models.append(model)
        
        return self

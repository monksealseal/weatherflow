
import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from ..utils import WeatherVisualizer

class SpatialUncertaintyAnalyzer:
    """Spatial uncertainty analysis from our experiments."""
    
    def __init__(self):
        self.visualizer = WeatherVisualizer()
    
    def analyze_spatial_patterns(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lat_grid: Optional[torch.Tensor] = None,
        lon_grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Analyze spatial patterns in prediction uncertainty."""
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        error = torch.abs(mean_pred - targets)
        
        # Spatial correlation
        spatial_corr = torch.corrcoef(
            std_pred.reshape(-1),
            error.reshape(-1)
        )[0, 1]
        
        # Latitude dependence
        if lat_grid is not None:
            error_by_lat = error.mean(dim=-1)  # Average over longitude
            uncertainty_by_lat = std_pred.mean(dim=-1)
        else:
            error_by_lat = None
            uncertainty_by_lat = None
        
        return {
            'error': error,
            'uncertainty': std_pred,
            'spatial_correlation': spatial_corr,
            'error_by_latitude': error_by_lat,
            'uncertainty_by_latitude': uncertainty_by_lat
        }
    
    def plot_spatial_analysis(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lat_grid: Optional[torch.Tensor] = None,
        lon_grid: Optional[torch.Tensor] = None,
        variable_idx: int = 0
    ):
        """Visualize spatial uncertainty analysis."""
        analysis = self.analyze_spatial_patterns(
            predictions, targets, lat_grid, lon_grid
        )
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid for subplots
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Error map
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(analysis['error'][variable_idx], cmap='viridis')
        ax1.set_title('Prediction Error')
        plt.colorbar(im1, ax=ax1)
        
        # Uncertainty map
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(analysis['uncertainty'][variable_idx], cmap='viridis')
        ax2.set_title('Prediction Uncertainty')
        plt.colorbar(im2, ax=ax2)
        
        # Error vs Uncertainty scatter
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(
            analysis['uncertainty'][variable_idx].numpy().flatten(),
            analysis['error'][variable_idx].numpy().flatten(),
            alpha=0.1
        )
        ax3.set_xlabel('Uncertainty')
        ax3.set_ylabel('Error')
        ax3.set_title(f'Correlation: {analysis["spatial_correlation"]:.3f}')
        
        # Latitude dependence
        if lat_grid is not None:
            ax4 = fig.add_subplot(gs[1, :])
            ax4.plot(
                lat_grid,
                analysis['error_by_latitude'][variable_idx],
                'b-', label='Error'
            )
            ax4.plot(
                lat_grid,
                analysis['uncertainty_by_latitude'][variable_idx],
                'r-', label='Uncertainty'
            )
            ax4.set_xlabel('Latitude')
            ax4.set_title('Latitude Dependence')
            ax4.legend()
        
        plt.tight_layout()
        return fig

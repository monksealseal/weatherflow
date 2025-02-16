import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class WeatherVisualizer:
    """Visualization tools for weather predictions"""
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def plot_prediction_comparison(self, true_state: Dict, pred_state: Dict,
                                 variables: Optional[List[str]] = None):
        """Plot comparison between true and predicted states"""
        if variables is None:
            variables = list(true_state.keys())
            
        n_vars = len(variables)
        fig, axes = plt.subplots(2, n_vars, figsize=self.figsize)
        
        for i, var in enumerate(variables):
            # True state
            im1 = axes[0, i].imshow(true_state[var], cmap='RdBu_r')
            plt.colorbar(im1, ax=axes[0, i])
            axes[0, i].set_title(f'True {var}')
            
            # Predicted state
            im2 = axes[1, i].imshow(pred_state[var], cmap='RdBu_r')
            plt.colorbar(im2, ax=axes[1, i])
            axes[1, i].set_title(f'Predicted {var}')
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, true_state: Dict, pred_state: Dict,
                              variables: Optional[List[str]] = None):
        """Plot error distribution for each variable"""
        if variables is None:
            variables = list(true_state.keys())
            
        fig, axes = plt.subplots(1, len(variables), figsize=self.figsize)
        if len(variables) == 1:
            axes = [axes]
            
        for ax, var in zip(axes, variables):
            error = pred_state[var] - true_state[var]
            ax.hist(error.flatten(), bins=50, density=True)
            ax.set_title(f'{var} Error Distribution')
            ax.set_xlabel('Error')
            ax.set_ylabel('Density')
            
        plt.tight_layout()
        return fig
    
    def plot_global_forecast(self, data: Dict, projection='PlateCarree'):
        """Plot global weather forecast with proper projection"""
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=getattr(ccrs, projection)())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Plot data
        for var, values in data.items():
            if var == 'temperature':
                plt.contourf(values, cmap='RdBu_r', transform=ccrs.PlateCarree())
            elif var in ['wind_u', 'wind_v']:
                plt.quiver(values, transform=ccrs.PlateCarree())
            else:
                plt.contour(values, transform=ccrs.PlateCarree())
                
        plt.colorbar()
        return fig
    
    def plot_time_series(self, predictions: List[Dict], times: List[float],
                        variable: str, lat_idx: int, lon_idx: int):
        """Plot time series for a specific location"""
        values = [pred[variable][lat_idx, lon_idx] for pred in predictions]
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, values, '-o')
        plt.xlabel('Time (hours)')
        plt.ylabel(variable)
        plt.title(f'{variable} Time Series at ({lat_idx}, {lon_idx})')
        plt.grid(True)
        
    def create_animation(self, predictions: List[Dict], variable: str,
                        interval: int = 200):
        """Create animation of prediction sequence"""
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=self.figsize)
        data = predictions[0][variable]
        im = ax.imshow(data, cmap='RdBu_r', animated=True)
        plt.colorbar(im)
        
        def update(frame):
            data = predictions[frame][variable]
            im.set_array(data)
            return [im]
            
        anim = FuncAnimation(fig, update, frames=len(predictions),
                           interval=interval, blit=True)
        return anim

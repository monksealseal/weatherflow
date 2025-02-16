
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

class WeatherVisualizer:
    """Visualization tools for weather predictions."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def plot_prediction_comparison(self, true_state, pred_state):
        """Plot comparison between true and predicted states."""
        fig, axes = plt.subplots(1, len(true_state), figsize=self.figsize)
        if len(true_state) == 1:
            axes = [axes]
            
        for ax, (var_name, true_var) in zip(axes, true_state.items()):
            pred_var = pred_state[var_name]
            diff = pred_var - true_var
            
            # Create three panel plot
            img = ax.imshow(diff, cmap='RdBu_r')
            plt.colorbar(img, ax=ax)
            ax.set_title(f'{var_name} Difference')
            
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, true_state, pred_state):
        """Plot error distribution for predictions."""
        fig, axes = plt.subplots(1, len(true_state), figsize=self.figsize)
        if len(true_state) == 1:
            axes = [axes]
            
        for ax, (var_name, true_var) in zip(axes, true_state.items()):
            pred_var = pred_state[var_name]
            errors = (pred_var - true_var).flatten()
            
            ax.hist(errors, bins=50, density=True)
            ax.set_title(f'{var_name} Error Distribution')
            
        plt.tight_layout()
        return fig
    
    def plot_global_forecast(self, forecast_state):
        """Plot global forecast on a map."""
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        for var_name, var_data in forecast_state.items():
            ax.contourf(var_data, transform=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines()
            
        plt.title('Global Forecast')
        return fig


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.animation import FuncAnimation
import xarray as xr

class WeatherVisualizer:
    """Advanced weather visualization tools from our successful experiments."""
    
    def __init__(self, projection='PlateCarree'):
        self.projection = getattr(ccrs, projection)()
        self.fig_size = (15, 10)
        
    def plot_weather_field(self, data, title=None, cmap='RdBu_r', 
                          add_colorbar=True, levels=20):
        """Plot a weather field with proper geographical context."""
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection=self.projection)
        
        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines()
        
        # Plot data
        if isinstance(data, xr.DataArray):
            lons, lats = np.meshgrid(data.longitude, data.latitude)
            data = data.values
        else:
            lons, lats = np.meshgrid(
                np.linspace(-180, 180, data.shape[1]),
                np.linspace(-90, 90, data.shape[0])
            )
        
        cs = ax.contourf(lons, lats, data, levels=levels,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap)
        
        if add_colorbar:
            plt.colorbar(cs, ax=ax, orientation='horizontal',
                        pad=0.05, label=title)
        
        if title:
            ax.set_title(title)
        
        return fig, ax
    
    def create_weather_animation(self, data_sequence, title=None, 
                               interval=200, save_path=None):
        """Create animation of weather evolution."""
        fig, ax = plt.subplots(figsize=self.fig_size,
                              subplot_kw={'projection': self.projection})
        
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines()
        
        # First frame for initialization
        data = data_sequence[0]
        if isinstance(data, xr.DataArray):
            lons, lats = np.meshgrid(data.longitude, data.latitude)
            data = data.values
        else:
            lons, lats = np.meshgrid(
                np.linspace(-180, 180, data.shape[1]),
                np.linspace(-90, 90, data.shape[0])
            )
        
        cs = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree())
        plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05)
        
        def update(frame):
            ax.clear()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines()
            
            if isinstance(frame, xr.DataArray):
                frame = frame.values
            cs = ax.contourf(lons, lats, frame,
                           transform=ccrs.PlateCarree())
            if title:
                ax.set_title(f"{title} - Frame {frame_idx}")
            return cs,
        
        anim = FuncAnimation(fig, update, frames=data_sequence,
                           interval=interval, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        return anim
    
    def plot_comparison(self, true, predicted, difference=True,
                       title="Prediction Comparison"):
        """Plot true vs predicted fields with optional difference."""
        if difference:
            fig, axs = plt.subplots(1, 3, figsize=(20, 6),
                                   subplot_kw={'projection': self.projection})
            diff = predicted - true
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 6),
                                   subplot_kw={'projection': self.projection})
        
        # True field
        self._plot_on_axis(axs[0], true, "True")
        
        # Predicted field
        self._plot_on_axis(axs[1], predicted, "Predicted")
        
        # Difference field if requested
        if difference:
            self._plot_on_axis(axs[2], diff, "Difference",
                              cmap='RdBu_r', center_zero=True)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig, axs
    
    def _plot_on_axis(self, ax, data, title, cmap='viridis',
                      center_zero=False):
        """Helper function for plotting on a specific axis."""
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines()
        
        if isinstance(data, xr.DataArray):
            lons, lats = np.meshgrid(data.longitude, data.latitude)
            data = data.values
        else:
            lons, lats = np.meshgrid(
                np.linspace(-180, 180, data.shape[1]),
                np.linspace(-90, 90, data.shape[0])
            )
        
        if center_zero:
            vmax = max(abs(data.min()), abs(data.max()))
            vmin = -vmax
        else:
            vmin, vmax = data.min(), data.max()
        
        cs = ax.contourf(lons, lats, data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05)
        ax.set_title(title)

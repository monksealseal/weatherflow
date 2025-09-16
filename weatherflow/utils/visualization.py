import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Union, Tuple, Any
import io
from PIL import Image
import base64
import xarray as xr

class WeatherVisualizer:
    """Comprehensive visualization tools for weather prediction.
    
    This class provides a variety of visualization methods for weather data,
    including:
    - Global maps with proper projections
    - Comparison plots between predicted and true values
    - Error distributions and metrics
    - Animations of weather evolution
    - Flow field visualizations
    """
    
    VAR_LABELS = {
        'temperature': 'Temperature (K)',
        'geopotential': 'Geopotential (m²/s²)',
        'u_component_of_wind': 'U-Wind (m/s)',
        'v_component_of_wind': 'V-Wind (m/s)',
        'specific_humidity': 'Specific Humidity (kg/kg)',
        'wind_speed': 'Wind Speed (m/s)',
        'vorticity': 'Vorticity (1/s)',
        't': 'Temperature (K)',
        'z': 'Geopotential (m²/s²)',
        'u': 'U-Wind (m/s)',
        'v': 'V-Wind (m/s)',
        'q': 'Specific Humidity (kg/kg)'
    }
    
    VAR_CMAPS = {
        'temperature': 'RdBu_r',
        'geopotential': 'viridis',
        'u_component_of_wind': 'RdBu_r',
        'v_component_of_wind': 'RdBu_r',
        'specific_humidity': 'Blues',
        'wind_speed': 'YlOrRd',
        'vorticity': 'RdBu_r',
        't': 'RdBu_r',
        'z': 'viridis',
        'u': 'RdBu_r',
        'v': 'RdBu_r',
        'q': 'Blues',
        'error': 'RdBu_r',
        'difference': 'RdBu_r'
    }
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 10),
        projection: str = 'PlateCarree',
        save_dir: Optional[str] = None
    ):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
            projection: Default cartopy projection
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.projection = getattr(ccrs, projection)()
        self.save_dir = save_dir
    
    def _get_latlons(
        self, 
        data: Union[np.ndarray, torch.Tensor, xr.DataArray],
        lat_range: Tuple[float, float] = (-90, 90),
        lon_range: Tuple[float, float] = (-180, 180)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude grids for the data."""
        # Handle different data types
        if isinstance(data, xr.DataArray):
            if 'latitude' in data.coords and 'longitude' in data.coords:
                lats = data.latitude.values
                lons = data.longitude.values
                return np.meshgrid(lons, lats)
            
        # For torch tensors and numpy arrays, create evenly spaced grid
        if isinstance(data, torch.Tensor):
            shape = data.shape[-2:]  # Assuming (lat, lon) are last dimensions
        else:
            shape = data.shape[-2:]
            
        lats = np.linspace(lat_range[0], lat_range[1], shape[0])
        lons = np.linspace(lon_range[0], lon_range[1], shape[1])
        return np.meshgrid(lons, lats)
    
    def _prep_data(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def plot_field(
        self,
        data: Union[np.ndarray, torch.Tensor, xr.DataArray],
        title: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
        add_colorbar: bool = True,
        levels: int = 20,
        coastlines: bool = True,
        grid: bool = True,
        var_name: Optional[str] = None,
        center_zero: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a single weather field on a map.
        
        Args:
            data: The data to plot (2D array)
            title: Plot title
            cmap: Colormap (if None, selected based on var_name)
            vmin, vmax: Color scale limits
            ax: Existing axis to plot on
            add_colorbar: Whether to add a colorbar
            levels: Number of contour levels
            coastlines: Whether to add coastlines
            grid: Whether to add gridlines
            var_name: Variable name for automatic colormap selection
            center_zero: Whether to center the colormap around zero
            
        Returns:
            Figure and Axes objects
        """
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = plt.axes(projection=self.projection)
        else:
            fig = ax.figure
        
        # Convert to numpy if needed
        data_np = self._prep_data(data)
        
        # Get lat/lon grid
        lons, lats = self._get_latlons(data)
        
        # Select colormap
        if cmap is None and var_name is not None:
            cmap = self.VAR_CMAPS.get(var_name, 'viridis')
        elif cmap is None:
            cmap = 'viridis'
            
        # Set color limits
        if center_zero and vmin is None and vmax is None:
            abs_max = np.max(np.abs(data_np))
            vmin, vmax = -abs_max, abs_max
        
        # Add map features
        if coastlines:
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            
        if grid:
            ax.gridlines(draw_labels=True)
        
        # Plot data
        cs = ax.contourf(
            lons, lats, data_np, 
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add colorbar
        if add_colorbar:
            label = self.VAR_LABELS.get(var_name, '') if var_name else ''
            plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05, label=label)
        
        # Set title
        if title:
            ax.set_title(title)
            
        return fig, ax
    
    def plot_comparison(
        self,
        true_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        pred_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        var_name: Optional[str] = None,
        level_idx: int = 0,
        title: str = "Prediction Comparison",
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot comparison between true and predicted fields with difference.
        
        Args:
            true_data: True weather state
            pred_data: Predicted weather state
            var_name: Variable name (if data is a dictionary)
            level_idx: Level index to plot
            title: Overall plot title
            save_path: Path to save the figure
            
        Returns:
            Figure and list of Axes objects
        """
        fig = plt.figure(figsize=(18, 6))
        
        # Extract data
        if isinstance(true_data, dict) and isinstance(pred_data, dict):
            if var_name is None:
                # Use first available variable
                var_name = list(true_data.keys())[0]
                
            true_field = true_data[var_name]
            pred_field = pred_data[var_name]
        else:
            true_field = true_data
            pred_field = pred_data
            
        # Convert to numpy
        true_np = self._prep_data(true_field)
        pred_np = self._prep_data(pred_field)
        
        # Select specific level if data has level dimension
        if true_np.ndim > 2:
            true_np = true_np[level_idx]
        if pred_np.ndim > 2:
            pred_np = pred_np[level_idx]
            
        # Calculate difference
        diff = pred_np - true_np
        
        # Get common color scale for true and predicted
        vmin = min(np.min(true_np), np.min(pred_np))
        vmax = max(np.max(true_np), np.max(pred_np))
        
        # Plot true field
        ax1 = fig.add_subplot(1, 3, 1, projection=self.projection)
        self.plot_field(
            true_np, 
            title="True", 
            ax=ax1, 
            var_name=var_name,
            vmin=vmin,
            vmax=vmax
        )
        
        # Plot predicted field
        ax2 = fig.add_subplot(1, 3, 2, projection=self.projection)
        self.plot_field(
            pred_np, 
            title="Predicted", 
            ax=ax2, 
            var_name=var_name,
            vmin=vmin,
            vmax=vmax
        )
        
        # Plot difference
        ax3 = fig.add_subplot(1, 3, 3, projection=self.projection)
        self.plot_field(
            diff, 
            title="Difference", 
            ax=ax3, 
            var_name="difference",
            center_zero=True
        )
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            
        return fig, [ax1, ax2, ax3]

    def plot_prediction_comparison(
        self,
        true_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        pred_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        var_name: Optional[str] = None,
        level_idx: int = 0,
        title: str = "Prediction Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Compare true and predicted fields using the default settings."""

        fig, _ = self.plot_comparison(
            true_data=true_data,
            pred_data=pred_data,
            var_name=var_name,
            level_idx=level_idx,
            title=title,
            save_path=save_path,
        )
        return fig
    
    def plot_error_metrics(
        self,
        true_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        pred_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        var_names: Optional[List[str]] = None,
        title: str = "Prediction Error Analysis",
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot error metrics for predictions.
        
        Args:
            true_data: True weather state
            pred_data: Predicted weather state
            var_names: List of variable names to analyze
            title: Overall plot title
            save_path: Path to save the figure
            
        Returns:
            Figure and list of Axes objects
        """
        # Extract variables to analyze
        if isinstance(true_data, dict) and var_names is None:
            var_names = list(true_data.keys())
        elif var_names is None:
            var_names = ['data']
            
        n_vars = len(var_names)
        
        # Create figure
        fig, axs = plt.subplots(2, n_vars, figsize=(5*n_vars, 10))
        if n_vars == 1:
            axs = axs.reshape(2, 1)
            
        for i, var in enumerate(var_names):
            # Extract data
            if isinstance(true_data, dict) and isinstance(pred_data, dict):
                true_field = true_data[var]
                pred_field = pred_data[var]
            else:
                true_field = true_data
                pred_field = pred_data
                
            # Convert to numpy
            true_np = self._prep_data(true_field)
            pred_np = self._prep_data(pred_field)
            
            # Calculate error
            error = pred_np - true_np
            
            # Plot error histogram
            axs[0, i].hist(error.flatten(), bins=50, density=True)
            axs[0, i].set_title(f"{var} Error Distribution")
            axs[0, i].set_xlabel("Error")
            axs[0, i].set_ylabel("Density")
            
            # Calculate error metrics
            rmse = np.sqrt(np.mean(error**2))
            mae = np.mean(np.abs(error))
            
            # Add text with metrics
            axs[0, i].text(
                0.95, 0.95, 
                f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}",
                transform=axs[0, i].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Plot error vs true value
            axs[1, i].scatter(
                true_np.flatten(), 
                error.flatten(), 
                alpha=0.1, 
                s=1
            )
            axs[1, i].set_title(f"{var} Error vs True Value")
            axs[1, i].set_xlabel("True Value")
            axs[1, i].set_ylabel("Error")
            
            # Add zero line
            axs[1, i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            
        return fig, axs.flatten()

    def plot_error_distribution(
        self,
        true_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        pred_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        var_name: Optional[str] = None,
        title: str = "Prediction Error Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Wrapper returning the error analysis figure expected by the tests."""

        if isinstance(true_data, dict) and var_name is None:
            selected = None
        else:
            selected = [var_name] if var_name is not None else None

        fig, _ = self.plot_error_metrics(
            true_data=true_data,
            pred_data=pred_data,
            var_names=selected,
            title=title,
            save_path=save_path,
        )
        return fig
    
    def plot_global_forecast(
        self,
        forecast_data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        var_name: Optional[str] = None,
        time_index: int = 0,
        title: str = "Global Forecast",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a single forecast map for the provided data."""

        if isinstance(forecast_data, dict):
            if var_name is None:
                var_name = next(iter(forecast_data))
            data = forecast_data[var_name]
        else:
            data = forecast_data

        field = self._prep_data(data)
        if field.ndim > 2:
            field = field[time_index]

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1, projection=self.projection)
        self.plot_field(field, title=title, ax=ax, var_name=var_name)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig

    def create_prediction_animation(
        self,
        predictions: Union[np.ndarray, torch.Tensor, List],
        var_name: Optional[str] = None,
        level_idx: int = 0,
        interval: int = 200,
        title: str = "Weather Prediction",
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """Create animation of weather prediction over time.
        
        Args:
            predictions: Sequence of weather states
            var_name: Variable name for colormap and label
            level_idx: Level index to plot
            interval: Animation interval in milliseconds
            title: Animation title
            save_path: Path to save the animation
            
        Returns:
            Animation object
        """
        # Convert to list of numpy arrays
        if isinstance(predictions, torch.Tensor):
            # Assume shape [time, (batch), channel, lat, lon]
            preds_np = predictions.detach().cpu().numpy()
            # Handle batch dimension if present
            if preds_np.ndim > 4:
                preds_np = preds_np[:, 0]
            # Select variable if multi-channel
            if preds_np.ndim > 3:
                var_idx = 0
                if var_name is not None:
                    var_names = list(self.VAR_LABELS.keys())
                    if var_name in var_names:
                        var_idx = var_names.index(var_name)
                preds_np = preds_np[:, var_idx]
            # Select level if needed
            if preds_np.ndim > 3:
                preds_np = preds_np[:, level_idx]
        elif isinstance(predictions, np.ndarray):
            preds_np = predictions
        else:
            # Assume list of tensors or arrays
            preds_np = [self._prep_data(p) for p in predictions]
            
        # Create figure and initial plot
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=self.projection)
        
        # Get colormap
        cmap = self.VAR_CMAPS.get(var_name, 'viridis')
        
        # Get lat/lon grid
        lons, lats = self._get_latlons(preds_np[0] if isinstance(preds_np, list) else preds_np[0])
        
        # Find global min/max for consistent colormap
        if isinstance(preds_np, list):
            all_data = np.concatenate([p.flatten() for p in preds_np])
        else:
            all_data = preds_np.flatten()
            
        vmin, vmax = np.min(all_data), np.max(all_data)
        
        # Add map features
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.gridlines(draw_labels=True)
        
        # Initial frame
        frame = preds_np[0] if isinstance(preds_np, list) else preds_np[0]
        cont = ax.contourf(
            lons, lats, frame, 
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add colorbar
        label = self.VAR_LABELS.get(var_name, '') if var_name else ''
        plt.colorbar(cont, ax=ax, orientation='horizontal', pad=0.05, label=label)
        
        # Title
        time_title = ax.set_title(f"{title} - Time step 0")
        
        def update(frame_idx):
            """Update function for animation."""
            # Clear previous contours
            for c in cont.collections:
                c.remove()
                
            # Get current frame
            if isinstance(preds_np, list):
                current = preds_np[frame_idx]
            else:
                current = preds_np[frame_idx]
                
            # Plot new frame
            new_cont = ax.contourf(
                lons, lats, current, 
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            
            # Update title
            time_title.set_text(f"{title} - Time step {frame_idx}")
            
            return new_cont.collections
        
        # Create animation
        n_frames = len(preds_np) if isinstance(preds_np, list) else preds_np.shape[0]
        anim = FuncAnimation(
            fig, 
            update, 
            frames=range(n_frames),
            interval=interval, 
            blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5, dpi=100)
            
        return anim
    
    def plot_flow_vectors(
        self,
        u: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        background: Optional[Union[np.ndarray, torch.Tensor]] = None,
        var_name: Optional[str] = None,
        title: str = "Flow Field",
        scale: float = 1.0,
        density: float = 1.0,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot flow vectors (e.g., wind field).
        
        Args:
            u: U-component of the vector field
            v: V-component of the vector field
            background: Optional background field
            var_name: Variable name for background
            title: Plot title
            scale: Vector scale factor
            density: Vector density
            save_path: Path to save the figure
            
        Returns:
            Figure and Axes objects
        """
        # Convert to numpy
        u_np = self._prep_data(u)
        v_np = self._prep_data(v)
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=self.projection)
        
        # Get lat/lon grid
        lons, lats = self._get_latlons(u_np)
        
        # Add map features
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.gridlines(draw_labels=True)
        
        # Plot background if provided
        if background is not None:
            bg_np = self._prep_data(background)
            cmap = self.VAR_CMAPS.get(var_name, 'viridis')
            bg = ax.contourf(
                lons, lats, bg_np, 
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                alpha=0.7
            )
            plt.colorbar(bg, ax=ax, orientation='horizontal', pad=0.05)
        
        # Sub-sample for cleaner plot
        n_lat, n_lon = u_np.shape
        step_lat = max(1, int(n_lat / (30 * density)))
        step_lon = max(1, int(n_lon / (60 * density)))
        
        # Plot vectors
        q = ax.quiver(
            lons[::step_lon, ::step_lat],
            lats[::step_lon, ::step_lat],
            u_np[::step_lat, ::step_lon],
            v_np[::step_lat, ::step_lon],
            transform=ccrs.PlateCarree(),
            scale=50/scale,
            scale_units='inches'
        )
        
        ax.quiverkey(q, 0.9, 0.05, 10, r'$10 \frac{m}{s}$', labelpos='E',
                     coordinates='figure', fontproperties={'size': 10})
        
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            
        return fig, ax
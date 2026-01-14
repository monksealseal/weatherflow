import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    following the classic Wallace & Battisti / textbook atmospheric science
    visualization style with:
    - Global Robinson projection showing the whole world
    - Professional publication-quality colormaps
    - Clear coastlines and gridlines
    - Properly centered diverging colormaps for anomaly fields
    
    The style is inspired by classic atmospheric science textbooks like
    "Atmospheric Science" by Wallace and Hobbs.
    """
    
    VAR_LABELS = {
        'temperature': 'Temperature (K)',
        'geopotential': 'Geopotential Height (m)',
        'geopotential_height': 'Geopotential Height (m)',
        'u_component_of_wind': 'Zonal Wind (m/s)',
        'v_component_of_wind': 'Meridional Wind (m/s)',
        'specific_humidity': 'Specific Humidity (g/kg)',
        'wind_speed': 'Wind Speed (m/s)',
        'vorticity': 'Relative Vorticity (10⁻⁵ s⁻¹)',
        'potential_vorticity': 'Potential Vorticity (PVU)',
        't': 'Temperature (K)',
        'z': 'Geopotential Height (m)',
        'u': 'Zonal Wind (m/s)',
        'v': 'Meridional Wind (m/s)',
        'q': 'Specific Humidity (g/kg)',
        'slp': 'Sea Level Pressure (hPa)',
        'mslp': 'Mean Sea Level Pressure (hPa)',
        'precip': 'Precipitation (mm/day)',
        'omega': 'Vertical Velocity (Pa/s)',
    }
    
    # Wallace & Battisti style colormaps - classic atmospheric science choices
    # Temperature: warm colors for warm, cool colors for cold
    # Geopotential: sequential colormap showing height variations
    # Wind components: diverging colormap centered on zero
    # Humidity: sequential blues for moisture content
    # Vorticity: diverging colormap (cyclonic vs anticyclonic)
    VAR_CMAPS = {
        'temperature': 'coolwarm',
        'geopotential': 'BuPu',
        'geopotential_height': 'BuPu',
        'u_component_of_wind': 'PuOr',
        'v_component_of_wind': 'PuOr',
        'specific_humidity': 'GnBu',
        'wind_speed': 'YlOrRd',
        'vorticity': 'PiYG',
        'potential_vorticity': 'PiYG',
        't': 'coolwarm',
        'z': 'BuPu',
        'u': 'PuOr',
        'v': 'PuOr',
        'q': 'GnBu',
        'slp': 'RdYlBu_r',
        'mslp': 'RdYlBu_r',
        'precip': 'YlGnBu',
        'omega': 'RdBu',
        'error': 'seismic',
        'difference': 'seismic',
        'anomaly': 'seismic',
    }
    
    # Variables that should use diverging colormaps centered on zero
    DIVERGING_VARS = {
        'u', 'v', 'u_component_of_wind', 'v_component_of_wind',
        'vorticity', 'potential_vorticity', 'omega', 
        'error', 'difference', 'anomaly',
    }
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        projection: str = 'Robinson',
        save_dir: Optional[str] = None
    ):
        """Initialize the visualizer with Wallace & Battisti style defaults.
        
        Args:
            figsize: Default figure size for plots (14, 8) for global maps
            projection: Default cartopy projection ('Robinson' for global views)
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.projection = getattr(ccrs, projection)()
        self.save_dir = save_dir
        
        # Set matplotlib style for publication-quality figures
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': 150,
        })
    
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
        levels: int = 15,
        coastlines: bool = False,
        grid: bool = True,
        var_name: Optional[str] = None,
        center_zero: bool = False,
        global_extent: bool = True,
        extend: str = 'both'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a single weather field on a global map.
        
        Uses Wallace & Battisti / textbook atmospheric science style with:
        - Robinson projection for global views
        - Appropriate colormaps for each variable type
        - Clean gridlines and professional formatting
        - Diverging colormaps centered on zero for appropriate variables
        
        Args:
            data: The data to plot (2D array)
            title: Plot title
            cmap: Colormap (if None, selected based on var_name)
            vmin, vmax: Color scale limits
            ax: Existing axis to plot on
            add_colorbar: Whether to add a colorbar
            levels: Number of contour levels (default 15 for cleaner look)
            coastlines: Whether to add coastlines (requires network on first use)
            grid: Whether to add gridlines
            var_name: Variable name for automatic colormap selection
            center_zero: Whether to center the colormap around zero
            global_extent: Whether to show the entire globe
            extend: Colorbar extension style ('both', 'min', 'max', 'neither')
            
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
        
        # Select colormap based on variable type
        if cmap is None and var_name is not None:
            cmap = self.VAR_CMAPS.get(var_name, 'coolwarm')
        elif cmap is None:
            cmap = 'coolwarm'
        
        # Automatically center on zero for diverging variables
        should_center = center_zero or (var_name in self.DIVERGING_VARS)
        
        # Set color limits
        if should_center and vmin is None and vmax is None:
            abs_max = np.nanmax(np.abs(data_np))
            vmin, vmax = -abs_max, abs_max
        elif vmin is None and vmax is None:
            vmin = np.nanmin(data_np)
            vmax = np.nanmax(data_np)
        
        # Create discrete levels for cleaner contours (Wallace & Battisti style)
        contour_levels = np.linspace(vmin, vmax, levels)
        
        # Set global extent to show the whole world
        if global_extent:
            ax.set_global()
        
        # Add map features with Wallace & Battisti styling
        # Note: coastlines and land/ocean features require network access to download
        # the Natural Earth data on first use. Skip silently if unavailable.
        if coastlines:
            try:
                # Try to add coastlines (lower resolution for faster loading)
                ax.coastlines(resolution='110m', color='#333333', linewidth=0.8)
            except Exception:
                pass  # Skip if data unavailable
            # Note: LAND and OCEAN features are skipped as they require additional
            # data downloads that may not be available in all environments
            
        if grid:
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.5,
                color='gray',
                alpha=0.5,
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
        
        # Plot data with filled contours
        cs = ax.contourf(
            lons, lats, data_np, 
            levels=contour_levels,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            extend=extend
        )
        
        # Add contour lines for better readability (classic textbook style)
        ax.contour(
            lons, lats, data_np,
            levels=contour_levels[::2],  # Every other level
            transform=ccrs.PlateCarree(),
            colors='black',
            linewidths=0.3,
            alpha=0.5
        )
        
        # Add colorbar with proper formatting
        if add_colorbar:
            label = self.VAR_LABELS.get(var_name, '') if var_name else ''
            cbar = plt.colorbar(
                cs, ax=ax, 
                orientation='horizontal', 
                pad=0.08,
                shrink=0.8,
                aspect=35,
                extend=extend
            )
            cbar.set_label(label, fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Set title with proper formatting
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            
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
        
        Uses Wallace & Battisti style with three-panel layout showing:
        - True (observed) field
        - Predicted field
        - Difference (prediction - truth) with centered colormap
        
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
        fig = plt.figure(figsize=(16, 5))
        
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
        vmin = min(np.nanmin(true_np), np.nanmin(pred_np))
        vmax = max(np.nanmax(true_np), np.nanmax(pred_np))
        
        # Plot true field
        ax1 = fig.add_subplot(1, 3, 1, projection=self.projection)
        self.plot_field(
            true_np, 
            title="(a) Observed", 
            ax=ax1, 
            var_name=var_name,
            vmin=vmin,
            vmax=vmax
        )
        
        # Plot predicted field
        ax2 = fig.add_subplot(1, 3, 2, projection=self.projection)
        self.plot_field(
            pred_np, 
            title="(b) Predicted", 
            ax=ax2, 
            var_name=var_name,
            vmin=vmin,
            vmax=vmax
        )
        
        # Plot difference with diverging colormap centered on zero
        ax3 = fig.add_subplot(1, 3, 3, projection=self.projection)
        self.plot_field(
            diff, 
            title="(c) Difference (Pred - Obs)", 
            ax=ax3, 
            var_name="difference",
            center_zero=True
        )
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
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
        save_path: Optional[str] = None,
        coastlines: bool = False
    ) -> FuncAnimation:
        """Create animation of weather prediction over time.
        
        Uses Wallace & Battisti style with global Robinson projection,
        appropriate colormaps, and clean presentation.
        
        Args:
            predictions: Sequence of weather states
            var_name: Variable name for colormap and label
            level_idx: Level index to plot
            interval: Animation interval in milliseconds
            title: Animation title
            save_path: Path to save the animation
            coastlines: Whether to add coastlines (requires network on first use)
            
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
        
        # Get colormap (Wallace & Battisti style)
        cmap = self.VAR_CMAPS.get(var_name, 'coolwarm')
        
        # Get lat/lon grid
        lons, lats = self._get_latlons(preds_np[0] if isinstance(preds_np, list) else preds_np[0])
        
        # Find global min/max for consistent colormap
        if isinstance(preds_np, list):
            all_data = np.concatenate([p.flatten() for p in preds_np])
        else:
            all_data = preds_np.flatten()
            
        vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
        levels = np.linspace(vmin, vmax, 15)
        
        # Set global extent
        ax.set_global()
        
        # Add map features (Wallace & Battisti style)
        if coastlines:
            try:
                ax.coastlines(resolution='110m', color='#333333', linewidth=0.8)
            except Exception:
                pass  # Skip if coastline data unavailable
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        
        # Initial frame
        frame = preds_np[0] if isinstance(preds_np, list) else preds_np[0]
        cont = ax.contourf(
            lons, lats, frame, 
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            extend='both'
        )
        
        # Add colorbar
        label = self.VAR_LABELS.get(var_name, '') if var_name else ''
        cbar = plt.colorbar(
            cont, ax=ax, 
            orientation='horizontal', 
            pad=0.08,
            shrink=0.8,
            aspect=35
        )
        cbar.set_label(label, fontsize=10)
        
        # Title
        time_title = ax.set_title(f"{title} - Time step 0", fontsize=13, fontweight='bold')
        
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
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                extend='both'
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
            anim.save(save_path, writer='pillow', fps=5, dpi=150)
            
        return anim
    
    def plot_flow_vectors(
        self,
        u: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        background: Optional[Union[np.ndarray, torch.Tensor]] = None,
        var_name: Optional[str] = None,
        title: str = "Wind Field",
        scale: float = 1.0,
        density: float = 1.0,
        save_path: Optional[str] = None,
        global_extent: bool = True,
        coastlines: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot flow vectors (e.g., wind field) in Wallace & Battisti style.
        
        Creates a global map with wind vectors overlaid on an optional
        background field (e.g., geopotential height or temperature).
        
        Args:
            u: U-component (zonal) of the vector field
            v: V-component (meridional) of the vector field
            background: Optional background field to plot beneath vectors
            var_name: Variable name for background colormap selection
            title: Plot title
            scale: Vector scale factor (larger = shorter arrows)
            density: Vector density (larger = more arrows)
            save_path: Path to save the figure
            global_extent: Whether to show the entire globe
            coastlines: Whether to add coastlines (requires network on first use)
            
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
        
        # Set global extent
        if global_extent:
            ax.set_global()
        
        # Add map features (Wallace & Battisti style)
        if coastlines:
            try:
                ax.coastlines(resolution='110m', color='#333333', linewidth=0.8)
            except Exception:
                pass  # Skip if coastline data unavailable
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot background if provided
        if background is not None:
            bg_np = self._prep_data(background)
            cmap = self.VAR_CMAPS.get(var_name, 'coolwarm')
            vmin, vmax = np.nanmin(bg_np), np.nanmax(bg_np)
            levels = np.linspace(vmin, vmax, 15)
            
            bg = ax.contourf(
                lons, lats, bg_np, 
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                alpha=0.8,
                extend='both'
            )
            cbar = plt.colorbar(
                bg, ax=ax, 
                orientation='horizontal', 
                pad=0.08,
                shrink=0.8,
                aspect=35
            )
            label = self.VAR_LABELS.get(var_name, '') if var_name else ''
            cbar.set_label(label, fontsize=10)
        
        # Sub-sample for cleaner plot (Wallace & Battisti style uses sparse vectors)
        n_lat, n_lon = u_np.shape
        step_lat = max(1, int(n_lat / (25 * density)))
        step_lon = max(1, int(n_lon / (50 * density)))
        
        # Calculate wind speed for coloring vectors
        wind_speed = np.sqrt(u_np**2 + v_np**2)
        
        # Plot vectors with color based on wind speed
        q = ax.quiver(
            lons[::step_lat, ::step_lon],
            lats[::step_lat, ::step_lon],
            u_np[::step_lat, ::step_lon],
            v_np[::step_lat, ::step_lon],
            wind_speed[::step_lat, ::step_lon],
            transform=ccrs.PlateCarree(),
            scale=40/scale,
            scale_units='inches',
            cmap='YlOrRd',
            alpha=0.9,
            width=0.003
        )
        
        # Add reference vector
        ax.quiverkey(
            q, 0.9, -0.1, 10, 
            r'10 m/s', 
            labelpos='E',
            coordinates='axes', 
            fontproperties={'size': 10}
        )
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            
        return fig, ax
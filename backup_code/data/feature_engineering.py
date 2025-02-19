
import numpy as np
import xarray as xr
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler

class WeatherFeatureEngineer:
    """Advanced feature engineering from our successful experiments."""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        
    def create_feature_vectors(
        self,
        data: xr.Dataset,
        variables: List[str],
        spatial_features: bool = True,
        temporal_features: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Create comprehensive feature vectors for ML models."""
        features = []
        
        for var in variables:
            # Basic features
            var_data = data[var].values
            if normalize:
                var_data = self._normalize_variable(var, var_data)
            features.append(var_data)
            
            if spatial_features:
                # Spatial gradients
                gradients = self._compute_spatial_gradients(data[var])
                features.extend(gradients)
                
                # Spatial statistics
                stats = self._compute_spatial_statistics(data[var])
                features.extend(stats)
            
            if temporal_features:
                # Temporal features
                temp_features = self._compute_temporal_features(data[var])
                features.extend(temp_features)
        
        return np.stack(features, axis=-1)
    
    def _normalize_variable(self, var_name: str, data: np.ndarray) -> np.ndarray:
        """Normalize variables using stored scalers."""
        if var_name not in self.scalers:
            self.scalers[var_name] = StandardScaler()
            # Reshape to 2D for scaling
            shape = data.shape
            flat_data = data.reshape(-1, 1)
            self.scalers[var_name].fit(flat_data)
        
        # Scale the data
        shape = data.shape
        flat_data = data.reshape(-1, 1)
        scaled = self.scalers[var_name].transform(flat_data)
        return scaled.reshape(shape)
    
    def _compute_spatial_gradients(self, data: xr.DataArray) -> List[np.ndarray]:
        """Compute spatial gradients and higher-order features."""
        # First-order gradients
        dy, dx = np.gradient(data)
        
        # Gradient magnitude
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Laplacian
        laplacian = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
        
        # Curl (for vector fields)
        if 'vector_component' in data.dims:
            curl = np.gradient(data.sel(vector_component='v'), axis=1) -                    np.gradient(data.sel(vector_component='u'), axis=0)
            return [dx, dy, gradient_mag, laplacian, curl]
        
        return [dx, dy, gradient_mag, laplacian]
    
    def _compute_spatial_statistics(self, data: xr.DataArray) -> List[np.ndarray]:
        """Compute local spatial statistics."""
        # Local statistics with different window sizes
        windows = [3, 5, 7]
        stats = []
        
        for window in windows:
            # Moving average
            mean = self._moving_average(data, window)
            stats.append(mean)
            
            # Local standard deviation
            std = self._moving_std(data, window)
            stats.append(std)
            
            # Local extrema
            min_val = self._moving_min(data, window)
            max_val = self._moving_max(data, window)
            stats.extend([min_val, max_val])
        
        return stats
    
    def _compute_temporal_features(self, data: xr.DataArray) -> List[np.ndarray]:
        """Compute temporal features."""
        # Ensure we have time dimension
        if 'time' not in data.dims:
            return []
        
        features = []
        
        # Rolling statistics
        windows = [6, 12, 24]  # Different time windows
        for window in windows:
            rolling = data.rolling(time=window, center=True)
            features.extend([
                rolling.mean().values,
                rolling.std().values,
                rolling.min().values,
                rolling.max().values
            ])
        
        # Temporal derivatives
        dt = np.gradient(data, axis=data.get_axis_num('time'))
        features.append(dt)
        
        # Acceleration (second derivative)
        d2t = np.gradient(dt, axis=data.get_axis_num('time'))
        features.append(d2t)
        
        return features
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average with given window size."""
        kernel = np.ones((window, window)) / (window * window)
        return self._convolve2d(data, kernel)
    
    def _moving_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving standard deviation."""
        mean = self._moving_average(data, window)
        mean_sq = self._moving_average(data**2, window)
        return np.sqrt(mean_sq - mean**2)
    
    def _moving_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving minimum."""
        from scipy.ndimage import minimum_filter
        return minimum_filter(data, size=window)
    
    def _moving_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving maximum."""
        from scipy.ndimage import maximum_filter
        return maximum_filter(data, size=window)
    
    def _convolve2d(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution with edge handling."""
        from scipy.signal import convolve2d
        return convolve2d(data, kernel, mode='same', boundary='wrap')

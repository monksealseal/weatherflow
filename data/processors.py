
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from ..physics.atmospheric import AtmosphericPhysics

class WeatherDataProcessor:
    """Advanced weather data processing from our successful experiments."""
    
    def __init__(self, pressure_levels=None, grid_spacing=2.5):
        self.physics = AtmosphericPhysics(
            pressure_levels=pressure_levels,
            grid_spacing=grid_spacing
        )
        
    def calculate_derived_variables(self, data: xr.Dataset) -> xr.Dataset:
        """Calculate all derived meteorological variables."""
        derived = data.copy()
        
        # 1. Thermal variables
        if all(v in data.variables for v in ['temperature', 'pressure']):
            # Potential temperature
            derived['theta'] = self.physics.potential_temperature(
                data.temperature, data.pressure
            )
            
            # Static stability
            derived['stability'] = self.physics.static_stability(
                data.temperature, data.pressure
            )
            
            # Temperature advection
            if all(v in data.variables for v in ['u', 'v']):
                derived['temp_advection'] = self._calculate_advection(
                    data.temperature, data.u, data.v
                )
        
        # 2. Dynamic variables
        if all(v in data.variables for v in ['u', 'v']):
            # Vorticity
            derived['vorticity'] = self.physics.vorticity(
                data.u, data.v, data.latitude
            )
            
            # Divergence
            derived['divergence'] = self._calculate_divergence(
                data.u, data.v, data.latitude
            )
            
            # Kinetic energy
            derived['kinetic_energy'] = 0.5 * (data.u**2 + data.v**2)
        
        # 3. Moisture variables
        if 'specific_humidity' in data.variables:
            # Moisture flux
            if all(v in data.variables for v in ['u', 'v']):
                qu = data.specific_humidity * data.u
                qv = data.specific_humidity * data.v
                derived['moisture_flux_magnitude'] = np.sqrt(qu**2 + qv**2)
                
                # Moisture flux convergence
                derived['moisture_flux_convergence'] = -(
                    self._calculate_divergence(qu, qv, data.latitude)
                )
        
        return derived
    
    def extract_temporal_features(self, data: xr.Dataset, window_size: int = 24) -> xr.Dataset:
        """Extract temporal features using our successful methods."""
        features = data.copy()
        
        for var in data.data_vars:
            # Rolling statistics
            rolling = data[var].rolling(time=window_size, center=True)
            features[f"{var}_mean"] = rolling.mean()
            features[f"{var}_std"] = rolling.std()
            
            # Temporal gradients
            features[f"{var}_trend"] = self._calculate_trend(data[var])
            
            # Diurnal components if hourly data
            if 'time' in data[var].dims and data.time.dt.hour.size > 24:
                features[f"{var}_diurnal"] = self._extract_diurnal_component(data[var])
        
        return features
    
    def _calculate_advection(self, scalar, u, v):
        """Calculate horizontal advection of a scalar field."""
        dx = self.physics.dx * np.pi/180 * self.physics.earth_radius
        dy = self.physics.dy * np.pi/180 * self.physics.earth_radius
        
        dsdx = np.gradient(scalar, dx, axis=-1)
        dsdy = np.gradient(scalar, dy, axis=-2)
        
        return -(u * dsdx + v * dsdy)
    
    def _calculate_divergence(self, u, v, lat):
        """Calculate horizontal divergence."""
        dx = self.physics.dx * np.pi/180 * self.physics.earth_radius * np.cos(np.deg2rad(lat))
        dy = self.physics.dy * np.pi/180 * self.physics.earth_radius
        
        dudx = np.gradient(u, dx, axis=-1)
        dvdy = np.gradient(v, dy, axis=-2)
        
        return dudx + dvdy
    
    def _calculate_trend(self, data, window=24):
        """Calculate local trend using linear regression."""
        x = np.arange(window)
        rolling = data.rolling(time=window, center=True)
        
        def fit_slope(y):
            mask = ~np.isnan(y)
            if np.sum(mask) > window//2:
                return np.polyfit(x[mask], y[mask], 1)[0]
            return np.nan
        
        return rolling.reduce(fit_slope)
    
    def _extract_diurnal_component(self, data):
        """Extract diurnal cycle component."""
        # Group by hour and calculate mean
        hourly_mean = data.groupby('time.hour').mean()
        
        # Subtract daily mean to get diurnal anomaly
        daily_mean = hourly_mean.mean('hour')
        return hourly_mean - daily_mean

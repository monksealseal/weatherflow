import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import fsspec
import gcsfs
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

logger = logging.getLogger(__name__)

class ERA5Dataset(Dataset):
    """Enhanced ERA5 dataset with robust data loading from multiple sources.
    
    This dataset supports:
    1. WeatherBench2 data from Google Cloud Storage
    2. Local NetCDF files
    3. Custom Zarr datasets
    
    It handles multiple access methods with robust fallbacks and provides
    detailed error reporting.
    """
    
    VARIABLE_MAP = {
        't': 'temperature',
        'z': 'geopotential',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'q': 'specific_humidity',
        'r': 'relative_humidity'
    }
    
    NORMALIZE_STATS = {
        'temperature': {'mean': 285.0, 'std': 15.0},
        'geopotential': {'mean': 50000.0, 'std': 5000.0},
        'u_component_of_wind': {'mean': 0.0, 'std': 10.0},
        'v_component_of_wind': {'mean': 0.0, 'std': 10.0},
        'specific_humidity': {'mean': 0.005, 'std': 0.005},
        'relative_humidity': {'mean': 0.7, 'std': 0.3}
    }
    
    DEFAULT_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    
    def __init__(
        self,
        variables: List[str] = ['z'],
        pressure_levels: List[int] = [500],
        data_path: Optional[str] = None,
        time_slice: Union[slice, str, Tuple[str, str]] = slice('2015', '2016'),
        normalize: bool = True,
        add_physics_features: bool = False,
        cache_data: bool = False,
        verbose: bool = True
    ):
        """Initialize ERA5Dataset with flexible data loading options.
        
        Args:
            variables: List of variables to load (using short names: 't', 'z', 'u', 'v')
            pressure_levels: List of pressure levels in hPa to include
            data_path: Path to dataset (GCS URL, local zarr, or netCDF)
            time_slice: Time period to load (slice, string, or tuple of strings)
            normalize: Whether to normalize data using predefined statistics
            add_physics_features: Whether to add derived physics-based variables
            cache_data: Whether to cache loaded data in memory (faster but uses more RAM)
            verbose: Whether to print detailed information
        """
        super().__init__()
        
        self.variables = [self.VARIABLE_MAP.get(v, v) for v in variables]
        self.pressure_levels = pressure_levels
        self.data_path = data_path or self.DEFAULT_URL
        self.normalize = normalize
        self.add_physics_features = add_physics_features
        self.cache_data = cache_data
        self.verbose = verbose
        
        # Handle time slice formatting
        if isinstance(time_slice, tuple):
            time_slice = slice(*time_slice)
        self.time_slice = time_slice
        
        # Configure logging
        if self.verbose:
            logger.setLevel(logging.INFO)
        
        # Load data
        self._load_data()
        
        # Cache for faster access
        self._cache = {} if self.cache_data else None
        
    def _log(self, message):
        if self.verbose:
            logger.info(message)
        
    def _load_data(self):
        """Robust data loading with multiple fallback methods."""
        self._log(f"Loading data from: {self.data_path}")
        
        methods = [
            # Method 1: Simple anonymous access
            lambda: {
                'method': xr.open_zarr,
                'args': [self.data_path],
                'kwargs': {
                    'storage_options': {'anon': True},
                    'consolidated': True
                }
            },
            
            # Method 2: Direct HTTP access for GCS paths
            lambda: {
                'method': xr.open_zarr,
                'args': [fsspec.filesystem(
                    'http',
                    client_kwargs={
                        'trust_env': False,
                        'timeout': 30
                    }
                ).get_mapper(self.data_path.replace('gs://', 'https://storage.googleapis.com/'))],
                'kwargs': {'consolidated': True}
            },
            
            # Method 3: GCS anonymous access
            lambda: {
                'method': xr.open_zarr,
                'args': [gcsfs.GCSFileSystem(token='anon').get_mapper(self.data_path)],
                'kwargs': {'consolidated': True}
            },
            
            # Method 4: Local file system access for netCDF
            lambda: {
                'method': xr.open_dataset,
                'args': [self.data_path],
                'kwargs': {'engine': 'netcdf4'}
            },
            
            # Method 5: Local zarr storage
            lambda: {
                'method': xr.open_zarr,
                'args': [self.data_path],
                'kwargs': {}
            }
        ]

        last_exception = None
        for i, method_factory in enumerate(methods):
            try:
                method_info = method_factory()
                self._log(f"Trying method {i+1}: {method_info['method'].__name__}")
                
                self.ds = method_info['method'](
                    *method_info['args'],
                    **method_info['kwargs']
                )
                
                # Standardize dimension names
                dim_map = {
                    'valid_time': 'time',
                    'pressure_level': 'level'
                }
                self.ds = self.ds.rename({old: new for old, new in dim_map.items() 
                                       if old in self.ds.dims})
                
                # Select time period
                self.times = self.ds.time.sel(time=self.time_slice)
                
                self._log(f"Successfully loaded data.")
                self._log(f"Time period: {self.times[0].values} to {self.times[-1].values}")
                self._log(f"Variables: {self.variables}")
                self._log(f"Pressure levels: {self.pressure_levels}")
                
                # Add derived variables if requested
                if self.add_physics_features:
                    self._add_derived_variables()
                
                return  # Success!
                
            except Exception as e:
                last_exception = e
                self._log(f"Method {i+1} failed with error: {str(e)}")
                continue
        
        # If we get here, all methods failed
        raise RuntimeError(f"All methods to load data failed. Last error: {str(last_exception)}")
    
    def _add_derived_variables(self):
        """Add physics-based derived variables."""
        self._log("Adding derived variables...")
        
        # Check if we have wind components
        if all(v in self.ds for v in ['u_component_of_wind', 'v_component_of_wind']):
            # Add wind speed
            self._log("Adding wind_speed variable")
            u = self.ds['u_component_of_wind']
            v = self.ds['v_component_of_wind']
            self.ds['wind_speed'] = np.sqrt(u**2 + v**2)
            
            # Add vorticity if possible
            if 'latitude' in self.ds.dims and 'longitude' in self.ds.dims:
                self._log("Adding vorticity variable")
                try:
                    # Get grid spacing
                    lat_spacing = float(np.diff(self.ds.latitude)[0])
                    lon_spacing = float(np.diff(self.ds.longitude)[0])
                    
                    # Compute gradients - simple approach for demonstration
                    dvdx = u.differentiate('longitude') / lon_spacing
                    dudy = v.differentiate('latitude') / lat_spacing
                    
                    # Relative vorticity
                    self.ds['vorticity'] = dvdx - dudy
                except Exception as e:
                    self._log(f"Failed to compute vorticity: {str(e)}")
        
        # Add more derived variables as needed
    
    def __len__(self) -> int:
        """Return number of samples (time steps - 1 for input/target pairs)."""
        return len(self.times) - 1
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Get a single sample with current and next timestep.
        
        Returns:
            Dictionary containing:
                - 'input': Tensor of shape [n_vars, n_levels, lat, lon]
                - 'target': Tensor of shape [n_vars, n_levels, lat, lon]
                - 'metadata': Dictionary with sample metadata
        """
        # Check if we have this item cached
        if self.cache_data and idx in self._cache:
            return self._cache[idx]
        
        # Get timestamps
        t0 = self.times[idx].values
        t1 = self.times[idx + 1].values
        
        # Get data slices for both timesteps
        data_t0 = {}
        data_t1 = {}
        
        for var in self.variables:
            try:
                # Select data for specific time and pressure levels
                if var in self.ds:
                    if 'level' in self.ds[var].dims:
                        data_t0[var] = self.ds[var].sel(
                            time=t0,
                            level=self.pressure_levels
                        ).values
                        
                        data_t1[var] = self.ds[var].sel(
                            time=t1,
                            level=self.pressure_levels
                        ).values
                    else:
                        # Handle variables without pressure levels
                        data_t0[var] = self.ds[var].sel(time=t0).values
                        data_t1[var] = self.ds[var].sel(time=t1).values
                else:
                    self._log(f"Warning: Variable {var} not found in dataset.")
                    # Use zeros as placeholder
                    shape = (len(self.pressure_levels), 
                             self.ds.latitude.size, 
                             self.ds.longitude.size)
                    data_t0[var] = np.zeros(shape)
                    data_t1[var] = np.zeros(shape)
                
                # Normalize if requested
                if self.normalize and var in self.NORMALIZE_STATS:
                    stats = self.NORMALIZE_STATS[var]
                    data_t0[var] = (data_t0[var] - stats['mean']) / stats['std']
                    data_t1[var] = (data_t1[var] - stats['mean']) / stats['std']
                    
            except Exception as e:
                self._log(f"Error processing variable {var}: {str(e)}")
                # Use zeros as placeholder
                shape = (len(self.pressure_levels), 
                         self.ds.latitude.size, 
                         self.ds.longitude.size)
                data_t0[var] = np.zeros(shape)
                data_t1[var] = np.zeros(shape)
        
        # Convert to tensors
        input_data = torch.tensor(np.stack([data_t0[var] for var in self.variables]))
        target_data = torch.tensor(np.stack([data_t1[var] for var in self.variables]))
        
        result = {
            'input': input_data.float(),
            'target': target_data.float(),
            'metadata': {
                't0': str(t0),
                't1': str(t1),
                'variables': self.variables,
                'pressure_levels': self.pressure_levels
            }
        }
        
        # Cache if requested
        if self.cache_data:
            self._cache[idx] = result
            
        return result
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the data (n_vars, n_levels, lat, lon)."""
        return (len(self.variables), len(self.pressure_levels), 
                self.ds.latitude.size, self.ds.longitude.size)
    
    def get_coords(self) -> Dict[str, np.ndarray]:
        """Return coordinate arrays for the dataset."""
        return {
            'latitude': self.ds.latitude.values,
            'longitude': self.ds.longitude.values,
            'pressure_levels': np.array(self.pressure_levels)
        }


def create_data_loaders(
    variables: List[str] = ['z'],
    pressure_levels: List[int] = [500],
    data_path: str = None,
    train_slice: Union[slice, Tuple[str, str]] = ('2015', '2016'),
    val_slice: Union[slice, Tuple[str, str]] = ('2017', '2017'),
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        variables: List of variables to load (using short names: 't', 'z', 'u', 'v')
        pressure_levels: List of pressure levels in hPa
        data_path: Path to dataset (defaults to WeatherBench2 GCS path)
        train_slice: Time period for training data
        val_slice: Time period for validation data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = ERA5Dataset(
        variables=variables,
        pressure_levels=pressure_levels,
        data_path=data_path,
        time_slice=train_slice,
        normalize=normalize
    )
    
    val_dataset = ERA5Dataset(
        variables=variables,
        pressure_levels=pressure_levels,
        data_path=data_path,
        time_slice=val_slice,
        normalize=normalize
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
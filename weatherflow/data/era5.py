
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import fsspec
import os
from typing import List, Dict, Tuple, Union

class ERA5Dataset(Dataset):
    """Enhanced ERA5 dataset with support for both WeatherBench2 and local files."""
    
    VARIABLE_MAP = {
        't': 'temperature',
        'z': 'geopotential',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind'
    }
    
    NORMALIZE_STATS = {
        'temperature': {'mean': 285.0, 'std': 15.0},
        'geopotential': {'mean': 50000.0, 'std': 5000.0},
        'u_component_of_wind': {'mean': 0.0, 'std': 10.0},
        'v_component_of_wind': {'mean': 0.0, 'std': 10.0}
    }
    
    def __init__(
        self,
        variables: List[str] = ['z'],
        pressure_levels: List[int] = [500],
        data_dir: str = 'weatherbench2/datasets/era5',
        year: Union[int, str] = 2020,
        local_file_path: str = None,
        normalize: bool = True,
        add_physics_features: bool = True
    ):
        super().__init__()
        self.variables = [self.VARIABLE_MAP.get(v, v) for v in variables]
        self.pressure_levels = pressure_levels
        self.data_dir = data_dir
        self.year = year
        self.local_file_path = local_file_path
        self.normalize = normalize
        self.add_physics_features = add_physics_features
        
        # Load data
        if self.local_file_path:
            self._load_local_data()
        else:
            self._load_wb2_data()
            
        # Add physics-based features if requested
        if self.add_physics_features:
            self._add_derived_variables()
    
    def _load_wb2_data(self):
        """Load data from WeatherBench2 GCS."""
        try:
            path = f'gs://{self.data_dir}/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
            mapper = fsspec.get_mapper(path)
            ds = xr.open_zarr(mapper, consolidated=True)
            
            # Select time period
            start_date = f'{self.year}-01-01'
            end_date = f'{self.year}-12-31'
            self.ds = ds.sel(time=slice(start_date, end_date))
            
            print(f"WeatherBench2 data loaded: {self.ds.time[0].values} to {self.ds.time[-1].values}")
            
        except Exception as e:
            print(f"Failed to load WeatherBench2 dataset: {str(e)}")
            raise
    
    def _load_local_data(self):
        """Load data from local NetCDF file."""
        if not os.path.exists(self.local_file_path):
            raise FileNotFoundError(f"Local file not found: {self.local_file_path}")
        
        try:
            self.ds = xr.open_dataset(self.local_file_path, engine='netcdf4')
            
            # Standardize dimension names
            dim_map = {
                'valid_time': 'time',
                'pressure_level': 'level'
            }
            self.ds = self.ds.rename({old: new for old, new in dim_map.items() 
                                    if old in self.ds.dims})
            
            # Select time period
            if isinstance(self.year, (int, str)):
                start_date = f'{self.year}-01-01'
                end_date = f'{self.year}-12-31'
                self.ds = self.ds.sel(time=slice(start_date, end_date))
                
            print(f"Local ERA5 data loaded from {self.local_file_path}")
            print(f"Time period: {self.ds.time[0].values} to {self.ds.time[-1].values}")
            
        except Exception as e:
            print(f"Error loading local dataset: {str(e)}")
            raise
    
    def _add_derived_variables(self):
        """Add physics-based derived variables."""
        if all(v in self.ds for v in ['u_component_of_wind', 'v_component_of_wind']):
            # Add wind speed
            u = self.ds['u_component_of_wind']
            v = self.ds['v_component_of_wind']
            self.ds['wind_speed'] = np.sqrt(u**2 + v**2)
            
            # Add vorticity
            dx = 2.5  # grid spacing in degrees
            dvdx = u.differentiate('longitude') / dx
            dudy = v.differentiate('latitude') / dx
            self.ds['vorticity'] = dvdx - dudy
    
    def __len__(self) -> int:
        return len(self.ds.time) - 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input-target pair with optional normalization."""
        data_t0 = []
        data_t1 = []
        
        for var in self.variables:
            for level in self.pressure_levels:
                # Get data at t and t+1
                if 'level' in self.ds[var].dims:
                    x0 = self.ds[var].sel(level=level).isel(time=idx).values
                    x1 = self.ds[var].sel(level=level).isel(time=idx + 1).values
                else:
                    x0 = self.ds[var].isel(time=idx).values
                    x1 = self.ds[var].isel(time=idx + 1).values
                
                # Normalize if requested
                if self.normalize and var in self.NORMALIZE_STATS:
                    stats = self.NORMALIZE_STATS[var]
                    x0 = (x0 - stats['mean']) / stats['std']
                    x1 = (x1 - stats['mean']) / stats['std']
                
                data_t0.append(x0)
                data_t1.append(x1)
        
        # Stack all variables and levels
        x0 = np.stack(data_t0)
        x1 = np.stack(data_t1)
        
        return torch.from_numpy(x0).float(), torch.from_numpy(x1).float()

def create_data_loaders(
    variables: List[str] = ['z'],
    pressure_levels: List[int] = [500],
    data_dir: str = 'weatherbench2/datasets/era5',
    local_file_path: str = None,
    train_year: int = 2020,
    val_year: int = 2021,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Create datasets
    train_dataset = ERA5Dataset(
        variables=variables,
        pressure_levels=pressure_levels,
        data_dir=data_dir,
        local_file_path=local_file_path,
        year=train_year
    )
    
    val_dataset = ERA5Dataset(
        variables=variables,
        pressure_levels=pressure_levels,
        data_dir=data_dir,
        local_file_path=local_file_path,
        year=val_year
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

class WeatherDataset(ERA5Dataset):
    """Alias for ERA5Dataset for backward compatibility."""
    pass

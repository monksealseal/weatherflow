import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
import fsspec

class WeatherDataset:
    """Dataset class for loading weather data from HDF5 files."""

    def __init__(self, data_path: str, variables: List[str]):
        self.data_path = Path(data_path)
        self.variables = variables
        self._load_data()

    def _load_data(self):
        self.data = {}
        for var in self.variables:
            file_path = self.data_path / f"{var}_train.h5"
            if file_path.exists():
                with h5py.File(file_path, "r") as f:
                    self.data[var] = np.array(f[var])

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {var: data[idx] for var, data in self.data.items()}

class ERA5Dataset(Dataset):
    """Dataset class for loading ERA5 reanalysis data from WeatherBench 2."""
    
    VARIABLE_MAPPING = {
        't': 'temperature',
        'z': 'geopotential',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind'
    }
    
    DEFAULT_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: str = "64x32",
        variables: List[str] = ['t', 'z'],
        pressure_levels: List[int] = [850, 700, 500, 300, 200],
        time_slice: Union[slice, str, Tuple[str, str]] = slice('2015', '2016')
    ):
        """
        Initialize ERA5Dataset.
        
        Args:
            data_path: Optional path to zarr dataset. If None, uses default GCS path
            resolution: Spatial resolution of the data ('64x32' or '128x64')
            variables: List of variables to load (using short names: 't', 'z', 'u', 'v')
            pressure_levels: List of pressure levels in hPa to include
            time_slice: Time period to load (slice, string, or tuple of strings)
        """
        super().__init__()
        
        # Set up data path
        self.data_path = data_path or self.DEFAULT_URL
        if isinstance(time_slice, tuple):
            time_slice = slice(*time_slice)
            
        # Map short variable names to full names
        self.variables = [self.VARIABLE_MAPPING[v] for v in variables]
        self.pressure_levels = pressure_levels
        
        print(f"Loading data from: {self.data_path}")
        self._load_data(time_slice)
        
    def _load_data(self, time_slice: slice):
        """Load the dataset and select time period."""
        try:
            self.ds = xr.open_zarr(self.data_path)
            self.times = self.ds.time.sel(time=time_slice)
            print(f"Selected time period: {self.times[0].values} to {self.times[-1].values}")
            print(f"Variables: {self.variables}")
            print(f"Pressure levels: {self.pressure_levels}")
            
        except Exception as e:
            print(f"Error loading ERA5 data: {str(e)}")
            raise
    
    def __len__(self) -> int:
        """Return number of samples (time steps - 1 for input/target pairs)."""
        return len(self.times) - 1
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Get a single sample with current and next timestep.
        
        Returns:
            Dictionary containing:
                - 'input': Tensor of shape [n_vars, n_levels, lat, lon]
                - 'target': Tensor of shape [n_vars, n_levels, lat, lon]
                - 'metadata': Dictionary with sample metadata
        """
        t0 = self.times[idx].values
        t1 = self.times[idx + 1].values
        
        # Get data slices for both timesteps
        data_t0 = {}
        data_t1 = {}
        
        for var in self.variables:
            # Select data for specific time and pressure levels
            data_t0[var] = self.ds[var].sel(
                time=t0,
                level=self.pressure_levels
            ).values
            
            data_t1[var] = self.ds[var].sel(
                time=t1,
                level=self.pressure_levels
            ).values
        
        # Convert to tensors
        input_data = torch.tensor(np.stack([data_t0[var] for var in self.variables]))
        target_data = torch.tensor(np.stack([data_t1[var] for var in self.variables]))
        
        return {
            'input': input_data,
            'target': target_data,
            'metadata': {
                't0': t0,
                't1': t1,
                'variables': self.variables,
                'pressure_levels': self.pressure_levels
            }
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the data (n_vars, n_levels, lat, lon)."""
        return (len(self.variables), len(self.pressure_levels), 
                self.ds.latitude.size, self.ds.longitude.size)

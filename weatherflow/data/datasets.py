import torch
from torch.utils.data import Dataset  # This line imports Dataset from PyTorch
import xarray as xr
import numpy as np
import h5py
import fsspec
import gcsfs
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

class WeatherDataset:
    """Dataset class for loading weather data from HDF5 files."""

    def __init__(self, data_path: str, variables: List[str]):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            variables: List of variable names to load
        """
        self.data_path = Path(data_path)
        self.variables = variables
        self._load_data()

    def _load_data(self):
        """Load data from HDF5 files."""
        self.data = {}
        for var in self.variables:
            file_path = self.data_path / f"{var}_train.h5"
            if file_path.exists():
                with h5py.File(file_path, "r") as f:
                    self.data[var] = np.array(f[var])
            else:
                print(f"Warning: File {file_path} not found.")

    def __len__(self) -> int:
        """Return the number of samples."""
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a sample from the dataset."""
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
        """Initialize ERA5Dataset."""
        super().__init__()
        
        self.data_path = data_path or self.DEFAULT_URL
        if isinstance(time_slice, tuple):
            time_slice = slice(*time_slice)
            
        self.variables = [self.VARIABLE_MAPPING[v] for v in variables]
        self.pressure_levels = pressure_levels
        
        print(f"Loading data from: {self.data_path}")
        self._load_data(time_slice)
        
    def _load_data(self, time_slice: slice):
        """Load the dataset and select time period."""
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
            
            # Method 2: Direct HTTP access
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
            
            # Method 4: Configured timeouts
            lambda: {
                'method': xr.open_zarr,
                'args': [self.data_path],
                'kwargs': {
                    'storage_options': {
                        'anon': True,
                        'timeout': 30,
                        'retries': 10,
                        'default_fill_cache': False
                    },
                    'consolidated': True
                }
            }
        ]

        last_exception = None
        for method_factory in methods:
            try:
                method_info = method_factory()
                logging.info(f"Attempting to open dataset with method: {method_info}")
                self.ds = method_info['method'](
                    *method_info['args'],
                    **method_info['kwargs']
                )
                self.times = self.ds.time.sel(time=time_slice)
                print(f"Selected time period: {self.times[0].values} to {self.times[-1].values}")
                print(f"Variables: {self.variables}")
                print(f"Pressure levels: {self.pressure_levels}")
                return  # Success! Exit the method
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Method failed with error: {str(e)}")
                continue
        
        # If we get here, all methods failed
        raise RuntimeError(f"All methods to load data failed. Last error: {str(last_exception)}")

    def __len__(self) -> int:
        """Return number of samples (time steps - 1 for input/target pairs)."""
        return len(self.times) - 1
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Get a single sample with current and next timestep."""
        t0 = self.times[idx].values
        t1 = self.times[idx + 1].values
        
        data_t0 = {}
        data_t1 = {}
        
        for var in self.variables:
            data_t0[var] = self.ds[var].sel(
                time=t0,
                level=self.pressure_levels
            ).values
            
            data_t1[var] = self.ds[var].sel(
                time=t1,
                level=self.pressure_levels
            ).values
        
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

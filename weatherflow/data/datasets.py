import xarray as xr
import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
import fsspec

class WeatherDataset:
    """Dataset class for loading weather data from HDF5 files."""
    
    def __init__(self, data_path: str, variables: List[str]):
        """
        Initialize WeatherDataset.
        
        Args:
            data_path: Path to data directory
            variables: List of variables to load (e.g., ["temperature"])
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
    
    def __len__(self):
        return len(next(iter(self.data.values())))
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {var: data[idx] for var, data in self.data.items()}

class ERA5Dataset:
    """Dataset class for loading ERA5 reanalysis data from WeatherBench 2."""
    
    DEFAULT_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: str = "64x32",
        variables: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ):
        """
        Initialize ERA5Dataset.
        
        Args:
            data_path: Path to ERA5 data. Can be local path or GCS URL.
                      If None, uses default WeatherBench 2 URL.
            resolution: One of "64x32", "240x121", "1440x721"
            variables: List of variables to load. If None, loads all.
            years: List of years to load. If None, loads all.
        """
        self.data_path = data_path or self.DEFAULT_URL
        self.variables = variables
        self.years = years
        self._load_data()
    
    def _load_data(self):
        """Load ERA5 data from local or GCS path."""
        try:
            # Handle both local and GCS paths
            if self.data_path.startswith("gs://"):
                mapper = fsspec.get_mapper(self.data_path)
                ds = xr.open_zarr(mapper)
            else:
                ds = xr.open_zarr(self.data_path)
            
            # Filter by variables if specified
            if self.variables:
                ds = ds[self.variables]
            
            # Filter by years if specified
            if self.years:
                ds = ds.sel(
                    time=ds.time.dt.year.isin(self.years)
                )
            
            self.data = ds
            
        except Exception as e:
            print(f"Error loading ERA5 data: {str(e)}")
            self.data = None
    
    def __len__(self):
        return len(self.data.time) - 1 if self.data is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise ValueError("No data loaded")
            
        current = self.data.isel(time=idx)
        next_state = self.data.isel(time=idx + 1)
        
        return current.to_array().values, next_state.to_array().values

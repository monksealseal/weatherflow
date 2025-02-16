import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from pathlib import Path
import h5py

class WeatherDataset(Dataset):
    """Dataset for weather prediction with multiple variables"""
    def __init__(self, root_dir, mode='train', variables=['temperature', 'pressure'], 
                 sequence_length=24, stride=6):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.variables = variables
        self.sequence_length = sequence_length
        self.stride = stride
        self._load_data()
    
    def _load_data(self):
        self.data = {}
        for var in self.variables:
            file_path = self.root_dir / f"{var}_{self.mode}.h5"
            with h5py.File(file_path, 'r') as f:
                self.data[var] = f[var][:]
        self.valid_indices = range(0, len(self.data[self.variables[0]]) - self.sequence_length, 
                                 self.stride)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        x = {var: torch.FloatTensor(self.data[var][start_idx:start_idx+self.sequence_length])
             for var in self.variables}
        if self.mode == 'train':
            return ({var: x[var][:-1] for var in self.variables},
                    {var: x[var][1:] for var in self.variables})
        return x

class ERA5Dataset(Dataset):
    """Dataset specifically for ERA5 reanalysis data"""
    def __init__(self, data_path, years=range(1979, 2022), variables=['z500']):
        self.data_path = Path(data_path)
        self.years = years
        self.variables = variables
        self._load_era5_data()
    
    def _load_era5_data(self):
        datasets = []
        for year in self.years:
            try:
                ds = xr.open_dataset(self.data_path / f"era5_{year}.nc")
                datasets.append(ds[self.variables])
            except FileNotFoundError:
                print(f"Warning: No data found for year {year}")
        self.data = xr.concat(datasets, dim='time')
    
    def __len__(self):
        return len(self.data.time) - 1
    
    def __getitem__(self, idx):
        current = {var: torch.FloatTensor(self.data[var].isel(time=idx).values)
                  for var in self.variables}
        next_step = {var: torch.FloatTensor(self.data[var].isel(time=idx+1).values)
                    for var in self.variables}
        return current, next_step

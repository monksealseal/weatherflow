import pytest
import numpy as np
from weatherflow.data import WeatherDataset, ERA5Dataset
from pathlib import Path

@pytest.fixture
def sample_data_path(tmp_path):
    # Create temporary data for testing
    data_dir = tmp_path / "weather_data"
    data_dir.mkdir()
    return data_dir

def test_weather_dataset(sample_data_path):
    # Create sample data file
    import h5py
    with h5py.File(sample_data_path / "temperature_train.h5", "w") as f:
        f.create_dataset("temperature", data=np.random.randn(100, 32, 64))
    
    dataset = WeatherDataset(sample_data_path, variables=["temperature"])
    assert len(dataset) > 0
    
    # Test data loading
    sample = dataset[0]
    assert isinstance(sample, dict) or isinstance(sample, tuple)

def test_era5_dataset(sample_data_path):
    import xarray as xr
    
    # Create sample ERA5 file
    ds = xr.Dataset(
        {
            "z500": (("time", "latitude", "longitude"), 
                    np.random.randn(10, 32, 64))
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=10),
            "latitude": np.linspace(-90, 90, 32),
            "longitude": np.linspace(-180, 180, 64)
        }
    )
    ds.to_netcdf(sample_data_path / "era5_2020.nc")
    
    dataset = ERA5Dataset(sample_data_path, years=[2020])
    assert len(dataset) > 0
    
    # Test data loading
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2  # current and next state

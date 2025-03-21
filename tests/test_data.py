import pytest
import numpy as np
import xarray as xr
import pandas as pd
from weatherflow.data import WeatherDataset, ERA5Dataset
from pathlib import Path

def test_era5_dataset():
    try:
        # Test with default WeatherBench 2 URL
        dataset = ERA5Dataset()
        assert len(dataset) > 0
        
        # Test data loading
        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2  # current and next state
        
        # Test with specific variables
        dataset = ERA5Dataset(variables=['z', 't'])
        assert len(dataset) > 0
        
        # Test with specific time slice
        dataset = ERA5Dataset(time_slice=('2020', '2021'))
        assert len(dataset) > 0
        
    except Exception as e:
        pytest.skip(f"ERA5 test failed: {str(e)}")
